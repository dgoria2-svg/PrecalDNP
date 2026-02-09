package com.dg.precaldnp.vision

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * Detecta la guía impresa:
 *  - Círculo Ø100 mm -> escala px/mm y centro (C).
 *  - Barra 100 mm    -> orientación (θ) “centro -> barra”.
 *
 * Devuelve Ref75 con:
 *  - pxPerMm (escala media),
 *  - pxPerMmX, pxPerMmY (escalas efectivas en X/Y),
 *  - (cx, cy), rPx,
 *  - thetaPageDeg (0° = +X imagen; sentido CCW),
 *  - barMid (opcional), barLenMm (opcional),
 *  - coverageCircle (soporte de borde), sharpness, confidence.
 *
 * Invariante a rotación de la foto (mientras se vea el patrón).
 */
object RefCalibrator75 {

    data class Ref75(
        val pxPerMm: Float,        // escala media (círculo Ø100)
        val cx: Float,
        val cy: Float,
        val rPx: Float,
        val thetaPageDeg: Float,   // 0° = +X imagen; CCW; vector C -> barMid
        val barMidX: Float?,       // puede ser null si no se halló barra
        val barMidY: Float?,
        val barLenMm: Float?,      // longitud de la barra en mm (según escala X)
        val coverageCircle: Float, // 0..1: % de muestras del borde presentes
        val sharpness: Float,      // var(Laplaciano)
        val confidence: Float,     // 0..1: heurística simple
        val pxPerMmX: Float,       // escala en X (horizontal/barra)
        val pxPerMmY: Float        // escala en Y (vertical)
    )

    private const val TAG = "RefCalibrator75"

    fun detect(bmp: Bitmap): Ref75? {
        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "OpenCV initLocal() = false")
            return null
        }

        val w = bmp.width
        val h = bmp.height

        // --- Bitmap -> Mat, prepro ---
        val rgba = Mat(h, w, CvType.CV_8UC4)
        Utils.bitmapToMat(bmp, rgba)

        val gray = Mat()
        Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)

        // CLAHE suave para contraste estable
        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        clahe.apply(gray, gray)

        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)

        // Canny
        val edges = Mat()
        Imgproc.Canny(gray, edges, 50.0, 150.0)

        // ---------------- CÍRCULO Ø100 ----------------
        val minDim = min(w, h).toDouble()
        val minR = (0.15 * minDim).toInt()
        val maxR = (0.50 * minDim).toInt()

        val circles = Mat()
        Imgproc.HoughCircles(
            gray, circles,
            Imgproc.HOUGH_GRADIENT,
            1.2,                   // dp
            minDim / 4.0,          // minDist entre centros
            120.0, 40.0,           // param1 (Canny alto), param2 (acumulador)
            minR, maxR
        )

        if (circles.empty()) {
            Log.e(TAG, "No se detectó círculo")
            releaseAll(rgba, gray, edges, circles)
            return null
        }

        // Elegimos el círculo con mejor "coverage" de borde + cercanía al centro
        var bestScore = Double.NEGATIVE_INFINITY
        var bestC = Triple(0.0, 0.0, 0.0)

        for (i in 0 until circles.cols()) {
            val data = circles.get(0, i) ?: continue
            val cx = data[0]
            val cy = data[1]
            val r = data[2]
            val cov = circleEdgeCoverage(edges, cx, cy, r)
            val centerBias = 1.0 - (hypot(cx - w / 2.0, cy - h / 2.0) / (minDim / 2.0)).coerceIn(0.0, 1.0)
            val score = 0.8 * cov + 0.2 * centerBias
            if (score > bestScore) {
                bestScore = score
                bestC = Triple(cx, cy, r)
            }
        }

        val (cxD, cyD, rPxD) = bestC
        val coverage = circleEdgeCoverage(edges, cxD, cyD, rPxD)

        // Escala media desde el aro Ø100
        val pxPerMmCircle = (2.0 * rPxD / 100.0).toFloat()

        // nitidez global
        val sharp = laplacianVar(gray)

        // ---------------- BARRA 100 (debajo del círculo) ----------------
        val barTriple = detectBar(edges, cxD, cyD, rPxD)
        val barMidX: Double?
        val barMidY: Double?
        val barLenPx: Double?

        if (barTriple != null) {
            barMidX = barTriple.first
            barMidY = barTriple.second
            barLenPx = barTriple.third
        } else {
            barMidX = null
            barMidY = null
            barLenPx = null
        }

        // Escalas X/Y:
        // - X: basada directamente en la barra 100 mm.
        // - Y: ajustada para no alejarse demasiado de la escala del aro.
        var pxPerMmX = pxPerMmCircle
        var pxPerMmY = pxPerMmCircle

        if (barLenPx != null && barLenPx > 1.0) {
            val pxPerMmBar = (barLenPx / 100.0).toFloat()   // px/mm a lo largo de la barra 100 mm
            pxPerMmX = pxPerMmBar
            pxPerMmY = (2f * pxPerMmCircle - pxPerMmX).coerceAtLeast(pxPerMmCircle * 0.5f)
        }

        // Longitud de barra en mm usando la escala de X
        val barLenMm = if (barLenPx != null && pxPerMmX > 0f) {
            (barLenPx.toFloat() / pxPerMmX)
        } else {
            null
        }

        // θ página: vector desde C -> barMid (si no hay barra, 0°)
        val theta = if (barMidX != null && barMidY != null) {
            Math.toDegrees(atan2(barMidY - cyD, barMidX - cxD)).toFloat()
        } else {
            0f
        }

        // Heurística de confianza:
        // - coverage del círculo
        // - proximidad de la barra a 100 mm
        val confCircle = coverage.coerceIn(0.0, 1.0)
        val confBar = if (barLenMm != null) {
            (1.0 - abs(barLenMm - 100.0) / 100.0).coerceIn(0.0, 1.0)
        } else {
            0.0
        }
        val conf = (0.7 * confCircle + 0.3 * confBar).toFloat()

        releaseAll(rgba, gray, edges, circles)

        return Ref75(
            pxPerMm = pxPerMmCircle,
            cx = cxD.toFloat(),
            cy = cyD.toFloat(),
            rPx = rPxD.toFloat(),
            thetaPageDeg = theta,
            barMidX = barMidX?.toFloat(),
            barMidY = barMidY?.toFloat(),
            barLenMm = barLenMm,
            coverageCircle = coverage.toFloat(),
            sharpness = sharp,
            confidence = conf,
            pxPerMmX = pxPerMmX,
            pxPerMmY = pxPerMmY
        )
    }

    // ---------- helpers ----------

    private fun circleEdgeCoverage(edges: Mat, cx: Double, cy: Double, r: Double): Double {
        val N = 360
        var hit = 0
        for (k in 0 until N) {
            val th = 2.0 * Math.PI * k / N
            val x = (cx + r * cos(th)).roundToInt().coerceIn(0, edges.cols() - 1)
            val y = (cy + r * sin(th)).roundToInt().coerceIn(0, edges.rows() - 1)
            if ((edges.get(y, x)?.get(0)?.toInt() ?: 0) > 0) hit++
        }
        return hit.toDouble() / N.toDouble()
    }

    private fun laplacianVar(gray: Mat): Float {
        val lap = Mat()
        Imgproc.Laplacian(gray, lap, CvType.CV_64F)
        val mean = MatOfDouble()
        val std = MatOfDouble()
        Core.meanStdDev(lap, mean, std)
        val sigma = std.toArray()[0]
        lap.release(); mean.release(); std.release()
        return (sigma * sigma).toFloat()
    }

    /**
     * Detecta la barra con HoughLinesP y devuelve (midX, midY, lengthPx).
     * Ajuste fino para tu PDF: barra HORIZONTAL y DEBAJO del círculo.
     */
    private fun detectBar(edges: Mat, cx: Double, cy: Double, r: Double): Triple<Double, Double, Double>? {
        val lines = Mat()
        Imgproc.HoughLinesP(
            edges,
            lines,
            1.0, Math.PI / 180.0,
            80,                 // threshold (más permisivo)
            r * 1.1,            // minLineLength ~ barra 100mm ≈ diámetro del círculo
            20.0                // maxLineGap
        )
        if (lines.empty()) {
            lines.release()
            return null
        }

        var best: Triple<Double, Double, Double>? = null
        var bestScore = Double.NEGATIVE_INFINITY

        for (i in 0 until lines.rows()) {
            val l = lines.get(i, 0) ?: continue
            val x1 = l[0]; val y1 = l[1]; val x2 = l[2]; val y2 = l[3]

            val dx = x2 - x1
            val dy = y2 - y1
            val len = hypot(dx, dy)
            if (len <= 1.0) continue

            val mx = (x1 + x2) / 2.0
            val my = (y1 + y2) / 2.0

            // barra casi horizontal
            val horizOk = abs(dy) <= 0.15 * len

            // barra debajo del círculo
            val belowOk = my >= (cy + 0.35 * r)

            // distancia radial del midpoint a C (solo como filtro suave)
            val dC = hypot(mx - cx, my - cy)
            val radialOk = dC in (0.9 * r)..(3.0 * r)

            val wHoriz = if (horizOk) 1.0 else 0.35
            val wBelow = if (belowOk) 1.0 else 0.25
            val wRadial = if (radialOk) 1.0 else 0.6

            val score = len * wHoriz * wBelow * wRadial

            if (score > bestScore) {
                bestScore = score
                best = Triple(mx, my, len)
            }
        }

        lines.release()
        return best
    }

    private fun releaseAll(vararg mats: Mat) {
        for (m in mats) m.release()
    }
}
