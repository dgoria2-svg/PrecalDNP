package com.dg.precaldnp.vision
import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
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
 * RefCalibrator75 (3250)
 *
 * Versión SIN ArUco-Contrib:
 * - Usa la hoja aprobada con "ArUco" pero los detecta como CUADRADOS por contornos.
 * - Con 4 marcadores (uno por cuadrante) calcula homografía y rectifica el plano.
 * - Luego corre tu detección actual (círculo Ø100 + barra 100) sobre la imagen rectificada.
 *
 * Ventaja: no requiere org.opencv.aruco.* (OpenCV contrib).
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

    private const val ARUCO_SQUARE_SIDE_MM = 150.0
    private const val BAR_LEN_MM = 100.0
    private const val CIRCLE_DIAM_MM = 100.0

    // Canvas rectificado nominal (solo para warp estable; la escala real la calculás con círculo/barra)
    private const val RECT_PX_PER_MM_NOM = 10.0

    fun detect(bmp: Bitmap): Ref75? {
        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "OpenCV initLocal() = false")
            return null
        }

        val w = bmp.width
        val h = bmp.height

        // --- Bitmap -> Mat ---
        val rgba = Mat(h, w, CvType.CV_8UC4)
        Utils.bitmapToMat(bmp, rgba)

        // --- Intento rectificar por 4 "cuadrados" (ArUco-as-squares). Si falla, seguimos legacy. ---
        val rect = tryRectifyWithSquares3250(rgba)
        val rgbaWork = rect?.rgbaRect ?: rgba
        val Hinv = rect?.Hinv

        val wWork = rgbaWork.cols()
        val hWork = rgbaWork.rows()

        // --- Gray + prepro ---
        val gray = Mat()
        Imgproc.cvtColor(rgbaWork, gray, Imgproc.COLOR_RGBA2GRAY)

        // CLAHE suave para contraste estable
        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        clahe.apply(gray, gray)
        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)

        // Canny
        val edges = Mat()
        Imgproc.Canny(gray, edges, 50.0, 150.0)

        // ---------------- CÍRCULO Ø100 ----------------
        val minDim = min(wWork, hWork).toDouble()
        val minR = (0.15 * minDim).toInt()
        val maxR = (0.50 * minDim).toInt()

        val circles = Mat()
        Imgproc.HoughCircles(
            gray, circles,
            Imgproc.HOUGH_GRADIENT,
            1.2,
            minDim / 4.0,
            120.0, 40.0,
            minR, maxR
        )

        if (circles.empty()) {
            Log.e(TAG, "No se detectó círculo")
            releaseAll(circles, edges, gray)
            rect?.release()
            rgba.release()
            return null
        }

        // Elegimos el círculo con mejor "coverage" de borde + cercanía al centro
        var bestScore = Double.NEGATIVE_INFINITY
        var bestC = Triple(0.0, 0.0, 0.0)

        for (i in 0 until circles.cols()) {
            val data = circles.get(0, i) ?: continue
            val cxD = data[0]
            val cyD = data[1]
            val rD = data[2]
            val cov = circleEdgeCoverage(edges, cxD, cyD, rD)
            val centerBias =
                1.0 - (hypot(cxD - wWork / 2.0, cyD - hWork / 2.0) / (minDim / 2.0)).coerceIn(0.0, 1.0)
            val score = 0.8 * cov + 0.2 * centerBias
            if (score > bestScore) {
                bestScore = score
                bestC = Triple(cxD, cyD, rD)
            }
        }

        val (cxDet, cyDet, rPxDet) = bestC
        val coverage = circleEdgeCoverage(edges, cxDet, cyDet, rPxDet)

        // Escala media desde el aro Ø100 (sobre imagen "work")
        val pxPerMmCircle = (2.0 * rPxDet / CIRCLE_DIAM_MM).toFloat()

        // nitidez global
        val sharp = laplacianVar(gray)

        // ---------------- BARRA 100 (debajo del círculo) ----------------
        val barTriple = detectBar(edges, cxDet, cyDet, rPxDet)

        val barMidXDet = barTriple?.first
        val barMidYDet = barTriple?.second
        val barLenPx = barTriple?.third

        // Escalas X/Y:
        // - X: basada en la barra 100 mm.
        // - Y: fallback a la del círculo (esta hoja no incluye barra vertical).
        var pxPerMmX = pxPerMmCircle
        var pxPerMmY = pxPerMmCircle

        if (barLenPx != null && barLenPx > 1.0) {
            val pxPerMmBar = (barLenPx / BAR_LEN_MM).toFloat()
            pxPerMmX = pxPerMmBar
            pxPerMmY = pxPerMmCircle
        }

        // Longitud de barra en mm usando la escala de X (debe dar ~100)
        val barLenMm = if (barLenPx != null && pxPerMmX > 0f) (barLenPx.toFloat() / pxPerMmX) else null

        // θ página: vector desde C -> barMid (si no hay barra, 0°)
        val theta = if (barMidXDet != null && barMidYDet != null) {
            Math.toDegrees(atan2(barMidYDet - cyDet, barMidXDet - cxDet)).toFloat()
        } else 0f

        // Confianza:
        val confCircle = coverage.coerceIn(0.0, 1.0)
        val confBar = if (barLenMm != null) (1.0 - abs(barLenMm - BAR_LEN_MM) / BAR_LEN_MM).coerceIn(0.0, 1.0) else 0.0
        val conf = (0.7 * confCircle + 0.3 * confBar).toFloat()

        // Si rectificamos con homografía, devolvemos coords del círculo en sistema ORIGINAL
        // (para no romper overlays/pipeline).
        val (cxOut, cyOut, rOut, barMidXOut, barMidYOut) = if (Hinv != null) {
            val cOrig = mapPoint(Hinv, cxDet, cyDet)
            val pOrig = mapPoint(Hinv, cxDet + rPxDet, cyDet)
            val rOrig = hypot(pOrig.x - cOrig.x, pOrig.y - cOrig.y)

            val bmOrig = if (barMidXDet != null && barMidYDet != null) mapPoint(Hinv, barMidXDet, barMidYDet) else null

            Quint(
                cOrig.x.toFloat(),
                cOrig.y.toFloat(),
                rOrig.toFloat(),
                bmOrig?.x?.toFloat(),
                bmOrig?.y?.toFloat()
            )
        } else {
            Quint(
                cxDet.toFloat(),
                cyDet.toFloat(),
                rPxDet.toFloat(),
                barMidXDet?.toFloat(),
                barMidYDet?.toFloat()
            )
        }

        releaseAll(circles, edges, gray)
        rect?.release()
        rgba.release()

        return Ref75(
            pxPerMm = pxPerMmCircle,
            cx = cxOut,
            cy = cyOut,
            rPx = rOut,
            thetaPageDeg = theta,
            barMidX = barMidXOut,
            barMidY = barMidYOut,
            barLenMm = barLenMm,
            coverageCircle = coverage.toFloat(),
            sharpness = sharp,
            confidence = conf,
            pxPerMmX = pxPerMmX,
            pxPerMmY = pxPerMmY
        )
    }

    // ---------------------------------------------------------------------------------------------
    // Rectificación por 4 CUADRADOS (ArUco-as-squares)
    // ---------------------------------------------------------------------------------------------

    private data class RectifyPack3250(
        val rgbaRect: Mat,
        val H: Mat,
        val Hinv: Mat
    ) {
        fun release() {
            rgbaRect.release()
            H.release()
            Hinv.release()
        }
    }

    private data class Quint(
        val a: Float, val b: Float, val c: Float, val d: Float?, val e: Float?
    )

    private data class SquareCand(
        val center: Point,
        val outerCorner: Point,   // esquina más alejada del centro global (para homografía)
        val score: Double
    )

    /**
     * Detecta 4 "markers" como cuadrados por contornos (uno por cuadrante),
     * arma homografía a un cuadrado 150mm y rectifica.
     */
    private fun tryRectifyWithSquares3250(rgbaSrc: Mat): RectifyPack3250? {
        val w = rgbaSrc.cols()
        val h = rgbaSrc.rows()
        val c0 = Point(w / 2.0, h / 2.0)

        val gray = Mat()
        Imgproc.cvtColor(rgbaSrc, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)

        // binario (invertido) para agarrar cuadrados negros
        val bin = Mat()
        Imgproc.threshold(gray, bin, 0.0, 255.0, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU)

        // cerramos un poco para compactar
        val k = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_CLOSE, k, Point(-1.0, -1.0), 1)

        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(bin, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        if (contours.isEmpty()) {
            releaseAll(gray, bin, hierarchy)
            contours.forEach { it.release() }
            return null
        }

        val imgArea = w.toDouble() * h.toDouble()
        val cands = ArrayList<SquareCand>(32)

        for (cnt in contours) {
            val area = Imgproc.contourArea(cnt)
            if (area < 800.0) continue
            if (area > 0.20 * imgArea) continue

            val peri = Imgproc.arcLength(MatOfPoint2f(*cnt.toArray()), true)
            if (peri <= 1.0) continue

            val approx2f = MatOfPoint2f()
            Imgproc.approxPolyDP(MatOfPoint2f(*cnt.toArray()), approx2f, 0.02 * peri, true)
            val pts = approx2f.toArray()
            if (pts.size != 4) {
                approx2f.release()
                continue
            }

            // convex?
            val approx = MatOfPoint(*pts)
            if (!Imgproc.isContourConvex(approx)) {
                approx2f.release(); approx.release()
                continue
            }

            // squareness: ángulos ~90° y lados similares
            val maxCos = maxAngleCos(pts)
            if (maxCos > 0.35) { // cuanto más chico, más recto
                approx2f.release(); approx.release()
                continue
            }

            val rect = Imgproc.boundingRect(approx)
            val ar = rect.width.toDouble() / rect.height.toDouble()
            if (ar < 0.70 || ar > 1.30) {
                approx2f.release(); approx.release()
                continue
            }

            val center = Point(rect.x + rect.width / 2.0, rect.y + rect.height / 2.0)

            // outer corner = punto más alejado del centro global
            var bestP = pts[0]
            var bestD = -1.0
            for (p in pts) {
                val d = hypot(p.x - c0.x, p.y - c0.y)
                if (d > bestD) { bestD = d; bestP = p }
            }

            // Score: prioriza área (legibilidad) + rectitud
            val score = area * (1.0 - maxCos).coerceAtLeast(0.0)

            cands.add(SquareCand(center = center, outerCorner = bestP, score = score))

            approx2f.release()
            approx.release()
        }

        // cleanup contornos
        contours.forEach { it.release() }
        hierarchy.release()
        k.release()
        bin.release()
        gray.release()

        if (cands.size < 4) return null

        // Elegir 1 por cuadrante (TL/TR/BL/BR) según score
        var tl: SquareCand? = null
        var tr: SquareCand? = null
        var bl: SquareCand? = null
        var br: SquareCand? = null

        for (s in cands) {
            val isLeft = s.center.x < c0.x
            val isTop = s.center.y < c0.y
            when {
                isLeft && isTop -> if (tl == null || s.score > tl.score) tl = s
                (!isLeft) && isTop -> if (tr == null || s.score > tr.score) tr = s
                isLeft -> if (bl == null || s.score > bl.score) bl = s
                else -> if (br == null || s.score > br.score) br = s
            }
        }

        if (tl == null || tr == null || bl == null || br == null) return null

        // Fuente: 4 esquinas externas (outer) en orden TL,TR,BR,BL
        val srcPts = MatOfPoint2f(
            tl.outerCorner,
            tr.outerCorner,
            br.outerCorner,
            bl.outerCorner
        )

        // Destino: cuadrado sidePx x sidePx
        val sidePx = ARUCO_SQUARE_SIDE_MM * RECT_PX_PER_MM_NOM
        val dstPts = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(sidePx, 0.0),
            Point(sidePx, sidePx),
            Point(0.0, sidePx)
        )

        val H = Imgproc.getPerspectiveTransform(srcPts, dstPts)
        val Hinv = H.inv()

        val outW = sidePx.toInt().coerceAtLeast(64)
        val outH = sidePx.toInt().coerceAtLeast(64)

        val rgbaRect = Mat(outH, outW, rgbaSrc.type())
        Imgproc.warpPerspective(
            rgbaSrc, rgbaRect, H, Size(outW.toDouble(), outH.toDouble()),
            Imgproc.INTER_LINEAR,
            Core.BORDER_CONSTANT,
            Scalar(255.0, 255.0, 255.0, 255.0)
        )

        srcPts.release()
        dstPts.release()

        return RectifyPack3250(rgbaRect = rgbaRect, H = H, Hinv = Hinv)
    }

    private fun maxAngleCos(pts: Array<Point>): Double {
        // pts: 4 puntos (orden cualquiera). Calculamos el máximo coseno de ángulo interno (cerca de 0 si es 90°).
        // Ordenamos por ángulo alrededor del centro para estabilidad.
        val c = Point(
            (pts[0].x + pts[1].x + pts[2].x + pts[3].x) / 4.0,
            (pts[0].y + pts[1].y + pts[2].y + pts[3].y) / 4.0
        )
        val ordered = pts.sortedBy { atan2(it.y - c.y, it.x - c.x) }.toTypedArray()

        fun cosAngle(a: Point, b: Point, c0: Point): Double {
            val abx = a.x - b.x
            val aby = a.y - b.y
            val cbx = c0.x - b.x
            val cby = c0.y - b.y
            val dot = abx * cbx + aby * cby
            val lab = hypot(abx, aby)
            val lcb = hypot(cbx, cby)
            if (lab < 1e-6 || lcb < 1e-6) return 1.0
            return abs(dot / (lab * lcb))
        }

        var maxCos = 0.0
        for (i in 0 until 4) {
            val a = ordered[(i + 3) % 4]
            val b = ordered[i]
            val c1 = ordered[(i + 1) % 4]
            val cs = cosAngle(a, b, c1)
            if (cs > maxCos) maxCos = cs
        }
        return maxCos
    }

    private fun mapPoint(H: Mat, x: Double, y: Double): Point {
        val src = MatOfPoint2f(Point(x, y))
        val dst = MatOfPoint2f()
        Core.perspectiveTransform(src, dst, H)
        val p = dst.toArray()[0]
        src.release(); dst.release()
        return p
    }

    // ---------------------------------------------------------------------------------------------
    // Tus helpers existentes (circulo/barra/nitidez)
    // ---------------------------------------------------------------------------------------------

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
     * Ajustado para tu PDF: barra HORIZONTAL y DEBAJO del círculo.
     */
    private fun detectBar(edges: Mat, cx: Double, cy: Double, r: Double): Triple<Double, Double, Double>? {
        val lines = Mat()
        Imgproc.HoughLinesP(
            edges,
            lines,
            1.0, Math.PI / 180.0,
            80,                 // threshold
            r * 1.1,            // minLineLength
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

            val horizOk = abs(dy) <= 0.15 * len
            val belowOk = my >= (cy + 0.35 * r)
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