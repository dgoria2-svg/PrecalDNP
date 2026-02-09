package com.dg.precaldnp.vision

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.provider.MediaStore
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfInt
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Herramientas de depuración visual + utilidades de construcción de outline.
 * Guarda PNGs en Pictures/medirDNP/DEBUG_*
 */
object TracerDebug {

    private const val TAG = "Precal/TraceDebug"

    data class Counters(
        var edgesInMask: Int = 0,
        var radialHits: Int = 0,
        var radialMiss: Int = 0,
        var serruchoRms: Double = 0.0
    )

    /** Guarda un Mat como PNG vía MediaStore (convierte a RGBA si es necesario). */
    fun saveMatPng(ctx: Context, name: String, m: Mat) {
        val rgba = when (m.type()) {
            CvType.CV_8UC1 -> {
                val tmp = Mat()
                Imgproc.cvtColor(m, tmp, Imgproc.COLOR_GRAY2RGBA)
                tmp
            }
            CvType.CV_8UC3 -> {
                val tmp = Mat()
                Imgproc.cvtColor(m, tmp, Imgproc.COLOR_BGR2RGBA)
                tmp
            }
            CvType.CV_8UC4 -> m
            else -> {
                val tmp = Mat()
                m.convertTo(tmp, CvType.CV_8UC1)
                val rgba2 = Mat()
                Imgproc.cvtColor(tmp, rgba2, Imgproc.COLOR_GRAY2RGBA)
                tmp.release()
                rgba2
            }
        }
        val bmp = Bitmap.createBitmap(rgba.cols(), rgba.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgba, bmp)
        rgba.release()
        saveBitmapPng(ctx, "DEBUG_$name", bmp)
        bmp.recycle()
    }

    /** Guarda un Bitmap como PNG vía MediaStore. */
    fun saveBitmapPng(ctx: Context, baseName: String, bmp: Bitmap): Uri? {
        return try {
            val name = "${baseName}.png"
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, name)
                put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/medirDNP")
            }
            val uri = ctx.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            ctx.contentResolver.openOutputStream(uri!!)?.use { out ->
                bmp.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
            uri
        } catch (e: Throwable) {
            Log.w(TAG, "saveBitmapPng($baseName) falló: ${e.message}")
            null
        }
    }

    /** Construye máscara de anillo centrada en (cx,cy) con radio r y espesor (thickness px). */
    fun buildRingMask(width: Int, height: Int, cx: Double, cy: Double, r: Double, thickness: Int): Mat {
        val mask = Mat.zeros(Size(width.toDouble(), height.toDouble()), CvType.CV_8UC1)
        Imgproc.circle(mask, Point(cx, cy), r.roundToInt(), Scalar(255.0), thickness)
        return mask
    }

    /** Aplica máscara (AND) a una imagen binaria (u8). */
    fun maskBinary(srcU8: Mat, maskU8: Mat): Mat {
        val out = Mat()
        Core.bitwise_and(srcU8, maskU8, out)
        return out
    }

    /** Canny con parámetros “prudentes” (auto-umbral por mediana de |∇|). */
    fun cannyAuto(grayU8: Mat): Mat {
        val blurred = Mat()
        Imgproc.GaussianBlur(grayU8, blurred, Size(5.0, 5.0), 0.0)

        // Estimación de umbrales por percentiles
        val hist = Mat()
        val histSize = MatOfInt(256)
        val ranges = MatOfFloat(0f, 256f)
        Imgproc.calcHist(listOf(blurred), MatOfInt(0), Mat(), hist, histSize, ranges)
        var total = 0.0
        for (i in 0 until 256) total += hist.get(i, 0)[0]
        var cumsum = 0.0
        var p10 = 0; var p90 = 255
        for (i in 0 until 256) {
            cumsum += hist.get(i, 0)[0]
            val frac = cumsum / total
            if (frac >= 0.10 && p10 == 0) p10 = i
            if (frac >= 0.90) { p90 = i; break }
        }
        hist.release()
        val low = max(10.0, p10.toDouble())
        val high = max(low + 10.0, p90.toDouble())

        val edges = Mat()
        Imgproc.Canny(blurred, edges, low, high)
        blurred.release()
        return edges
    }

    /**
     * Barrido radial desde rOuter hacia adentro hasta rMin buscando 1° edge.
     * Devuelve el radio encontrado (px) o null si no encuentra.
     */
    fun scanRadialOutsideIn(
        edgesU8: Mat,
        cx: Double,
        cy: Double,
        theta: Double,
        rOuter: Double,
        rMin: Double,
        stepPx: Double = 1.0
    ): Double? {
        var r = rOuter
        val cosT = cos(theta)
        val sinT = sin(theta)
        val h = edgesU8.rows()
        val w = edgesU8.cols()
        while (r >= rMin) {
            val x = (cx + r * cosT)
            val y = (cy + r * sinT)
            val xi = x.roundToInt()
            val yi = y.roundToInt()
            if (xi in 0 until w && yi in 0 until h) {
                val v = edgesU8.get(yi, xi)
                if (v != null && v.isNotEmpty() && v[0] > 0.0) return r
            }
            r -= stepPx
        }
        return null
    }

    /** Suavizado circular: mediana (ventana impar) + promedio móvil (ventana impar). */
    fun smoothCircular(values: DoubleArray, winMedian: Int = 5, winMean: Int = 7): DoubleArray {
        fun wrapGet(a: DoubleArray, i: Int): Double = a[(i % a.size + a.size) % a.size]
        val n = values.size
        val med = DoubleArray(n)
        val halfM = winMedian / 2
        for (i in 0 until n) {
            val window = DoubleArray(winMedian) { k -> wrapGet(values, i + k - halfM) }
            window.sort()
            med[i] = window[winMedian / 2]
        }
        val out = DoubleArray(n)
        val halfA = winMean / 2
        for (i in 0 until n) {
            var s = 0.0
            for (k in -halfA..halfA) s += wrapGet(med, i + k)
            out[i] = s / winMean
        }
        return out
    }

    /** RMS del “serrucho” (desviación respecto de suavizado). */
    fun serruchoRms(raw: DoubleArray, smooth: DoubleArray): Double {
        var s = 0.0
        for (i in raw.indices) {
            val d = raw[i] - smooth[i]
            s += d * d
        }
        return sqrt(s / raw.size.coerceAtLeast(1))
    }

    /** Dibuja el polilínea sobre RGBA en color (B,G,R,A) y lo devuelve. */
    fun drawPolylineOn(bgrOrRgba: Mat, points: List<Point>, color: Scalar, thickness: Int = 2): Mat {
        val out = bgrOrRgba.clone()
        for (i in 0 until points.size - 1) {
            Imgproc.line(out, points[i], points[i + 1], color, thickness)
        }
        return out
    }

    /** Heatmap del distance transform (útil para ver “cresta” del borde). */
    fun distanceHeat(edgesU8: Mat): Mat {
        val inv = Mat()
        Core.bitwise_not(edgesU8, inv)
        val dist = Mat()
        Imgproc.distanceTransform(inv, dist, Imgproc.DIST_L2, 3)
        Core.normalize(dist, dist, 0.0, 255.0, Core.NORM_MINMAX)
        val u8 = Mat()
        dist.convertTo(u8, CvType.CV_8UC1)
        val color = Mat()
        Imgproc.applyColorMap(u8, color, Imgproc.COLORMAP_JET)
        inv.release(); dist.release(); u8.release()
        return color
    }

    /** Log principal de chequeo. */
    fun logSummary(pxPerMm: Double, r75px: Double, cnt: Counters, n: Int) {
        Log.d(TAG, "px/mm=$pxPerMm  R75px=${"%.1f".format(r75px)}  edgesInMask=${cnt.edgesInMask} hits=${cnt.radialHits} miss=${cnt.radialMiss} serruchoRMS=${"%.2f".format(cnt.serruchoRms)} (n=$n)")
    }
}
