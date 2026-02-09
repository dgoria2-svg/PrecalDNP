package com.dg.precaldnp.vision

import android.graphics.Bitmap
import android.graphics.PointF
import android.util.Log
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.photo.Photo
import kotlin.math.max
import android.util.Size as AndroidSize

object OutlineExtractor {

    data class Result(
        val outlinePx: List<PointF>,   // hull del lente en coords absolutas
        val bboxPx: AndroidSize,       // bounding box axis-aligned (fallback)
        val sharpness: Float,          // nitidez
        val rotWidthPx: Float,         // ancho del minAreaRect
        val rotHeightPx: Float,        // alto del minAreaRect
        val rotAngleDeg: Float         // ángulo del minAreaRect
    )

    fun extractOutline(bmp: Bitmap): Result? {
        @Suppress("DEPRECATION")
        val ok = OpenCVLoader.initDebug()
        if (!ok) {
            Log.e("OutlineExtractor", "OpenCV no cargó")
            return null
        }

        // --- Bitmap -> Mat RGBA ---
        val rgba = Mat(bmp.height, bmp.width, CvType.CV_8UC4)
        org.opencv.android.Utils.bitmapToMat(bmp, rgba)

        // Pasamos a BGR para trabajar en HSV e inpaint
        val bgr = Mat()
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)

        // Reducimos reflejos en BGR (inpaint en zonas blancas/saturación baja)
        val bgrNoGlare = reduceGlareBgr(bgr)

        // Ya no necesitamos los originales
        rgba.release()
        bgr.release()

        // gris a partir de la imagen sin tanto reflejo
        val gray = Mat()
        Imgproc.cvtColor(bgrNoGlare, gray, Imgproc.COLOR_BGR2GRAY)
        bgrNoGlare.release()

        // blur leve
        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)

        // Canny
        val edges = Mat()
        Imgproc.Canny(gray, edges, 50.0, 150.0)

        // cerramos huecos para pegar bordes cortados
        val kernel = Imgproc.getStructuringElement(
            Imgproc.MORPH_ELLIPSE,
            Size(7.0, 7.0) // un poquito más grande que antes
        )
        Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, kernel)
        // un dilate suave ayuda a unir pedazos separados por tornillos / reflejos
        Imgproc.dilate(edges, edges, kernel)

        kernel.release()
        gray.release()

        // --- contornos externos ---
        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(
            edges,
            contours,
            Mat(),
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_NONE
        )
        edges.release()

        if (contours.isEmpty()) {
            Log.e("OutlineExtractor", "sin contornos detectables")
            contours.forEach { it.release() }
            return null
        }

        // filtramos ruido: nos quedamos con contornos "grandes"
        // regla burda: área > 1000 px (ajustable)
        val bigOnes = contours.filter { c ->
            Imgproc.contourArea(c) > 1000.0
        }
        if (bigOnes.isEmpty()) {
            Log.e("OutlineExtractor", "solo había mugre chiquita")
            contours.forEach { it.release() }
            return null
        }

        // juntamos TODOS los puntos de TODOS esos contornos grandes
        val allPtsList = ArrayList<Point>()
        for (c in bigOnes) {
            val pts = c.toArray()
            for (p in pts) {
                allPtsList.add(p)
            }
        }

        // ya no necesitamos los contornos individuales
        contours.forEach { it.release() }

        if (allPtsList.isEmpty()) {
            Log.e("OutlineExtractor", "allPtsList vacío")
            return null
        }

        // 1) hull convexo para dibujar outline "cerradito"
        val matPts = MatOfPoint2f()
        matPts.fromList(allPtsList.map { Point(it.x, it.y) })
        val hullIdx = MatOfInt()
        Imgproc.convexHull(MatOfPoint(*allPtsList.toTypedArray()), hullIdx)

        val hullPts = ArrayList<PointF>()
        val hullArrayIdx = hullIdx.toArray()
        for (idx in hullArrayIdx) {
            val p = allPtsList[idx]
            hullPts.add(PointF(p.x.toFloat(), p.y.toFloat()))
        }

        // 2) minAreaRect del conjunto total de puntos
        val rotRect = Imgproc.minAreaRect(matPts) // RotatedRect
        val rotSize = rotRect.size
        val rotWidthPx  = rotSize.width.toFloat()
        val rotHeightPx = rotSize.height.toFloat()
        val rotAngleDeg = rotRect.angle.toFloat()

        // 3) también sacamos AABB axis-aligned por si querés debug paralelo
        var minX = Float.POSITIVE_INFINITY
        var minY = Float.POSITIVE_INFINITY
        var maxX = Float.NEGATIVE_INFINITY
        var maxY = Float.NEGATIVE_INFINITY
        for (p in allPtsList) {
            val fx = p.x.toFloat()
            val fy = p.y.toFloat()
            if (fx < minX) minX = fx
            if (fy < minY) minY = fy
            if (fx > maxX) maxX = fx
            if (fy > maxY) maxY = fy
        }
        val boxW = max(1f, maxX - minX).toInt()
        val boxH = max(1f, maxY - minY).toInt()

        // nitidez (como antes, sobre el bitmap original)
        val sharp = estimateSharpnessForResult(bmp)

        matPts.release()
        hullIdx.release()

        return Result(
            outlinePx = hullPts,
            bboxPx = AndroidSize(boxW, boxH),
            sharpness = sharp,
            rotWidthPx = rotWidthPx,
            rotHeightPx = rotHeightPx,
            rotAngleDeg = rotAngleDeg
        )
    }

    // ---- NUEVO: reducción de reflejos en BGR ----
    private fun reduceGlareBgr(srcBgr: Mat): Mat {
        val hsv = Mat()
        Imgproc.cvtColor(srcBgr, hsv, Imgproc.COLOR_BGR2HSV)

        val channels = ArrayList<Mat>(3)
        Core.split(hsv, channels)
        val h = channels[0]
        val s = channels[1]
        val v = channels[2]

        // brillo alto
        val maskBright = Mat()
        Imgproc.threshold(
            v, maskBright,
            240.0, 255.0, Imgproc.THRESH_BINARY
        )

        // saturación baja (casi blanco)
        val maskLowSat = Mat()
        Imgproc.threshold(
            s, maskLowSat,
            40.0, 255.0, Imgproc.THRESH_BINARY_INV
        )

        val glareMask = Mat()
        Core.bitwise_and(maskBright, maskLowSat, glareMask)

        val glarePixels = Core.countNonZero(glareMask)
        val totalPixels = glareMask.rows() * glareMask.cols()
        val glareRatio = if (totalPixels > 0) {
            glarePixels.toFloat() / totalPixels.toFloat()
        } else 0f

        // limpieza de mats intermedios
        h.release()
        s.release()
        v.release()
        hsv.release()
        maskBright.release()
        maskLowSat.release()

        val out = Mat()
        if (glareRatio < 0.02f) {
            // glare casi inexistente: copiar tal cual
            srcBgr.copyTo(out)
            glareMask.release()
            return out
        }

        // glare significativo: inpaint sobre las zonas blancas
        Photo.inpaint(srcBgr, glareMask, out, 3.0, Photo.INPAINT_TELEA)
        glareMask.release()

        return out
    }

    // ---- nitidez igual que antes ----
    private fun estimateSharpnessForResult(bmp: Bitmap): Float {
        val mat = Mat(bmp.height, bmp.width, CvType.CV_8UC4)
        org.opencv.android.Utils.bitmapToMat(bmp, mat)

        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2GRAY)

        val lap = Mat()
        Imgproc.Laplacian(gray, lap, CvType.CV_64F)

        val mean = MatOfDouble()
        val std = MatOfDouble()
        Core.meanStdDev(lap, mean, std)
        val sigma = std.toArray()[0]
        val variance = (sigma * sigma).toFloat()

        lap.release()
        gray.release()
        mat.release()
        mean.release()
        std.release()

        return variance
    }
}
