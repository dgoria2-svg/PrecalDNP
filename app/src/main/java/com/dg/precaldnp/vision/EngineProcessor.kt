package com.dg.precaldnp.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.net.Uri
import android.util.Log
import androidx.core.content.FileProvider
import androidx.core.graphics.toColorInt
import com.dg.precaldnp.model.MeasurementResult
import java.io.File
import java.io.FileOutputStream
import kotlin.math.max

/**
 * 3250 – EngineProcessor (DRAW ONLY).
 * Dibuja:
 *  - FIL/contorno (naranja)
 *  - pupilas + midline (turquesa)
 *  - DNP (línea pupila->midline)
 *  - Altura (línea pupila->bottom rim)
 *  - Ø útil (círculo: maxDist + 2mm)
 *
 * NO escribe texto en la imagen.
 */
class EngineProcessor(private val context: Context) {

    companion object { private const val TAG = "EngineProcessorDNP" }

    private val geom = InnerRimSnapper3250()

    fun renderFromShot3250(
        stillBitmap: Bitmap,
        uri: Uri,
        baseResult: MeasurementResult,
        pupilOd: PointF,
        pupilOi: PointF,
        midlineX: Double,
        outlineOd: List<PointF>?,
        outlineOi: List<PointF>?
    ): MeasurementResult {
        return try {
            val annotated = annotateBitmap3250(
                src = stillBitmap,
                res = baseResult,
                midlineX = midlineX.toFloat(),
                pupilOd = pupilOd,
                pupilOi = pupilOi,
                outlineOd = outlineOd,
                outlineOi = outlineOi
            )

            val annotatedUri = saveTempPng(annotated)

            if (annotatedUri != null) {
                baseResult.copy(
                    annotatedPath = annotatedUri.toString(),
                    originalPath = uri.toString()
                )
            } else baseResult.copy(originalPath = uri.toString())

        } catch (t: Throwable) {
            Log.e(TAG, "renderFromShot3250", t)
            baseResult.copy(originalPath = uri.toString())
        }
    }

    private fun annotateBitmap3250(
        src: Bitmap,
        res: MeasurementResult,
        midlineX: Float,
        pupilOd: PointF,
        pupilOi: PointF,
        outlineOd: List<PointF>?,
        outlineOi: List<PointF>?
    ): Bitmap {

        val bmp = src.copy(Bitmap.Config.ARGB_8888, true)
        val c = Canvas(bmp)

        val colorFrame = "#FF9800".toColorInt() // contorno/FIL
        val colorPupil = "#00BCD4".toColorInt() // pupilas+midline+medidas

        val swLens = max(3f, bmp.width * 0.006f)
        val swMeas = max(2f, bmp.width * 0.0045f)

        val pLens = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = swLens
            color = colorFrame
        }
        val pMeas = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = swMeas
            color = colorPupil
        }
        val pPupil = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = swLens
            color = colorPupil
        }

        fun drawClosedPolyline(pts: List<PointF>?) {
            if (pts == null || pts.size < 2) return
            val path = Path()
            path.moveTo(pts[0].x, pts[0].y)
            for (i in 1 until pts.size) path.lineTo(pts[i].x, pts[i].y)
            path.close()
            c.drawPath(path, pLens)
        }

        // Contornos (naranja)
        drawClosedPolyline(outlineOd)
        drawClosedPolyline(outlineOi)

        // Midline (turquesa)
        if (midlineX.isFinite()) {
            c.drawLine(midlineX, 0f, midlineX, bmp.height.toFloat(), pMeas)
        }

        // Pupilas (turquesa)
        c.drawCircle(pupilOd.x, pupilOd.y, max(8f, bmp.width * 0.010f), pPupil)
        c.drawCircle(pupilOi.x, pupilOi.y, max(8f, bmp.width * 0.010f), pPupil)

        // DNP (línea pupila -> midline)
        drawDnpLine(c, pupilOd, midlineX, pMeas)
        drawDnpLine(c, pupilOi, midlineX, pMeas)

        // Altura + Ø útil con +2mm (derivado de res + bbox outline)
        val tolX = max(6f, bmp.width * 0.006f)

        drawHeightAndUseful(c, res, pupilOd, outlineOd, pMeas, tolX)
        drawHeightAndUseful(c, res, pupilOi, outlineOi, pMeas, tolX)

        return bmp
    }

    private fun drawDnpLine(canvas: Canvas, pupil: PointF, midX: Float, paint: Paint) {
        if (!pupil.x.isFinite() || !pupil.y.isFinite() || !midX.isFinite()) return
        canvas.drawLine(pupil.x, pupil.y, midX, pupil.y, paint)
    }

    private fun drawHeightAndUseful(
        canvas: Canvas,
        res: MeasurementResult,
        pupil: PointF,
        outline: List<PointF>?,
        paint: Paint,
        tolX: Float
    ) {
        if (outline.isNullOrEmpty()) return
        if (!pupil.x.isFinite() || !pupil.y.isFinite()) return

        val g = geom.computeDrawGeom3250(
            outlinePx = outline,
            pupilPx = pupil,
            res = res,
            usefulMarginMm = 2.0,   // ✅ tu regla
            tolXpx = tolX
        )

        // Altura: pupila -> bottom rim
        g.bottomYpx?.let { bottomY ->
            if (bottomY.isFinite()) {
                canvas.drawLine(pupil.x, pupil.y, pupil.x, bottomY, paint)
            }
        }

        // Ø útil: círculo centrado en pupila
        g.usefulRadiusPx?.let { r ->
            if (r.isFinite() && r > 2f) {
                canvas.drawCircle(pupil.x, pupil.y, r, paint)
            }
        }
    }

    private fun saveTempPng(bmp: Bitmap): Uri? {
        return try {
            val f = File(context.cacheDir, "measurement_${System.currentTimeMillis()}.png")
            FileOutputStream(f).use { out ->
                bmp.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
            FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", f)
        } catch (t: Throwable) {
            Log.e(TAG, "saveTempPng", t)
            null
        }
    }
}
