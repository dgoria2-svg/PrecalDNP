// app/src/main/java/com/dg/precaldnp/vision/DnpMeasurementPipeline3250.kt
package com.dg.precaldnp.vision

import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.RectF
import android.util.Log
import org.opencv.core.Rect

class DnpMeasurementPipeline3250(
    private val lensBoxEngine3250: LensBoxEngine3250,
    private val irisEngine3250: IrisDnpLandmarker3250?
) {
    companion object { private const val TAG = "DnpMeasPipeline3250" }

    data class Inputs(
        val faceBitmap: Bitmap,
        val trace: com.dg.precaldnp.model.ShapeTraceResult,
        val boxOdNorm: RectF?,   // del overlay (OD)
        val boxOiNorm: RectF?,   // del overlay (OI)
        val pxPerMmUsed: Double  // UNA sola escala para TODO (por ahora fallback)
    )

    data class Output(
        val filHboxMm: Double,
        val filVboxMm: Double,
        val pupilsOdPx: PointF,
        val pupilsOiPx: PointF,
        val boxesOdPx: Rect,
        val boxesOiPx: Rect,
        val metrics: LensBoxEngine3250.DnpMetrics3250?,
        val irisDistancePx: Float,
        val irisDistanceMm: Float?,
        val debug: String
    )

    fun run(input: Inputs): Output? {
        val bmp = input.faceBitmap
        if (input.pxPerMmUsed <= 0.0) return null

        // 1) FIL geometry (mm) desde tu trace
        val lensShape = LensShapeUtils3250.fromTrace(input.trace) ?: run {
            Log.w(TAG, "LensShapeUtils3250.fromTrace() = null")
            return null
        }
        val filHboxMm = lensShape.hboxMm.toDouble()
        val filVboxMm = lensShape.vboxMm.toDouble()
        if (filHboxMm <= 0.0 || filVboxMm <= 0.0) return null

        // 2) Boxes desde overlay (norm -> px). Si faltan, abortamos (porque vos ya dijiste que overlay es “verdad”)
        val odRect = input.boxOdNorm?.let { rectNormToPx(it, bmp.width, bmp.height) } ?: return null
        val oiRect = input.boxOiNorm?.let { rectNormToPx(it, bmp.width, bmp.height) } ?: return null

        // 3) Pupilas/Iris (px) en ESTE bitmap
        val iris = irisEngine3250?.estimateDnp(
            bitmap = bmp,
            timestampMs = System.currentTimeMillis(),
            pxPerMmOverride3250 = input.pxPerMmUsed.toFloat()
        ) ?: run {
            Log.w(TAG, "Iris engine sin resultado")
            return null
        }

        // OJO: MediaPipe devuelve “left/right iris” por anatomía, pero en tu flujo OD/OI se define por cajas.
        val p1 = iris.leftIrisCenterPx
        val p2 = iris.rightIrisCenterPx

        // 4) Asignar pupilas a OD/OI según qué caja las contiene (esto te elimina swaps y negativos raros)
        val (pOd, pOi) = assignPupilsToBoxes(p1, p2, odRect, oiRect)

        val pupilsForBoxes = LensBoxEngine3250.Pupils3250(
            left = pOd,   // “lado izquierdo en imagen” en tu overlay actual = OD
            right = pOi
        )
        val boxesForMetrics = LensBoxEngine3250.LensBoxes3250(
            leftBox = odRect,
            rightBox = oiRect
        )

        // 5) Métricas: A/B/Diag SIEMPRE desde FIL; midline = centro del bitmap
        val metrics = lensBoxEngine3250.computeDnpAndUsefulRadii(
            pupils = pupilsForBoxes,
            boxes = boxesForMetrics,
            pxPerMmUsed = input.pxPerMmUsed,
            filHboxMm = filHboxMm,
            filVboxMm = filVboxMm,
            midlineXOverridePx = bmp.width / 2.0
        )

        val dbg = buildString {
            append("FIL HBOX/VBOX=%.2f/%.2f mm | pxPerMm=%.4f"
                .format(filHboxMm, filVboxMm, input.pxPerMmUsed))
            append(" | ODbox=$odRect OIbox=$oiRect")
            append(" | ODp=(%.1f,%.1f) OIp=(%.1f,%.1f)"
                .format(pOd.x, pOd.y, pOi.x, pOi.y))
        }

        return Output(
            filHboxMm = filHboxMm,
            filVboxMm = filVboxMm,
            pupilsOdPx = pOd,
            pupilsOiPx = pOi,
            boxesOdPx = odRect,
            boxesOiPx = oiRect,
            metrics = metrics,
            irisDistancePx = iris.distancePx,
            irisDistanceMm = iris.distanceMm,
            debug = dbg
        )
    }

    private fun rectNormToPx(r: RectF, w: Int, h: Int): Rect {
        val l = (r.left * w).toInt().coerceIn(0, w - 2)
        val t = (r.top * h).toInt().coerceIn(0, h - 2)
        val rr = (r.right * w).toInt().coerceIn(l + 1, w - 1)
        val bb = (r.bottom * h).toInt().coerceIn(t + 1, h - 1)
        return Rect(l, t, rr - l, bb - t)
    }

    private fun assignPupilsToBoxes(
        p1: PointF,
        p2: PointF,
        boxOd: Rect,
        boxOi: Rect
    ): Pair<PointF, PointF> {
        val p1InOd = contains(boxOd, p1)
        val p2InOd = contains(boxOd, p2)
        val p1InOi = contains(boxOi, p1)
        val p2InOi = contains(boxOi, p2)

        // Caso ideal: cada pupila cae en su caja
        if (p1InOd && p2InOi) return p1 to p2
        if (p2InOd && p1InOi) return p2 to p1

        // Fallback: por X (si el overlay pone OD a la izquierda)
        return if (p1.x <= p2.x) p1 to p2 else p2 to p1
    }

    private fun contains(r: Rect, p: PointF): Boolean {
        return p.x >= r.x && p.x <= (r.x + r.width) &&
                p.y >= r.y && p.y <= (r.y + r.height)
    }
}
