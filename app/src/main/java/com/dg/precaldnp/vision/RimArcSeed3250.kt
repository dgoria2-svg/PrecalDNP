// app/src/main/java/com/dg/precaldnp/vision/RimArcSeed3250.kt
package com.dg.precaldnp.vision

import android.graphics.PointF
import org.opencv.core.Rect
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sign

object RimArcSeed3250 {

    data class ArcSeed3250(
        val originPxRoi: PointF,      // SIEMPRE ROI-local (x: 0..w-1, y: 0..h-1)
        val pxPerMmInitGuess: Double,
        val src3250: String
    )

    /**
     * Seed “oficial” para ARC-FIT (3250):
     * - Por defecto NO usa pupila.
     * - Planta el origen cerca del “centro esperado” a partir de HBOX/VBOX y pxPerMmGuess.
     *
     * OJO: midlineXpxGlobal está en coords GLOBAL (STILL). NO pasar midline ROI acá.
     */
    fun seedFromPupilOrRoi3250(
        roiCv: Rect,
        filHboxMm: Double,
        filVboxMm: Double,
        pxPerMmGuess: Float,
        midlineXpxGlobal: Float,
        pupilGlobal: PointF?,
        usePupil3250: Boolean = false,
        bridgeRowYpxGlobal: Float? = null
    ): ArcSeed3250 {

        val roiW = roiCv.width
        val roiH = roiCv.height
        if (roiW <= 1 || roiH <= 1) {
            return ArcSeed3250(
                originPxRoi = PointF(0f, 0f),
                pxPerMmInitGuess = 5.0,
                src3250 = "roi_invalid"
            )
        }

        val pxGuess = pxPerMmGuess.toDouble()
            .takeIf { it.isFinite() && it > 1e-6 }
            ?.coerceIn(2.5, 20.0)
            ?: 5.0

        val expWpx = (filHboxMm * pxGuess).toFloat().coerceAtLeast(60f)
        val expHpx = (filVboxMm * pxGuess).toFloat().coerceAtLeast(60f)

        // Lado del ROI respecto a la midline GLOBAL (STILL)
        val roiCenterXg = roiCv.x + roiCv.width * 0.5f
        val s = sign(roiCenterXg - midlineXpxGlobal)
        val sideSign = if (s == 0f) 1f else s  // +1: a la derecha de la midline, -1: a la izquierda

        val pupilAllowed = usePupil3250 && (pupilGlobal != null)

        val src: String
        val originXg: Float
        val originYg: Float

        if (pupilAllowed) {
            // (Hoy NO lo querés, pero lo dejamos disponible)
            src = "pupil"
            val offX = (0.25f * expWpx).coerceIn(20f, 0.45f * expWpx)
            originXg = pupilGlobal!!.x + sideSign * offX
            originYg = pupilGlobal.y + 0.10f * expHpx
        } else {
            // Opción oficial: seed esperado del FIL (NO depende de pupila)
            src = "expected"

            // separación desde midline hasta el interior del aro (gap) + medio ancho esperado
            val gapPx = max(18f, 0.10f * expWpx)
            originXg = midlineXpxGlobal + sideSign * (gapPx + 0.50f * expWpx)

            // yRef = bridgeRow (preferido) o centro del ROI
            val yRef = bridgeRowYpxGlobal ?: (roiCv.y + 0.52f * roiCv.height)
            originYg = yRef + 0.10f * expHpx
        }

        // Clamp GLOBAL dentro del ROI GLOBAL
        val xg = clamp(originXg, roiCv.x.toFloat(), (roiCv.x + roiCv.width - 1).toFloat())
        val yg = clamp(originYg, roiCv.y.toFloat(), (roiCv.y + roiCv.height - 1).toFloat())

        // Pasar a coords ROI-local
        val xRoi = clamp(xg - roiCv.x.toFloat(), 0f, (roiCv.width - 1).toFloat())
        val yRoi = clamp(yg - roiCv.y.toFloat(), 0f, (roiCv.height - 1).toFloat())

        return ArcSeed3250(
            originPxRoi = PointF(xRoi, yRoi),
            pxPerMmInitGuess = pxGuess,
            src3250 = src
        )
    }

    private fun clamp(v: Float, lo: Float, hi: Float): Float =
        max(lo, min(hi, v))
}
