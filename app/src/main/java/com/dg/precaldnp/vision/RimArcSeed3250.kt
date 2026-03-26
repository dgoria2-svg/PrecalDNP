package com.dg.precaldnp.vision

import android.graphics.PointF
import org.opencv.core.Rect
import kotlin.math.max
import kotlin.math.min

object RimArcSeed3250 {

    data class ArcSeed3250(
        val originPxRoi: PointF,      // ROI-local (x: 0..w-1, y: 0..h-1)
        val pxPerMmInitGuess: Double,
        val src3250: String
    )

    private fun clamp(v: Float, lo: Float, hi: Float): Float = max(lo, min(hi, v))

    /**
     * Fallback: seed desde pupil o ROI.
     *
     * Criterio coherente 3250:
     * - si usePupil3250=true y hay pupila válida -> usar pupilGlobal.x
     * - si no, inferir el lado del ojo respecto de la midline usando el ROI:
     *      ROI a la izquierda de midline => OD => seed a la izquierda
     *      ROI a la derecha de midline => OI => seed a la derecha
     *
     * Así evitamos plantar la seed “ciega” en el centro del ROI.
     *
     * Y:
     * - Y se apoya en bridgeRowY + una fracción de VBOX para caer cerca del centro vertical.
     */
    fun seedFromPupilOrRoi3250(
        roiCv: Rect,
        filHboxMm: Double,
        filVboxMm: Double,
        pxPerMmGuess: Float,
        midlineXpxGlobal: Float,
        pupilGlobal: PointF?,
        usePupil3250: Boolean,
        bridgeRowYpxGlobal: Float
    ): ArcSeed3250 {

        val w = roiCv.width.coerceAtLeast(2)
        val h = roiCv.height.coerceAtLeast(2)
        val l = roiCv.x.toFloat()
        val t = roiCv.y.toFloat()

        val pxmm = pxPerMmGuess.toDouble().takeIf { it.isFinite() && it > 1e-6 } ?: 5.0
        val expWpx = (filHboxMm * pxmm).toFloat()
        val expHpx = (filVboxMm * pxmm).toFloat()

        val xG = if (usePupil3250 && pupilGlobal != null && pupilGlobal.x.isFinite()) {
            pupilGlobal.x
        } else {
            val roiCenterX = l + 0.5f * w
            val eyeSide = if (roiCenterX < midlineXpxGlobal) -1f else +1f // OD=-1, OI=+1
            midlineXpxGlobal + eyeSide * 0.33f * expWpx
        }

        // Pupil/bridge está “alto” vs caja; empujamos hacia el centro usando VBOX.
        val yRef = bridgeRowYpxGlobal.takeIf { it.isFinite() } ?: (t + 0.5f * h)
        val yG = yRef + 0.10f * expHpx

        val xL = clamp(xG - l, 0f, (w - 1).toFloat())
        val yL = clamp(yG - t, 0f, (h - 1).toFloat())

        return ArcSeed3250(
            originPxRoi = PointF(xL, yL),
            pxPerMmInitGuess = pxmm,
            src3250 = if (usePupil3250) "pupilOrRoi_pupil" else "pupilOrRoi_midlineSeed"
        )
    }

    /**
     * Seed REAL: sale del RimDetectionResult (inner L/R + bottom).
     * - px/mm init sale de innerWidthPx/HBOX_inner_mm si está.
     * - originX = centro de inner (o fallback guiado por midline + eyeSideSign).
     * - originY = bottom - 0.52*VBOXpx (centro vertical aproximado, consistente con “ancla bottom”).
     */
    fun seedFromRimResult3250(
        roiCv: Rect,
        rim: RimDetectionResult,
        filHboxMm: Double,
        filVboxMm: Double,
        pxPerMmGuess: Float,
        midlineXpxGlobal: Float,
        bridgeRowYpxGlobal: Float,
        eyeSideSign3250: Int // OD=-1, OI=+1
    ): ArcSeed3250 {

        val w = roiCv.width.coerceAtLeast(2)
        val h = roiCv.height.coerceAtLeast(2)
        val l = roiCv.x.toFloat()
        val t = roiCv.y.toFloat()

        val pxmmGuess = pxPerMmGuess.toDouble().takeIf { it.isFinite() && it > 1e-6 } ?: 5.0

        val pxmmFromW = run {
            val wPx = rim.innerWidthPx
            if (rim.ok && wPx.isFinite() && wPx > 1f && filHboxMm.isFinite() && filHboxMm > 1.0) {
                (wPx.toDouble() / filHboxMm).takeIf { it.isFinite() && it > 1e-6 }
            } else {
                null
            }
        }

        val pxmm = pxmmFromW ?: pxmmGuess
        val expWpx = (filHboxMm * pxmm).toFloat()
        val expHpx = (filVboxMm * pxmm).toFloat()

        val cxG = run {
            val xL = rim.innerLeftXpx
            val xR = rim.innerRightXpx
            if (rim.ok && xL.isFinite() && xR.isFinite() && xR > xL) {
                0.5f * (xL + xR)
            } else {
                // Fallback guiado por midline:
                // OD=-1 => a la izquierda de la midline
                // OI=+1 => a la derecha de la midline
                val side = eyeSideSign3250.toFloat()
                midlineXpxGlobal + side * 0.33f * expWpx
            }
        }

        val cyG = run {
            val bot = rim.bottomYpx
            if (rim.ok && bot.isFinite()) {
                bot - 0.52f * expHpx
            } else {
                val yRef = bridgeRowYpxGlobal.takeIf { it.isFinite() } ?: (t + 0.5f * h)
                yRef + 0.10f * expHpx
            }
        }

        val xL = clamp(cxG - l, 0f, (w - 1).toFloat())
        val yL = clamp(cyG - t, 0f, (h - 1).toFloat())

        return ArcSeed3250(
            originPxRoi = PointF(xL, yL),
            pxPerMmInitGuess = pxmm,
            src3250 = if (pxmmFromW != null) "rimResult_wSeed" else "rimResult_guessSeed"
        )
    }
}