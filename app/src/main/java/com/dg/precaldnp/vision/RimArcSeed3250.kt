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
     * Fallback: seed desde pupil o centro del ROI (tu API actual).
     * Usa VBOX para poner un Y razonable (center-ish) en vez de quedarte pegado a bridge/pupila.
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
        val expHpx = (filVboxMm * pxmm).toFloat()

        val xG = if (usePupil3250 && pupilGlobal != null && pupilGlobal.x.isFinite()) {
            pupilGlobal.x
        } else {
            l + 0.5f * w
        }

        // Pupil/bridge está “alto” vs caja; empujamos hacia el centro usando VBOX (no inventamos top/bottom).
        val yRef = bridgeRowYpxGlobal.takeIf { it.isFinite() } ?: (t + 0.5f * h)
        val yG = (yRef + 0.10f * expHpx) // ~centro desde bridge

        val xL = clamp(xG - l, 0f, (w - 1).toFloat())
        val yL = clamp(yG - t, 0f, (h - 1).toFloat())

        return ArcSeed3250(
            originPxRoi = PointF(xL, yL),
            pxPerMmInitGuess = pxmm,
            src3250 = if (usePupil3250) "pupilOrRoi_pupil" else "pupilOrRoi_roi"
        )
    }

    /**
     * ✅ Seed REAL: sale del RimDetectionResult (inner L/R + bottom).
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
        eyeSideSign3250: Int // OD=-1, OI=+1 (nasal hacia midline)
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
            } else null
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
                // fallback guiado por midline: el centro del ojo está “del lado opuesto” al nasal.
                val side = (-eyeSideSign3250).toFloat() // OD => +1 (a la derecha), OI => -1 (a la izquierda)
                (midlineXpxGlobal + side * 0.33f * expWpx)
            }
        }

        val cyG = run {
            val bot = rim.bottomYpx
            if (rim.ok && bot.isFinite()) {
                (bot - 0.52f * expHpx)
            } else {
                val yRef = bridgeRowYpxGlobal.takeIf { it.isFinite() } ?: (t + 0.5f * h)
                (yRef + 0.10f * expHpx)
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