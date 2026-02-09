package com.dg.precaldnp.vision

import android.graphics.Matrix
import android.graphics.PointF
import kotlin.math.abs

object FilOutlineProjector3250 {

    data class Projection3250(
        val outlineOiPx: List<PointF>,  // base (sin espejo)
        val outlineOdPx: List<PointF>,  // espejo X
        val matrixOi: Matrix,
        val matrixOd: Matrix,
        val pxPerMmOi: Double,
        val pxPerMmOd: Double,
        val pxPerMmFace: Double
    )

    fun projectOiBaseAndOdMirrorXToRings(
        contourMmBaseOi: FilContourMm3250,
        innerOi: FilFitToRing.RingInner,
        innerOd: FilFitToRing.RingInner
    ): Projection3250 {

        val mOi = FilFitToRing.fit(contourMmBaseOi.maxAbsX, contourMmBaseOi.maxAbsY, innerOi)
        val mOd = FilFitToRing.fit(contourMmBaseOi.maxAbsX, contourMmBaseOi.maxAbsY, innerOd)

        // ✅ OI = base como viene del FIL
        val outlineOi = mapMmToPx(contourMmBaseOi.ptsMm, mOi, mirrorX = false)

        // ✅ OD = espejo en X de la MISMA base
        val outlineOd = mapMmToPx(contourMmBaseOi.ptsMm, mOd, mirrorX = true)

        val sOi = uniformScalePxPerMm(mOi)
        val sOd = uniformScalePxPerMm(mOd)
        val pxPerMmFace = ((sOi + sOd) * 0.5).coerceAtLeast(1e-6)

        return Projection3250(
            outlineOiPx = outlineOi,
            outlineOdPx = outlineOd,
            matrixOi = mOi,
            matrixOd = mOd,
            pxPerMmOi = sOi,
            pxPerMmOd = sOd,
            pxPerMmFace = pxPerMmFace
        )
    }

    private fun uniformScalePxPerMm(m: Matrix): Double {
        val v = FloatArray(9)
        m.getValues(v)
        return abs(v[Matrix.MSCALE_X].toDouble())
    }

    private fun mapMmToPx(
        ptsMm: List<PointF>,
        mmToPx: Matrix,
        mirrorX: Boolean
    ): List<PointF> {
        val out = ArrayList<PointF>(ptsMm.size)
        val tmp = FloatArray(2)
        for (p in ptsMm) {
            tmp[0] = if (mirrorX) -p.x else p.x
            tmp[1] = p.y
            mmToPx.mapPoints(tmp)
            out.add(PointF(tmp[0], tmp[1]))
        }
        return out
    }
}
