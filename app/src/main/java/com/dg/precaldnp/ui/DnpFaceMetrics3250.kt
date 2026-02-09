package com.dg.precaldnp.ui

import android.graphics.Bitmap
import android.graphics.PointF
import android.util.Log
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.sign

data class DnpMetrics3250(
    val mode3250: String,           // BINOC / MONO_OD / MONO_OI
    val pupilSrc: String,           // "iris" / "cascade"
    val pxPerMmFaceD: Double,

    val dnpOdMm: Double,
    val dnpOiMm: Double,
    val dnpTotalMm: Double,
    val npdMm: Double,              // si monocular

    val bridgeMm: Double,           // NaN si no aplica
    val pnOdMm: Double,             // NaN si no aplica
    val pnOiMm: Double,             // NaN si no aplica

    val heightOdMm: Double,         // pupil->bottom rim si rim disponible
    val heightOiMm: Double,

    val diamUtilOdMm: Double,       // Ø útil desde pupila->FIL (+2mm)
    val diamUtilOiMm: Double
)

object DnpFaceMetrics3250 {

    private const val TAG = "DnpFaceMetrics3250"

    fun computeMetrics3250(
        stillBmp: Bitmap,
        pm: DnpFacePipeline3250.PupilsMidPack3250,
        fit: DnpFacePipeline3250.FitPack3250,
        pupilSrc: String
    ): DnpMetrics3250 {

        val w = stillBmp.width
        val h = stillBmp.height

        // px/mm (no dejar NaN romper todo)
        val pxPerMmFaceD = fit.pxPerMmFaceD
            .takeIf { it.isFinite() && it > 1e-6 }
            ?.coerceIn(1.5, 20.0)
            ?: run {
                Log.e(TAG, "pxPerMmFaceD inválido=${fit.pxPerMmFaceD} -> fallback=6.0")
                6.0
            }

        val odPupil: PointF? = if (pm.odOkReal) pm.pupilOdDet else null
        val oiPupil: PointF? = if (pm.oiOkReal) pm.pupilOiDet else null

        val mode = when {
            odPupil != null && oiPupil != null -> "BINOC"
            odPupil != null -> "MONO_OD"
            oiPupil != null -> "MONO_OI"
            else -> "NO_PUPIL"
        }

        fun midXAt(y: Float): Float =
            pm.midline3250.xAt(y.coerceIn(0f, (h - 1).toFloat()), w)

        // =========================
        // 1) DNP / NPD (OFICIAL) usando midline xAt(pupilY) por ojo
        // =========================
        var dnpOdMm = Double.NaN
        var dnpOiMm = Double.NaN
        var dnpTotalMm = Double.NaN
        var npdMm = Double.NaN

        when (mode) {
            "BINOC" -> {
                val midOd = midXAt(odPupil!!.y)
                val midOi = midXAt(oiPupil!!.y)

                val dnpOdPx = abs(odPupil.x - midOd)
                val dnpOiPx = abs(oiPupil.x - midOi)

                dnpOdMm = dnpOdPx.toDouble() / pxPerMmFaceD
                dnpOiMm = dnpOiPx.toDouble() / pxPerMmFaceD
                dnpTotalMm = dnpOdMm + dnpOiMm

                Log.d(
                    TAG,
                    "DNP3250 BINOC midOd=%.1f midOi=%.1f od=(%.1f,%.1f) oi=(%.1f,%.1f) dPx=(%.1f,%.1f) mm=(%.2f,%.2f) tot=%.2f pxmm=%.4f"
                        .format(midOd, midOi, odPupil.x, odPupil.y, oiPupil.x, oiPupil.y, dnpOdPx, dnpOiPx, dnpOdMm, dnpOiMm, dnpTotalMm, pxPerMmFaceD)
                )
            }

            "MONO_OD" -> {
                val mid = midXAt(odPupil!!.y)
                val npdPx = abs(odPupil.x - mid)
                npdMm = npdPx.toDouble() / pxPerMmFaceD
                Log.d(TAG, "DNP3250 MONO_OD mid=%.1f odX=%.1f npdPx=%.1f npdMm=%.2f".format(mid, odPupil.x, npdPx, npdMm))
            }

            "MONO_OI" -> {
                val mid = midXAt(oiPupil!!.y)
                val npdPx = abs(oiPupil.x - mid)
                npdMm = npdPx.toDouble() / pxPerMmFaceD
                Log.d(TAG, "DNP3250 MONO_OI mid=%.1f oiX=%.1f npdPx=%.1f npdMm=%.2f".format(mid, oiPupil.x, npdPx, npdMm))
            }

            else -> error("NO_PUPIL: no hay pupila válida para métricas.")
        }

        // =========================
        // 2) PN y PUENTE (solo BINOC)
        // PN: pupila -> nasal hacia midline (desde FIL colocado)
        // PUENTE OFICIAL: bridge = dnpTotal - pnOd - pnOi + offset
        // =========================
        val bridgeOffsetMm = 0.3

        val bandHalfMm = 3.0
        val pnBandHalfPx = (bandHalfMm * pxPerMmFaceD).toFloat()

        var pnOdMm = Double.NaN
        var pnOiMm = Double.NaN
        var bridgeMm = Double.NaN

        if (mode == "BINOC") {
            // Usar yRef por ojo (midline oblicua) + banda que tolera poco tilt
            val yRefOd = odPupil!!.y
            val yRefOi = oiPupil!!.y

            val midOd = midXAt(yRefOd)
            val midOi = midXAt(yRefOi)

            val nasalOdX = nasalFromPlacedTowardMidline3250(
                placedPtsGlobal = fit.placedOdUsed,
                pupilX = odPupil.x,
                midXAtRef = midOd,
                yRefGlobal = yRefOd,
                bandHalfPx = pnBandHalfPx
            )
            val nasalOiX = nasalFromPlacedTowardMidline3250(
                placedPtsGlobal = fit.placedOiUsed,
                pupilX = oiPupil.x,
                midXAtRef = midOi,
                yRefGlobal = yRefOi,
                bandHalfPx = pnBandHalfPx
            )

            if (nasalOdX != null && nasalOiX != null) {
                val pnOdPx = pupilToNasalTowardMidlinePx3250(odPupil, nasalOdX, midOd)
                val pnOiPx = pupilToNasalTowardMidlinePx3250(oiPupil, nasalOiX, midOi)

                if (pnOdPx != null) pnOdMm = pnOdPx.toDouble() / pxPerMmFaceD
                if (pnOiPx != null) pnOiMm = pnOiPx.toDouble() / pxPerMmFaceD

                if (dnpTotalMm.isFinite() && pnOdMm.isFinite() && pnOiMm.isFinite()) {
                    bridgeMm = (dnpTotalMm - pnOdMm - pnOiMm + bridgeOffsetMm)
                        .takeIf { it.isFinite() } ?: Double.NaN
                }

                // Debug nasal-nasal (diagnóstico)
                val bridgeInnerPx = abs(nasalOdX - nasalOiX)
                val bridgeNasalMm = bridgeInnerPx.toDouble() / pxPerMmFaceD + bridgeOffsetMm

                Log.d(
                    TAG,
                    "BRIDGE3250 midOd=%.1f midOi=%.1f nasalOdX=%.1f nasalOiX=%.1f " +
                            "pnOd=%.2f pnOi=%.2f dnpTot=%.2f bridge(DNP)=%.2f bridge(nasal)=%.2f pxmm=%.4f"
                                .format(midOd, midOi, nasalOdX, nasalOiX, pnOdMm, pnOiMm, dnpTotalMm, bridgeMm, bridgeNasalMm, pxPerMmFaceD)
                )
            } else {
                Log.w(TAG, "BRIDGE3250 FAIL nasalOdX=$nasalOdX nasalOiX=$nasalOiX bandPx=$pnBandHalfPx")
            }
        }

        // =========================
        // 3) Alturas (pupila -> bottom rim) si rim disponible
        // =========================
        val heightOdMm = computeHeightMm(odPupil, fit.rimOd?.bottomYpx, pxPerMmFaceD)
        val heightOiMm = computeHeightMm(oiPupil, fit.rimOi?.bottomYpx, pxPerMmFaceD)

        // =========================
        // 4) Ø útil (pupila -> placed FIL) +2mm
        // =========================
        val diamUtilOdMm =
            if (odPupil != null) diameterUtilMmFromPlacedFil(odPupil, fit.placedOdUsed, pxPerMmFaceD) else Double.NaN
        val diamUtilOiMm =
            if (oiPupil != null) diameterUtilMmFromPlacedFil(oiPupil, fit.placedOiUsed, pxPerMmFaceD) else Double.NaN

        return DnpMetrics3250(
            mode3250 = mode,
            pupilSrc = pupilSrc,
            pxPerMmFaceD = pxPerMmFaceD,

            dnpOdMm = dnpOdMm,
            dnpOiMm = dnpOiMm,
            dnpTotalMm = dnpTotalMm,
            npdMm = npdMm,

            bridgeMm = bridgeMm,
            pnOdMm = pnOdMm,
            pnOiMm = pnOiMm,

            heightOdMm = heightOdMm,
            heightOiMm = heightOiMm,

            diamUtilOdMm = diamUtilOdMm,
            diamUtilOiMm = diamUtilOiMm
        )
    }

    private fun computeHeightMm(pupil: PointF?, bottomYpx: Float?, pxPerMm: Double): Double {
        if (pupil == null || bottomYpx == null) return Double.NaN
        if (!pupil.x.isFinite() || !pupil.y.isFinite() || !bottomYpx.isFinite()) return Double.NaN
        val hPx = (bottomYpx - pupil.y).coerceAtLeast(0f)
        return hPx.toDouble() / pxPerMm
    }

    private fun nasalFromPlacedTowardMidline3250(
        placedPtsGlobal: List<PointF>?,
        pupilX: Float,
        midXAtRef: Float,
        yRefGlobal: Float,
        bandHalfPx: Float
    ): Float? {
        val pts = placedPtsGlobal ?: return null
        if (pts.size < 6) return null
        if (!pupilX.isFinite() || !midXAtRef.isFinite() || !yRefGlobal.isFinite() || !bandHalfPx.isFinite()) return null

        // lado basado en la pupila (NO en el contorno)
        val s = sign(pupilX - midXAtRef).let { if (it == 0f) 1f else it } // +1: derecha, -1: izquierda

        val xs = ArrayList<Float>(64)
        for (p in pts) {
            if (!p.x.isFinite() || !p.y.isFinite()) continue
            if (abs(p.y - yRefGlobal) > bandHalfPx) continue

            // quedate con puntos del lado del ojo (descarta claramente el otro lado)
            if (((p.x - midXAtRef) * s) < -2f) continue
            xs.add(p.x)
        }

        if (xs.isEmpty()) return null
        xs.sort()

        // derecha -> queremos el MIN (cerca de midline)
        // izquierda -> queremos el MAX (cerca de midline)
        val q = if (s > 0f) 0.12f else 0.88f
        val idx = (q * (xs.size - 1)).toInt().coerceIn(0, xs.size - 1)

        val i0 = (idx - 1).coerceAtLeast(0)
        val i2 = (idx + 1).coerceAtMost(xs.size - 1)
        val picked = (xs[i0] + xs[idx] + xs[i2]) / 3f

        Log.d(
            TAG,
            "NASAL3250 n=${xs.size} side=${if (s > 0f) "RIGHT" else "LEFT"} q=$q idx=$idx x=%.1f bandPx=%.1f yRef=%.1f midX=%.1f"
                .format(picked, bandHalfPx, yRefGlobal, midXAtRef)
        )
        return picked
    }

    private fun pupilToNasalTowardMidlinePx3250(
        pupil: PointF?,
        nasalX: Float?,
        midXAtRef: Float
    ): Float? {
        val p = pupil ?: return null
        val nx = nasalX ?: return null
        if (!p.x.isFinite() || !nx.isFinite() || !midXAtRef.isFinite()) return null

        // +1: pupila a la derecha, -1: pupila a la izquierda
        val s = sign(p.x - midXAtRef).let { if (it == 0f) 1f else it }

        // derecha: nasalX < pupilX ; izquierda: nasalX > pupilX
        val d = if (s > 0f) (p.x - nx) else (nx - p.x)
        return d.takeIf { it.isFinite() && it > 0.5f }
    }

    private fun diameterUtilMmFromPlacedFil(
        pupilPx: PointF,
        placedFilPx: List<PointF>?,
        pxPerMmFaceD: Double
    ): Double {
        if (placedFilPx.isNullOrEmpty()) return Double.NaN
        if (!pxPerMmFaceD.isFinite() || pxPerMmFaceD <= 1e-9) return Double.NaN

        var maxDistPx = 0.0
        for (p in placedFilPx) {
            val d = hypot((p.x - pupilPx.x).toDouble(), (p.y - pupilPx.y).toDouble())
            if (d > maxDistPx) maxDistPx = d
        }

        val diamMm = (2.0 * maxDistPx) / pxPerMmFaceD
        return (diamMm + 2.0).coerceAtLeast(0.0) // +2mm margen fijo
    }
}
