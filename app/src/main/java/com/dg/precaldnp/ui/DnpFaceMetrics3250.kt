package com.dg.precaldnp.ui

import android.graphics.Bitmap
import android.graphics.PointF
import android.util.Log
import kotlin.math.abs
import kotlin.math.hypot

data class DnpMetrics3250(
    val mode3250: String,           // BINOC / MONO_OD / MONO_OI / NO_PUPIL
    val pupilSrc: String,           // "iris" / "cascade" / etc
    val pxPerMmFaceD: Double,

    val dnpOdMm: Double,
    val dnpOiMm: Double,
    val dnpTotalMm: Double,
    val npdMm: Double,              // si monocular

    val bridgeMm: Double,           // siempre que se pueda (fallback nasal-nasal)
    val pnOdMm: Double,             // NaN si no aplica/no hay pupila OD
    val pnOiMm: Double,             // NaN si no aplica/no hay pupila OI

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

        // ------------------------------------------------------------
        // px/mm (si viene inválido, no dejemos que rompa todo: fallback)
        // ------------------------------------------------------------
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

        fun midXAt(y: Float): Float {
            val yy = y.coerceIn(0f, (h - 1).toFloat())
            val x = pm.midline3250.xAt(yy, w)
            return x.coerceIn(0f, (w - 1).toFloat())
        }

        // ============================================================
        // 1) DNP / NPD (OFICIAL) -> solo si hay pupila(s)
        // ============================================================
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
                        .format(
                            midOd, midOi,
                            odPupil.x, odPupil.y,
                            oiPupil.x, oiPupil.y,
                            dnpOdPx, dnpOiPx,
                            dnpOdMm, dnpOiMm, dnpTotalMm,
                            pxPerMmFaceD
                        )
                )
            }

            "MONO_OD" -> {
                val mid = midXAt(odPupil!!.y)
                val npdPx = abs(odPupil.x - mid)
                npdMm = npdPx.toDouble() / pxPerMmFaceD
                Log.d(
                    TAG,
                    "DNP3250 MONO_OD mid=%.1f odX=%.1f npdPx=%.1f npdMm=%.2f"
                        .format(mid, odPupil.x, npdPx, npdMm)
                )
            }

            "MONO_OI" -> {
                val mid = midXAt(oiPupil!!.y)
                val npdPx = abs(oiPupil.x - mid)
                npdMm = npdPx.toDouble() / pxPerMmFaceD
                Log.d(
                    TAG,
                    "DNP3250 MONO_OI mid=%.1f oiX=%.1f npdPx=%.1f npdMm=%.2f"
                        .format(mid, oiPupil.x, npdPx, npdMm)
                )
            }

            "NO_PUPIL" -> {
                Log.w(
                    TAG,
                    "DNP3250 NO_PUPIL: sin pupilas -> DNP/NPD = NaN (pero puente puede salir por nasal-nasal)"
                )
            }
        }

        // ============================================================
        // 2) PN y PUENTE (TODO en yRef = línea pupilar)
        // ============================================================
        val bridgeOffsetMm = 0.3

        val bandHalfMm = 3.0
        val pnBandHalfPx = (bandHalfMm * pxPerMmFaceD).toFloat()

        var pnOdMm = Double.NaN
        var pnOiMm = Double.NaN
        var bridgeMm: Double

        // yCommonRef: SOLO para NO_PUPIL
        val yCommonRef = run {
            val y0 = listOfNotNull(
                fit.rimOd?.bottomYpx,
                fit.rimOi?.bottomYpx
            ).minOrNull()

            val guess = if (y0 != null && y0.isFinite()) (y0 - 0.20f * h) else (0.55f * h)
            guess.coerceIn(0f, (h - 1).toFloat())
        }

        // yRef OFICIAL: línea pupilar (promedio si hay 2)
        val yRef = run {
            val y = when {
                odPupil != null && oiPupil != null -> 0.5f * (odPupil.y + oiPupil.y)
                odPupil != null -> odPupil.y
                oiPupil != null -> oiPupil.y
                else -> yCommonRef
            }
            y.coerceIn(0f, (h - 1).toFloat())
        }

        val midRef = midXAt(yRef)

        val odSideSign = when {
            odPupil != null && odPupil.x < midRef -> -1f
            odPupil != null && odPupil.x >= midRef -> +1f
            else -> -1f
        }

        val oiSideSign = when {
            oiPupil != null && oiPupil.x < midRef -> -1f
            oiPupil != null && oiPupil.x >= midRef -> +1f
            else -> +1f
        }
        // ✅ IMPORTANTÍSIMO:
        // - Si te falta el placed de un ojo, lo espejamos sobre la midline para poder sacar nasal-nasal
        // - Esto te evita "Puente: -" cuando ArcFit te dio sólo 1 ojo en ese frame
        val placedOdEff: List<PointF>? =
            fit.placedOdUsed ?: fit.placedOiUsed?.let { mirrorPlacedAroundX3250(it, midRef) }
        val placedOiEff: List<PointF>? =
            fit.placedOiUsed ?: fit.placedOdUsed?.let { mirrorPlacedAroundX3250(it, midRef) }

        val nasalOdX = nasalFromPlacedTowardMidline3250(
            placedPtsGlobal = placedOdEff,
            sideSign = odSideSign,              // OD = derecha
            midXAtRef = midRef,
            yRefGlobal = yRef,
            bandHalfPx = pnBandHalfPx,
            pupilXOrNull = odPupil?.x
        )
        val nasalOiX = nasalFromPlacedTowardMidline3250(
            placedPtsGlobal = placedOiEff,
            sideSign = oiSideSign,              // OI = izquierda
            midXAtRef = midRef,
            yRefGlobal = yRef,
            bandHalfPx = pnBandHalfPx,
            pupilXOrNull = oiPupil?.x
        )

        // PN (solo si hay pupila en ese ojo y nasal salió)
        if (odPupil != null && nasalOdX != null) {
            val pnOdPx = pupilToNasalTowardMidlinePx3250(odPupil, nasalOdX, midRef)
            if (pnOdPx != null) pnOdMm = pnOdPx.toDouble() / pxPerMmFaceD
        }
        if (oiPupil != null && nasalOiX != null) {
            val pnOiPx = pupilToNasalTowardMidlinePx3250(oiPupil, nasalOiX, midRef)
            if (pnOiPx != null) pnOiMm = pnOiPx.toDouble() / pxPerMmFaceD
        }

        // Puente fallback nasal-nasal (aunque NO haya pupilas)
        val bridgeNasalMm = if (nasalOdX != null && nasalOiX != null) {
            val bridgeInnerPx = abs(nasalOdX - nasalOiX)
            bridgeInnerPx.toDouble() / pxPerMmFaceD + bridgeOffsetMm
        } else Double.NaN

        // Puente preferido (solo si BINOC y tenemos todo)
        val bridgeDnpMm =
            if (mode == "BINOC" && dnpTotalMm.isFinite() && pnOdMm.isFinite() && pnOiMm.isFinite()) {
                (dnpTotalMm - pnOdMm - pnOiMm + bridgeOffsetMm).takeIf { it.isFinite() }
                    ?: Double.NaN
            } else Double.NaN

        bridgeMm = if (bridgeDnpMm.isFinite()) bridgeDnpMm else bridgeNasalMm

        if (bridgeMm.isFinite()) {
            Log.d(
                TAG,
                "BRIDGE3250 mode=$mode yRef=%.1f mid=%.1f nasal(od/oi)=(%s,%s) pn(od/oi)=(%s,%s) dnpTot=%s bridge(DNP)=%s bridge(nasal)=%s -> used=%s pxmm=%.4f"
                    .format(
                        yRef, midRef,
                        nasalOdX?.let { "%.1f".format(it) } ?: "null",
                        nasalOiX?.let { "%.1f".format(it) } ?: "null",
                        pnOdMm.takeIf { it.isFinite() }?.let { "%.2f".format(it) } ?: "NaN",
                        pnOiMm.takeIf { it.isFinite() }?.let { "%.2f".format(it) } ?: "NaN",
                        dnpTotalMm.takeIf { it.isFinite() }?.let { "%.2f".format(it) } ?: "NaN",
                        bridgeDnpMm.takeIf { it.isFinite() }?.let { "%.2f".format(it) } ?: "NaN",
                        bridgeNasalMm.takeIf { it.isFinite() }?.let { "%.2f".format(it) } ?: "NaN",
                        "%.2f".format(bridgeMm),
                        pxPerMmFaceD
                    )
            )
        } else {
            Log.w(
                TAG,
                "BRIDGE3250 FAIL mode=$mode yRef=$yRef mid=$midRef nasalOdX=$nasalOdX nasalOiX=$nasalOiX bandPx=$pnBandHalfPx yCommonRef=$yCommonRef"
            )
        }

        // ============================================================
        // 3) Alturas (pupila -> bottom rim) si rim disponible
        // ============================================================
        val heightOdMm = computeHeightMm(odPupil, fit.rimOd?.bottomYpx, pxPerMmFaceD)
        val heightOiMm = computeHeightMm(oiPupil, fit.rimOi?.bottomYpx, pxPerMmFaceD)

        // ============================================================
        // 4) Ø útil (pupila -> placed FIL) +2mm
        // ============================================================
        val diamUtilOdMm =
            if (odPupil != null) diameterUtilMmFromPlacedFil(
                odPupil,
                placedOdEff,
                pxPerMmFaceD
            ) else Double.NaN
        val diamUtilOiMm =
            if (oiPupil != null) diameterUtilMmFromPlacedFil(
                oiPupil,
                placedOiEff,
                pxPerMmFaceD
            ) else Double.NaN

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

    private fun mirrorPlacedAroundX3250(pts: List<PointF>, xMid: Float): List<PointF> {
        val out = ArrayList<PointF>(pts.size)
        for (p in pts) out.add(PointF(2f * xMid - p.x, p.y))
        return out
    }

    private fun computeHeightMm(pupil: PointF?, bottomYpx: Float?, pxPerMm: Double): Double {
        if (pupil == null || bottomYpx == null) return Double.NaN
        if (!pupil.x.isFinite() || !pupil.y.isFinite() || !bottomYpx.isFinite()) return Double.NaN
        if (!pxPerMm.isFinite() || pxPerMm <= 1e-9) return Double.NaN

        val hPx = bottomYpx - pupil.y
        if (hPx <= 1.0f) return Double.NaN
        return hPx.toDouble() / pxPerMm
    }


    private fun nasalFromPlacedTowardMidline3250(
        placedPtsGlobal: List<PointF>?,
        sideSign: Float,          // OD=+1f (derecha), OI=-1f (izquierda)
        midXAtRef: Float,
        yRefGlobal: Float,
        bandHalfPx: Float,
        pupilXOrNull: Float?
    ): Float? {
        val pts = placedPtsGlobal ?: return null
        if (pts.size < 6) return null
        if (!midXAtRef.isFinite() || !yRefGlobal.isFinite() || !bandHalfPx.isFinite()) return null
        if (!sideSign.isFinite() || sideSign == 0f) return null

        val pupilDistToMid = pupilXOrNull?.takeIf { it.isFinite() }?.let { abs(it - midXAtRef) }
        val maxFromMid = (pupilDistToMid?.let { (it * 1.20f).coerceIn(30f, 260f) } ?: 220f)

        val xs = ArrayList<Float>(64)
        for (p in pts) {
            if (!p.x.isFinite() || !p.y.isFinite()) continue
            if (abs(p.y - yRefGlobal) > bandHalfPx) continue

            val dFromMidSigned = (p.x - midXAtRef) * sideSign
            if (dFromMidSigned < -2f) continue
            if (dFromMidSigned > maxFromMid) continue

            xs.add(p.x)
        }

        if (xs.isEmpty()) return null
        xs.sort()

        val q = if (sideSign > 0f) 0.12f else 0.88f
        val idx = (q * (xs.size - 1)).toInt().coerceIn(0, xs.size - 1)

        val i0 = (idx - 1).coerceAtLeast(0)
        val i2 = (idx + 1).coerceAtMost(xs.size - 1)
        val picked = (xs[i0] + xs[idx] + xs[i2]) / 3f

        Log.d(
            TAG,
            "NASAL3250 n=${xs.size} side=${if (sideSign > 0f) "OD_RIGHT" else "OI_LEFT"} q=$q idx=$idx " +
                    "x=%.1f bandPx=%.1f yRef=%.1f midX=%.1f maxFromMid=%.1f pupilDist=%.1f"
                        .format(
                            picked,
                            bandHalfPx,
                            yRefGlobal,
                            midXAtRef,
                            maxFromMid,
                            (pupilDistToMid ?: -1f)
                        )
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

        val sideRight = (p.x >= midXAtRef)
        val d = if (sideRight) (p.x - nx) else (nx - p.x)
        return d.takeIf { it.isFinite() && it > 0.5f }
    }

    private fun diameterUtilMmFromPlacedFil(
        pupilPx: PointF,
        placedFilPx: List<PointF>?,
        pxPerMmFaceD: Double
    ): Double {
        if (placedFilPx.isNullOrEmpty()) return Double.NaN
        if (!pxPerMmFaceD.isFinite() || pxPerMmFaceD <= 1e-9) return Double.NaN

        var maxDistPx = Double.NaN

        for (p in placedFilPx) {
            val d = hypot((p.x - pupilPx.x).toDouble(), (p.y - pupilPx.y).toDouble())
            if (!d.isFinite()) continue
            if (!maxDistPx.isFinite() || d > maxDistPx) {
                maxDistPx = d
            }
        }

        if (!maxDistPx.isFinite()) return Double.NaN

        val diamMm = (2.0 * maxDistPx) / pxPerMmFaceD
        return (diamMm + 2.0).coerceAtLeast(0.0)
    }
}