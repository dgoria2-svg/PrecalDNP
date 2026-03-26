package com.dg.precaldnp.vision

import android.graphics.PointF
import android.util.Log
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

/**
 * PxMmOfficialResolver3250
 *
 * Regla simple:
 * - si ArcFit final de OD midió -> sirve
 * - si ArcFit final de OI midió -> sirve
 * - si midieron ambos -> promedio
 * - si midió uno solo -> usar ese
 * - si no midió ninguno -> fallback a rim base
 *
 * NO comparar contra rim base para elegir.
 * NO elegir “el más cercano”.
 * NO descartar por comparación entre ojos.
 *
 * X manda.
 * Y queda solo como dato/log.
 */
object PxMmOfficialResolver3250 {

    private const val TAG = "PxMmOfficial3250"

    data class EyeInput3250(
        val eyeTag: String,                  // "OD" / "OI"
        val placedPtsGlobal: List<PointF>?,  // FINAL: placedOdUsed / placedOiUsed
        val originGlobal: PointF?,           // origen global del fit final
        val rotDeg: Double?,                 // rotación final del fit
        val isIndependent: Boolean = true    // false si vino espejado del otro ojo
    )

    data class EyePack3250(
        val eyeTag: String,
        val spanXpx: Double,
        val spanYpx: Double,
        val pxPerMmX: Double,
        val pxPerMmY: Double,
        val relErrXY: Double,
        val errVboxFromX: Double,
        val relToRimBase: Double,
        val isIndependent: Boolean,
        val acceptedForOfficial: Boolean,
        val rejectReason: String?
    )

    data class Result3250(
        val pxPerMmOfficial: Double,
        val source: String, // AVG_BOTH / OD_ONLY / OI_ONLY / RIM_BASE
        val odPack: EyePack3250?,
        val oiPack: EyePack3250?,
        val relBetweenEyes: Double = Double.NaN
    )

    @Suppress("UNUSED_PARAMETER")
    fun resolve3250(
        hboxInnerMm: Double,
        vboxInnerMm: Double,
        pxPerMmRimBase: Double,
        od: EyeInput3250?,
        oi: EyeInput3250?,
        maxRelErrXY: Double = 0.10,
        maxErrVboxFromX: Double = 0.15,
        maxRelToRimBase: Double = 0.12,
        maxRelBetweenEyes: Double = 0.08
    ): Result3250 {

        val rimBase = pxPerMmRimBase.takeIf { it.isFinite() && it > 1e-6 } ?: 5.0
        val hboxMm = hboxInnerMm.takeIf { it.isFinite() && it > 1e-6 } ?: 1.0
        val vboxMm = vboxInnerMm.takeIf { it.isFinite() && it > 1e-6 } ?: 1.0

        val odPack = buildEyePack3250(
            input = od,
            hboxInnerMm = hboxMm,
            vboxInnerMm = vboxMm,
            pxPerMmRimBase = rimBase
        )

        val oiPack = buildEyePack3250(
            input = oi,
            hboxInnerMm = hboxMm,
            vboxInnerMm = vboxMm,
            pxPerMmRimBase = rimBase
        )

        logEyePack3250("OD", odPack)
        logEyePack3250("OI", oiPack)

        val odOk = odPack?.takeIf { it.acceptedForOfficial }
        val oiOk = oiPack?.takeIf { it.acceptedForOfficial }

        val result = when {
            odOk != null && oiOk != null -> {
                val avg = 0.5 * (odOk.pxPerMmX + oiOk.pxPerMmX)
                val rel = abs(odOk.pxPerMmX - oiOk.pxPerMmX) / avg.coerceAtLeast(1e-6)

                Result3250(
                    pxPerMmOfficial = avg,
                    source = "AVG_BOTH",
                    odPack = odPack,
                    oiPack = oiPack,
                    relBetweenEyes = rel
                )
            }

            odOk != null -> {
                Result3250(
                    pxPerMmOfficial = odOk.pxPerMmX,
                    source = "OD_ONLY",
                    odPack = odPack,
                    oiPack = oiPack
                )
            }

            oiOk != null -> {
                Result3250(
                    pxPerMmOfficial = oiOk.pxPerMmX,
                    source = "OI_ONLY",
                    odPack = odPack,
                    oiPack = oiPack
                )
            }

            else -> {
                Result3250(
                    pxPerMmOfficial = rimBase,
                    source = "RIM_BASE",
                    odPack = odPack,
                    oiPack = oiPack
                )
            }
        }

        Log.d(
            TAG,
            "RESOLVE hbox=${f4(hboxMm)} vbox=${f4(vboxMm)} rimBase=${f4(rimBase)} " +
                    "=> official=${f4(result.pxPerMmOfficial)} src=${result.source} relBoth=${f4(result.relBetweenEyes)}"
        )

        return result
    }

    private fun buildEyePack3250(
        input: EyeInput3250?,
        hboxInnerMm: Double,
        vboxInnerMm: Double,
        pxPerMmRimBase: Double
    ): EyePack3250? {

        input ?: return null

        if (!input.isIndependent) {
            return EyePack3250(
                eyeTag = input.eyeTag,
                spanXpx = Double.NaN,
                spanYpx = Double.NaN,
                pxPerMmX = Double.NaN,
                pxPerMmY = Double.NaN,
                relErrXY = Double.NaN,
                errVboxFromX = Double.NaN,
                relToRimBase = Double.NaN,
                isIndependent = false,
                acceptedForOfficial = false,
                rejectReason = "NOT_INDEPENDENT"
            )
        }

        val placed = input.placedPtsGlobal
        val origin = input.originGlobal
        val rotDeg = input.rotDeg

        if (placed.isNullOrEmpty()) return rejectedEmpty3250(input.eyeTag, "NO_PLACED")
        if (origin == null || !origin.x.isFinite() || !origin.y.isFinite()) {
            return rejectedEmpty3250(input.eyeTag, "BAD_ORIGIN")
        }
        if (rotDeg == null || !rotDeg.isFinite()) {
            return rejectedEmpty3250(input.eyeTag, "BAD_ROT")
        }

        val span = measurePlacedSpan3250(
            placedPtsGlobal = placed,
            originGlobal = origin,
            rotDeg = rotDeg
        ) ?: return rejectedEmpty3250(input.eyeTag, "BAD_SPAN")

        val pxX = span.spanXpx / hboxInnerMm
        val pxY = span.spanYpx / vboxInnerMm

        if (!pxX.isFinite() || !pxY.isFinite() || pxX <= 1e-6 || pxY <= 1e-6) {
            return rejectedWithSpan3250(
                eyeTag = input.eyeTag,
                spanXpx = span.spanXpx,
                spanYpx = span.spanYpx,
                reason = "BAD_PXMM"
            )
        }

        val relErrXY = abs(pxY - pxX) / pxX.coerceAtLeast(1e-6)
        val expSpanYFromX = vboxInnerMm * pxX
        val errVboxFromX = abs(span.spanYpx - expSpanYFromX) / expSpanYFromX.coerceAtLeast(1e-6)
        val relToRimBase = abs(pxX - pxPerMmRimBase) / pxPerMmRimBase.coerceAtLeast(1e-6)

        return EyePack3250(
            eyeTag = input.eyeTag,
            spanXpx = span.spanXpx,
            spanYpx = span.spanYpx,
            pxPerMmX = pxX,
            pxPerMmY = pxY,
            relErrXY = relErrXY,
            errVboxFromX = errVboxFromX,
            relToRimBase = relToRimBase,
            isIndependent = true,
            acceptedForOfficial = true,
            rejectReason = null
        )
    }

    private data class Span3250(
        val spanXpx: Double,
        val spanYpx: Double
    )

    private fun measurePlacedSpan3250(
        placedPtsGlobal: List<PointF>,
        originGlobal: PointF,
        rotDeg: Double
    ): Span3250? {
        if (placedPtsGlobal.isEmpty()) return null
        if (!originGlobal.x.isFinite() || !originGlobal.y.isFinite()) return null
        if (!rotDeg.isFinite()) return null

        val theta = Math.toRadians(rotDeg)
        val c = cos(theta)
        val s = sin(theta)

        var minX = Double.POSITIVE_INFINITY
        var maxX = Double.NEGATIVE_INFINITY
        var minY = Double.POSITIVE_INFINITY
        var maxY = Double.NEGATIVE_INFINITY

        for (p in placedPtsGlobal) {
            val dx = (p.x - originGlobal.x).toDouble()
            val dyUp = (-(p.y - originGlobal.y)).toDouble()

            val xModel = c * dx + s * dyUp
            val yModel = -s * dx + c * dyUp

            if (!xModel.isFinite() || !yModel.isFinite()) continue

            if (xModel < minX) minX = xModel
            if (xModel > maxX) maxX = xModel
            if (yModel < minY) minY = yModel
            if (yModel > maxY) maxY = yModel
        }

        if (!minX.isFinite() || !maxX.isFinite() || !minY.isFinite() || !maxY.isFinite()) {
            return null
        }

        val spanX = maxX - minX
        val spanY = maxY - minY
        if (!spanX.isFinite() || !spanY.isFinite() || spanX <= 1e-6 || spanY <= 1e-6) {
            return null
        }

        return Span3250(
            spanXpx = spanX,
            spanYpx = spanY
        )
    }

    private fun rejectedEmpty3250(
        eyeTag: String,
        reason: String
    ): EyePack3250 =
        EyePack3250(
            eyeTag = eyeTag,
            spanXpx = Double.NaN,
            spanYpx = Double.NaN,
            pxPerMmX = Double.NaN,
            pxPerMmY = Double.NaN,
            relErrXY = Double.NaN,
            errVboxFromX = Double.NaN,
            relToRimBase = Double.NaN,
            isIndependent = true,
            acceptedForOfficial = false,
            rejectReason = reason
        )

    private fun rejectedWithSpan3250(
        eyeTag: String,
        spanXpx: Double,
        spanYpx: Double,
        reason: String
    ): EyePack3250 =
        EyePack3250(
            eyeTag = eyeTag,
            spanXpx = spanXpx,
            spanYpx = spanYpx,
            pxPerMmX = Double.NaN,
            pxPerMmY = Double.NaN,
            relErrXY = Double.NaN,
            errVboxFromX = Double.NaN,
            relToRimBase = Double.NaN,
            isIndependent = true,
            acceptedForOfficial = false,
            rejectReason = reason
        )

    private fun logEyePack3250(
        label: String,
        pack: EyePack3250?
    ) {
        if (pack == null) {
            Log.d(TAG, "EYE[$label] null")
            return
        }

        Log.d(
            TAG,
            "EYE[$label] independent=${pack.isIndependent} accepted=${pack.acceptedForOfficial} " +
                    "reason=${pack.rejectReason ?: "OK"} " +
                    "spanX=${f4(pack.spanXpx)} spanY=${f4(pack.spanYpx)} " +
                    "pxX=${f4(pack.pxPerMmX)} pxY=${f4(pack.pxPerMmY)} " +
                    "relXY=${f4(pack.relErrXY)} errV=${f4(pack.errVboxFromX)} relRim=${f4(pack.relToRimBase)}"
        )
    }

    private fun f4(v: Double?): String =
        if (v == null || !v.isFinite()) "NaN" else "%.4f".format(v)
}
