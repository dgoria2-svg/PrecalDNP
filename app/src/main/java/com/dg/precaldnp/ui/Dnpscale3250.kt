package com.dg.precaldnp.ui

import com.dg.precaldnp.vision.RimDetectionResult
import kotlin.math.abs

object DnpScale3250 {

    data class SeedOut3250(
        val seedPxPerMm: Float,
        val pxPerMmOd: Float?,
        val pxPerMmOi: Float?,
        val src: String
    )

    /**
     * ScaleONE (3250):
     * - Usa HBOX INNER mm (ya descontado lo que definas por lado)
     * - Acepta 1 ojo si el otro falla
     * - Si hay 2: promedia si son coherentes; si no, elige el más cercano al guess (o fallback estable)
     */
    fun pxPerMmSeedFromRimWidth3250(
        rimOd: RimDetectionResult?,
        rimOi: RimDetectionResult?,
        hboxInnerMm: Double,
        pxPerMmGuessFace: Float,
        tag: String
    ): SeedOut3250 {

        // Guardrail de entrada
        if (!hboxInnerMm.isFinite() || hboxInnerMm <= 1e-9) {
            val src0 = if (tag.isNotBlank()) "$tag:NONE_BAD_HBOX" else "NONE_BAD_HBOX"
            return SeedOut3250(Float.NaN, null, null, src0)
        }

        fun pxFrom(r: RimDetectionResult?): Float? {
            if (r == null || !r.ok) return null

            val wPx = r.innerWidthPx
            if (!wPx.isFinite() || wPx <= 1f) return null

            val pxmm = (wPx / hboxInnerMm.toFloat())
            return pxmm.takeIf { it.isFinite() && it > 1e-6f }
        }

        val od = pxFrom(rimOd)
        val oi = pxFrom(rimOi)

        // Si vienen ambos, definimos coherencia y/o elegimos el mejor contra el guess
        val tolRel = 0.18f // 18%: ajustalo si querés más estricto (0.12) o más laxo (0.25)

        val used: Float? = when {
            od != null && oi != null -> {
                val avg = (od + oi) * 0.5f
                val rel = abs(od - oi) / (avg.coerceAtLeast(1e-6f))

                if (rel <= tolRel) {
                    avg
                } else {
                    // incoherentes -> elegimos el más cercano al guess (si el guess es usable)
                    val g = if (pxPerMmGuessFace.isFinite() && pxPerMmGuessFace > 1e-6f) pxPerMmGuessFace else Float.NaN
                    if (g.isFinite()) {
                        val dOd = abs(od - g)
                        val dOi = abs(oi - g)
                        if (dOd <= dOi) od else oi
                    } else {
                        // sin guess confiable -> fallback determinístico: mediana simple (o elegí od)
                        avg
                    }
                }
            }

            od != null -> od
            oi != null -> oi
            else -> null
        }

        val srcCore = when {
            od != null && oi != null -> {
                val avg = (od + oi) * 0.5f
                val rel = abs(od - oi) / (avg.coerceAtLeast(1e-6f))
                if (rel <= tolRel) "rimBOTH_W" else {
                    // cuál se eligió (si no se promedió)
                    if (used == od) "rimBOTH_W_PICK_OD" else if (used == oi) "rimBOTH_W_PICK_OI" else "rimBOTH_W_PICK_AVG"
                }
            }
            od != null -> "rimOD_W"
            oi != null -> "rimOI_W"
            else -> "NONE"
        }

        val src = if (tag.isNotBlank()) "$tag:$srcCore" else srcCore

        return if (used == null || !used.isFinite() || used <= 1e-6f) {
            SeedOut3250(Float.NaN, od, oi, src)
        } else {
            SeedOut3250(used, od, oi, src)
        }
    }
}
