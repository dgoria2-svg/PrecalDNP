package com.dg.precaldnp.ui

import com.dg.precaldnp.vision.RimDetectionResult

object DnpScale3250 {

    data class SeedOut3250(
        val seedPxPerMm: Float,
        val pxPerMmOd: Float?,
        val pxPerMmOi: Float?,
        val src: String
    )

    /**
     * ScaleONE:
     * - Usa HBOX INNER mm (ya descontado 0.3mm/lado o lo que definas)
     * - Acepta 1 ojo si el otro falla.
     * - Si hay 2, promedia si son coherentes, si no elige el más confiable.
     */
    fun pxPerMmSeedFromRimWidth3250(
        rimOd: RimDetectionResult?,
        rimOi: RimDetectionResult?,
        hboxInnerMm: Double,
        pxPerMmGuessFace: Float,
        tag: String
    ): SeedOut3250 {

        fun pxFrom(r: RimDetectionResult?): Float? {
            if (r == null || !r.ok) return null
            val w = r.innerWidthPx
            if (!w.isFinite() || w <= 1f) return null
            val pxmm = (w / hboxInnerMm.toFloat())
            return pxmm.takeIf { it.isFinite() && it > 1e-6f }
        }

        val od = pxFrom(rimOd)
        val oi = pxFrom(rimOi)

// ✅ si uno solo existe, usamos ese
        val used = when {
            od != null && oi != null -> ((od + oi) * 0.5f)   // o mediana, pero promedio basta
            od != null -> od
            oi != null -> oi
            else -> null
        }

        val src = when {
            od != null && oi != null -> "rimBOTH_W"
            od != null -> "rimOD_W"
            oi != null -> "rimOI_W"
            else -> "NONE"
        }

        if (used == null) {
            return SeedOut3250(Float.NaN, null, null, src)
        }

// (opcional) si querés guardar también “qué ojo dio escala”
        return SeedOut3250(used, rimOd as Float?, rimOi as Float?, src)

    }
}