package com.dg.precaldnp.vision

import android.graphics.PointF
import android.graphics.RectF
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Cálculos numéricos de DNP / alturas / puente / diámetro útil
 * usando:
 *  - pupilas en píxeles,
 *  - midline en píxeles,
 *  - escala de cara (pxPerMmFace),
 *  - contorno del FIL (outline en píxeles) o rectángulo de fallback.
 */
object DnpMeasureComputer3250 {

    data class PupilPair(
        val od: PointF,
        val oi: PointF
    )

    data class ResultMm(
        val dnpOd: Double,
        val dnpOi: Double,
        val puente: Double,
        val altOd: Double,
        val altOi: Double,
        val diamUtilOd: Double,
        val diamUtilOi: Double
    )

    fun computeDnpAltBridgeAndDiameter(
        pupils: PupilPair,
        midlineX: Double,
        pxPerMmFace: Double,
        filFace: FilFace3250,
        boxOd: RectF,
        boxOi: RectF,
        outlineOd: List<PointF>?,
        outlineOi: List<PointF>?
    ): ResultMm {
        require(pxPerMmFace > 0.0) { "pxPerMmFace debe ser > 0" }

        val pxToMm = 1.0 / pxPerMmFace

        // --- DNP OD/OI ---------------------------------------------------
        val dnpOd = abs(pupils.od.x.toDouble() - midlineX) * pxToMm
        val dnpOi = abs(pupils.oi.x.toDouble() - midlineX) * pxToMm

        // --- Puente: borde interno del FIL (HBOX/2) ----------------------
        val filHalfWpx = (filFace.hboxMm * filFace.pxPerMmFace * 0.5)
        val cxOd = boxOd.centerX().toDouble()
        val cxOi = boxOi.centerX().toDouble()
        val filRightOdX = cxOd + filHalfWpx
        val filLeftOiX = cxOi - filHalfWpx
        val puentePx = filLeftOiX - filRightOdX
        val puente = max(0.0, puentePx * pxToMm)

        // --- Alturas: pupila -> base FIL (VBOX/2) ------------------------
        val filHalfHpx = (filFace.vboxMm * filFace.pxPerMmFace * 0.5)
        val bottomOdY = cxCenterY(boxOd) + filHalfHpx
        val bottomOiY = cxCenterY(boxOi) + filHalfHpx

        val altOd = max(0.0, (bottomOdY - pupils.od.y.toDouble()) * pxToMm)
        val altOi = max(0.0, (bottomOiY - pupils.oi.y.toDouble()) * pxToMm)

        // --- Diámetro útil contra FIL (fallback a caja si no hay outline) -
        val diamOd = diameterUtilMm(
            pupil = pupils.od,
            outline = outlineOd,
            fallbackRect = boxOd,
            pxPerMmFace = pxPerMmFace
        )
        val diamOi = diameterUtilMm(
            pupil = pupils.oi,
            outline = outlineOi,
            fallbackRect = boxOi,
            pxPerMmFace = pxPerMmFace
        )

        return ResultMm(
            dnpOd = dnpOd,
            dnpOi = dnpOi,
            puente = puente,
            altOd = altOd,
            altOi = altOi,
            diamUtilOd = diamOd,
            diamUtilOi = diamOi
        )
    }

    // --------------------------------------------------------------------
    // Helpers
    // --------------------------------------------------------------------

    private fun cxCenterY(r: RectF): Double =
        (r.top + r.bottom).toDouble() * 0.5

    private fun diameterUtilMm(
        pupil: PointF,
        outline: List<PointF>?,
        fallbackRect: RectF,
        pxPerMmFace: Double
    ): Double {
        val dPx = outline
            ?.takeIf { it.size >= 2 }
            ?.let { minDistToOutline(pupil, it) }
            ?: minDistToRectEdge(pupil, fallbackRect)

        return 2.0 * dPx / pxPerMmFace
    }

    // Distancia mínima punto-contorno poligonal cerrado (píxeles)
    private fun minDistToOutline(pupil: PointF, outline: List<PointF>): Float {
        if (outline.size < 2) return Float.POSITIVE_INFINITY

        var best = Float.POSITIVE_INFINITY
        var prev = outline[0]
        for (i in 1 until outline.size) {
            val cur = outline[i]
            val d = distPointToSegment(pupil, prev, cur)
            if (d < best) best = d
            prev = cur
        }
        // cierre
        val dClose = distPointToSegment(pupil, outline.last(), outline.first())
        if (dClose < best) best = dClose

        return best
    }

    private fun distPointToSegment(p: PointF, a: PointF, b: PointF): Float {
        val vx = b.x - a.x
        val vy = b.y - a.y
        val wx = p.x - a.x
        val wy = p.y - a.y

        val c1 = vx * wx + vy * wy
        if (c1 <= 0f) return hypot(p.x - a.x, p.y - a.y)

        val c2 = vx * vx + vy * vy
        if (c2 <= c1) return hypot(p.x - b.x, p.y - b.y)

        val t = c1 / c2
        val projX = a.x + t * vx
        val projY = a.y + t * vy
        return hypot(p.x - projX, p.y - projY)
    }

    // Fallback: distancia mínima a los lados del rectángulo (píxeles)
    private fun minDistToRectEdge(p: PointF, r: RectF): Float {
        val dxLeft = abs(p.x - r.left)
        val dxRight = abs(r.right - p.x)
        val dyTop = abs(p.y - r.top)
        val dyBottom = abs(r.bottom - p.y)
        return min(
            min(dxLeft, dxRight),
            min(dyTop, dyBottom)
        )
    }
}
