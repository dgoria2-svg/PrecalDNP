package com.dg.precaldnp.vision

import android.graphics.PointF
import com.dg.precaldnp.model.MeasurementResult
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * InnerRimSnapper3250 (REINVENTADO) – DRAW GEOMETRY HELPER
 *
 * Ya NO “snapea” contornos.
 * Dado un outline (px) + pupila (px) + MeasurementResult (mm),
 * devuelve geometría para dibujar:
 *  - bottomYpx en X≈pupil.x (intersección con el contorno)
 *  - usefulRadiusPx = maxDist(pupil, outline) + (2mm * pxPerMmDerived)
 *
 * pxPerMmDerived se estima SOLO para el +2mm:
 *  - preferimos ancho: bboxWpx / res.anchoMm
 *  - fallback: bboxHpx / res.altoMm
 * Si no hay mm finitos, marginPx = 0.
 */
class InnerRimSnapper3250 {

    data class DrawGeom3250(
        val bottomYpx: Float?,
        val farPointPx: PointF?,
        val usefulRadiusPx: Float?,
        val marginPx: Float,
        val reason: String
    )

    fun computeDrawGeom3250(
        outlinePx: List<PointF>?,
        pupilPx: PointF,
        res: MeasurementResult?,
        usefulMarginMm: Double = 2.0,
        tolXpx: Float = 8f
    ): DrawGeom3250 {

        if (outlinePx.isNullOrEmpty() || outlinePx.size < 3) {
            return DrawGeom3250(
                bottomYpx = null,
                farPointPx = null,
                usefulRadiusPx = null,
                marginPx = 0f,
                reason = "outline=null/short"
            )
        }
        if (!pupilPx.x.isFinite() || !pupilPx.y.isFinite()) {
            return DrawGeom3250(
                bottomYpx = null,
                farPointPx = null,
                usefulRadiusPx = null,
                marginPx = 0f,
                reason = "pupil invalid"
            )
        }

        val bbox = bboxOf(outlinePx)
        val pxPerMm = derivePxPerMmForMargin(bbox, res)
        val marginPx = if (pxPerMm.isFinite() && pxPerMm > 1e-6) {
            (usefulMarginMm * pxPerMm).toFloat()
        } else 0f

        val bottomY = bottomIntersectionYAtX(outlinePx, pupilPx.x, tolXpx)

        val far = farthestPoint(outlinePx, pupilPx)
        val rPx = far?.let {
            val r = hypot((it.x - pupilPx.x).toDouble(), (it.y - pupilPx.y).toDouble()).toFloat()
            if (r.isFinite() && r > 1f) (r + marginPx) else null
        }

        return DrawGeom3250(
            bottomYpx = bottomY,
            farPointPx = far,
            usefulRadiusPx = rPx,
            marginPx = marginPx,
            reason = "ok"
        )
    }

    // ---------------- helpers ----------------

    private data class BBox(val minX: Float, val minY: Float, val maxX: Float, val maxY: Float) {
        val w: Float get() = (maxX - minX).coerceAtLeast(0f)
        val h: Float get() = (maxY - minY).coerceAtLeast(0f)
    }

    private fun bboxOf(pts: List<PointF>): BBox {
        var minX = Float.POSITIVE_INFINITY
        var minY = Float.POSITIVE_INFINITY
        var maxX = Float.NEGATIVE_INFINITY
        var maxY = Float.NEGATIVE_INFINITY
        for (p in pts) {
            if (!p.x.isFinite() || !p.y.isFinite()) continue
            minX = min(minX, p.x)
            minY = min(minY, p.y)
            maxX = max(maxX, p.x)
            maxY = max(maxY, p.y)
        }
        if (!minX.isFinite() || !minY.isFinite() || !maxX.isFinite() || !maxY.isFinite()) {
            return BBox(0f, 0f, 0f, 0f)
        }
        return BBox(minX, minY, maxX, maxY)
    }

    /** SOLO para convertir 2mm→px del margen. Preferimos ancho. */
    private fun derivePxPerMmForMargin(b: BBox, res: MeasurementResult?): Double {
        if (res == null) return Double.NaN
        val aMm = res.anchoMm
        val bMm = res.altoMm

        val candW = if (aMm.isFinite() && aMm > 1e-6 && b.w > 2f) b.w.toDouble() / aMm else Double.NaN
        val candH = if (bMm.isFinite() && bMm > 1e-6 && b.h > 2f) b.h.toDouble() / bMm else Double.NaN

        return when {
            candW.isFinite() -> candW
            candH.isFinite() -> candH
            else -> Double.NaN
        }
    }

    private fun farthestPoint(pts: List<PointF>, pupil: PointF): PointF? {
        var bestD2 = -1.0
        var best: PointF? = null
        for (p in pts) {
            if (!p.x.isFinite() || !p.y.isFinite()) continue
            val dx = (p.x - pupil.x).toDouble()
            val dy = (p.y - pupil.y).toDouble()
            val d2 = dx * dx + dy * dy
            if (d2 > bestD2) {
                bestD2 = d2
                best = p
            }
        }
        return best
    }

    /**
     * Intersección robusta de vertical x=c con el polígono (cerrado).
     * Devuelve la Y máxima (bottom) dentro de tolerancia.
     */
    private fun bottomIntersectionYAtX(pts: List<PointF>, x: Float, tolX: Float): Float? {
        if (pts.size < 3) return null
        val n = pts.size

        var bestY: Float? = null

        fun considerY(y: Float) {
            if (!y.isFinite()) return
            if (bestY == null || y > bestY!!) bestY = y
        }

        for (i in 0 until n) {
            val p1 = pts[i]
            val p2 = pts[(i + 1) % n]
            if (!p1.x.isFinite() || !p1.y.isFinite() || !p2.x.isFinite() || !p2.y.isFinite()) continue

            val x1 = p1.x
            val y1 = p1.y
            val x2 = p2.x
            val y2 = p2.y

            val dx = x2 - x1
            // segmento casi vertical
            if (abs(dx) < 1e-6f) {
                if (abs(x - x1) <= tolX) {
                    considerY(y1)
                    considerY(y2)
                }
                continue
            }

            // ¿x cae dentro del rango del segmento (con tol)?
            val minX = min(x1, x2) - tolX
            val maxX = max(x1, x2) + tolX
            if (x < minX || x > maxX) continue

            // t para x=c
            val t = (x - x1) / dx
            if (t < -0.05f || t > 1.05f) continue

            val y = y1 + t * (y2 - y1)
            considerY(y)
        }

        return bestY
    }
}
