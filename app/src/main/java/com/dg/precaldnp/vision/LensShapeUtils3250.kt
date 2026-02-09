// app/src/main/java/com/dg/precaldnp/vision/LensShapeUtils3250.kt
package com.dg.precaldnp.vision

import android.graphics.PointF
import com.dg.precaldnp.model.ShapeTraceResult
import kotlin.math.sqrt

/**
 * Representación de la lente en mm, centrada en (0,0) con eje Y hacia arriba.
 * El origen (0,0) es el centro geométrico de la HBOX/VBOX (bbox del contorno).
 */
data class LensShapeMm3250(
    val outlineMm: List<PointF>, // polilínea cerrada
    val hboxMm: Float,
    val vboxMm: Float
)

/**
 * Métricas “resumidas” del trazado:
 * - hboxMm / vboxMm: caja del FIL en mm.
 * - eyesizMm: diagonal efectiva de la lente (2·Rmax sobre el contorno).
 */
data class LensShapeInfo3250(
    val hboxMm: Double,
    val vboxMm: Double,
    val eyesizMm: Double,
)

/**
 * Utilidades para construir la forma de lente en mm a partir del ShapeTraceResult.
 */
object LensShapeUtils3250 {

    /**
     * Toma el contorno en píxeles del PRECAL (ShapeTraceResult) y lo pasa a mm
     * centrado en el centro de la HBOX/VBOX, con Y hacia arriba.
     *
     * - Usa pxPerMm del propio ShapeTraceResult.
     * - Devuelve null si falta escala o no hay puntos.
     */
    fun fromTrace(trace: ShapeTraceResult): LensShapeMm3250? {
        val ptsPx = trace.outlinePx
        val ppm = trace.pxPerMm
        if (ptsPx.isEmpty() || ppm <= 0f) return null

        // ---- 1) BBox en PIXELES ------------------------------------------
        var minXpx = Float.POSITIVE_INFINITY
        var maxXpx = Float.NEGATIVE_INFINITY
        var minYpx = Float.POSITIVE_INFINITY
        var maxYpx = Float.NEGATIVE_INFINITY

        for (p in ptsPx) {
            if (p.x < minXpx) minXpx = p.x
            if (p.x > maxXpx) maxXpx = p.x
            if (p.y < minYpx) minYpx = p.y
            if (p.y > maxYpx) maxYpx = p.y
        }

        val cxPx = (minXpx + maxXpx) * 0.5f
        val cyPx = (minYpx + maxYpx) * 0.5f

        // ---- 2) Pasamos a mm centrando en el centro de la caja -----------
        val invPpm = 1f / ppm
        val out = ArrayList<PointF>(ptsPx.size + 1)

        var minXmm = Float.POSITIVE_INFINITY
        var maxXmm = Float.NEGATIVE_INFINITY
        var minYmm = Float.POSITIVE_INFINITY
        var maxYmm = Float.NEGATIVE_INFINITY

        for (p in ptsPx) {
            val dxMm = (p.x - cxPx) * invPpm
            val dyMm = (p.y - cyPx) * invPpm

            // Convención mm: Y hacia ARRIBA
            val xMm = dxMm
            val yMm = -dyMm

            if (xMm < minXmm) minXmm = xMm
            if (xMm > maxXmm) maxXmm = xMm
            if (yMm < minYmm) minYmm = yMm
            if (yMm > maxYmm) maxYmm = yMm

            out.add(PointF(xMm, yMm))
        }

        // Cierre exacto de la polilínea
        if (out.isNotEmpty()) {
            val first = out.first()
            val last = out.last()
            if (first.x != last.x || first.y != last.y) {
                out.add(PointF(first.x, first.y))
            }
        }

        val hbox = (maxXmm - minXmm).coerceAtLeast(0f)
        val vbox = (maxYmm - minYmm).coerceAtLeast(0f)

        return LensShapeMm3250(
            outlineMm = out,
            hboxMm = hbox,
            vboxMm = vbox
        )
    }

    /**
     * Versión “compacta”: devuelve sólo HBOX/VBOX/EYESIZ del trazado.
     *
     * - EYESIZ se calcula como 2·Rmax, donde Rmax es la distancia máxima
     *   del centro (0,0) a cualquier punto del contorno en mm.
     */
    fun infoFromTrace(trace: ShapeTraceResult): LensShapeInfo3250? {
        val shapeMm = fromTrace(trace) ?: return null

        var maxR2 = 0.0
        for (p in shapeMm.outlineMm) {
            val x = p.x.toDouble()
            val y = p.y.toDouble()
            val r2 = x * x + y * y
            if (r2 > maxR2) maxR2 = r2
        }

        val eyesizMm = if (maxR2 > 0.0) 2.0 * sqrt(maxR2) else 0.0

        return LensShapeInfo3250(
            hboxMm = shapeMm.hboxMm.toDouble(),
            vboxMm = shapeMm.vboxMm.toDouble(),
            eyesizMm = eyesizMm
        )
    }

    /**
     * Proyecta la forma de lente en mm (centrada en 0,0; Y hacia arriba)
     * a coordenadas de la FOTO (0,0 arriba-izquierda; Y hacia abajo).
     *
     * @param pxPerMmFactor  escala de la CARA (px por mm).
     * @param centerXpx      centro del aro en la foto (en píxeles).
     * @param centerYpx      centro del aro en la foto (en píxeles).
     * @param mirrorHoriz    si true, espeja horizontalmente (para usar mismo FIL en OI).
     */
    fun outlineMmToFacePx(
        shape: LensShapeMm3250,
        pxPerMmFactor: Float,
        centerXpx: Float,
        centerYpx: Float,
        mirrorHoriz: Boolean
    ): List<PointF> {
        val s = pxPerMmFactor.toDouble()
        val cx = centerXpx.toDouble()
        val cy = centerYpx.toDouble()

        val out = ArrayList<PointF>(shape.outlineMm.size)

        for (pMm in shape.outlineMm) {
            var xMm = pMm.x.toDouble()
            val yMm = pMm.y.toDouble()

            if (mirrorHoriz) {
                xMm = -xMm
            }

            // En mm el eje Y va hacia arriba; en imagen, hacia abajo.
            val xPx = (cx + xMm * s).toFloat()
            val yPx = (cy - yMm * s).toFloat()

            out.add(PointF(xPx, yPx))
        }

        // Aseguramos cierre si venía cerrado
        if (out.isNotEmpty()) {
            val first = out.first()
            val last = out.last()
            if (first.x != last.x || first.y != last.y) {
                out.add(PointF(first.x, first.y))
            }
        }

        return out
    }
}

/**
 * Rota la forma en mm alrededor del origen (0,0) un ángulo en grados.
 */
fun rotateLensShape3250(shape: LensShapeMm3250, degrees: Float): LensShapeMm3250 {
    if (degrees == 0f) return shape

    val rad = Math.toRadians(degrees.toDouble())
    val cosT = kotlin.math.cos(rad).toFloat()
    val sinT = kotlin.math.sin(rad).toFloat()

    val rotated = ArrayList<PointF>(shape.outlineMm.size)
    var minX = Float.POSITIVE_INFINITY
    var maxX = Float.NEGATIVE_INFINITY
    var minY = Float.POSITIVE_INFINITY
    var maxY = Float.NEGATIVE_INFINITY

    for (p in shape.outlineMm) {
        val x = p.x
        val y = p.y
        val xr = x * cosT - y * sinT
        val yr = x * sinT + y * cosT

        rotated.add(PointF(xr, yr))

        if (xr < minX) minX = xr
        if (xr > maxX) maxX = xr
        if (yr < minY) minY = yr
        if (yr > maxY) maxY = yr
    }

    val hbox = (maxX - minX).coerceAtLeast(0f)
    val vbox = (maxY - minY).coerceAtLeast(0f)

    return LensShapeMm3250(
        outlineMm = rotated,
        hboxMm = hbox,
        vboxMm = vbox
    )
}

/**
 * Distancia mínima desde un punto (pupila) al contorno del FIL en píxeles.
 *
 * - `outlinePx` es una polilínea (idealmente cerrada) en coords de la foto.
 * - Se calcula la mínima distancia punto–segmento.
 */
fun minDistToOutlinePxFil3250(pupil: PointF, outlinePx: List<PointF>): Float {
    if (outlinePx.size < 2) return 0f

    var best = Float.POSITIVE_INFINITY
    val px = pupil.x.toDouble()
    val py = pupil.y.toDouble()

    fun distPointToSeg(ax: Double, ay: Double, bx: Double, by: Double): Double {
        val vx = bx - ax
        val vy = by - ay
        val wx = px - ax
        val wy = py - ay

        val c1 = vx * wx + vy * wy
        val c2 = vx * vx + vy * vy
        val t = if (c2 > 0.0) (c1 / c2).coerceIn(0.0, 1.0) else 0.0

        val projX = ax + t * vx
        val projY = ay + t * vy

        val dx = px - projX
        val dy = py - projY
        return sqrt(dx * dx + dy * dy)
    }

    for (i in 0 until outlinePx.size - 1) {
        val a = outlinePx[i]
        val b = outlinePx[i + 1]
        val d = distPointToSeg(a.x.toDouble(), a.y.toDouble(), b.x.toDouble(), b.y.toDouble())
        if (d < best) best = d.toFloat()
    }

    return if (best.isFinite()) best else 0f
}

/**
 * Alias viejo por si quedó alguna llamada sin sufijo.
 */
fun minDistToOutlinePx(pupil: PointF, outlinePx: List<PointF>): Float =
    minDistToOutlinePxFil3250(pupil, outlinePx)
