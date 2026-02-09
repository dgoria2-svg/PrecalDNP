// app/src/main/java/com/dg/precaldnp/vision/FilContourBuilder3250.kt
package com.dg.precaldnp.vision

import android.graphics.PointF
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

object FilContourBuilder3250 {

    data class ContourMm3250(
        val ptsMm: List<PointF>,   // mm, centro (0,0), Y-up
        val minX: Double,
        val maxX: Double,
        val minY: Double,
        val maxY: Double
    )

    data class OdOiContours3250(
        val od: ContourMm3250,
        val oi: ContourMm3250
    )

    /**
     * LEY 3250 AS-IS:
     * idx=0 (R1)  => +X
     * idx=200     => +Y (arriba)
     * idx=400     => -X
     * idx=600     => -Y
     *
     * OJO: esto es coord mm Y-up. En Canvas se invierte Y al dibujar.
     */
    fun buildOdAsIs3250(radiiMm: DoubleArray): ContourMm3250 {
        val n = radiiMm.size
        require(n >= 10) { "radiiMm inválido (n=$n)" }

        val delta = 2.0 * PI / n.toDouble()

        val pts = ArrayList<PointF>(n)
        var minX = Double.POSITIVE_INFINITY
        var maxX = Double.NEGATIVE_INFINITY
        var minY = Double.POSITIVE_INFINITY
        var maxY = Double.NEGATIVE_INFINITY

        for (i in 0 until n) {
            val r = radiiMm[i]
            val th = i.toDouble() * delta

            // AS-IS: idx 0 => +X, idx 200 => +Y
            val x = r * cos(th)
            val y = r * sin(th)

            pts.add(PointF(x.toFloat(), y.toFloat()))
            minX = min(minX, x); maxX = max(maxX, x)
            minY = min(minY, y); maxY = max(maxY, y)
        }

        return ContourMm3250(pts, minX, maxX, minY, maxY)
    }

    /** OI = espejo de OD en X (solo X cambia de signo, Y se mantiene). */
    fun flipOdToOi3250(od: ContourMm3250): ContourMm3250 {
        val ptsOi = ArrayList<PointF>(od.ptsMm.size)
        var minX = Double.POSITIVE_INFINITY
        var maxX = Double.NEGATIVE_INFINITY
        var minY = Double.POSITIVE_INFINITY
        var maxY = Double.NEGATIVE_INFINITY

        for (p in od.ptsMm) {
            val x = -p.x.toDouble()
            val y =  p.y.toDouble()
            ptsOi.add(PointF(x.toFloat(), y.toFloat()))
            minX = min(minX, x); maxX = max(maxX, x)
            minY = min(minY, y); maxY = max(maxY, y)
        }
        return ContourMm3250(ptsOi, minX, maxX, minY, maxY)
    }

    fun buildOdOiFromFilTextAsIs3250(filText: String): OdOiContours3250 {
        val parsed = FilParser3250.parseFromText(filText)
        val od = buildOdAsIs3250(parsed.radiiMm)
        val oi = flipOdToOi3250(od)
        return OdOiContours3250(od, oi)
    }

    /** Cardinales FIJOS por índice (no por extremos). */
    fun fixedIndexCardinals3250(n: Int): IntArray {
        fun idx(v800: Int): Int = ((v800 / 800.0) * n).toInt().coerceIn(0, n - 1)
        return intArrayOf(idx(0), idx(200), idx(400), idx(600))
    }
}
