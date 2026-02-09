package com.dg.precaldnp.fil

import android.graphics.PointF
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

object FilR {
    fun parseRHundredths(text: String): IntArray {
        val out = ArrayList<Int>()
        text.lineSequence().forEach { raw ->
            val line = raw.trim()
            if (line.startsWith("R=")) {
                val body = line.substring(2)
                body.split(';').forEach { t -> t.trim().toIntOrNull()?.let(out::add) }
            }
        }
        return out.toIntArray()
    }

    fun rotateR(r: IntArray, angleDeg: Int): IntArray {
        if (r.isEmpty()) return r
        val n = r.size
        val shift = ((angleDeg / 360.0) * n).toInt().mod(n)
        if (shift == 0) return r.clone()
        val out = IntArray(n)
        for (i in 0 until n) out[i] = r[(i - shift + n) % n]
        return out
    }

    /** Convierte R (centro 0,0 en mm) a polilínea (mm). Convenio: x=-r*cos θ ; y= r*sin θ */
    fun rToPolylineMm(r: IntArray): List<PointF> {
        val n = r.size
        val dth = 2.0 * Math.PI / n
        val pts = ArrayList<PointF>(n + 1)
        for (i in 0 until n) {
            val mm = r[i] / 100.0
            val a  = dth * i
            pts.add(PointF((-mm * cos(a)).toFloat(), (mm * sin(a)).toFloat()))
        }
        if (n >= 1) pts.add(pts.first())
        return pts
    }

    /** Mapea mm→px usando centro (cx,cy) de buffer y px/mm. */
    fun mmToPx(ptsMm: List<PointF>, cx: Float, cy: Float, pxPerMm: Float): List<PointF> =
        ptsMm.map { p -> PointF(cx + p.x * pxPerMm, cy - p.y * pxPerMm) }

    data class Geo(val hbox: Double, val vbox: Double, val fed: Double)
    fun geoFromR(r: IntArray): Geo {
        val pts = rToPolylineMm(r)
        var minX=1e9; var maxX=-1e9; var minY=1e9; var maxY=-1e9; var rMax=0.0
        for (p in pts) {
            val x=p.x.toDouble(); val y=p.y.toDouble()
            if (x<minX) minX=x; if (x>maxX) maxX=x
            if (y<minY) minY=y; if (y>maxY) maxY=y
            rMax = maxOf(rMax, hypot(x,y))
        }
        return Geo(maxX-minX, maxY-minY, 2.0*rMax)
    }
}
