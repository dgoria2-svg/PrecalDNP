package com.dg.precaldnp.vision

import android.graphics.PointF
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.imgproc.Imgproc

object FaceMaskHull3250 {
    fun convexHull(points: List<PointF>): List<PointF> {
        if (points.size < 3) return emptyList()
        var mop: MatOfPoint? = null
        var hullIdx: MatOfInt? = null
        return try {
            val pts = points.map { Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray()
            mop = MatOfPoint(*pts)
            hullIdx = MatOfInt()
            Imgproc.convexHull(mop, hullIdx)

            val idx = hullIdx.toArray()
            val arr = mop.toArray()
            val out = ArrayList<PointF>(idx.size)
            for (i in idx) {
                val p = arr[i]
                out.add(PointF(p.x.toFloat(), p.y.toFloat()))
            }
            out
        } catch (_: Throwable) {
            emptyList()
        } finally {
            try { hullIdx?.release() } catch (_: Throwable) {}
            try { mop?.release() } catch (_: Throwable) {}
        }
    }
}
