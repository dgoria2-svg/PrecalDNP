@file:Suppress("SameParameterValue")

package com.dg.precaldnp.vision

import android.graphics.PointF
import android.util.Log
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * RimDtRefiner3250
 *
 * Core MDPI-style matcher:
 *   edge map -> distanceTransform -> score = mean(distance) over polyline points
 *
 * - score: MENOR es mejor.
 * - edges01: ByteArray w*h con 0/1 o 0/255 (cualquier nonzero se trata como edge).
 */
object RimDtRefiner3250 {

    private const val TAG = "RimDtRefiner3250"

    data class DtField3250(
        val w: Int,
        val h: Int,
        val dist: FloatArray // w*h, distancia (px) al borde más cercano
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false

            other as DtField3250

            if (w != other.w) return false
            if (h != other.h) return false
            if (!dist.contentEquals(other.dist)) return false

            return true
        }

        override fun hashCode(): Int {
            var result = w
            result = 31 * result + h
            result = 31 * result + dist.contentHashCode()
            return result
        }
    }

    /**
     * Crea campo de distancia a borde desde edge map binario.
     * distanceTransform mide distancia a los PIXELES CERO, así que invertimos:
     *   inv = 255 donde NO hay borde, 0 donde hay borde.
     */
    fun buildDtFieldFromEdgeMap3250(edges01: ByteArray, w: Int, h: Int): DtField3250? {
        if (w <= 1 || h <= 1) return null
        if (edges01.size < w * h) return null

        var edgeMat: Mat? = null
        var inv: Mat? = null
        var distMat: Mat? = null

        try {
            // edgeMat: 0/255
            edgeMat = Mat(h, w, CvType.CV_8U)
            val edge255 = ByteArray(w * h)
            for (i in 0 until w * h) {
                val e = edges01[i].toInt() and 0xFF
                edge255[i] = if (e != 0) 255.toByte() else 0.toByte()
            }
            edgeMat.put(0, 0, edge255)

            // inv: fondo 255, borde 0
            inv = Mat(h, w, CvType.CV_8U)
            Imgproc.threshold(edgeMat, inv, 0.0, 255.0, Imgproc.THRESH_BINARY_INV)

            // close leve para micro-gaps (opcional, pero ayuda)
            val k = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0, 3.0))
            Imgproc.morphologyEx(inv, inv, Imgproc.MORPH_CLOSE, k)

            // distance transform (CV_32F)
            distMat = Mat()
            Imgproc.distanceTransform(inv, distMat, Imgproc.DIST_L2, 3)

            val out = FloatArray(w * h)
            distMat.get(0, 0, out)

            // sanity (no rompe)
            var maxD = 0f
            for (v in out) if (v.isFinite() && v > maxD) maxD = v
            if (maxD <= 1e-4f) {
                Log.w(TAG, "DT max ~ 0 (edge map vacío o invertido mal)")
            }

            return DtField3250(w, h, out)
        } catch (t: Throwable) {
            Log.e(TAG, "buildDtFieldFromEdgeMap3250 error", t)
            return null
        } finally {
            try { distMat?.release() } catch (_: Throwable) {}
            try { inv?.release() } catch (_: Throwable) {}
            try { edgeMat?.release() } catch (_: Throwable) {}
        }
    }

    /**
     * Score = media(distancia) en los puntos de la polilínea.
     * oobPenalty se suma si el punto cae fuera del ROI.
     */
    fun scorePolylineDt3250(
        field: DtField3250,
        pts: List<PointF>,
        oobPenalty: Float = 12f
    ): Float {
        if (pts.isEmpty()) return Float.POSITIVE_INFINITY

        val w = field.w
        val h = field.h
        val dist = field.dist

        var s = 0f
        var n = 0

        for (p in pts) {
            val x = p.x.roundToInt()
            val y = p.y.roundToInt()
            val v = if (x in 0 until w && y in 0 until h) dist[y * w + x] else oobPenalty
            s += if (v.isFinite()) v else oobPenalty
            n++
        }
        return if (n > 0) s / n.toFloat() else Float.POSITIVE_INFINITY
    }

    /**
     * Selecciona sólo la parte inferior del contorno por Y.
     * frac = 0.60 => conserva ~40% inferior (Y grandes).
     */
    fun selectBottomByYFrac3250(pts: List<PointF>, frac: Float): List<PointF> {
        if (pts.isEmpty()) return pts
        val f = frac.coerceIn(0f, 1f)
        if (f <= 0.01f) return pts

        val ys = pts.map { it.y }.sorted()
        val idx = ((ys.size - 1) * f).roundToInt().coerceIn(0, ys.size - 1)
        val thr = ys[idx]
        return pts.filter { it.y >= thr }
    }

    /**
     * Pivot recomendado (ROI-local) para rot/DT: bottom-center detectado.
     * rim.roiPx es GLOBAL; convertimos a ROI-local restando roiPx.left/top.
     */
    fun pivotFromRimResultToLocal3250(rim: RimDetectionResult): PointF {
        val cxG = 0.5f * (rim.innerLeftXpx + rim.innerRightXpx)
        val cyG = rim.bottomYpx
        return PointF(cxG - rim.roiPx.left, cyG - rim.roiPx.top)
    }

    // ------------------------------------------------------------
    // (Opcional) búsqueda local completa por DT (si más adelante la querés)
    // ------------------------------------------------------------

    data class DtCandidate3250(
        val score: Float,
        val dx: Float,
        val dy: Float,
        val scale: Float,
        val rotDeg: Float,
        val ptsOut: List<PointF>
    )

    fun refineByDtLocalSearch3250(
        field: DtField3250,
        ptsIn: List<PointF>,
        pivot: PointF,
        dxMax: Int = 18,
        dyMax: Int = 14,
        dxStep: Int = 2,
        dyStep: Int = 2,
        scaleMin: Float = 0.96f,
        scaleMax: Float = 1.04f,
        scaleStep: Float = 0.01f,
        rotMinDeg: Float = -3.0f,
        rotMaxDeg: Float = +3.0f,
        rotStepDeg: Float = 0.5f,
        oobPenalty: Float = 12f,
        weightBottomFrac: Float? = null
    ): DtCandidate3250? {
        if (ptsIn.isEmpty()) return null

        val ptsEval = if (weightBottomFrac != null) selectBottomByYFrac3250(ptsIn, weightBottomFrac) else ptsIn

        var bestScore = Float.POSITIVE_INFINITY
        var bestDx = 0f
        var bestDy = 0f
        var bestS = 1f
        var bestR = 0f
        var bestPts: List<PointF> = ptsIn

        var sc = scaleMin
        while (sc <= scaleMax + 1e-6f) {
            var rr = rotMinDeg
            while (rr <= rotMaxDeg + 1e-6f) {
                var dy = -dyMax
                while (dy <= dyMax) {
                    var dx = -dxMax
                    while (dx <= dxMax) {

                        val ptsT = transformPts3250(ptsIn, pivot, dx.toFloat(), dy.toFloat(), sc, rr)

                        val ptsTscore = if (ptsEval === ptsIn) {
                            ptsT
                        } else {
                            val sub = selectBottomByYFrac3250(ptsIn, weightBottomFrac ?: 1f)
                            transformPts3250(sub, pivot, dx.toFloat(), dy.toFloat(), sc, rr)
                        }

                        val sVal = scorePolylineDt3250(field, ptsTscore, oobPenalty)

                        if (sVal < bestScore) {
                            bestScore = sVal
                            bestDx = dx.toFloat()
                            bestDy = dy.toFloat()
                            bestS = sc
                            bestR = rr
                            bestPts = ptsT
                        }

                        dx += dxStep
                    }
                    dy += dyStep
                }
                rr += rotStepDeg
            }
            sc += scaleStep
        }

        return DtCandidate3250(
            score = bestScore,
            dx = bestDx,
            dy = bestDy,
            scale = bestS,
            rotDeg = bestR,
            ptsOut = bestPts
        )
    }

    private fun transformPts3250(
        pts: List<PointF>,
        pivot: PointF,
        dx: Float,
        dy: Float,
        scale: Float,
        rotDeg: Float
    ): List<PointF> {
        val rad = Math.toRadians(rotDeg.toDouble())
        val c = cos(rad).toFloat()
        val s = sin(rad).toFloat()

        val out = ArrayList<PointF>(pts.size)
        for (p in pts) {
            val x0 = p.x - pivot.x
            val y0 = p.y - pivot.y

            val xs = x0 * scale
            val ys = y0 * scale

            val xr = xs * c - ys * s
            val yr = xs * s + ys * c

            out.add(PointF(xr + pivot.x + dx, yr + pivot.y + dy))
        }
        return out
    }
}
