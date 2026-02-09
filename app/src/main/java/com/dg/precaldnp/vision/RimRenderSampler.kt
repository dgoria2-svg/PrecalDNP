package com.dg.precaldnp.vision

import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.RectF
import androidx.core.graphics.get
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

object RimRenderSampler3250 {

    // =========================================================================
    // Params / Results (Template Sampling)
    // =========================================================================

    data class Params3250(
        val sampleStepPx: Float = 2.0f,
        val normalSampleDistPx: Float = 5.0f,
        val thrResp: Float = 8.0f,
        val minCoverage: Float = 0.35f,
        val clampToRoi: Boolean = true,
        val respRef: Float = 20f,

        // refine tiny (template only)
        val refineDxPx: Int = 2,
        val refineDyPx: Int = 2,
        val refineScaleSteps: FloatArray = floatArrayOf(0.985f, 1.0f, 1.015f),
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false

            other as Params3250

            if (sampleStepPx != other.sampleStepPx) return false
            if (normalSampleDistPx != other.normalSampleDistPx) return false
            if (thrResp != other.thrResp) return false
            if (minCoverage != other.minCoverage) return false
            if (clampToRoi != other.clampToRoi) return false
            if (respRef != other.respRef) return false
            if (refineDxPx != other.refineDxPx) return false
            if (refineDyPx != other.refineDyPx) return false
            if (!refineScaleSteps.contentEquals(other.refineScaleSteps)) return false

            return true
        }

        override fun hashCode(): Int {
            var result = sampleStepPx.hashCode()
            result = 31 * result + normalSampleDistPx.hashCode()
            result = 31 * result + thrResp.hashCode()
            result = 31 * result + minCoverage.hashCode()
            result = 31 * result + clampToRoi.hashCode()
            result = 31 * result + respRef.hashCode()
            result = 31 * result + refineDxPx
            result = 31 * result + refineDyPx
            result = 31 * result + refineScaleSteps.contentHashCode()
            return result
        }
    }

    data class ScoreDebug3250(
        val samples: Int,
        val inliers: Int,
        val coverage: Float,
        val medianResp: Float,
        val meanResp: Float,
        val p10Resp: Float,
        val p90Resp: Float
    )

    data class Candidate3250(
        val dx: Int,
        val dy: Int,
        val scale: Float,
        val score: Float,
        val debug: ScoreDebug3250,
        val transformedPtsGlobal: List<PointF>
    )

    data class RefinementResult3250(
        val best: Candidate3250,
        val all: List<Candidate3250>
    )

    // =========================================================================
    // Public API (Template sampling)
    // =========================================================================

    fun scorePlacedBySampling3250(
        stillBmp: Bitmap,
        roiGlobal: RectF,
        placedPtsGlobal: List<PointF>,
        pupilYGlobal: Float,
        useBottomOnly: Boolean = true,
        params: Params3250 = Params3250()
    ): Pair<Float, ScoreDebug3250> {

        val ptsPrepared = preparePtsForSampling3250(
            roiGlobal = roiGlobal,
            ptsGlobal = placedPtsGlobal,
            pupilYGlobal = pupilYGlobal,
            useBottomOnly = useBottomOnly,
            params = params
        )

        val dbg = scorePolylineOnBitmap3250(
            bmp = stillBmp,
            roi = roiGlobal,
            pts = ptsPrepared,
            params = params
        )

        val score = toScore3250(dbg, params)
        return score to dbg
    }

    fun refinePlacedBySampling3250(
        stillBmp: Bitmap,
        roiGlobal: RectF,
        placedPtsGlobal: List<PointF>,
        pupilYGlobal: Float,
        useBottomOnly: Boolean = true,
        params: Params3250 = Params3250()
    ): RefinementResult3250 {

        val center = centroid3250(placedPtsGlobal)

        val all = ArrayList<Candidate3250>()

        for (s in params.refineScaleSteps) {
            for (dy in -params.refineDyPx..params.refineDyPx) {
                for (dx in -params.refineDxPx..params.refineDxPx) {

                    val moved = transformPts3250(
                        pts = placedPtsGlobal,
                        center = center,
                        scale = s,
                        dx = dx.toFloat(),
                        dy = dy.toFloat()
                    )

                    val (score, dbg) = scorePlacedBySampling3250(
                        stillBmp = stillBmp,
                        roiGlobal = roiGlobal,
                        placedPtsGlobal = moved,
                        pupilYGlobal = pupilYGlobal,
                        useBottomOnly = useBottomOnly,
                        params = params
                    )

                    all += Candidate3250(
                        dx = dx,
                        dy = dy,
                        scale = s,
                        score = score,
                        debug = dbg,
                        transformedPtsGlobal = moved
                    )
                }
            }
        }

        val best = all.maxByOrNull { it.score }
            ?: Candidate3250(0, 0, 1.0f, 0f,
                ScoreDebug3250(0, 0, 0f, 0f, 0f, 0f, 0f),
                placedPtsGlobal
            )

        return RefinementResult3250(best, all)
    }

    // =========================================================================
    // Template scoring core
    // =========================================================================

    private fun toScore3250(dbg: ScoreDebug3250, params: Params3250): Float {
        if (dbg.samples <= 0) return 0f
        if (dbg.coverage < params.minCoverage * 0.5f) return 0f

        val respNorm = clamp01(dbg.medianResp / params.respRef)
        return clamp01(dbg.coverage) * respNorm
    }

    private fun scorePolylineOnBitmap3250(
        bmp: Bitmap,
        roi: RectF,
        pts: List<PointF>,
        params: Params3250
    ): ScoreDebug3250 {

        if (pts.size < 6) return ScoreDebug3250(0, 0, 0f, 0f, 0f, 0f, 0f)

        val responses = ArrayList<Float>(1024)

        var prev = pts[0]
        for (i in 1 until pts.size) {

            val cur = pts[i]
            val segLen = dist3250(prev, cur)

            if (segLen <= 1e-3f) {
                prev = cur
                continue
            }

            val steps = max(1, (segLen / params.sampleStepPx).roundToInt())

            for (k in 0..steps) {
                val t = k.toFloat() / steps.toFloat()
                val p = lerp3250(prev, cur, t)

                if (params.clampToRoi && !roi.contains(p.x, p.y)) continue

                val tang = approxTangent3250(pts, i)
                val n = normalFromTangent3250(tang)

                val d = params.normalSampleDistPx
                val p1 = PointF(p.x - n.x * d, p.y - n.y * d)
                val p2 = PointF(p.x + n.x * d, p.y + n.y * d)

                val i1 = sampleLumaBilinear3250(bmp, p1.x, p1.y)
                val i2 = sampleLumaBilinear3250(bmp, p2.x, p2.y)

                val resp = abs(i2 - i1)
                responses += resp
            }

            prev = cur
        }

        if (responses.isEmpty()) return ScoreDebug3250(0, 0, 0f, 0f, 0f, 0f, 0f)

        responses.sort()

        val samples = responses.size
        val inliers = responses.count { it >= params.thrResp }
        val coverage = inliers.toFloat() / samples.toFloat()

        val median = percentileSorted3250(responses, 0.50f)
        val p10 = percentileSorted3250(responses, 0.10f)
        val p90 = percentileSorted3250(responses, 0.90f)
        val mean = responses.sum() / samples.toFloat()

        return ScoreDebug3250(
            samples = samples,
            inliers = inliers,
            coverage = coverage,
            medianResp = median,
            meanResp = mean,
            p10Resp = p10,
            p90Resp = p90
        )
    }

    // =========================================================================
    // Prep (bottom only)
    // =========================================================================

    private fun preparePtsForSampling3250(
        roiGlobal: RectF,
        ptsGlobal: List<PointF>,
        pupilYGlobal: Float,
        useBottomOnly: Boolean,
        params: Params3250
    ): List<PointF> {

        if (ptsGlobal.isEmpty()) return emptyList()

        var minY = Float.POSITIVE_INFINITY
        var maxY = Float.NEGATIVE_INFINITY
        for (p in ptsGlobal) {
            minY = min(minY, p.y)
            maxY = max(maxY, p.y)
        }

        val midY = (minY + maxY) * 0.5f
        val yCut = max(midY, pupilYGlobal)

        val filtered = ArrayList<PointF>()
        for (p in ptsGlobal) {
            if (useBottomOnly && p.y < yCut) continue
            if (params.clampToRoi && !roiGlobal.contains(p.x, p.y)) continue
            filtered += p
        }

        return if (filtered.size >= 6) filtered else ptsGlobal
    }

    // =========================================================================
    // Geometry scoring (NORMAL CROSS)
    // =========================================================================

    data class GeomDebug3250(
        val samples: Int,
        val hits: Int,
        val coverageGeom: Float,
        val rmsGeomPx: Float,
        val thrGrad: Float,
        val medHitDistPx: Float,
        val p90HitDistPx: Float
    )

    data class GeomScore3250(
        val scoreGeom: Float,
        val dbg: GeomDebug3250
    )

    data class GeomParams3250(
        val dMaxPx: Float = 14f,
        val stepPx: Float = 1f,

        val minGrad: Float = 6f,
        val thrFracOfMedian: Float = 0.55f,
        val maxHitDistPx: Float = 12f,

        val rmsRefPx: Float = 4.5f,
        val bottomKeepFrac: Float = 0.60f
    )

    fun scorePlacedBySamplingGeom3250(
        stillBmp: Bitmap,
        roiGlobal: RectF,
        placedPtsGlobal: List<PointF>,
        pupilYGlobal: Float,
        params: GeomParams3250 = GeomParams3250()
    ): GeomScore3250 {

        val pts = prepareBottomOnly3250(
            placedPtsGlobal,
            pupilYGlobal,
            roiGlobal,
            params.bottomKeepFrac
        )

        if (pts.size < 8) {
            return GeomScore3250(0f,
                GeomDebug3250(0, 0, 0f, Float.NaN, 0f, Float.NaN, Float.NaN)
            )
        }

        // ---- estimate gradient median ----
        val gradBuf = FloatArray(pts.size)
        var gN = 0

        for (i in pts.indices) {
            val nrm = normalAtIndex3250(pts, i) ?: continue
            val maxGrad = maxGradientAlongNormal3250(stillBmp, pts[i], nrm, params)
            if (maxGrad > 0f) gradBuf[gN++] = maxGrad
        }

        if (gN <= 0) {
            return GeomScore3250(0f,
                GeomDebug3250(pts.size, 0, 0f, Float.NaN, 0f, Float.NaN, Float.NaN)
            )
        }

        gradBuf.sort(0, gN)
        val medGrad = gradBuf[gN / 2]
        val thrGrad = max(params.minGrad, medGrad * params.thrFracOfMedian)

        // ---- hits ----
        val distBuf = FloatArray(pts.size)
        var dN = 0

        for (i in pts.indices) {
            val nrm = normalAtIndex3250(pts, i) ?: continue
            val dHit = findFirstEdge3250(stillBmp, pts[i], nrm, thrGrad, params)
            if (dHit != null) distBuf[dN++] = dHit
        }

        val samples = pts.size
        val hits = dN
        val coverageGeom = hits.toFloat() / samples.toFloat()

        // RMS
        var rms = 0.0
        for (k in 0 until hits) rms += (distBuf[k] * distBuf[k]).toDouble()
        val rmsGeomPx = if (hits > 0) sqrt(rms / hits.toDouble()).toFloat() else Float.NaN

        distBuf.sort(0, dN)
        val medD = if (hits > 0) distBuf[hits / 2] else Float.NaN
        val p90D = if (hits > 0) distBuf[(hits * 0.90f).toInt().coerceIn(0, hits - 1)] else Float.NaN

        val scoreGeom = coverageGeom * clamp01(1f - (rmsGeomPx / params.rmsRefPx))

        return GeomScore3250(
            scoreGeom,
            GeomDebug3250(samples, hits, coverageGeom, rmsGeomPx, thrGrad, medD, p90D)
        )
    }

    // =========================================================================
    // Helpers (Geom)
    // =========================================================================

    private fun maxGradientAlongNormal3250(
        bmp: Bitmap,
        p: PointF,
        nrm: PointF,
        params: GeomParams3250
    ): Float {

        fun scan(sign: Float): Float {
            var prev = Float.NaN
            var best = 0f
            var d = 0f
            while (d <= params.dMaxPx) {
                val I = sampleLumaBilinear3250(
                    bmp,
                    p.x + nrm.x * d * sign,
                    p.y + nrm.y * d * sign
                )
                if (prev.isFinite()) best = max(best, abs(I - prev))
                prev = I
                d += params.stepPx
            }
            return best
        }

        return max(scan(+1f), scan(-1f))
    }

    private fun findFirstEdge3250(
        bmp: Bitmap,
        p: PointF,
        nrm: PointF,
        thrGrad: Float,
        params: GeomParams3250
    ): Float? {

        fun scan(sign: Float): Float? {
            var prev = Float.NaN
            var d = 0f
            while (d <= params.dMaxPx) {
                val I = sampleLumaBilinear3250(
                    bmp,
                    p.x + nrm.x * d * sign,
                    p.y + nrm.y * d * sign
                )
                if (prev.isFinite()) {
                    if (abs(I - prev) >= thrGrad) return d
                }
                prev = I
                d += params.stepPx
            }
            return null
        }

        val d1 = scan(+1f)
        val d2 = scan(-1f)
        val best = listOfNotNull(d1, d2).minOrNull()

        return best?.takeIf { it <= params.maxHitDistPx }
    }

    private fun prepareBottomOnly3250(
        pts: List<PointF>,
        pupilYGlobal: Float,
        roi: RectF,
        keepFrac: Float
    ): List<PointF> {

        val filtered = pts.filter {
            it.y >= pupilYGlobal &&
                    roi.contains(it.x, it.y)
        }

        if (filtered.isEmpty()) return emptyList()

        val sorted = filtered.sortedBy { it.y }
        val start = (sorted.size * (1f - keepFrac)).toInt()
            .coerceIn(0, sorted.size - 1)

        return sorted.subList(start, sorted.size)
    }

    // =========================================================================
    // Shared utilities
    // =========================================================================

    private fun centroid3250(pts: List<PointF>): PointF {
        var sx = 0f
        var sy = 0f
        for (p in pts) {
            sx += p.x
            sy += p.y
        }
        val inv = 1f / pts.size.toFloat()
        return PointF(sx * inv, sy * inv)
    }

    private fun transformPts3250(
        pts: List<PointF>,
        center: PointF,
        scale: Float,
        dx: Float,
        dy: Float
    ): List<PointF> {
        val out = ArrayList<PointF>(pts.size)
        for (p in pts) {
            val x0 = p.x - center.x
            val y0 = p.y - center.y
            out += PointF(
                center.x + x0 * scale + dx,
                center.y + y0 * scale + dy
            )
        }
        return out
    }

    private fun dist3250(a: PointF, b: PointF): Float {
        return hypot((b.x - a.x).toDouble(), (b.y - a.y).toDouble()).toFloat()
    }

    private fun lerp3250(a: PointF, b: PointF, t: Float): PointF {
        return PointF(
            a.x + (b.x - a.x) * t,
            a.y + (b.y - a.y) * t
        )
    }

    private fun approxTangent3250(pts: List<PointF>, i: Int): PointF {
        val i0 = max(0, i - 1)
        val i1 = min(pts.size - 1, i + 1)
        val a = pts[i0]
        val b = pts[i1]
        val tx = b.x - a.x
        val ty = b.y - a.y
        val norm = max(1e-6f, sqrt(tx * tx + ty * ty))
        return PointF(tx / norm, ty / norm)
    }

    private fun normalFromTangent3250(t: PointF): PointF {
        return PointF(-t.y, t.x)
    }

    private fun normalAtIndex3250(pts: List<PointF>, i: Int): PointF? {
        if (pts.size < 3) return null
        val i0 = max(0, i - 1)
        val i1 = min(pts.size - 1, i + 1)
        val p0 = pts[i0]
        val p1 = pts[i1]
        val tx = p1.x - p0.x
        val ty = p1.y - p0.y
        val len = sqrt(tx * tx + ty * ty)
        if (len <= 1e-3f) return null
        return PointF(-ty / len, tx / len)
    }

    private fun sampleLumaBilinear3250(bmp: Bitmap, x: Float, y: Float): Float {
        val w = bmp.width
        val h = bmp.height
        if (w <= 1 || h <= 1) return 0f

        val xf = x.coerceIn(0f, (w - 1).toFloat())
        val yf = y.coerceIn(0f, (h - 1).toFloat())

        val x0 = xf.toInt()
        val y0 = yf.toInt()
        val x1 = min(x0 + 1, w - 1)
        val y1 = min(y0 + 1, h - 1)

        val dx = xf - x0
        val dy = yf - y0

        val l00 = lumaFromPixel3250(bmp[x0, y0])
        val l10 = lumaFromPixel3250(bmp[x1, y0])
        val l01 = lumaFromPixel3250(bmp[x0, y1])
        val l11 = lumaFromPixel3250(bmp[x1, y1])

        val a0 = l00 + (l10 - l00) * dx
        val a1 = l01 + (l11 - l01) * dx
        return a0 + (a1 - a0) * dy
    }

    private fun lumaFromPixel3250(argb: Int): Float {
        val r = (argb shr 16) and 0xFF
        val g = (argb shr 8) and 0xFF
        val b = (argb) and 0xFF
        return 0.299f * r + 0.587f * g + 0.114f * b
    }

    private fun percentileSorted3250(sorted: List<Float>, q: Float): Float {
        val idx = (q * (sorted.size - 1))
        val i0 = idx.toInt()
        val i1 = min(i0 + 1, sorted.size - 1)
        val t = idx - i0
        return sorted[i0] + (sorted[i1] - sorted[i0]) * t
    }

    private fun clamp01(x: Float): Float = x.coerceIn(0f, 1f)
}
