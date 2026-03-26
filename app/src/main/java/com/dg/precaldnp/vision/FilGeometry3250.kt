package com.dg.precaldnp.vision

import android.graphics.PointF
import android.graphics.RectF
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

object FilGeometry3250 {

    private const val CLOSE_RADIUS_PX_3250 = 1
    private const val DILATE_RADIUS_PX_3250 = 0
    private const val MAX_POLY_POINTS_3250 = 320
    private const val FALLBACK_SUPERELLIPSE_N_3250 = 240

    class FilGeometrySeed3250(
        val roiGlobal: RectF,
        val pxPerMmGuess: Float,
        val filHboxInnerMm: Float,
        val filVboxInnerMm: Float,
        val yRefGlobal: Float,
        val midAtRefGlobal: Float,
        val pupilGlobal: PointF? = null,

        val pupilXFracFromNasal: Float = 0.35f,
        val pupilYFracFromTop: Float = 0.45f,

        val annulusHalfThicknessMm: Float = 1.25f,
        val extraPadMmX: Float = 0f,
        val extraPadMmY: Float = 0f,

        val isLeftEyeInPhoto: Boolean,

        val filOutlineMm3250: List<PointF>? = null,
        val filR800CentiMm3250: IntArray? = null,

        val filRotationDeg3250: Float = 0f,
        val filMirrorX3250: Boolean = false
    )

    class FilGeometryPack3250(
        val seed: FilGeometrySeed3250,
        val roiGlobal: RectF,
        val pxPerMmUsed: Float,

        val targetCxGlobal: Float,
        val targetCyGlobal: Float,

        val polylineGlobal3250: List<PointF>,

        val rawMaskFullU83250: ByteArray,
        val maskFullU83250: ByteArray,

        val boundsGlobal3250: RectF
    )

    class FilGeometryMultiPack3250(
        val packs: List<FilGeometryPack3250>,
        val mergedMaskFullU83250: ByteArray
    )

    fun buildPackFull3250(
        w: Int,
        h: Int,
        seed: FilGeometrySeed3250
    ): FilGeometryPack3250 {
        if (w <= 0 || h <= 0) {
            return FilGeometryPack3250(
                seed = seed,
                roiGlobal = seed.roiGlobal,
                pxPerMmUsed = seed.pxPerMmGuess.coerceAtLeast(1.0f),
                targetCxGlobal = 0f,
                targetCyGlobal = 0f,
                polylineGlobal3250 = emptyList(),
                rawMaskFullU83250 = ByteArray(0),
                maskFullU83250 = ByteArray(0),
                boundsGlobal3250 = RectF()
            )
        }

        val roiL = seed.roiGlobal.left.coerceIn(0f, (w - 1).toFloat())
        val roiT = seed.roiGlobal.top.coerceIn(0f, (h - 1).toFloat())
        val roiR = seed.roiGlobal.right.coerceIn(roiL + 1f, w.toFloat())
        val roiB = seed.roiGlobal.bottom.coerceIn(roiT + 1f, h.toFloat())

        val roiSafe = RectF(roiL, roiT, roiR, roiB)

        val pxPerMm = seed.pxPerMmGuess.coerceAtLeast(1.0f)
        val expW = (seed.filHboxInnerMm * pxPerMm).coerceAtLeast(16f)
        val expH = (seed.filVboxInnerMm * pxPerMm).coerceAtLeast(12f)

        val pupilX = seed.pupilGlobal?.x ?: ((roiL + roiR) * 0.5f)
        val pupilY = seed.pupilGlobal?.y ?: seed.yRefGlobal

        val pupilFracFromLeft = if (seed.isLeftEyeInPhoto) {
            1f - seed.pupilXFracFromNasal
        } else {
            seed.pupilXFracFromNasal
        }.coerceIn(0.20f, 0.80f)

        var leftExp = pupilX - pupilFracFromLeft * expW

        var topExp =
            ((pupilY + seed.yRefGlobal) * 0.5f) -
                    seed.pupilYFracFromTop.coerceIn(0.20f, 0.80f) * expH

        if (seed.isLeftEyeInPhoto) {
            val rightLimit = min(roiR, seed.midAtRefGlobal + expW * 0.20f)
            if (leftExp + expW > rightLimit) {
                leftExp = rightLimit - expW
            }
        } else {
            val leftLimit = max(roiL, seed.midAtRefGlobal - expW * 0.20f)
            if (leftExp < leftLimit) {
                leftExp = leftLimit
            }
        }

        if (leftExp < roiL) {
            leftExp = roiL
        }

        if (leftExp + expW > roiR) {
            leftExp = roiR - expW
        }

        if (topExp < roiT) {
            topExp = roiT
        }

        if (topExp + expH > roiB) {
            topExp = roiB - expH
        }

        val targetCx = leftExp + expW * 0.5f
        val targetCy = topExp + expH * 0.5f

        val polyGlobal = buildExpectedFilPolylineGlobal3250(
            seed = seed,
            pxPerMm = pxPerMm,
            targetCx = targetCx,
            targetCy = targetCy
        )

        val bandPx = (seed.annulusHalfThicknessMm * pxPerMm).coerceAtLeast(2f)
        val padPx = max(seed.extraPadMmX, seed.extraPadMmY) * pxPerMm
        val halfTubePx = (bandPx + padPx).coerceAtLeast(2f)

        val rawMask = buildTubeMaskFromClosedPolylineFull3250(
            w = w,
            h = h,
            roiL = roiL,
            roiT = roiT,
            roiR = roiR,
            roiB = roiB,
            poly = polyGlobal,
            halfThicknessPx = halfTubePx
        )

        val closedMask = closeMaskDiskU83250(
            src = rawMask,
            w = w,
            h = h,
            radius = CLOSE_RADIUS_PX_3250
        )

        val finalMask = if (DILATE_RADIUS_PX_3250 > 0) {
            dilateMaskDiskU83250(
                src = closedMask,
                w = w,
                h = h,
                radius = DILATE_RADIUS_PX_3250
            )
        } else {
            closedMask
        }

        val bounds = computePolylineBounds3250(polyGlobal)

        return FilGeometryPack3250(
            seed = seed,
            roiGlobal = roiSafe,
            pxPerMmUsed = pxPerMm,
            targetCxGlobal = targetCx,
            targetCyGlobal = targetCy,
            polylineGlobal3250 = polyGlobal,
            rawMaskFullU83250 = rawMask,
            maskFullU83250 = finalMask,
            boundsGlobal3250 = bounds
        )
    }

    fun buildPackFull3250(
        w: Int,
        h: Int,
        seeds: List<FilGeometrySeed3250>
    ): FilGeometryMultiPack3250 {
        if (w <= 0 || h <= 0) {
            return FilGeometryMultiPack3250(
                packs = emptyList(),
                mergedMaskFullU83250 = ByteArray(0)
            )
        }

        val packs = ArrayList<FilGeometryPack3250>(seeds.size)
        val merged = ByteArray(w * h)

        for (seed in seeds) {
            val pack = buildPackFull3250(w, h, seed)
            packs += pack

            val mask = pack.maskFullU83250
            val n = min(merged.size, mask.size)

            for (i in 0 until n) {
                if ((mask[i].toInt() and 0xFF) != 0) {
                    merged[i] = 0xFF.toByte()
                }
            }
        }

        return FilGeometryMultiPack3250(
            packs = packs,
            mergedMaskFullU83250 = merged
        )
    }

    fun cropMaskToRoiU83250(
        fullMask: ByteArray,
        fullW: Int,
        fullH: Int,
        roiGlobal: RectF
    ): Pair<ByteArray, RectF> {
        if (fullW <= 0 || fullH <= 0 || fullMask.isEmpty()) {
            return ByteArray(0) to RectF()
        }

        val l = floor(roiGlobal.left.coerceIn(0f, (fullW - 1).toFloat())).toInt()
        val t = floor(roiGlobal.top.coerceIn(0f, (fullH - 1).toFloat())).toInt()
        val r = ceil(roiGlobal.right.coerceIn((l + 1).toFloat(), fullW.toFloat())).toInt()
        val b = ceil(roiGlobal.bottom.coerceIn((t + 1).toFloat(), fullH.toFloat())).toInt()

        val cw = max(0, r - l)
        val ch = max(0, b - t)

        if (cw <= 0 || ch <= 0) {
            return ByteArray(0) to RectF()
        }

        val out = ByteArray(cw * ch)

        for (y in 0 until ch) {
            val srcRow = (t + y) * fullW
            val dstRow = y * cw
            for (x in 0 until cw) {
                out[dstRow + x] = fullMask[srcRow + l + x]
            }
        }

        return out to RectF(l.toFloat(), t.toFloat(), r.toFloat(), b.toFloat())
    }

    private fun buildExpectedFilPolylineGlobal3250(
        seed: FilGeometrySeed3250,
        pxPerMm: Float,
        targetCx: Float,
        targetCy: Float
    ): List<PointF> {
        val outline = seed.filOutlineMm3250
        val radii = seed.filR800CentiMm3250

        val mmPolyline: List<PointF> =
            when {
                outline != null && outline.size >= 8 -> {
                    decimateClosedPolyline3250(
                        poly = outline,
                        maxPoints = MAX_POLY_POINTS_3250
                    )
                }

                radii != null && radii.isNotEmpty() && radii.size >= 32 -> {
                    radii800ToClosedPolylineMm3250(
                        radiiCentiMm = radii,
                        maxPoints = MAX_POLY_POINTS_3250
                    )
                }

                else -> {
                    buildFallbackSuperellipseMm3250(
                        wMm = seed.filHboxInnerMm,
                        hMm = seed.filVboxInnerMm,
                        n = FALLBACK_SUPERELLIPSE_N_3250
                    )
                }
            }

        if (mmPolyline.size < 3) {
            return emptyList()
        }

        val centered = centerPolylineAtOrigin3250(mmPolyline)

        val rot = Math.toRadians(seed.filRotationDeg3250.toDouble())
        val cosR = cos(rot).toFloat()
        val sinR = sin(rot).toFloat()

        val out = ArrayList<PointF>(centered.size)

        for (p in centered) {
            val mx = if (seed.filMirrorX3250) -p.x else p.x
            val my = p.y

            val rx = mx * cosR - my * sinR
            val ry = mx * sinR + my * cosR

            out += PointF(
                targetCx + rx * pxPerMm,
                targetCy + ry * pxPerMm
            )
        }

        return out
    }

    private fun radii800ToClosedPolylineMm3250(
        radiiCentiMm: IntArray,
        maxPoints: Int
    ): List<PointF> {
        if (radiiCentiMm.isEmpty()) {
            return emptyList()
        }

        val nSrc = radiiCentiMm.size
        val step = max(1, ceil(nSrc / maxPoints.toFloat()).toInt())
        val out = ArrayList<PointF>((nSrc + step - 1) / step)

        var i = 0
        while (i < nSrc) {
            val rMm = radiiCentiMm[i].coerceAtLeast(0) / 100f
            val ang = (2.0 * PI * i.toDouble()) / nSrc.toDouble()

            val x = (rMm * cos(ang)).toFloat()
            val y = (-rMm * sin(ang)).toFloat()

            out += PointF(x, y)
            i += step
        }

        return out
    }

    private fun buildFallbackSuperellipseMm3250(
        wMm: Float,
        hMm: Float,
        n: Int
    ): List<PointF> {
        if (wMm <= 0f || hMm <= 0f || n < 8) {
            return emptyList()
        }

        val a = wMm * 0.5f
        val b = hMm * 0.5f
        val m = 4.0

        val out = ArrayList<PointF>(n)

        for (i in 0 until n) {
            val t = (2.0 * PI * i.toDouble()) / n.toDouble()
            val ct = cos(t)
            val st = sin(t)

            val x = signPow3250(ct, 2.0 / m).toFloat() * a
            val y = signPow3250(st, 2.0 / m).toFloat() * b

            out += PointF(x, y)
        }

        return out
    }

    private fun signPow3250(
        v: Double,
        p: Double
    ): Double {
        val a = abs(v).coerceAtLeast(0.0)
        val r = Math.pow(a, p)
        return if (v < 0.0) -r else r
    }

    private fun centerPolylineAtOrigin3250(
        poly: List<PointF>
    ): List<PointF> {
        if (poly.isEmpty()) {
            return emptyList()
        }

        var minX = Float.POSITIVE_INFINITY
        var minY = Float.POSITIVE_INFINITY
        var maxX = Float.NEGATIVE_INFINITY
        var maxY = Float.NEGATIVE_INFINITY

        for (p in poly) {
            if (p.x < minX) minX = p.x
            if (p.y < minY) minY = p.y
            if (p.x > maxX) maxX = p.x
            if (p.y > maxY) maxY = p.y
        }

        val cx = (minX + maxX) * 0.5f
        val cy = (minY + maxY) * 0.5f

        val out = ArrayList<PointF>(poly.size)

        for (p in poly) {
            out += PointF(
                p.x - cx,
                p.y - cy
            )
        }

        return out
    }

    private fun decimateClosedPolyline3250(
        poly: List<PointF>,
        maxPoints: Int
    ): List<PointF> {
        if (poly.size <= maxPoints) {
            return poly.toList()
        }

        val step = ceil(poly.size / maxPoints.toFloat()).toInt().coerceAtLeast(1)
        val out = ArrayList<PointF>((poly.size + step - 1) / step)

        var i = 0
        while (i < poly.size) {
            out += poly[i]
            i += step
        }

        return out
    }

    private fun buildTubeMaskFromClosedPolylineFull3250(
        w: Int,
        h: Int,
        roiL: Float,
        roiT: Float,
        roiR: Float,
        roiB: Float,
        poly: List<PointF>,
        halfThicknessPx: Float
    ): ByteArray {
        val out = ByteArray(w * h)

        if (poly.size < 3) {
            return out
        }

        var minX = Float.POSITIVE_INFINITY
        var minY = Float.POSITIVE_INFINITY
        var maxX = Float.NEGATIVE_INFINITY
        var maxY = Float.NEGATIVE_INFINITY

        for (p in poly) {
            if (p.x < minX) minX = p.x
            if (p.y < minY) minY = p.y
            if (p.x > maxX) maxX = p.x
            if (p.y > maxY) maxY = p.y
        }

        val pad = halfThicknessPx + 1f

        val x0 = max(0, floor(max(minX - pad, roiL)).toInt())
        val y0 = max(0, floor(max(minY - pad, roiT)).toInt())
        val x1 = min(w - 1, ceil(min(maxX + pad, roiR - 1f)).toInt())
        val y1 = min(h - 1, ceil(min(maxY + pad, roiB - 1f)).toInt())

        if (x1 < x0 || y1 < y0) {
            return out
        }

        val dist2Max = halfThicknessPx * halfThicknessPx

        for (y in y0..y1) {
            val yc = y + 0.5f
            val row = y * w

            for (x in x0..x1) {
                val xc = x + 0.5f
                var best2 = Float.POSITIVE_INFINITY

                var i = 0
                while (i < poly.size) {
                    val a = poly[i]
                    val b = poly[(i + 1) % poly.size]

                    val d2 = pointToSegmentDistanceSq3250(
                        px = xc,
                        py = yc,
                        ax = a.x,
                        ay = a.y,
                        bx = b.x,
                        by = b.y
                    )

                    if (d2 < best2) {
                        best2 = d2
                    }

                    if (best2 <= dist2Max) {
                        break
                    }

                    i++
                }

                if (best2 <= dist2Max) {
                    out[row + x] = 0xFF.toByte()
                }
            }
        }

        return out
    }

    private fun pointToSegmentDistanceSq3250(
        px: Float,
        py: Float,
        ax: Float,
        ay: Float,
        bx: Float,
        by: Float
    ): Float {
        val abx = bx - ax
        val aby = by - ay
        val apx = px - ax
        val apy = py - ay

        val ab2 = abx * abx + aby * aby

        if (ab2 <= 1e-6f) {
            val dx = px - ax
            val dy = py - ay
            return dx * dx + dy * dy
        }

        val t = ((apx * abx) + (apy * aby)) / ab2
        val tc = t.coerceIn(0f, 1f)

        val qx = ax + tc * abx
        val qy = ay + tc * aby

        val dx = px - qx
        val dy = py - qy

        return dx * dx + dy * dy
    }

    private fun computePolylineBounds3250(
        poly: List<PointF>
    ): RectF {
        if (poly.isEmpty()) {
            return RectF()
        }

        var minX = Float.POSITIVE_INFINITY
        var minY = Float.POSITIVE_INFINITY
        var maxX = Float.NEGATIVE_INFINITY
        var maxY = Float.NEGATIVE_INFINITY

        for (p in poly) {
            if (p.x < minX) minX = p.x
            if (p.y < minY) minY = p.y
            if (p.x > maxX) maxX = p.x
            if (p.y > maxY) maxY = p.y
        }

        return RectF(minX, minY, maxX, maxY)
    }

    private fun closeMaskDiskU83250(
        src: ByteArray,
        w: Int,
        h: Int,
        radius: Int
    ): ByteArray {
        if (radius <= 0 || src.isEmpty()) {
            return src.copyOf()
        }

        val dil = dilateMaskDiskU83250(
            src = src,
            w = w,
            h = h,
            radius = radius
        )

        return erodeMaskDiskU83250(
            src = dil,
            w = w,
            h = h,
            radius = radius
        )
    }

    private fun dilateMaskDiskU83250(
        src: ByteArray,
        w: Int,
        h: Int,
        radius: Int
    ): ByteArray {
        if (radius <= 0 || src.isEmpty()) {
            return src.copyOf()
        }

        val out = ByteArray(src.size)
        val offsets = diskOffsets3250(radius)

        for (y in 0 until h) {
            val row = y * w

            for (x in 0 until w) {
                var on = false

                for (o in offsets) {
                    val xx = x + o.first
                    val yy = y + o.second

                    if (xx < 0 || xx >= w || yy < 0 || yy >= h) {
                        continue
                    }

                    if ((src[yy * w + xx].toInt() and 0xFF) != 0) {
                        on = true
                        break
                    }
                }

                if (on) {
                    out[row + x] = 0xFF.toByte()
                }
            }
        }

        return out
    }

    private fun erodeMaskDiskU83250(
        src: ByteArray,
        w: Int,
        h: Int,
        radius: Int
    ): ByteArray {
        if (radius <= 0 || src.isEmpty()) {
            return src.copyOf()
        }

        val out = ByteArray(src.size)
        val offsets = diskOffsets3250(radius)

        for (y in 0 until h) {
            val row = y * w

            for (x in 0 until w) {
                var keep = true

                for (o in offsets) {
                    val xx = x + o.first
                    val yy = y + o.second

                    if (xx < 0 || xx >= w || yy < 0 || yy >= h) {
                        keep = false
                        break
                    }

                    if ((src[yy * w + xx].toInt() and 0xFF) == 0) {
                        keep = false
                        break
                    }
                }

                if (keep) {
                    out[row + x] = 0xFF.toByte()
                }
            }
        }

        return out
    }

    private fun diskOffsets3250(
        radius: Int
    ): List<Pair<Int, Int>> {
        if (radius <= 0) {
            return listOf(0 to 0)
        }

        val out = ArrayList<Pair<Int, Int>>()
        val r2 = radius * radius

        for (dy in -radius..radius) {
            for (dx in -radius..radius) {
                if (dx * dx + dy * dy <= r2) {
                    out += dx to dy
                }
            }
        }

        return out
    }
}
