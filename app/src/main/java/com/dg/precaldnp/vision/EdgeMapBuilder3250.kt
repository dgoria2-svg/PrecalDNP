@file:Suppress("SameParameterValue")

package com.dg.precaldnp.vision

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.provider.MediaStore
import android.util.Log
import androidx.core.graphics.createBitmap
import java.io.OutputStream
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min

/**
 * ✅ ÚNICA fuente de EDGE MAP 3250.
 *
 * Regla: still -> edgeMapFull (binario 0/255) UNA VEZ.
 * - RimDetector trabaja SOLO sobre edgeMap (recortando ROI)
 * - ArcFit trabaja SOLO sobre edgeMap (nunca regenera bordes)
 *
 * Cambio 2026-03-25:
 * - El path Bitmap ahora puede pre-segmentar una región positiva de anteojo
 *   para aplanar textura de piel/fondo ANTES del Scharr.
 * - La API pública no cambia: sigue devolviendo EdgeMapOut3250.
 * - El path GrayU8 se mantiene compatible y no inventa color.
 */
object EdgeMapBuilder3250 {

    private const val TAG = "EdgeMapBuilder3250"

    // ------------------------------------------------------------
    // Parámetros base (FULL)
    // ------------------------------------------------------------

    private const val K_H = 0.60f
    private const val K_V = 0.60f

    private const val EDGE_THR_FRAC = 0.10f
    private const val EDGE_THR_MIN = 6

    private const val P95_CAP_K = 1.10f

    private const val TAN22_NUM = 41
    private const val TAN22_DEN = 100
    private const val TAN67_NUM = 241
    private const val TAN67_DEN = 100

    private const val THICKEN_RADIUS_PX = 0
    private const val THICKEN_ITERS = 0

    private const val DIR_MAX_ANGLE_DEG = 25f
    @Suppress("unused")
    private val DIR_COS_TOL: Float = cos(Math.toRadians(DIR_MAX_ANGLE_DEG.toDouble())).toFloat()

    private const val DEBUG_MAX_DIM_PX = 1600

    // ============================================================
    // MASK DEBUG / compat
    // ============================================================

    private const val MASK_DILATE_RADIUS_PX = 0

    @Suppress("unused")
    private const val MASK_FLATTEN_BLEND = 0.35f
    @Suppress("unused")
    private const val MASK_SCORE_WEIGHT = 0.85f

    // ============================================================
    // Pre-segmentación positiva desde Bitmap (sin romper API)
    // ============================================================

    @Suppress("unused")
    private const val ENABLE_BITMAP_PRESEG_3250 = true
    private const val PRESEG_SMALL_MAX_DIM_PX = 640
    private const val PRESEG_SMALL_BLOCK_PX = 8
    private const val PRESEG_FULL_FLAT_BLOCK_PX = 16
    private const val PRESEG_LOCAL_DARK_DELTA = 12
    private const val PRESEG_DARK_EXTRA = 18
    private const val PRESEG_KEEP_X0_FRAC = 0.08f
    private const val PRESEG_KEEP_X1_FRAC = 0.92f
    private const val PRESEG_KEEP_Y0_FRAC = 0.10f
    private const val PRESEG_KEEP_Y1_FRAC = 0.72f
    private const val PRESEG_MIN_COVERAGE = 0.0035f
    private const val PRESEG_MAX_COVERAGE = 0.45f
    private const val PRESEG_MIN_CENTER_HITS = 48

    private const val EXPECTED_MASK_MIN_PIXELS = 96
    private const val EXPECTED_MASK_MAX_COVERAGE = 0.25f
    private const val EXPECTED_MASK_DILATE_RX = 3
    private const val EXPECTED_MASK_DILATE_RY = 2
    private const val EXPECTED_MASK_CLOSE_RX = 2
    private const val EXPECTED_MASK_CLOSE_RY = 1


    data class EdgeStats(
        val maxScore: Float,
        val p95Score: Float,
        val thr: Float,
        val nnz: Int,
        val density: Float,
        val bandPx: Float,
        val samples: Int,
        val topKillY: Int,
        val maskedN: Int,
        val maskedFrac: Float,
        val maskDilateR: Int
    )

    class EdgeMapOut3250(
        val w: Int,
        val h: Int,
        val edgesU8: ByteArray,
        val dirU8: ByteArray,
        val stats: EdgeStats
    )

    private data class BitmapPreseg3250(
        val grayWorkU8: ByteArray,
        val gateMaskU8: ByteArray?,
        val rawGrayU8: ByteArray,
        val used: Boolean
    )

    private fun u8(b: Byte): Int = b.toInt() and 0xFF


    // ============================================================
    // FULL FRAME: Bitmap -> EdgeMapOut3250
    // ============================================================

    fun buildFullFrameEdgePackFromBitmap3250(
        stillBmp: Bitmap,
        borderKillPx: Int,
        topKillY: Int = 0,
        maskFull: ByteArray? = null,
        regionMaskFull3250: ByteArray? = null,
        forceRegion3250: Boolean = false,
        debugTag: String = "FULL",
        statsOut: ((EdgeStats) -> Unit)? = null,
        ctx: Context? = null,
        debugSaveToGallery3250: Boolean = false,
        debugAlsoSaveGray3250: Boolean = false
    ): EdgeMapOut3250
    {

        val w = stillBmp.width
        val h = stillBmp.height
        if (w <= 0 || h <= 0) {
            val st = EdgeStats(0f, 0f, 0f, 0, 0f, 0f, 0, topKillY, 0, 0f, 0)
            return EdgeMapOut3250(0, 0, ByteArray(0), ByteArray(0), st)
        }

        val grayRaw = bitmapToGrayU83250(stillBmp)
        val pre = prepareBitmapPreseg3250(
            stillBmp = stillBmp,
            grayRawU8 = grayRaw,
            maskFull = maskFull,
            debugTag = debugTag,
            regionMaskFull3250 = regionMaskFull3250,
            forceRegion3250 = forceRegion3250
        )

        val pack = buildfullframeedgepackcoreu83250(
            w = w,
            h = h,
            grayU8 = pre.grayWorkU8,
            borderKillPx = borderKillPx,
            topKillY = topKillY,
            maskFull = maskFull,
            regionMaskFull3250 = pre.gateMaskU8,
            debugTag = debugTag,
            statsOut = statsOut
        )

        if (debugSaveToGallery3250 && ctx != null) {
            try {
                debugDumpEdgeTriplet3250(
                    ctx = ctx,
                    w = w,
                    h = h,
                    edgesFullU8 = pack.edgesU8,
                    maskFullU8 = maskFull,
                    debugTag = debugTag,
                    grayRawU8 = if (debugAlsoSaveGray3250) pre.rawGrayU8 else null,
                    grayWorkU8 = if (debugAlsoSaveGray3250 && pre.used) pre.grayWorkU8 else null,
                    gateMaskU8 = pre.gateMaskU8
                )
            } catch (t: Throwable) {
                Log.e(TAG, "EDGE dump save failed", t)
            }
        }

        return pack
    }


    // ============================================================
    // FULL FRAME: GrayU8 -> EdgeMapOut3250
    // ============================================================

    fun buildfullframeedgepackfromgrayu83250(
        w: Int,
        h: Int,
        grayU8: ByteArray,
        borderKillPx: Int,
        topKillY: Int = 0,
        maskFull: ByteArray? = null,
        debugTag: String = "FULL",
        statsOut: ((EdgeStats) -> Unit)? = null
    ): EdgeMapOut3250 {
        return buildfullframeedgepackcoreu83250(
            w = w,
            h = h,
            grayU8 = grayU8,
            borderKillPx = borderKillPx,
            topKillY = topKillY,
            maskFull = maskFull,
            regionMaskFull3250 = null,
            debugTag = debugTag,
            statsOut = statsOut
        )
    }

    fun buildfullframeedgepackfromgrayu8andregion3250(
        w: Int,
        h: Int,
        grayU8: ByteArray,
        borderKillPx: Int,
        regionMaskFull3250: ByteArray,
        topKillY: Int = 0,
        maskFull: ByteArray? = null,
        debugTag: String = "FULL",
        statsOut: ((EdgeStats) -> Unit)? = null
    ): EdgeMapOut3250 {
        return buildfullframeedgepackcoreu83250(
            w = w,
            h = h,
            grayU8 = grayU8,
            borderKillPx = borderKillPx,
            topKillY = topKillY,
            maskFull = maskFull,
            regionMaskFull3250 = regionMaskFull3250,
            debugTag = debugTag,
            statsOut = statsOut
        )
    }

    private fun buildfullframeedgepackcoreu83250(
        w: Int,
        h: Int,
        grayU8: ByteArray,
        borderKillPx: Int,
        topKillY: Int = 0,
        maskFull: ByteArray? = null,
        regionMaskFull3250: ByteArray? = null,
        debugTag: String = "FULL",
        statsOut: ((EdgeStats) -> Unit)? = null
    ): EdgeMapOut3250 {

        if (w <= 0 || h <= 0) {
            val st = EdgeStats(0f, 0f, 0f, 0, 0f, 0f, 0, topKillY, 0, 0f, 0)
            return EdgeMapOut3250(0, 0, ByteArray(0), ByteArray(0), st)
        }

        val N = w * h

        if (grayU8.size != N) {
            Log.w(TAG, "EDGE_FULL[$debugTag] bad gray size=${grayU8.size} N=$N")
            val st = EdgeStats(0f, 0f, 0f, 0, 0f, 0f, 0, topKillY.coerceIn(0, h), 0, 0f, 0)
            return EdgeMapOut3250(w, h, ByteArray(N), ByteArray(N), st)
        }
        if (maskFull != null && maskFull.size != N) {
            Log.w(TAG, "EDGE_FULL[$debugTag] bad mask size=${maskFull.size} N=$N")
            val st = EdgeStats(0f, 0f, 0f, 0, 0f, 0f, 0, topKillY.coerceIn(0, h), 0, 0f, 0)
            return EdgeMapOut3250(w, h, ByteArray(N), ByteArray(N), st)
        }
        if (regionMaskFull3250 != null && regionMaskFull3250.size != N) {
            Log.w(TAG, "EDGE_FULL[$debugTag] bad regionMask size=${regionMaskFull3250.size} N=$N")
            val st = EdgeStats(0f, 0f, 0f, 0, 0f, 0f, 0, topKillY.coerceIn(0, h), 0, 0f, 0)
            return EdgeMapOut3250(w, h, ByteArray(N), ByteArray(N), st)
        }

        val B = borderKillPx.coerceIn(0, min(w, h) / 3)
        val yKill = 0

        val maskWork: ByteArray? = maskFull
        val hasMask = (maskWork != null)
        val hasRegionMask = (regionMaskFull3250 != null)

        fun isMasked(i: Int): Boolean =
            maskWork != null && ((maskWork[i].toInt() and 0xFF) != 0)

        fun inRegion(i: Int): Boolean =
            regionMaskFull3250 == null || ((regionMaskFull3250[i].toInt() and 0xFF) != 0)

        var maskedN = 0
        if (hasMask) {
            for (i in 0 until N) if (isMasked(i)) maskedN++
        }
        val maskedFrac = maskedN.toFloat() / N.toFloat().coerceAtLeast(1f)

        fun pix(i: Int): Int = u8(grayU8[i])

        val score = ShortArray(N)
        val dir = ByteArray(N)
        var maxScore = 0
        var validCount = 0

        val kH100 = (K_H * 100f).toInt()
        val kV100 = (K_V * 100f).toInt()

        for (y in 1 until (h - 1)) {
            if (y < B || y >= h - B) continue

            val off = y * w
            val offUp = (y - 1) * w
            val offDn = (y + 1) * w

            for (x in 1 until (w - 1)) {
                if (x < B || x >= w - B) continue
                val i = off + x
                if (!inRegion(i)) continue

                val p00 = pix(offUp + (x - 1))
                val p01 = pix(offUp + x)
                val p02 = pix(offUp + (x + 1))
                val p10 = pix(off + (x - 1))
                val p12 = pix(off + (x + 1))
                val p20 = pix(offDn + (x - 1))
                val p21 = pix(offDn + x)
                val p22 = pix(offDn + (x + 1))

                val gx = 3 * (p02 - p00) + 10 * (p12 - p10) + 3 * (p22 - p20)
                val gy = 3 * (p20 - p00) + 10 * (p21 - p01) + 3 * (p22 - p02)

                val ax = abs(gx)
                val ay = abs(gy)

                val hs100 = ay * 100 - kH100 * ax
                val vs100 = ax * 100 - kV100 * ay
                val s100 = max(0, max(hs100, vs100))
                val s = (s100 / 100).coerceIn(0, 32767)
                if (s == 0) continue

                score[i] = s.toShort()
                validCount++
                if (s > maxScore) maxScore = s

                dir[i] = quantizeDir3250(gx, gy, ax, ay).toByte()
            }
        }

        if (validCount < 256 || maxScore <= 0) {
            Log.w(
                TAG,
                "EDGE_FULL[$debugTag] not enough signal valid=$validCount max=$maxScore topKillY=$yKill B=$B hasMask=$hasMask hasRegion=$hasRegionMask"
            )
            val st = EdgeStats(
                maxScore = maxScore.toFloat(),
                p95Score = 0f,
                thr = 0f,
                nnz = 0,
                density = 0f,
                bandPx = 0f,
                samples = validCount,
                topKillY = yKill,
                maskedN = maskedN,
                maskedFrac = maskedFrac,
                maskDilateR = MASK_DILATE_RADIUS_PX
            )
            statsOut?.invoke(st)
            return EdgeMapOut3250(w, h, ByteArray(N), dir, st)
        }

        val bins = 2048
        val hist = IntArray(bins)
        for (i in 0 until N) {
            if (!inRegion(i)) continue
            val s = score[i].toInt()
            if (s <= 0) continue
            val bin = (s * (bins - 1)) / maxScore
            hist[bin]++
        }

        val p95 = percentileFromHist3250(hist, bins, maxScore, 0.99f)
        val thrHigh = computeThrHigh3250(maxScore, p95)
        val thrLow = max(EDGE_THR_MIN, (thrHigh * 0.80f).toInt())

        val classMap = ByteArray(N)
        var candCount = 0
        var strongCount = 0

        for (y in 1 until (h - 1)) {
            if (y < B || y >= h - B) continue
            val off = y * w
            for (x in 1 until (w - 1)) {
                if (x < B || x >= w - B) continue
                val i = off + x
                if (!inRegion(i)) continue

                val s = score[i].toInt()
                if (s < thrLow) continue

                val d = dir[i].toInt() and 0xFF
                val (i1, i2) = nmsNeighbors3250(i, x, y, w, d)
                val s1 = if (inRegion(i1)) score[i1].toInt() else 0
                val s2 = if (inRegion(i2)) score[i2].toInt() else 0
                if (s < s1 || s < s2) continue

                if (s >= thrHigh) {
                    classMap[i] = 2
                    strongCount++
                } else {
                    classMap[i] = 1
                }
                candCount++
            }
        }

        val out = ByteArray(N)
        if (strongCount == 0 || candCount == 0) {
            Log.w(
                TAG,
                "EDGE_FULL[$debugTag] no strong/candidates (strong=$strongCount cand=$candCount) thrH=$thrHigh thrL=$thrLow hasRegion=$hasRegionMask"
            )
            val st = EdgeStats(
                maxScore = maxScore.toFloat(),
                p95Score = p95.toFloat(),
                thr = thrHigh.toFloat(),
                nnz = 0,
                density = 0f,
                bandPx = 0f,
                samples = validCount,
                topKillY = yKill,
                maskedN = maskedN,
                maskedFrac = maskedFrac,
                maskDilateR = MASK_DILATE_RADIUS_PX
            )
            statsOut?.invoke(st)
            return EdgeMapOut3250(w, h, out, dir, st)
        }

        val q = IntArray(candCount.coerceAtLeast(1))
        var qh = 0
        var qt = 0

        for (i in 0 until N) {
            if (classMap[i].toInt() == 2) {
                out[i] = 0xFF.toByte()
                q[qt++] = i
            }
        }

        while (qh < qt) {
            val i = q[qh++]
            val x = i % w
            val y = i / w
            for (dy in -1..1) for (dx in -1..1) {
                if (dx == 0 && dy == 0) continue
                val xx = x + dx
                val yy = y + dy
                if (xx <= 0 || xx >= w - 1) continue
                if (yy <= 0 || yy >= h - 1) continue
                if (xx < B || xx >= w - B) continue
                if (yy < B || yy >= h - B) continue
                val j = yy * w + xx
                if (!inRegion(j)) continue
                if (classMap[j].toInt() == 1) {
                    classMap[j] = 2
                    out[j] = 0xFF.toByte()
                    q[qt++] = j
                }
            }
        }

        val outBridged = bridgehorizontalgapsu83250(
            edges = out,
            dir = dir,
            w = w,
            h = h,
            maxGapPx = 0
        )

        val outFinal =
            thickenedgemapu83250(outBridged, w, h, radius = THICKEN_RADIUS_PX, iters = THICKEN_ITERS)

        if (regionMaskFull3250 != null) {
            for (i in 0 until N) {
                if ((regionMaskFull3250[i].toInt() and 0xFF) == 0) {
                    outFinal[i] = 0
                }
            }
        }

        val dirFinal = dir.copyOf()
        for (i in 0 until N) {
            val eNew = outFinal[i].toInt() and 0xFF
            if (eNew == 0) continue
            val eOld = out[i].toInt() and 0xFF
            if (eOld == 0) dirFinal[i] = 0xFF.toByte()
        }

        var nnzFinal = 0
        for (b in outFinal) {
            if ((b.toInt() and 0xFF) != 0) nnzFinal++
        }
        val dens = nnzFinal.toFloat() / N.toFloat().coerceAtLeast(1f)

        Log.d(
            TAG,
            "EDGE_FULL[$debugTag] max=$maxScore p95=$p95 thrH=$thrHigh thrL=$thrLow " +
                    "strong=$strongCount cand=$candCount nnz=$nnzFinal/$N dens=${"%.4f".format(dens)} " +
                    "topKillY=$yKill B=$B hasMask=$hasMask hasRegion=$hasRegionMask maskDilR=$MASK_DILATE_RADIUS_PX maskedN=$maskedN frac=${"%.4f".format(maskedFrac)} " +
                    "thickR=$THICKEN_RADIUS_PX thickI=$THICKEN_ITERS"
        )

        val st = EdgeStats(
            maxScore = maxScore.toFloat(),
            p95Score = p95.toFloat(),
            thr = thrHigh.toFloat(),
            nnz = nnzFinal,
            density = dens,
            bandPx = 0f,
            samples = validCount,
            topKillY = yKill,
            maskedN = maskedN,
            maskedFrac = maskedFrac,
            maskDilateR = MASK_DILATE_RADIUS_PX
        )

        statsOut?.invoke(st)

        return EdgeMapOut3250(
            w = w,
            h = h,
            edgesU8 = outFinal,
            dirU8 = dirFinal,
            stats = st
        )
    }

    // ============================================================
    // DEBUG: guardar U8 como PNG en Galería
    // ============================================================

    private fun saveU8ToGalleryAsPng3250(
        ctx: Context,
        u8: ByteArray,
        w: Int,
        h: Int,
        displayName: String,
        invert: Boolean
    ): Uri? {
        if (u8.size != w * h || w <= 0 || h <= 0) return null

        val scale = computeDebugScale3250(w, h)
        val outW = (w * scale).toInt().coerceAtLeast(1)
        val outH = (h * scale).toInt().coerceAtLeast(1)

        val bmp = createBitmap(outW, outH)
        val row = IntArray(outW)

        for (yy in 0 until outH) {
            val ySrc = ((yy.toFloat() / outH) * h).toInt().coerceIn(0, h - 1)
            val offSrc = ySrc * w
            for (xx in 0 until outW) {
                val xSrc = ((xx.toFloat() / outW) * w).toInt().coerceIn(0, w - 1)
                var v = u8[offSrc + xSrc].toInt() and 0xFF
                if (invert) v = 255 - v
                row[xx] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
            }
            bmp.setPixels(row, 0, outW, 0, yy, outW, 1)
        }

        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, displayName)
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/debug3250")
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }

        val resolver = ctx.contentResolver
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: run {
            try { bmp.recycle() } catch (_: Throwable) {}
            return null
        }

        try {
            resolver.openOutputStream(uri)?.use { os: OutputStream ->
                bmp.compress(Bitmap.CompressFormat.PNG, 100, os)
            }
            values.clear()
            values.put(MediaStore.Images.Media.IS_PENDING, 0)
            resolver.update(uri, values, null, null)
            return uri
        } catch (t: Throwable) {
            Log.e(TAG, "saveU8ToGalleryAsPng3250 failed uri=$uri", t)
            try { resolver.delete(uri, null, null) } catch (_: Throwable) {}
            return null
        } finally {
            try { bmp.recycle() } catch (_: Throwable) {}
        }
    }

    private fun computeDebugScale3250(w: Int, h: Int): Float {
        val maxDim = max(w, h).toFloat()
        return if (maxDim <= DEBUG_MAX_DIM_PX) 1f else (DEBUG_MAX_DIM_PX / maxDim).coerceIn(0.10f, 1f)
    }

    // ============================================================
    // DEBUG TRIPLETE: EDGE_FULL / MASK / EDGE_USED(post-edge)
    // ============================================================

    private fun applyMaskPostEdge3250(edgesU8: ByteArray, maskU8: ByteArray): ByteArray {
        val out = edgesU8.copyOf()
        val n = min(out.size, maskU8.size)
        for (i in 0 until n) {
            if ((maskU8[i].toInt() and 0xFF) != 0) out[i] = 0
        }
        return out
    }

    private fun debugDumpEdgeTriplet3250(
        ctx: Context,
        w: Int,
        h: Int,
        edgesFullU8: ByteArray,
        maskFullU8: ByteArray?,
        debugTag: String,
        grayRawU8: ByteArray? = null,
        grayWorkU8: ByteArray? = null,
        gateMaskU8: ByteArray? = null
    ) {
        if (edgesFullU8.size != w * h) return

        val ts = System.currentTimeMillis()

        if (grayRawU8 != null && grayRawU8.size == w * h) {
            saveU8ToGalleryAsPng3250(
                ctx = ctx,
                u8 = grayRawU8,
                w = w,
                h = h,
                displayName = "EDGE_GRAY_RAW_${debugTag}_${ts}.png",
                invert = false
            )
        }

        if (grayWorkU8 != null && grayWorkU8.size == w * h) {
            saveU8ToGalleryAsPng3250(
                ctx = ctx,
                u8 = grayWorkU8,
                w = w,
                h = h,
                displayName = "EDGE_GRAY_WORK_${debugTag}_${ts}.png",
                invert = false
            )
        }

        if (gateMaskU8 != null && gateMaskU8.size == w * h) {
            saveU8ToGalleryAsPng3250(
                ctx = ctx,
                u8 = gateMaskU8,
                w = w,
                h = h,
                displayName = "EDGE_GATE_${debugTag}_${ts}.png",
                invert = false
            )
        }

        saveU8ToGalleryAsPng3250(
            ctx = ctx,
            u8 = edgesFullU8,
            w = w,
            h = h,
            displayName = "EDGE_FULL_${debugTag}_${ts}.png",
            invert = false
        )

        if (maskFullU8 != null && maskFullU8.size == w * h) {
            saveU8ToGalleryAsPng3250(
                ctx = ctx,
                u8 = maskFullU8,
                w = w,
                h = h,
                displayName = "EDGE_MASK_${debugTag}_${ts}.png",
                invert = false
            )

            val edgesUsed = applyMaskPostEdge3250(edgesFullU8, maskFullU8)
            saveU8ToGalleryAsPng3250(
                ctx = ctx,
                u8 = edgesUsed,
                w = w,
                h = h,
                displayName = "EDGE_USED_${debugTag}_${ts}.png",
                invert = false
            )
        }
    }

    // ============================================================
    // Pre-segmentación / aplanado previo (Bitmap path)
    // ============================================================

    private fun bitmapToGrayU83250(stillBmp: Bitmap): ByteArray {
        val w = stillBmp.width
        val h = stillBmp.height
        val gray = ByteArray(w * h)
        val row = IntArray(w)
        for (y in 0 until h) {
            stillBmp.getPixels(row, 0, w, 0, y, w, 1)
            val off = y * w
            for (x in 0 until w) {
                val c = row[x]
                val r = (c shr 16) and 0xFF
                val g = (c shr 8) and 0xFF
                val b = c and 0xFF
                val yv = (77 * r + 150 * g + 29 * b + 128) shr 8
                gray[off + x] = yv.toByte()
            }
        }
        return gray
    }

    private fun prepareBitmapPreseg3250(
        stillBmp: Bitmap,
        grayRawU8: ByteArray,
        maskFull: ByteArray?,
        debugTag: String,
        regionMaskFull3250: ByteArray? = null,
        forceRegion3250: Boolean = false
    ): BitmapPreseg3250 {
        val w = stillBmp.width
        val h = stillBmp.height
        val n = w * h

        if (w <= 0 || h <= 0 || grayRawU8.size != n) {
            return BitmapPreseg3250(grayRawU8, null, grayRawU8, false)
        }

        val preparedRegion = prepareRegionMaskFull3250(
            regionMaskFull3250 = regionMaskFull3250,
            w = w,
            h = h,
            debugTag = debugTag
        )

        if (preparedRegion != null) {
            val grayWork = applyRegionFlattenPre3250(
                grayRawU8 = grayRawU8,
                w = w,
                h = h,
                gateFull = preparedRegion,
                maskFull = maskFull
            )

            val cov = countNonZeroU83250(preparedRegion).toFloat() /
                    n.toFloat().coerceAtLeast(1f)

            Log.d(
                TAG,
                "EDGE_PRE[$debugTag] use region gate cov=${"%.4f".format(cov)} force=$forceRegion3250"
            )

            return BitmapPreseg3250(
                grayWorkU8 = grayWork,
                gateMaskU8 = preparedRegion,
                rawGrayU8 = grayRawU8,
                used = true
            )
        }

        if (forceRegion3250) {
            return BitmapPreseg3250(grayRawU8, null, grayRawU8, false)
        }

        val maxDim = max(w, h)
        val scale =
            if (maxDim <= PRESEG_SMALL_MAX_DIM_PX) 1f
            else PRESEG_SMALL_MAX_DIM_PX.toFloat() / maxDim.toFloat()

        val ws = max(1, (w * scale).toInt())
        val hs = max(1, (h * scale).toInt())

        val graySmall = resizeU8Nearest3250(grayRawU8, w, h, ws, hs)
        val coarseSmall = buildCoarseMeanGrayU83250(graySmall, ws, hs, PRESEG_SMALL_BLOCK_PX)

        val histGray = IntArray(256)
        for (b in graySmall) histGray[u8(b)]++

        val p35 = percentileFromGrayHist3250(histGray, 0.35f)
        val darkAbsMax = min(160, p35 + PRESEG_DARK_EXTRA)

        val keepX0 = (ws * PRESEG_KEEP_X0_FRAC).toInt().coerceIn(0, ws - 1)
        val keepX1 = (ws * PRESEG_KEEP_X1_FRAC).toInt().coerceIn(keepX0 + 1, ws)
        val keepY0 = (hs * PRESEG_KEEP_Y0_FRAC).toInt().coerceIn(0, hs - 1)
        val keepY1 = (hs * PRESEG_KEEP_Y1_FRAC).toInt().coerceIn(keepY0 + 1, hs)

        val cand = ByteArray(ws * hs)
        for (y in 0 until hs) {
            val off = y * ws
            for (x in 0 until ws) {
                if (x !in keepX0..<keepX1 || y < keepY0 || y >= keepY1) continue

                val i = off + x
                val g = u8(graySmall[i])
                val local = u8(coarseSmall[i])
                val delta = local - g

                val isDark = g <= darkAbsMax
                val isLocalDark = delta >= PRESEG_LOCAL_DARK_DELTA
                val isVeryDark = g <= max(32, darkAbsMax - 10)

                if (isDark && (isLocalDark || isVeryDark)) {
                    cand[i] = 0xFF.toByte()
                }
            }
        }

        var mat = closeMaskRectU83250(cand, ws, hs, 4, 2)
        mat = openMaskRectU83250(mat, ws, hs, 1, 1)
        mat = closeMaskRectU83250(mat, ws, hs, 6, 2)
        mat = keepLargestCentralComponentU83250(mat, ws, hs)

        if (countNonZeroU83250(mat) == 0) {
            Log.d(TAG, "EDGE_PRE[$debugTag] no central component")
            return BitmapPreseg3250(grayRawU8, null, grayRawU8, false)
        }

        var region = fillHolesU83250(mat, ws, hs)
        region = dilateMaskRectU83250(region, ws, hs, 5, 4)
        region = closeMaskRectU83250(region, ws, hs, 6, 3)

        val gateFull = resizeU8Nearest3250(region, ws, hs, w, h)
        val cov = countNonZeroU83250(gateFull).toFloat() / n.toFloat().coerceAtLeast(1f)
        val centerHits = countCenterHitsU83250(gateFull, w, h)

        if (cov !in PRESEG_MIN_COVERAGE..PRESEG_MAX_COVERAGE || centerHits < PRESEG_MIN_CENTER_HITS) {
            Log.d(
                TAG,
                "EDGE_PRE[$debugTag] reject cov=${"%.4f".format(cov)} centerHits=$centerHits ws=$ws hs=$hs"
            )
            return BitmapPreseg3250(grayRawU8, null, grayRawU8, false)
        }

        val grayWork = applyRegionFlattenPre3250(
            grayRawU8 = grayRawU8,
            w = w,
            h = h,
            gateFull = gateFull,
            maskFull = maskFull
        )

        Log.d(
            TAG,
            "EDGE_PRE[$debugTag] use photometric gate cov=${"%.4f".format(cov)} centerHits=$centerHits ws=$ws hs=$hs"
        )

        return BitmapPreseg3250(
            grayWorkU8 = grayWork,
            gateMaskU8 = gateFull,
            rawGrayU8 = grayRawU8,
            used = true
        )
    }
    private fun prepareRegionMaskFull3250(
        regionMaskFull3250: ByteArray?,
        w: Int,
        h: Int,
        debugTag: String
    ): ByteArray?
    {
        val n = w * h
        if (regionMaskFull3250 == null || regionMaskFull3250.size != n) return null

        var gate = closeMaskRectU83250(
            regionMaskFull3250,
            w,
            h,
            EXPECTED_MASK_CLOSE_RX,
            EXPECTED_MASK_CLOSE_RY
        )
        gate = dilateMaskRectU83250(
            gate,
            w,
            h,
            EXPECTED_MASK_DILATE_RX,
            EXPECTED_MASK_DILATE_RY
        )

        val nnz = countNonZeroU83250(gate)
        val cov = nnz.toFloat() / n.toFloat().coerceAtLeast(1f)
        if (nnz < EXPECTED_MASK_MIN_PIXELS || cov <= 0f || cov > EXPECTED_MASK_MAX_COVERAGE) {
            Log.d(
                TAG,
                "EDGE_PRE[$debugTag] reject expected gate nnz=$nnz cov=${"%.4f".format(cov)}"
            )
            return null
        }
        return gate
    }

    private fun applyRegionFlattenPre3250(
        grayRawU8: ByteArray,
        w: Int,
        h: Int,
        gateFull: ByteArray,
        maskFull: ByteArray?
    ): ByteArray {
        val n = w * h
        val coarseFull = buildCoarseMeanGrayU83250(grayRawU8, w, h, PRESEG_FULL_FLAT_BLOCK_PX)
        val grayWork = grayRawU8.copyOf()
        for (i in 0 until n) {
            val inGate = (gateFull[i].toInt() and 0xFF) != 0
            val veto = maskFull != null && (maskFull[i].toInt() and 0xFF) != 0
            if (!inGate || veto) {
                grayWork[i] = coarseFull[i]
            }
        }
        return grayWork
    }

    private fun buildCoarseMeanGrayU83250(
        grayU8: ByteArray,
        w: Int,
        h: Int,
        blockPx: Int
    ): ByteArray {
        if (grayU8.size != w * h || w <= 0 || h <= 0) return grayU8
        val b = blockPx.coerceAtLeast(1)
        val gw = (w + b - 1) / b
        val gh = (h + b - 1) / b
        val sums = IntArray(gw * gh)
        val counts = IntArray(gw * gh)

        for (y in 0 until h) {
            val gy = y / b
            val off = y * w
            for (x in 0 until w) {
                val gx = x / b
                val gi = gy * gw + gx
                sums[gi] += u8(grayU8[off + x])
                counts[gi]++
            }
        }

        val avg = ByteArray(gw * gh)
        for (i in avg.indices) {
            val c = counts[i].coerceAtLeast(1)
            avg[i] = (sums[i] / c).coerceIn(0, 255).toByte()
        }

        val out = ByteArray(w * h)
        for (y in 0 until h) {
            val gy = y / b
            val off = y * w
            for (x in 0 until w) {
                val gx = x / b
                out[off + x] = avg[gy * gw + gx]
            }
        }
        return out
    }

    private fun resizeU8Nearest3250(
        src: ByteArray,
        srcW: Int,
        srcH: Int,
        dstW: Int,
        dstH: Int
    ): ByteArray {
        if (src.size != srcW * srcH || srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0) {
            return ByteArray(0)
        }
        if (srcW == dstW && srcH == dstH) return src.copyOf()
        val out = ByteArray(dstW * dstH)
        for (y in 0 until dstH) {
            val ySrc = ((y.toLong() * srcH) / dstH).toInt().coerceIn(0, srcH - 1)
            val offDst = y * dstW
            val offSrc = ySrc * srcW
            for (x in 0 until dstW) {
                val xSrc = ((x.toLong() * srcW) / dstW).toInt().coerceIn(0, srcW - 1)
                out[offDst + x] = src[offSrc + xSrc]
            }
        }
        return out
    }

    private fun percentileFromGrayHist3250(hist: IntArray, q: Float): Int {
        var total = 0
        for (c in hist) total += c
        if (total <= 0) return 0
        val target = (total * q).toInt().coerceIn(0, total - 1)
        var acc = 0
        for (i in hist.indices) {
            acc += hist[i]
            if (acc > target) return i
        }
        return hist.lastIndex
    }

    private fun countNonZeroU83250(src: ByteArray): Int {
        var n = 0
        for (b in src) if ((b.toInt() and 0xFF) != 0) n++
        return n
    }

    private fun countCenterHitsU83250(src: ByteArray, w: Int, h: Int): Int {
        if (src.size != w * h || w <= 0 || h <= 0) return 0
        val x0 = (w * 0.30f).toInt().coerceIn(0, w - 1)
        val x1 = (w * 0.70f).toInt().coerceIn(x0 + 1, w)
        val y0 = (h * 0.18f).toInt().coerceIn(0, h - 1)
        val y1 = (h * 0.62f).toInt().coerceIn(y0 + 1, h)
        var n = 0
        for (y in y0 until y1) {
            val off = y * w
            for (x in x0 until x1) {
                if ((src[off + x].toInt() and 0xFF) != 0) n++
            }
        }
        return n
    }

    private fun dilateMaskRectU83250(src: ByteArray, w: Int, h: Int, rx: Int, ry: Int): ByteArray {
        if (src.size != w * h || w <= 0 || h <= 0) return src
        if (rx <= 0 && ry <= 0) return src.copyOf()
        val out = ByteArray(w * h)
        for (y in 0 until h) {
            val y0 = max(0, y - ry)
            val y1 = min(h - 1, y + ry)
            val offOut = y * w
            for (x in 0 until w) {
                val x0 = max(0, x - rx)
                val x1 = min(w - 1, x + rx)
                var hit = false
                loop@ for (yy in y0..y1) {
                    val off = yy * w
                    for (xx in x0..x1) {
                        if ((src[off + xx].toInt() and 0xFF) != 0) {
                            hit = true
                            break@loop
                        }
                    }
                }
                if (hit) out[offOut + x] = 0xFF.toByte()
            }
        }
        return out
    }

    private fun erodeMaskRectU83250(src: ByteArray, w: Int, h: Int, rx: Int, ry: Int): ByteArray {
        if (src.size != w * h || w <= 0 || h <= 0) return src
        if (rx <= 0 && ry <= 0) return src.copyOf()
        val out = ByteArray(w * h)
        for (y in 0 until h) {
            val offOut = y * w
            for (x in 0 until w) {
                if (x - rx < 0 || x + rx >= w || y - ry < 0 || y + ry >= h) {
                    out[offOut + x] = 0
                    continue
                }
                var keep = true
                loop@ for (yy in (y - ry)..(y + ry)) {
                    val off = yy * w
                    for (xx in (x - rx)..(x + rx)) {
                        if ((src[off + xx].toInt() and 0xFF) == 0) {
                            keep = false
                            break@loop
                        }
                    }
                }
                if (keep) out[offOut + x] = 0xFF.toByte()
            }
        }
        return out
    }

    private fun openMaskRectU83250(src: ByteArray, w: Int, h: Int, rx: Int, ry: Int): ByteArray {
        return dilateMaskRectU83250(erodeMaskRectU83250(src, w, h, rx, ry), w, h, rx, ry)
    }

    private fun closeMaskRectU83250(src: ByteArray, w: Int, h: Int, rx: Int, ry: Int): ByteArray {
        return erodeMaskRectU83250(dilateMaskRectU83250(src, w, h, rx, ry), w, h, rx, ry)
    }

    private fun keepLargestCentralComponentU83250(src: ByteArray, w: Int, h: Int): ByteArray {
        if (src.size != w * h || w <= 0 || h <= 0) return src
        val visited = ByteArray(w * h)
        val queue = IntArray(w * h)
        val out = ByteArray(w * h)
        val cxRef = w * 0.5f
        val cyRef = h * 0.38f

        var bestScore = Float.NEGATIVE_INFINITY
        var bestPixels: IntArray? = null
        var bestCount = 0

        for (i0 in src.indices) {
            if ((src[i0].toInt() and 0xFF) == 0) continue
            if ((visited[i0].toInt() and 0xFF) != 0) continue

            var qh = 0
            var qt = 0
            queue[qt++] = i0
            visited[i0] = 1

            var area = 0
            var minX = w
            var maxX = 0
            var minY = h
            var maxY = 0
            var sumX = 0f
            var sumY = 0f
            val pix = IntArray(w * h)
            var pixN = 0

            while (qh < qt) {
                val i = queue[qh++]
                pix[pixN++] = i
                area++
                val x = i % w
                val y = i / w
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y
                sumX += x.toFloat()
                sumY += y.toFloat()

                for (dy in -1..1) for (dx in -1..1) {
                    if (dx == 0 && dy == 0) continue
                    val xx = x + dx
                    val yy = y + dy
                    if (xx !in 0..<w || yy < 0 || yy >= h) continue
                    val j = yy * w + xx
                    if ((src[j].toInt() and 0xFF) == 0) continue
                    if ((visited[j].toInt() and 0xFF) != 0) continue
                    visited[j] = 1
                    queue[qt++] = j
                }
            }

            if (area <= 0) continue
            val bw = (maxX - minX + 1).coerceAtLeast(1)
            val bh = (maxY - minY + 1).coerceAtLeast(1)
            val cx = sumX / area.toFloat()
            val cy = sumY / area.toFloat()
            val dist = abs(cx - cxRef) + abs(cy - cyRef) * 1.5f
            val widthBonus = bw * 2.0f
            val aspectBonus = if (bw >= bh) 60f else 0f
            val score = area.toFloat() + widthBonus + aspectBonus - dist * 6f

            if (score > bestScore) {
                bestScore = score
                bestPixels = pix.copyOfRange(0, pixN)
                bestCount = pixN
            }
        }

        if (bestPixels != null && bestCount > 0) {
            for (k in 0 until bestCount) {
                out[bestPixels[k]] = 0xFF.toByte()
            }
        }
        return out
    }

    private fun fillHolesU83250(src: ByteArray, w: Int, h: Int): ByteArray {
        if (src.size != w * h || w <= 0 || h <= 0) return src
        val inv = ByteArray(w * h)
        for (i in src.indices) {
            inv[i] = if ((src[i].toInt() and 0xFF) == 0) 0xFF.toByte() else 0
        }

        val visited = ByteArray(w * h)
        val queue = IntArray(w * h)
        var qh = 0
        var qt = 0

        fun push(i: Int) {
            if ((inv[i].toInt() and 0xFF) == 0) return
            if ((visited[i].toInt() and 0xFF) != 0) return
            visited[i] = 1
            queue[qt++] = i
        }

        for (x in 0 until w) {
            push(x)
            push((h - 1) * w + x)
        }
        for (y in 0 until h) {
            push(y * w)
            push(y * w + (w - 1))
        }

        while (qh < qt) {
            val i = queue[qh++]
            val x = i % w
            val y = i / w
            if (x > 0) push(i - 1)
            if (x < w - 1) push(i + 1)
            if (y > 0) push(i - w)
            if (y < h - 1) push(i + w)
        }

        val out = src.copyOf()
        for (i in out.indices) {
            if ((inv[i].toInt() and 0xFF) != 0 && (visited[i].toInt() and 0xFF) == 0) {
                out[i] = 0xFF.toByte()
            }
        }
        return out
    }

    // ============================================================
    // Internals edge core
    // ============================================================

    private fun thickenedgemapu83250(
        edges: ByteArray,
        w: Int,
        h: Int,
        radius: Int = 1,
        iters: Int = 1
    ): ByteArray {
        if (edges.size != w * h || w <= 0 || h <= 0) return edges
        if (radius <= 0 || iters <= 0) return edges

        var src = edges
        repeat(iters) {
            val dst = ByteArray(w * h)
            for (y in 0 until h) {
                val y0 = max(0, y - radius)
                val y1 = min(h - 1, y + radius)
                val rowOut = y * w
                for (x in 0 until w) {
                    val x0 = max(0, x - radius)
                    val x1 = min(w - 1, x + radius)
                    var hit = false
                    run {
                        var yy = y0
                        while (yy <= y1) {
                            val row = yy * w
                            var xx = x0
                            while (xx <= x1) {
                                if ((src[row + xx].toInt() and 0xFF) != 0) { hit = true; return@run }
                                xx++
                            }
                            yy++
                        }
                    }
                    if (hit) dst[rowOut + x] = 0xFF.toByte()
                }
            }
            src = dst
        }
        return src
    }

    private fun percentileFromHist3250(hist: IntArray, bins: Int, maxScore: Int, q: Float): Int {
        var total = 0
        for (c in hist) total += c
        if (total <= 0) return 0
        val target = (total * q).toInt().coerceIn(0, total - 1)

        var acc = 0
        for (b in 0 until bins) {
            acc += hist[b]
            if (acc > target) return (b * maxScore) / (bins - 1).coerceAtLeast(1)
        }
        return maxScore
    }

    private fun computeThrHigh3250(maxScore: Int, p95: Int): Int {
        val thrBase = max(EDGE_THR_MIN, (EDGE_THR_FRAC * maxScore.toFloat()).toInt())
        val thrCap = max(EDGE_THR_MIN, (P95_CAP_K * p95.toFloat()).toInt())
        return min(thrBase, thrCap).coerceAtLeast(EDGE_THR_MIN)
    }

    private fun quantizeDir3250(gx: Int, gy: Int, ax: Int, ay: Int): Int {
        if (ax == 0 && ay == 0) return 0
        if (ay * TAN22_DEN <= ax * TAN22_NUM) return 0
        if (ay * TAN67_DEN >= ax * TAN67_NUM) return 2
        return if ((gx xor gy) >= 0) 1 else 3
    }

    private fun nmsNeighbors3250(
        i: Int,
        x: Int,
        y: Int,
        w: Int,
        dir: Int
    ): Pair<Int, Int> {
        return when (dir) {
            0 -> (i - 1) to (i + 1)
            2 -> (i - w) to (i + w)
            1 -> (i - w - 1) to (i + w + 1)
            else -> (i - w + 1) to (i + w - 1)
        }
    }

    private fun isHorizontalEdgeSeed3250(
        edges: ByteArray,
        dir: ByteArray,
        idx: Int
    ): Boolean {
        if ((edges[idx].toInt() and 0xFF) == 0) return false
        return (dir[idx].toInt() and 0xFF) == 2
    }

    private fun bridgehorizontalgapsu83250(
        edges: ByteArray,
        dir: ByteArray,
        w: Int,
        h: Int,
        maxGapPx: Int = 5
    ): ByteArray {
        if (edges.size != w * h || dir.size != w * h || w <= 2 || h <= 2) return edges
        if (maxGapPx <= 0) return edges

        val out = edges.copyOf()

        for (y in 1 until h - 1) {
            var x = 1
            while (x < w - 1) {
                val i0 = y * w + x
                if (!isHorizontalEdgeSeed3250(edges, dir, i0)) {
                    x++
                    continue
                }

                var bridged = false
                val xMax = min(w - 2, x + maxGapPx + 1)

                var xr = x + 2
                while (xr <= xMax) {
                    val ir = y * w + xr
                    if (isHorizontalEdgeSeed3250(edges, dir, ir)) {
                        for (xx in (x + 1) until xr) {
                            out[y * w + xx] = 0xFF.toByte()
                        }
                        x = xr
                        bridged = true
                        break
                    }
                    xr++
                }

                if (!bridged) x++
            }
        }

        return out
    }
}
