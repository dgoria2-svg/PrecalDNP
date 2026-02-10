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
 * maskFull (si se pasa) está en coords FULL (w*h) y se usa:
 * - ✅ SOLO para stats y dumps DEBUG (ver qué pasaría con censura post-edge).
 * - ❌ NO altera el edgeMapFull (NO flatten, NO scoreWeight, NO hard-kill).
 *
 * La censura/marker (ojo/ceja) se aplica en RimDetector (post-edge), no acá.
 */
object EdgeMapBuilder3250 {

    private const val TAG = "EdgeMapBuilder3250"

    // ------------------------------------------------------------
    // Parámetros (FULL)
    // ------------------------------------------------------------

    private const val K_H = 0.60f
    private const val K_V = 0.60f

    private const val EDGE_THR_FRAC = 0.16f
    private const val EDGE_THR_MIN = 6

    private const val P95_CAP_K = 1.10f

    private const val TAN22_NUM = 41
    private const val TAN22_DEN = 100
    private const val TAN67_NUM = 241
    private const val TAN67_DEN = 100

    // ✅ Engrosar edgeMap (dilate) DESPUÉS de hysteresis (0/255)
    private const val THICKEN_RADIUS_PX = 1   // 1 => kernel 3x3
    private const val THICKEN_ITERS = 1

    private const val DIR_MAX_ANGLE_DEG = 25f
    @Suppress("unused")
    private val DIR_COS_TOL: Float = cos(Math.toRadians(DIR_MAX_ANGLE_DEG.toDouble())).toFloat()

    private const val DEBUG_MAX_DIM_PX = 1600

    // ✅ TopKill suave: rampa lineal desde y=0 hasta y=topKillY.
    private const val TOPKILL_MIN_WEIGHT = 0.10f

    // ============================================================
    // ✅ MASK DEBUG (se mantiene para no romper nombres / lógica)
    // ============================================================

    // 0 = sin dilatar mask (RECOMENDADO para no comerte el aro)
    private const val MASK_DILATE_RADIUS_PX = 0

    // Estos dos quedan “sin efecto” en la construcción del edgeMap (por diseño).
    @Suppress("unused")
    private const val MASK_FLATTEN_BLEND = 0.35f   // (NO usado)
    @Suppress("unused")
    private const val MASK_SCORE_WEIGHT = 0.85f    // (NO usado)

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

    /**
     * ✅ Salida ÚNICA del builder: edges + dir + stats (FULL).
     */
    class EdgeMapOut3250(
        val w: Int,
        val h: Int,
        val edgesU8: ByteArray, // 0/255 post-hysteresis (+thicken)
        val dirU8: ByteArray,   // 0..3 para píxeles NMS; 255 para píxeles agregados por thicken
        val stats: EdgeStats
    )

    private fun u8(b: Byte): Int = b.toInt() and 0xFF

    // ============================================================
    // ✅ FULL FRAME: Bitmap -> EdgeMapOut3250 (edges + dir + stats)
    // ============================================================

    fun buildFullFrameEdgePackFromBitmap3250(
        stillBmp: Bitmap,
        borderKillPx: Int,
        topKillY: Int = 0,
        maskFull: ByteArray? = null,
        debugTag: String = "FULL",
        statsOut: ((EdgeStats) -> Unit)? = null,

        // DEBUG opcional
        ctx: Context? = null,
        debugSaveToGallery3250: Boolean = false,
        debugAlsoSaveGray3250: Boolean = false
    ): EdgeMapOut3250 {

        val w = stillBmp.width
        val h = stillBmp.height
        if (w <= 0 || h <= 0) {
            val st = EdgeStats(0f, 0f, 0f, 0, 0f, 0f, 0, topKillY, 0, 0f, 0)
            return EdgeMapOut3250(0, 0, ByteArray(0), ByteArray(0), st)
        }

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

        val pack = buildFullFrameEdgePackFromGrayU8_3250(
            w = w,
            h = h,
            grayU8 = gray,
            borderKillPx = borderKillPx,
            topKillY = topKillY,
            maskFull = maskFull,
            debugTag = debugTag,
            statsOut = statsOut
        )

        // ✅ DEBUG: triplete limpio (GRAY opcional + EDGE_FULL + MASK + EDGE_USED post-edge)
        if (debugSaveToGallery3250 && ctx != null) {
            try {
                if (debugAlsoSaveGray3250) {
                    saveU8ToGalleryAsPng3250(
                        ctx = ctx,
                        u8 = gray,
                        w = w,
                        h = h,
                        displayName = "EDGE_GRAY_${debugTag}_${System.currentTimeMillis()}.png",
                        invert = false
                    )
                }
                debugDumpEdgeTriplet3250(
                    ctx = ctx,
                    w = w,
                    h = h,
                    edgesFullU8 = pack.edgesU8,
                    maskFullU8 = maskFull,
                    debugTag = debugTag
                )
            } catch (t: Throwable) {
                Log.e(TAG, "EDGE dump save failed", t)
            }
        }

        return pack
    }

    // ============================================================
    // ✅ FULL FRAME: GrayU8 -> EdgeMapOut3250
    // ============================================================

    fun buildFullFrameEdgePackFromGrayU8_3250(
        w: Int,
        h: Int,
        grayU8: ByteArray,
        borderKillPx: Int,
        topKillY: Int = 0,
        maskFull: ByteArray? = null,
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

        val B = borderKillPx.coerceIn(0, min(w, h) / 3)

        val yKill = topKillY.coerceIn(0, h)
        val hasTopKill = (yKill in 1 until h)

        // ✅ mask dilatada opcional (por defecto 0) — SOLO stats/debug
        val hasMaskIn = (maskFull != null)
        val maskWork: ByteArray? = if (hasMaskIn && MASK_DILATE_RADIUS_PX > 0) {
            dilateMaskU8_3250(maskFull, w, h, radius = MASK_DILATE_RADIUS_PX)
        } else maskFull

        val hasMask = (maskWork != null)

        fun isMasked(i: Int): Boolean =
            hasMask && ((maskWork[i].toInt() and 0xFF) != 0)

        // stats mask (NO afecta edges)
        var maskedN = 0
        if (hasMask) {
            for (i in 0 until N) if (isMasked(i)) maskedN++
        }
        val maskedFrac = maskedN.toFloat() / N.toFloat().coerceAtLeast(1f)

        fun pix(i: Int): Int {
            // ✅ EDGE_FULL “puro”: NO flatten, NO scoreWeight por máscara
            return u8(grayU8[i])
        }

        fun topWeight(y: Int): Float {
            if (!hasTopKill) return 1f
            if (y >= yKill) return 1f
            val t = y.toFloat() / yKill.toFloat().coerceAtLeast(1f) // 0..1
            return (TOPKILL_MIN_WEIGHT + (1f - TOPKILL_MIN_WEIGHT) * t)
                .coerceIn(TOPKILL_MIN_WEIGHT, 1f)
        }

        val score = ShortArray(N)
        val dir = ByteArray(N)
        var maxScore = 0
        var validCount = 0

        val kH100 = (K_H * 100f).toInt()
        val kV100 = (K_V * 100f).toInt()

        // Pass 1: Scharr + score + dir + max (sin máscara)
        for (y in 1 until (h - 1)) {
            if (y < B || y >= h - B) continue
            val tw = topWeight(y)

            val off = y * w
            val offUp = (y - 1) * w
            val offDn = (y + 1) * w

            for (x in 1 until (w - 1)) {
                if (x < B || x >= w - B) continue
                val i = off + x

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
                var s = (s100 / 100).coerceIn(0, 32767)
                if (s == 0) continue

                s = (s.toFloat() * tw).toInt().coerceIn(0, 32767)
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
                "EDGE_FULL[$debugTag] not enough signal valid=$validCount max=$maxScore topKillY=$yKill B=$B hasMask=$hasMask"
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

        // Pass 2: hist p95 (preferible sin masked, así el thr refleja el aro)
        // Pass 2: hist p95 (SIN máscara; máscara NO altera edgeMapFull)
        val bins = 2048
        val hist = IntArray(bins)
        var histN = 0
        for (i in 0 until N) {
            val s = score[i].toInt()
            if (s <= 0) continue
            val bin = (s * (bins - 1)) / maxScore
            hist[bin]++
            histN++
        }

        val p95 = percentileFromHist3250(hist, bins, maxScore, 0.995f)  // antes 0.95f
        val thrHigh = computeThrHigh3250(maxScore, p95)
        val thrLow  = max(EDGE_THR_MIN, (thrHigh * 0.85f).toInt())     // subí LOW (0.60–0.70)

        // Pass 3: NMS + weak/strong
        val classMap = ByteArray(N) // 0 none, 1 weak, 2 strong
        var candCount = 0
        var strongCount = 0

        for (y in 1 until (h - 1)) {
            if (y < B || y >= h - B) continue
            val off = y * w
            for (x in 1 until (w - 1)) {
                if (x < B || x >= w - B) continue
                val i = off + x
                val s = score[i].toInt()
                if (s < thrLow) continue

                val d = dir[i].toInt() and 0xFF
                val (i1, i2) = nmsNeighbors3250(i, x, y, w, d)
                val s1 = score[i1].toInt()
                val s2 = score[i2].toInt()
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

        // Pass 4: hysteresis BFS
        val out = ByteArray(N)
        if (strongCount == 0 || candCount == 0) {
            Log.w(
                TAG,
                "EDGE_FULL[$debugTag] no strong/candidates (strong=$strongCount cand=$candCount) thrH=$thrHigh thrL=$thrLow"
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

        val q = IntArray(candCount)
        var qh = 0
        var qt = 0

        // seed: strong
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
                if (classMap[j].toInt() == 1) {
                    classMap[j] = 2
                    out[j] = 0xFF.toByte()
                    q[qt++] = j
                }
            }
        }

        // ✅ Engrosar (dilate) el edgeMap FINAL (0/255)
        val outThick = thickenEdgeMapU8_3250(out, w, h, radius = THICKEN_RADIUS_PX, iters = THICKEN_ITERS)

        // ✅ Dir: para píxeles "nuevos" agregados por el engrosado, 255 para NO filtrar orientación
        val dirThick = dir.copyOf()
        for (i in 0 until N) {
            val eNew = outThick[i].toInt() and 0xFF
            if (eNew == 0) continue
            val eOld = out[i].toInt() and 0xFF
            if (eOld == 0) dirThick[i] = 0xFF.toByte()
        }

        var nnzFinal = 0
        for (b in outThick) if ((b.toInt() and 0xFF) != 0) nnzFinal++
        val dens = nnzFinal.toFloat() / N.toFloat().coerceAtLeast(1f)

        Log.d(
            TAG,
            "EDGE_FULL[$debugTag] max=$maxScore p95=$p95 thrH=$thrHigh thrL=$thrLow " +
                    "strong=$strongCount cand=$candCount nnz=$nnzFinal/$N dens=${"%.4f".format(dens)} " +
                    "topKillY=$yKill B=$B hasMask=$hasMask maskDilR=$MASK_DILATE_RADIUS_PX maskedN=$maskedN frac=${"%.4f".format(maskedFrac)} " +
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

        return EdgeMapOut3250(w, h, outThick, dirThick, st)
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
    // ✅ DEBUG TRIPLETE: EDGE_FULL / MASK / EDGE_USED(post-edge)
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
        debugTag: String
    ) {
        if (edgesFullU8.size != w * h) return

        // 1) EDGE_FULL (verdad)
        saveU8ToGalleryAsPng3250(
            ctx = ctx,
            u8 = edgesFullU8,
            w = w,
            h = h,
            displayName = "EDGE_FULL_${debugTag}_${System.currentTimeMillis()}.png",
            invert = false
        )

        // 2) MASK + 3) EDGE_USED (post-edge) si hay máscara
        if (maskFullU8 != null && maskFullU8.size == w * h) {
            saveU8ToGalleryAsPng3250(
                ctx = ctx,
                u8 = maskFullU8,
                w = w,
                h = h,
                displayName = "EDGE_MASK_${debugTag}_${System.currentTimeMillis()}.png",
                invert = false
            )

            val edgesUsed = applyMaskPostEdge3250(edgesFullU8, maskFullU8)
            saveU8ToGalleryAsPng3250(
                ctx = ctx,
                u8 = edgesUsed,
                w = w,
                h = h,
                displayName = "EDGE_USED_${debugTag}_${System.currentTimeMillis()}.png",
                invert = false
            )
        }
    }

    // ============================================================
    // Internals
    // ============================================================

    private fun dilateMaskU8_3250(mask: ByteArray, w: Int, h: Int, radius: Int): ByteArray {
        if (radius <= 0) return mask
        val out = ByteArray(w * h)
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
                            if ((mask[row + xx].toInt() and 0xFF) != 0) { hit = true; return@run }
                            xx++
                        }
                        yy++
                    }
                }
                if (hit) out[rowOut + x] = 1
            }
        }
        return out
    }

    private fun thickenEdgeMapU8_3250(
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

    /**
     * Cuantiza dirección (0,45,90,135) sin atan2 (rápido).
     */
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
}
