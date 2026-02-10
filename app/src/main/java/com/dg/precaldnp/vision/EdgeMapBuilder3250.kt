@file:Suppress("SameParameterValue")

package com.dg.precaldnp.vision

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import java.io.OutputStream
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min

/**
 * ✅ ÚNICA fuente de EDGE MAP 3250.
 *
 * Regla: still -> edgeMapFull (binario 0/255) UNA VEZ.
 * - RimDetector trabaja SOLO sobre edgeMap (recortando ROIs)
 * - ArcFit trabaja SOLO sobre edgeMap (nunca regenera bordes)
 *
 * DEBUG: puede dumpear el edgeMap (y opcionalmente gray) a Galería desde acá mismo.
 */
object EdgeMapBuilder3250 {

    private const val TAG = "EdgeMapBuilder3250"

    enum class EdgeMode3250 { LEGACY, DIVINE }

    // ------------------------------------------------------------
    // Parámetros
    // ------------------------------------------------------------

    private const val BAND_MIN_PX = 6f
    private const val BAND_MAX_PX = 42f
    private const val BAND_K_T = 1.75f

    private const val K_H = 0.60f
    private const val K_V = 0.60f

    private const val EDGE_THR_FRAC = 0.16f
    private const val EDGE_THR_MIN = 6

    private const val HYST_LOW_FRAC = 0.50f
    private const val P95_CAP_K = 1.10f

    private const val TAN22_NUM = 41
    private const val TAN22_DEN = 100
    private const val TAN67_NUM = 241
    private const val TAN67_DEN = 100

    private const val DIR_MAX_ANGLE_DEG = 25f
    private val DIR_COS_TOL: Float = cos(Math.toRadians(DIR_MAX_ANGLE_DEG.toDouble())).toFloat()

    // DEBUG: si el still es gigante, conviene bajar tamaño del PNG debug
    private const val DEBUG_MAX_DIM_PX = 1600

    data class EdgeStats(
        val maxScore: Float,
        val p95Score: Float,
        val thr: Float,
        val nnz: Int,
        val density: Float,
        val bandPx: Float,
        val samples: Int,
        val topKillY: Int
    )

    private fun u8(b: Byte): Int = b.toInt() and 0xFF

    // ============================================================
    // ✅ FULL FRAME: Bitmap -> EdgeMap (0/255)
    // ============================================================

    fun buildFullFrameEdgeMapFromBitmap3250(
        stillBmp: Bitmap,
        borderKillPx: Int,
        topKillY: Int = 0,
        maskFull: ByteArray? = null,
        debugTag: String = "FULL",
        statsOut: ((EdgeStats) -> Unit)? = null,

        // ✅ DEBUG integrado (no “otra llamada”)
        ctx: Context? = null,
        debugSaveToGallery3250: Boolean = false,
        debugAlsoSaveGray3250: Boolean = false,
    ): ByteArray {
        val w = stillBmp.width
        val h = stillBmp.height
        val bb = listOfNotNull(iris.browLeftBottomYpx, iris.browRightBottomYpx)
            .filter { it.isFinite() }
            .minOrNull()

        val topKillY = if (bb != null) {
            // anti-ceja: mata solo arriba de browBottom - pad, pero nunca más abajo de 30% de la imagen
            (bb - 18f).toInt().coerceIn(0, (h * 0.30f).toInt())
        } else 0

        if (w <= 0 || h <= 0) return ByteArray(0)

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

        val out = buildFullFrameEdgeMapFromGrayU8_3250(
            w = w,
            h = h,
            grayU8 = gray,
            borderKillPx = borderKillPx,
            topKillY = topKillY,
            maskFull = maskFull,
            debugTag = debugTag,
            statsOut = statsOut
        )

        // ✅ Dump integrado (FULL)
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
                saveU8ToGalleryAsPng3250(
                    ctx = ctx,
                    u8 = out,
                    w = w,
                    h = h,
                    displayName = "EDGE_FULL_${debugTag}_${System.currentTimeMillis()}.png",
                    invert = false
                )
                Log.d(TAG, "EDGE dump saved to gallery (tag=$debugTag)")
            } catch (t: Throwable) {
                Log.e(TAG, "EDGE dump save failed", t)
            }
        }

        return out
    }

    fun buildFullFrameEdgeMapFromGrayU8_3250(
        w: Int,
        h: Int,
        grayU8: ByteArray,
        borderKillPx: Int,
        topKillY: Int = 0,
        maskFull: ByteArray? = null,
        debugTag: String = "FULL",
        statsOut: ((EdgeStats) -> Unit)? = null
    ): ByteArray {

        val N = w * h
        if (w <= 0 || h <= 0) return ByteArray(0)
        if (grayU8.size != N) {
            Log.w(TAG, "EDGE_FULL[$debugTag] bad gray size=${grayU8.size} N=$N")
            return ByteArray(0)
        }
        if (maskFull != null && maskFull.size != N) {
            Log.w(TAG, "EDGE_FULL[$debugTag] bad mask size=${maskFull.size} N=$N")
            return ByteArray(0)
        }

        val B = borderKillPx.coerceIn(0, min(w, h) / 3)
        val yKill = topKillY.coerceIn(0, h)

        val blurU8 = boxBlur3x3U8(grayU8, w, h)

        val score = ShortArray(N)
        val dir = ByteArray(N)
        var maxScore = 0
        var validCount = 0

        val hasMask = (maskFull != null)

        fun pix(idx: Int): Int {
            return if (hasMask && (maskFull!![idx].toInt() and 0xFF) != 0) u8(blurU8[idx]) else u8(grayU8[idx])
        }

        // Pass 1
        for (y in 1 until (h - 1)) {
            if (y < yKill) continue
            if (y < B || y >= h - B) continue
            val off = y * w
            val offUp = (y - 1) * w
            val offDn = (y + 1) * w

            for (x in 1 until (w - 1)) {
                if (x < B || x >= w - B) continue
                val i = off + x
                if (hasMask && (maskFull!![i].toInt() and 0xFF) != 0)

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

                val hs100 = ay * 100 - (K_H * 100f).toInt() * ax
                val vs100 = ax * 100 - (K_V * 100f).toInt() * ay
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
            Log.w(TAG, "EDGE_FULL[$debugTag] not enough signal valid=$validCount max=$maxScore yKill=$yKill B=$B")
            return ByteArray(N)
        }

        // Pass 2: hist p95
        val bins = 2048
        val hist = IntArray(bins)
        for (i in 0 until N) {
            val s = score[i].toInt()
            if (s <= 0) continue
            val bin = (s * (bins - 1)) / maxScore
            hist[bin]++
        }

        val p95 = percentileFromHist3250(hist, bins, maxScore, 0.95f)
        val thrHigh = computeThrHigh3250(maxScore, p95)
        val thrLow = max(EDGE_THR_MIN, (thrHigh * HYST_LOW_FRAC).toInt())

        // Pass 3: NMS + weak/strong
        val classMap = ByteArray(N)
        var candCount = 0
        var strongCount = 0

        for (y in 1 until (h - 1)) {
            if (y < yKill) continue
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
            Log.w(TAG, "EDGE_FULL[$debugTag] no strong/candidates (strong=$strongCount cand=$candCount) thrH=$thrHigh thrL=$thrLow")
            return out
        }

        val q = IntArray(candCount)
        var qh = 0
        var qt = 0
        var nnz = 0

        for (i in 0 until N) {
            if (classMap[i].toInt() == 2) {
                out[i] = 0xFF.toByte()
                q[qt++] = i
                nnz++
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
                if (yy < yKill) continue
                if (xx < B || xx >= w - B) continue
                if (yy < B || yy >= h - B) continue
                val j = yy * w + xx
                if (classMap[j].toInt() == 1) {
                    classMap[j] = 2
                    out[j] = 0xFF.toByte()
                    q[qt++] = j
                    nnz++
                }
            }
        }

        val dens = nnz.toFloat() / N.toFloat().coerceAtLeast(1f)
        Log.d(TAG, "EDGE_FULL[$debugTag] max=$maxScore p95=$p95 thrH=$thrHigh thrL=$thrLow strong=$strongCount cand=$candCount nnz=$nnz/$N dens=${"%.4f".format(dens)} yKill=$yKill B=$B")

        statsOut?.invoke(
            EdgeStats(
                maxScore = maxScore.toFloat(),
                p95Score = p95.toFloat(),
                thr = thrHigh.toFloat(),
                nnz = nnz,
                density = dens,
                bandPx = 0f,
                samples = validCount,
                topKillY = yKill
            )
        )

        return out
    }

    // ============================================================
    // DEBUG: guardar U8 (gray o edge) como PNG en Galería
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

        // downscale para debug si es gigante
        val scale = computeDebugScale3250(w, h)
        val outW = (w * scale).toInt().coerceAtLeast(1)
        val outH = (h * scale).toInt().coerceAtLeast(1)

        val bmp = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
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
            if (Build.VERSION.SDK_INT >= 29) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/debug3250")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
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
            if (Build.VERSION.SDK_INT >= 29) {
                values.clear()
                values.put(MediaStore.Images.Media.IS_PENDING, 0)
                resolver.update(uri, values, null, null)
            }
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
    // Internals
    // ============================================================

    private fun boxBlur3x3U8(grayU8: ByteArray, w: Int, h: Int): ByteArray {
        val out = ByteArray(w * h)
        if (w <= 0 || h <= 0) return out

        var prev = IntArray(w)
        var curr = IntArray(w)
        var next = IntArray(w)

        fun buildHsumRow(y: Int, dst: IntArray) {
            val yy = y.coerceIn(0, h - 1)
            val base = yy * w
            for (x in 0 until w) {
                val xm1 = if (x > 0) x - 1 else 0
                val xp1 = if (x < w - 1) x + 1 else w - 1
                dst[x] = u8(grayU8[base + xm1]) + u8(grayU8[base + x]) + u8(grayU8[base + xp1])
            }
        }

        buildHsumRow(0, curr)
        buildHsumRow(0, prev)
        buildHsumRow(if (h > 1) 1 else 0, next)

        for (y in 0 until h) {
            val base = y * w
            for (x in 0 until w) {
                val sum = prev[x] + curr[x] + next[x]
                out[base + x] = (sum / 9).coerceIn(0, 255).toByte()
            }
            val tmp = prev
            prev = curr
            curr = next
            next = tmp
            buildHsumRow(y + 2, next)
        }

        return out
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

    private fun nmsNeighbors3250(i: Int, x: Int, y: Int, w: Int, dir: Int): Pair<Int, Int> {
        return when (dir) {
            0 -> (i - 1) to (i + 1)
            2 -> (i - w) to (i + w)
            1 -> (i - w - 1) to (i + w + 1)
            else -> (i - w + 1) to (i + w - 1)
        }
    }
}
