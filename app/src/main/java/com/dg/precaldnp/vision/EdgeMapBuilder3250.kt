@file:Suppress("SameParameterValue")

package com.dg.precaldnp.vision

import android.graphics.Bitmap
import android.util.Log
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * ✅ ÚNICA fuente de EDGE MAP 3250.
 *
 * Regla: la foto (still) -> edgeMapFull (binario 0/255) UNA VEZ, y luego:
 * - RimDetector trabaja SOLO sobre este edgeMap (recortando ROIs)
 * - ArcFit trabaja SOLO sobre edgeMap (nunca regenera bordes desde la imagen)
 *
 * Este builder incluye:
 * - Flattening (anti-ojo/ceja) usando mask (reemplazo por blur) para NO crear "halo edges"
 * - Scharr SIGNADO (implementación pura en Kotlin, sin OpenCV)
 * - Score tipo “tus PNG” (sesgo H/V)
 * - NMS (thin edges)
 * - Hysteresis (conectividad fuerte-débil) estilo Canny
 *
 * Y conserva tu función ROI/annulus (ArcFit) pero mejorada con NMS + hysteresis + umbral robusto.
 */
object EdgeMapBuilder3250 {

    private const val TAG = "EdgeMapBuilder3250"

    enum class EdgeMode3250 { LEGACY, DIVINE }

    // ============================================================
    // Parámetros (ajustables, pero estos defaults son razonables)
    // ============================================================

    // Banda elíptica alrededor del aro esperado (solo para buildArcFit... ROI)
    private const val BAND_MIN_PX = 6f
    private const val BAND_MAX_PX = 42f
    private const val BAND_K_T = 1.75f

    // Score tipo “tus PNG” (a partir de abs(gx), abs(gy))
    private const val K_H = 0.60f
    private const val K_V = 0.60f

    // Umbral alto base (se combina con p95 para cap)
    private const val EDGE_THR_FRAC = 0.16f
    private const val EDGE_THR_MIN = 6    // en unidades "score" (aprox)

    // Hysteresis
    private const val HYST_LOW_FRAC = 0.50f  // thrLow = thrHigh * frac
    private const val P95_CAP_K = 1.10f      // thrCap = 1.10 * p95

    // NMS: cuantización de dirección (0,45,90,135) usando aprox tan(22.5)=0.414 y tan(67.5)=2.414
    private const val TAN22_NUM = 41
    private const val TAN22_DEN = 100
    private const val TAN67_NUM = 241
    private const val TAN67_DEN = 100

    // Divine: gradiente alineado con normal elíptica (solo buildArcFit... ROI)
    private const val DIR_MAX_ANGLE_DEG = 25f
    private val DIR_COS_TOL: Float = cos(Math.toRadians(DIR_MAX_ANGLE_DEG.toDouble())).toFloat()

    data class EdgeStats(
        val maxScore: Float,
        val p95Score: Float,
        val thr: Float,      // thrHigh
        val nnz: Int,        // nnz final (post-hysteresis)
        val density: Float,
        val bandPx: Float,   // 0f en FULL
        val samples: Int,
        val topKillY: Int
    )

    // ============================================================
    // Helpers U8
    // ============================================================

    private fun u8(b: Byte): Int = b.toInt() and 0xFF

    // ============================================================
    // ✅ FULL FRAME: Bitmap -> EdgeMap (0/255) del mismo tamaño (w*h)
    // ============================================================

    /**
     * FULL-FRAME edge map binario (0/255), misma resolución que el still.
     *
     * @param stillBmp ARGB_8888 (o convertible)
     * @param borderKillPx kill duro alrededor del frame (recomendado 2..6)
     * @param topKillY kill duro anti-ceja por Y (global). y < topKillY => 0
     * @param maskFull ByteArray w*h; !=0 mata/flatten (ojos/ceja/zonas prohibidas)
     */

    fun buildFullFrameEdgeMapFromBitmap3250(
        stillBmp: Bitmap,
        borderKillPx: Int,
        topKillY: Int = 0,
        maskFull: ByteArray? = null,
        debugTag: String = "FULL",
        statsOut: ((EdgeStats) -> Unit)? = null
    ): ByteArray {
        val w = stillBmp.width
        val h = stillBmp.height
        if (w <= 0 || h <= 0) return ByteArray(0)

        val gray = ByteArray(w * h)
        val row = IntArray(w)

        // Gray (luma) sin asignar un IntArray gigante full-frame
        for (y in 0 until h) {
            stillBmp.getPixels(row, 0, w, 0, y, w, 1)
            val off = y * w
            for (x in 0 until w) {
                val c = row[x]
                val r = (c shr 16) and 0xFF
                val g = (c shr 8) and 0xFF
                val b = c and 0xFF
                // y ≈ 0.299R + 0.587G + 0.114B (fixed-point)
                val yv = (77 * r + 150 * g + 29 * b + 128) shr 8
                gray[off + x] = yv.toByte()
            }
        }

        return buildFullFrameEdgeMapFromGrayU8_3250(
            w = w,
            h = h,
            grayU8 = gray,
            borderKillPx = borderKillPx,
            topKillY = topKillY,
            maskFull = maskFull,
            debugTag = debugTag,
            statsOut = statsOut
        )
    }

    /**
     * FULL-FRAME edge map desde gray U8 (w*h).
     * (Útil si tu stage EDGE ya construye gray por OpenCV u otro camino).
     */
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

        // Blur 3x3 para flattening (barato y suficiente para anti-halo en máscara)
        val blurU8 = boxBlur3x3U8(grayU8, w, h)

        // Score y dirección (para NMS)
        val score = ShortArray(N)     // score (0..~8192)
        val dir = ByteArray(N)        // 0/1/2/3 => 0°,45°,90°,135°
        var maxScore = 0
        var validCount = 0

        val hasMask = (maskFull != null)

        fun pix(idx: Int): Int {
            // Flattening: si masked => usar blur; si no => gray
            return if (hasMask && (maskFull[idx].toInt() and 0xFF) != 0) u8(blurU8[idx]) else u8(
                grayU8[idx]
            )
        }

        // Pass 1: Scharr + score + dir + max
        for (y in 1 until (h - 1)) {
            if (y < yKill) continue
            if (y < B || y >= h - B) continue
            val off = y * w
            val offUp = (y - 1) * w
            val offDn = (y + 1) * w

            for (x in 1 until (w - 1)) {
                if (x < B || x >= w - B) continue
                val i = off + x
                if (hasMask && (maskFull[i].toInt() and 0xFF) != 0) continue

                val p00 = pix(offUp + (x - 1))
                val p01 = pix(offUp + x)
                val p02 = pix(offUp + (x + 1))
                val p10 = pix(off + (x - 1))
                val p12 = pix(off + (x + 1))
                val p20 = pix(offDn + (x - 1))
                val p21 = pix(offDn + x)
                val p22 = pix(offDn + (x + 1))

                // Scharr 3x3 SIGNADO
                val gx = 3 * (p02 - p00) + 10 * (p12 - p10) + 3 * (p22 - p20)
                val gy = 3 * (p20 - p00) + 10 * (p21 - p01) + 3 * (p22 - p02)

                val ax = abs(gx)
                val ay = abs(gy)

                // Score tipo “tus PNG”: hs = ay - K_H*ax; vs = ax - K_V*ay
                val hs100 = ay * 100 - (K_H * 100f).toInt() * ax
                val vs100 = ax * 100 - (K_V * 100f).toInt() * ay
                val s100 = max(0, max(hs100, vs100))
                val s = (s100 / 100).coerceIn(0, 32767)

                if (s == 0) continue

                score[i] = s.toShort()
                validCount++
                if (s > maxScore) maxScore = s

                // Dirección para NMS (cuantizada)
                val d = quantizeDir3250(gx, gy, ax, ay)
                dir[i] = d.toByte()
            }
        }

        if (validCount < 256 || maxScore <= 0) {
            Log.w(
                TAG,
                "EDGE_FULL[$debugTag] not enough signal valid=$validCount max=$maxScore yKill=$yKill B=$B"
            )
            return ByteArray(N)
        }

        // Pass 2: histogram -> p95 (robusto, no sesgado por orden)
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

        // Pass 3: NMS + clasificar weak/strong (sin guardar otro mapa de magnitudes)
        // classMap: 0=none, 1=weak, 2=strong
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

                // NMS
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

        // Pass 4: hysteresis (BFS) => output final 0/255
        val out = ByteArray(N)
        if (strongCount == 0 || candCount == 0) {
            Log.w(
                TAG,
                "EDGE_FULL[$debugTag] no strong/candidates (strong=$strongCount cand=$candCount) thrH=$thrHigh thrL=$thrLow"
            )
            return out
        }

        val q = IntArray(candCount) // cola máxima = candidatos (cada uno entra una vez)
        var qh = 0
        var qt = 0
        var nnz = 0

        // seed con todos los strong
        for (i in 0 until N) {
            if (classMap[i].toInt() == 2) {
                out[i] = 0xFF.toByte()
                q[qt++] = i
                nnz++
            }
        }

        // 8-neighborhood
        while (qh < qt) {
            val i = q[qh++]
            val x = i % w
            val y = i / w
            // vecinos
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
                    // promover weak -> strong (evita re-encolar)
                    classMap[j] = 2
                    out[j] = 0xFF.toByte()
                    q[qt++] = j
                    nnz++
                }
            }
        }

        val dens = nnz.toFloat() / N.toFloat().coerceAtLeast(1f)
        Log.d(
            TAG,
            "EDGE_FULL[$debugTag] max=$maxScore p95=$p95 thrH=$thrHigh thrL=$thrLow " +
                    "strong=$strongCount cand=$candCount nnz=$nnz/$N dens=${"%.4f".format(dens)} yKill=$yKill B=$B"
        )

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
    // ✅ ROI/ANNULUS (ArcFit): gx/gy SIGNADOS (CV_32F) -> edgeMap binario 0/255
    // (mejorado: hist p95 + NMS + hysteresis)
    // ============================================================

    /**
     * EdgeMap binario (0/255) para ARC-FIT en ROI.
     *
     * Inputs:
     *  - gxArr/gyArr: Scharr con SIGNO (CV_32F) del ROI, tamaño w*h
     *  - bestL/R/T/B: bbox interno aproximado (ROI local) para el annulus elíptico
     *  - tPxEst: espesor estimado del aro (px) => define bandPx
     *  - borderKillPx: kill duro alrededor del ROI
     *  - topKillY: kill duro anti-ceja (ROI local). Todo y < topKillY muere (sólido).
     *  - maskRoi: ByteArray w*h, !=0 mata (ojos/ceja/zonas prohibidas) (sólido).
     */
    fun buildArcFitEdgeMapFromScharrSigned3250(
        mode: EdgeMode3250,
        w: Int,
        h: Int,
        gxArr: FloatArray,
        gyArr: FloatArray,
        bestLeft: Int,
        bestRight: Int,
        bestTop: Int,
        bestBottom: Int,
        tPxEst: Float,
        borderKillPx: Int,
        topKillY: Int = 0,
        maskRoi: ByteArray? = null,
        debugTag: String = "",
        statsOut: ((EdgeStats) -> Unit)? = null
    ): ByteArray {

        val N = w * h
        if (w <= 0 || h <= 0) return ByteArray(0)
        if (gxArr.size != N || gyArr.size != N) {
            Log.w(TAG, "EDGE_ROI[$debugTag|$mode] bad sizes gx=${gxArr.size} gy=${gyArr.size} N=$N")
            return ByteArray(0)
        }
        if (maskRoi != null && maskRoi.size != N) {
            Log.w(TAG, "EDGE_ROI[$debugTag|$mode] bad mask size=${maskRoi.size} N=$N")
            return ByteArray(0)
        }

        val B = borderKillPx.coerceIn(0, min(w, h) / 3)
        val yKill = topKillY.coerceIn(0, h)

        // Clamp box para no romper annulus
        val l0 = bestLeft.coerceIn(0, w - 2)
        val r0 = bestRight.coerceIn(l0 + 1, w - 1)
        val t0 = bestTop.coerceIn(0, h - 2)
        val b0 = bestBottom.coerceIn(t0 + 1, h - 1)

        // Elipse esperada (desde best box)
        val cx = 0.5f * (l0 + r0).toFloat()
        val cy = 0.5f * (t0 + b0).toFloat()
        val rx = 0.5f * (r0 - l0).toFloat().coerceAtLeast(1f)
        val ry = 0.5f * (b0 - t0).toFloat().coerceAtLeast(1f)
        val rMin = min(rx, ry)

        val bandPx = max(BAND_MIN_PX, BAND_K_T * tPxEst).coerceAtMost(BAND_MAX_PX)
        val invRx2 = 1f / (rx * rx)
        val invRy2 = 1f / (ry * ry)

        fun insideAnnulus(x: Int, y: Int): Boolean {
            val dx = (x - cx) / rx
            val dy = (y - cy) / ry
            val rr = sqrt(dx * dx + dy * dy)
            val distPx = abs(rr - 1f) * rMin
            return distPx <= bandPx
        }

        fun passesDirectional(i: Int, x: Int, y: Int): Boolean {
            if (mode != EdgeMode3250.DIVINE) return true

            val gx = gxArr[i]
            val gy = gyArr[i]
            val gmag2 = gx * gx + gy * gy
            if (gmag2 < 1e-6f) return false
            val gmag = sqrt(gmag2)

            // normal esperada de elipse: n ~ (dx/rx^2, dy/ry^2)
            var nx = (x - cx) * invRx2
            var ny = (y - cy) * invRy2
            val nmag2 = nx * nx + ny * ny
            if (nmag2 < 1e-10f) return false
            val nmag = sqrt(nmag2)
            nx /= nmag
            ny /= nmag

            val dot = abs((gx / gmag) * nx + (gy / gmag) * ny)
            return dot >= DIR_COS_TOL
        }

        // Pass A: score + dir (para NMS) dentro de annulus y kills sólidos
        val score = ShortArray(N)
        val dir = ByteArray(N)

        var maxScore = 0
        var validCount = 0

        for (y in B until (h - B)) {
            if (y < yKill) continue
            val off = y * w
            for (x in B until (w - B)) {
                val i = off + x
                if (maskRoi != null && (maskRoi[i].toInt() and 0xFF) != 0) continue
                if (!insideAnnulus(x, y)) continue
                if (!passesDirectional(i, x, y)) continue

                val gx = gxArr[i]
                val gy = gyArr[i]
                val ax = abs(gx)
                val ay = abs(gy)

                val hs = max(0f, ay - K_H * ax)
                val vs = max(0f, ax - K_V * ay)
                val s = max(hs, vs)

                val si = s.toInt().coerceIn(0, 32767)
                if (si == 0) continue

                score[i] = si.toShort()
                validCount++
                if (si > maxScore) maxScore = si

                // dir para NMS (cuantizada)
                val gxi = gx.toInt()
                val gyi = gy.toInt()
                val axi = abs(gxi)
                val ayi = abs(gyi)
                dir[i] = quantizeDir3250(gxi, gyi, axi, ayi).toByte()
            }
        }

        if (validCount < 128 || maxScore <= 0) {
            Log.w(
                TAG,
                "EDGE_ROI[$debugTag|$mode] not enough signal valid=$validCount max=$maxScore yKill=$yKill"
            )
            return ByteArray(N)
        }

        // Pass B: hist -> p95
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

        // Pass C: NMS + classificar
        val classMap = ByteArray(N) // 0 none, 1 weak, 2 strong
        var candCount = 0
        var strongCount = 0

        for (y in B until (h - B)) {
            if (y < yKill) continue
            val off = y * w
            for (x in B until (w - B)) {
                val i = off + x
                if (score[i].toInt() < thrLow) continue
                if (maskRoi != null && (maskRoi[i].toInt() and 0xFF) != 0) continue
                if (!insideAnnulus(x, y)) continue
                if (!passesDirectional(i, x, y)) continue

                val s = score[i].toInt()
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

        // Pass D: hysteresis BFS
        val out = ByteArray(N)
        if (strongCount == 0 || candCount == 0) {
            Log.w(
                TAG,
                "EDGE_ROI[$debugTag|$mode] no strong/candidates (strong=$strongCount cand=$candCount) thrH=$thrHigh thrL=$thrLow"
            )
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
                if (xx <= B || xx >= w - B) continue
                if (yy <= B || yy >= h - B) continue
                if (yy < yKill) continue
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
        Log.d(
            TAG,
            "EDGE_ROI[$debugTag|$mode] max=$maxScore p95=$p95 thrH=$thrHigh thrL=$thrLow " +
                    "strong=$strongCount cand=$candCount nnz=$nnz/$N dens=${"%.4f".format(dens)} bandPx=${
                        "%.1f".format(
                            bandPx
                        )
                    } " +
                    "yKill=$yKill B=$B cosTol=${"%.2f".format(DIR_COS_TOL)}"
        )


        statsOut?.invoke(
            EdgeStats(
                maxScore = maxScore.toFloat(),
                p95Score = p95.toFloat(),
                thr = thrHigh.toFloat(),
                nnz = nnz,
                density = dens,
                bandPx = bandPx,
                samples = validCount,
                topKillY = yKill
            )
        )

        return out
    }

    // ============================================================
    // Internals: blur, histogram->p95, thresholds, NMS helpers
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

        // init rolling rows
        buildHsumRow(0, curr)
        buildHsumRow(0, prev)
        buildHsumRow(if (h > 1) 1 else 0, next)

        for (y in 0 until h) {
            val base = y * w
            for (x in 0 until w) {
                val sum = prev[x] + curr[x] + next[x]
                out[base + x] = (sum / 9).coerceIn(0, 255).toByte()
            }
            // roll
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
            if (acc > target) {
                // bin -> score approx
                return (b * maxScore) / (bins - 1).coerceAtLeast(1)
            }
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
     * 0 => comparar L/R
     * 1 => comparar diag (x-1,y-1) y (x+1,y+1)
     * 2 => comparar U/D
     * 3 => comparar diag (x+1,y-1) y (x-1,y+1)
     */
    private fun quantizeDir3250(gx: Int, gy: Int, ax: Int, ay: Int): Int {
        if (ax == 0 && ay == 0) return 0
        // ay <= 0.414*ax  => 0°
        if (ay * TAN22_DEN <= ax * TAN22_NUM) return 0
        // ay >= 2.414*ax  => 90°
        if (ay * TAN67_DEN >= ax * TAN67_NUM) return 2
        // diagonales
        return if ((gx xor gy) >= 0) 1 else 3
    }

    private fun nmsNeighbors3250(i: Int, x: Int, y: Int, w: Int, dir: Int): Pair<Int, Int> {
        return when (dir) {
            0 -> (i - 1) to (i + 1)                 // left/right
            2 -> (i - w) to (i + w)                 // up/down
            1 -> (i - w - 1) to (i + w + 1)         // diag \
            else -> (i - w + 1) to (i + w - 1)      // diag /
        }
    }

}
