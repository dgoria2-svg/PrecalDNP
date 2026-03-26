package com.dg.precaldnp.vision

import android.graphics.Bitmap
import androidx.core.graphics.createBitmap
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

object RoiAppearanceNormalizer3250 {

    enum class FrameProfileId3250 {
        DARK,
        METAL,
        CLEAR,
        PATTERNED
    }

    data class Config3250(
        val targetLuma: Int = 96,
        val weightColor: Float = 0.00f,     // compat
        val weightGray: Float = 0.00f,      // compat
        val weightAnnulus: Float = 1.00f,   // compat
        val highlightCompressP90Factor: Float = 0.55f,
        val highlightCompressP98Factor: Float = 0.22f,
        val probThresholdLow: Float = 0.18f,    // compat
        val probThresholdHigh: Float = 0.78f,   // compat
        val samplePercentileLow: Float = 0.20f, // compat
        val samplePercentileHigh: Float = 0.85f,// compat
        val canonDetailRetainFactor: Float = 1.00f, // compat
        val contrastGainWhenProbable: Float = 1.04f,
        val lowerBlendStartFrac: Float = 0.42f,
        val lowerBlendEndFrac: Float = 0.66f,

        // banda geométrica suave
        val annulusInnerPx: Float = 0.60f,
        val annulusOuterPx: Float = 2.40f,

        // control de textura / continuidad
        val detailGainInsideAnnulus: Float = 1.18f,
        val detailGainOutsideAnnulus: Float = 0.88f,
        val textureSuppressInsideAnnulus: Float = 0.18f,
        val textureSuppressOutsideAnnulus: Float = 0.34f,
        val coherenceMin: Float = 0.28f,
        val edgeSupportMin: Float = 0.16f
    )

    object FrameProfiles3250 {

        val DARK = Config3250(
            targetLuma = 92,
            highlightCompressP90Factor = 0.55f,
            highlightCompressP98Factor = 0.22f,
            contrastGainWhenProbable = 1.05f,
            annulusInnerPx = 0.75f,
            annulusOuterPx = 2.50f,
            detailGainInsideAnnulus = 1.18f,
            detailGainOutsideAnnulus = 0.90f,
            textureSuppressInsideAnnulus = 0.18f,
            textureSuppressOutsideAnnulus = 0.30f,
            coherenceMin = 0.26f,
            edgeSupportMin = 0.16f
        )

        val METAL = Config3250(
            targetLuma = 108,
            highlightCompressP90Factor = 0.45f,
            highlightCompressP98Factor = 0.16f,
            contrastGainWhenProbable = 1.03f,
            annulusInnerPx = 0.50f,
            annulusOuterPx = 2.25f,
            detailGainInsideAnnulus = 1.12f,
            detailGainOutsideAnnulus = 0.92f,
            textureSuppressInsideAnnulus = 0.16f,
            textureSuppressOutsideAnnulus = 0.26f,
            coherenceMin = 0.24f,
            edgeSupportMin = 0.14f
        )

        val CLEAR = Config3250(
            targetLuma = 116,
            highlightCompressP90Factor = 0.36f,
            highlightCompressP98Factor = 0.12f,
            contrastGainWhenProbable = 1.01f,
            annulusInnerPx = 0.50f,
            annulusOuterPx = 2.00f,
            detailGainInsideAnnulus = 1.18f,
            detailGainOutsideAnnulus = 0.68f,
            textureSuppressInsideAnnulus = 0.18f,
            textureSuppressOutsideAnnulus = 0.52f,
            coherenceMin = 0.30f,
            edgeSupportMin = 0.18f
        )

        val PATTERNED = Config3250(
            targetLuma = 100,
            highlightCompressP90Factor = 0.50f,
            highlightCompressP98Factor = 0.20f,
            contrastGainWhenProbable = 1.04f,
            annulusInnerPx = 0.75f,
            annulusOuterPx = 2.50f,
            detailGainInsideAnnulus = 1.16f,
            detailGainOutsideAnnulus = 0.88f,
            textureSuppressInsideAnnulus = 0.18f,
            textureSuppressOutsideAnnulus = 0.32f,
            coherenceMin = 0.26f,
            edgeSupportMin = 0.15f
        )

        fun get3250(id: FrameProfileId3250): Config3250 = when (id) {
            FrameProfileId3250.DARK -> DARK
            FrameProfileId3250.METAL -> METAL
            FrameProfileId3250.CLEAR -> CLEAR
            FrameProfileId3250.PATTERNED -> PATTERNED
        }
    }

    data class Stats3250(
        val sampleN: Int,
        val meanR: Int,
        val meanG: Int,
        val meanB: Int,
        val meanGray: Int,
        val p10Lower: Int,
        val p90Lower: Int,
        val p98Lower: Int,
        val hiFracLower: Float,
        val targetLuma: Int,
        val normCoverage: Float,
        val blurRadius: Int,
        val profileId3250: String,
        val sampleColorMad: Float,
        val annulusInnerPx: Float,
        val annulusOuterPx: Float,
        val meanGrad3250: Float,
        val meanCoherence3250: Float,
        val meanPreserve3250: Float,
        val annulusCoverage3250: Float,
        val textureSuppression3250: Float
    )

    data class Out3250(
        val bitmap: Bitmap,
        val stats: Stats3250
    )

    /**
     * Versión corregida:
     * - NO usa AUTO
     * - NO centra el FIL en el centro del ROI por defecto
     * - USA HBOX/VBOX INNER y centro esperado del detector
     * - Aplana iluminación
     * - Suprime textura
     * - Preserva detalle coherente cerca del annulus
     * - NO inventa continuidad
     */
    fun normalizeForRim3250(
        roiBmp: Bitmap,
        filHboxInnerMm: Double,
        filVboxInnerMm: Double,
        pxPerMmGuess: Double,
        expectedCxLocal: Float,
        expectedCyLocal: Float,
        eyeMaskU8: ByteArray? = null,
        profileId3250: FrameProfileId3250 = FrameProfileId3250.CLEAR
    ): Out3250 {

        val w = roiBmp.width
        val h = roiBmp.height
        if (w <= 8 || h <= 8) {
            val cfg = FrameProfiles3250.get3250(profileId3250)
            return Out3250(
                bitmap = roiBmp,
                stats = Stats3250(
                    sampleN = 0,
                    meanR = 0,
                    meanG = 0,
                    meanB = 0,
                    meanGray = 0,
                    p10Lower = 0,
                    p90Lower = 0,
                    p98Lower = 0,
                    hiFracLower = 0f,
                    targetLuma = cfg.targetLuma,
                    normCoverage = 0f,
                    blurRadius = 0,
                    profileId3250 = profileId3250.name,
                    sampleColorMad = 0f,
                    annulusInnerPx = cfg.annulusInnerPx,
                    annulusOuterPx = cfg.annulusOuterPx,
                    meanGrad3250 = 0f,
                    meanCoherence3250 = 0f,
                    meanPreserve3250 = 0f,
                    annulusCoverage3250 = 0f,
                    textureSuppression3250 = 0f
                )
            )
        }

        val n = w * h
        val argb = IntArray(n)
        roiBmp.getPixels(argb, 0, w, 0, 0, w, h)

        val rArr = IntArray(n)
        val gArr = IntArray(n)
        val bArr = IntArray(n)
        val gray = IntArray(n)

        for (i in 0 until n) {
            val c = argb[i]
            val r = (c ushr 16) and 0xFF
            val g = (c ushr 8) and 0xFF
            val b = c and 0xFF
            rArr[i] = r
            gArr[i] = g
            bArr[i] = b
            gray[i] = ((77 * r + 150 * g + 29 * b + 128) ushr 8).coerceIn(0, 255)
        }

        fun isMasked(i: Int): Boolean =
            eyeMaskU8 != null &&
                    i in eyeMaskU8.indices &&
                    ((eyeMaskU8[i].toInt() and 0xFF) != 0)

        val cfg = FrameProfiles3250.get3250(profileId3250)
        val targetLuma = cfg.targetLuma
        val annulusInnerPx = cfg.annulusInnerPx.coerceIn(0f, 2f)
        val annulusOuterPx = cfg.annulusOuterPx.coerceIn(1f, 5f)

        // ------------------------------------------------------------
        // 1) Flatten suave de iluminación
        // ------------------------------------------------------------
        val blurRadius = (min(w, h) / 14).coerceIn(10, 32)
        val illum = boxBlurGray3250(gray, w, h, blurRadius)

        val flatGray = IntArray(n)
        for (i in 0 until n) {
            flatGray[i] = (gray[i] - illum[i] + 128).coerceIn(0, 255)
        }

        // ------------------------------------------------------------
        // 2) Geometría esperada del aro (CORREGIDA)
        //    - usa INNER
        //    - usa centro esperado del detector
        // ------------------------------------------------------------
        val expW = (filHboxInnerMm * pxPerMmGuess).toFloat().coerceAtLeast(40f)
        val expH = (filVboxInnerMm * pxPerMmGuess).toFloat().coerceAtLeast(32f)

        val cx = expectedCxLocal.coerceIn(0f, (w - 1).toFloat())
        val cy = expectedCyLocal.coerceIn(0f, (h - 1).toFloat())

        val a = (expW * 0.50f).coerceAtLeast(28f)
        val b = (expH * 0.50f).coerceAtLeast(24f)

        fun signedAnnulusDistPx3250(x: Int, y: Int): Float {
            val dx = (x - cx) / a
            val dy = (y - cy) / b
            val rho = sqrt(dx * dx + dy * dy)
            val meanRadiusPx = 0.5f * (a + b)
            return (rho - 1f) * meanRadiusPx
        }

        // ------------------------------------------------------------
        // 3) Stats globales suaves
        // ------------------------------------------------------------
        var sumR = 0L
        var sumG = 0L
        var sumB = 0L
        var sumGray = 0L
        var count = 0

        for (i in 0 until n) {
            if (isMasked(i)) continue
            sumR += rArr[i].toLong()
            sumG += gArr[i].toLong()
            sumB += bArr[i].toLong()
            sumGray += flatGray[i].toLong()
            count++
        }

        val meanR = if (count > 0) (sumR / count).toInt().coerceIn(0, 255) else 0
        val meanG = if (count > 0) (sumG / count).toInt().coerceIn(0, 255) else 0
        val meanB = if (count > 0) (sumB / count).toInt().coerceIn(0, 255) else 0
        val meanGray = if (count > 0) (sumGray / count).toInt().coerceIn(0, 255) else 0

        var sampleColorMad = 0f
        if (count > 0) {
            var madAcc = 0f
            for (i in 0 until n) {
                if (isMasked(i)) continue
                val dr = abs(rArr[i] - meanR)
                val dg = abs(gArr[i] - meanG)
                val db = abs(bArr[i] - meanB)
                madAcc += (dr + dg + db) / 3f
            }
            sampleColorMad = madAcc / count.toFloat().coerceAtLeast(1f)
        }

        // ------------------------------------------------------------
        // 4) Stats de highlights en mitad inferior
        // ------------------------------------------------------------
        val lowerVals = ArrayList<Int>(w * max(1, h / 2))
        val yLower0 = (h * 0.52f).toInt().coerceIn(0, h - 1)

        for (y in yLower0 until h) {
            val off = y * w
            for (x in 0 until w) {
                val i = off + x
                if (isMasked(i)) continue
                lowerVals.add(flatGray[i])
            }
        }

        val p10Lower = percentile3250(lowerVals, 0.10f)
        val p90Lower = percentile3250(lowerVals, 0.90f)
        val p98Lower = percentile3250(lowerVals, 0.98f)

        var hiCount = 0
        for (v in lowerVals) if (v >= p98Lower) hiCount++
        val hiFracLower = hiCount.toFloat() / lowerVals.size.toFloat().coerceAtLeast(1f)

        // ------------------------------------------------------------
        // 5) Compresión de highlights global suave
        // ------------------------------------------------------------
        val compGray = IntArray(n)
        for (i in 0 until n) {
            compGray[i] = compressHighlights3250(flatGray[i], p90Lower, p98Lower, cfg)
        }

        // ------------------------------------------------------------
        // 6) Gradientes + coherencia local + soporte de borde
        // ------------------------------------------------------------
        val gx = sobelGradX3250(compGray, w, h)
        val gy = sobelGradY3250(compGray, w, h)

        val gradMag = FloatArray(n)
        var maxGrad = 1f
        for (i in 0 until n) {
            val mag = sqrt(gx[i] * gx[i] + gy[i] * gy[i])
            gradMag[i] = mag
            if (mag > maxGrad) maxGrad = mag
        }

        val coherence = localCoherence3250(gx, gy, w, h, radius = 2)
        val edgeSupport = localEdgeSupport3250(
            gradMag = gradMag,
            w = w,
            h = h,
            thrFrac = 0.24f,
            maxGrad = maxGrad
        )

        // ------------------------------------------------------------
        // 7) Build preserve score
        //    preserveScore alto => conservar detalle
        //    preserveScore bajo => suprimir textura
        // ------------------------------------------------------------
        val preserveScore = FloatArray(n)
        var annulusCount = 0
        var preserveAcc = 0f
        var coherenceAcc = 0f
        var gradAcc = 0f
        var textureSuppressionAcc = 0f

        for (y in 0 until h) {
            val off = y * w
            for (x in 0 until w) {
                val i = off + x
                if (isMasked(i)) {
                    preserveScore[i] = 0f
                    continue
                }

                val dSignedPx = signedAnnulusDistPx3250(x, y)
                val annScore = when {
                    dSignedPx >= 0f ->
                        (1f - dSignedPx / annulusOuterPx).coerceIn(0f, 1f)
                    else ->
                        (1f - abs(dSignedPx) / annulusInnerPx.coerceAtLeast(1e-3f)).coerceIn(0f, 1f)
                }

                val inAnnulusBand = dSignedPx >= -annulusInnerPx && dSignedPx <= annulusOuterPx
                if (inAnnulusBand) annulusCount++

                val gradN = (gradMag[i] / maxGrad).coerceIn(0f, 1f)
                val coh = coherence[i].coerceIn(0f, 1f)
                val sup = edgeSupport[i].coerceIn(0f, 1f)

                val textureNoise = (gradN * (1f - coh) * (1f - sup)).coerceIn(0f, 1f)

                val preserve = (
                        0.20f * gradN +
                                0.36f * coh +
                                0.28f * sup +
                                0.16f * annScore
                        ).coerceIn(0f, 1f)

                preserveScore[i] = smoothstep3250(
                    cfg.coherenceMin,
                    1.00f - cfg.edgeSupportMin * 0.25f,
                    preserve
                ).coerceIn(0f, 1f)

                preserveAcc += preserveScore[i]
                coherenceAcc += coh
                gradAcc += gradN
                textureSuppressionAcc += textureNoise
            }
        }

        val annulusCoverage = annulusCount.toFloat() / n.toFloat().coerceAtLeast(1f)
        val meanPreserve = preserveAcc / n.toFloat().coerceAtLeast(1f)
        val meanCoherence = coherenceAcc / n.toFloat().coerceAtLeast(1f)
        val meanGrad = gradAcc / n.toFloat().coerceAtLeast(1f)
        val textureSuppression = textureSuppressionAcc / n.toFloat().coerceAtLeast(1f)

        // ------------------------------------------------------------
        // 8) Salida final:
        //    base plana + detalle real * preserveScore
        //    sin inventar continuidad
        // ------------------------------------------------------------
        val outGray = IntArray(n)

        for (y in 0 until h) {
            val off = y * w

            val yW = when {
                y < h * cfg.lowerBlendStartFrac -> 0f
                y > h * cfg.lowerBlendEndFrac -> 1f
                else -> (
                        (y - h * cfg.lowerBlendStartFrac) /
                                (h * (cfg.lowerBlendEndFrac - cfg.lowerBlendStartFrac))
                        ).coerceIn(0f, 1f)
            }

            for (x in 0 until w) {
                val i = off + x

                if (isMasked(i)) {
                    outGray[i] = compGray[i]
                    continue
                }

                val dSignedPx = signedAnnulusDistPx3250(x, y)
                val inAnnulusBand = dSignedPx >= -annulusInnerPx && dSignedPx <= annulusOuterPx

                val p = preserveScore[i]
                val detail = compGray[i] - 128

                val gainDetail = if (inAnnulusBand) {
                    cfg.detailGainInsideAnnulus
                } else {
                    cfg.detailGainOutsideAnnulus
                }

                val suppressTexture = if (inAnnulusBand) {
                    cfg.textureSuppressInsideAnnulus
                } else {
                    cfg.textureSuppressOutsideAnnulus
                }

                val textureKill = suppressTexture * (1f - p)
                val lowerBias = 0.03f * yW

                val detailMix = (gainDetail * p - textureKill + lowerBias).coerceIn(0f, 1.10f)

                val v = (128f + detail * detailMix).toInt().coerceIn(0, 255)
                outGray[i] = applySoftContrast3250(
                    v,
                    if (p > 0.22f && inAnnulusBand) cfg.contrastGainWhenProbable else 1.00f
                )
            }
        }

        val outBmp = createBitmap(w, h)
        val outArgb = IntArray(n)
        for (i in 0 until n) {
            val v = outGray[i]
            outArgb[i] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
        }
        outBmp.setPixels(outArgb, 0, w, 0, 0, w, h)

        return Out3250(
            bitmap = outBmp,
            stats = Stats3250(
                sampleN = count,
                meanR = meanR,
                meanG = meanG,
                meanB = meanB,
                meanGray = meanGray,
                p10Lower = p10Lower,
                p90Lower = p90Lower,
                p98Lower = p98Lower,
                hiFracLower = hiFracLower,
                targetLuma = targetLuma,
                normCoverage = meanPreserve,
                blurRadius = blurRadius,
                profileId3250 = profileId3250.name,
                sampleColorMad = sampleColorMad,
                annulusInnerPx = annulusInnerPx,
                annulusOuterPx = annulusOuterPx,
                meanGrad3250 = meanGrad,
                meanCoherence3250 = meanCoherence,
                meanPreserve3250 = meanPreserve,
                annulusCoverage3250 = annulusCoverage,
                textureSuppression3250 = textureSuppression
            )
        )
    }

    // ============================================================
    // Internals
    // ============================================================

    private fun compressHighlights3250(
        v: Int,
        p90: Int,
        p98: Int,
        cfg: Config3250
    ): Int {
        if (v <= p90 || p98 <= p90) return v
        return if (v <= p98) {
            val dv = v - p90
            (p90 + cfg.highlightCompressP90Factor * dv).toInt().coerceIn(0, 255)
        } else {
            val mid = p90 + cfg.highlightCompressP90Factor * (p98 - p90)
            val dv = v - p98
            (mid + cfg.highlightCompressP98Factor * dv).toInt().coerceIn(0, 255)
        }
    }

    private fun applySoftContrast3250(v: Int, gain: Float): Int {
        return (128f + gain * (v - 128f)).toInt().coerceIn(0, 255)
    }

    private fun percentile3250(values: List<Int>, q: Float): Int {
        if (values.isEmpty()) return 0
        val arr = values.toIntArray()
        arr.sort()
        val idx = (q.coerceIn(0f, 1f) * (arr.size - 1)).toInt().coerceIn(0, arr.size - 1)
        return arr[idx]
    }

    private fun smoothstep3250(edge0: Float, edge1: Float, x: Float): Float {
        if (edge1 <= edge0) return if (x >= edge1) 1f else 0f
        val t = ((x - edge0) / (edge1 - edge0)).coerceIn(0f, 1f)
        return t * t * (3f - 2f * t)
    }

    private fun sobelGradX3250(src: IntArray, w: Int, h: Int): FloatArray {
        val out = FloatArray(w * h)
        if (w < 3 || h < 3) return out
        for (y in 1 until h - 1) {
            val ym = y - 1
            val yp = y + 1
            for (x in 1 until w - 1) {
                val xm = x - 1
                val xp = x + 1

                val g =
                    -1f * src[ym * w + xm] + 1f * src[ym * w + xp] +
                            -2f * src[y * w + xm] + 2f * src[y * w + xp] +
                            -1f * src[yp * w + xm] + 1f * src[yp * w + xp]

                out[y * w + x] = g
            }
        }
        return out
    }

    private fun sobelGradY3250(src: IntArray, w: Int, h: Int): FloatArray {
        val out = FloatArray(w * h)
        if (w < 3 || h < 3) return out
        for (y in 1 until h - 1) {
            val ym = y - 1
            val yp = y + 1
            for (x in 1 until w - 1) {
                val xm = x - 1
                val xp = x + 1

                val g =
                    -1f * src[ym * w + xm] + -2f * src[ym * w + x] + -1f * src[ym * w + xp] +
                            1f * src[yp * w + xm] + 2f * src[yp * w + x] + 1f * src[yp * w + xp]

                out[y * w + x] = g
            }
        }
        return out
    }

    /**
     * coherencia local de dirección de gradiente:
     * 0 = ruido / textura
     * 1 = estructura direccional consistente
     */
    private fun localCoherence3250(
        gx: FloatArray,
        gy: FloatArray,
        w: Int,
        h: Int,
        radius: Int = 2
    ): FloatArray {
        val out = FloatArray(w * h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                var sxx = 0f
                var syy = 0f
                var sxy = 0f
                var n = 0

                val y0 = max(0, y - radius)
                val y1 = min(h - 1, y + radius)
                val x0 = max(0, x - radius)
                val x1 = min(w - 1, x + radius)

                for (yy in y0..y1) {
                    val off = yy * w
                    for (xx in x0..x1) {
                        val i = off + xx
                        val dx = gx[i]
                        val dy = gy[i]
                        sxx += dx * dx
                        syy += dy * dy
                        sxy += dx * dy
                        n++
                    }
                }

                if (n <= 0) {
                    out[y * w + x] = 0f
                    continue
                }

                val trace = sxx + syy
                val detTerm = sqrt(max(0f, (sxx - syy) * (sxx - syy) + 4f * sxy * sxy))
                val coh = if (trace > 1e-3f) (detTerm / trace).coerceIn(0f, 1f) else 0f
                out[y * w + x] = coh
            }
        }
        return out
    }

    /**
     * soporte local de borde:
     * cuenta vecindad con gradiente por encima de umbral relativo
     */
    private fun localEdgeSupport3250(
        gradMag: FloatArray,
        w: Int,
        h: Int,
        thrFrac: Float,
        maxGrad: Float
    ): FloatArray {
        val out = FloatArray(w * h)
        val thr = (thrFrac * maxGrad).coerceAtLeast(1f)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val y0 = max(0, y - 1)
                val y1 = min(h - 1, y + 1)
                val x0 = max(0, x - 1)
                val x1 = min(w - 1, x + 1)

                var hits = 0
                var cnt = 0
                for (yy in y0..y1) {
                    val off = yy * w
                    for (xx in x0..x1) {
                        cnt++
                        if (gradMag[off + xx] >= thr) hits++
                    }
                }
                out[y * w + x] = (hits.toFloat() / cnt.toFloat().coerceAtLeast(1f)).coerceIn(0f, 1f)
            }
        }
        return out
    }

    /**
     * Box blur usando imagen integral.
     */
    private fun boxBlurGray3250(src: IntArray, w: Int, h: Int, radius: Int): IntArray {
        if (radius <= 0) return src.copyOf()

        val integral = LongArray((w + 1) * (h + 1))

        for (y in 1..h) {
            var rowSum = 0L
            val srcOff = (y - 1) * w
            val intOff = y * (w + 1)
            val intPrev = (y - 1) * (w + 1)
            for (x in 1..w) {
                rowSum += src[srcOff + (x - 1)].toLong()
                integral[intOff + x] = integral[intPrev + x] + rowSum
            }
        }

        val dst = IntArray(w * h)

        for (y in 0 until h) {
            val y0 = max(0, y - radius)
            val y1 = min(h - 1, y + radius)

            for (x in 0 until w) {
                val x0 = max(0, x - radius)
                val x1 = min(w - 1, x + radius)

                val xb = x1 + 1
                val yb = y1 + 1

                val sum =
                    integral[yb * (w + 1) + xb] -
                            integral[y0 * (w + 1) + xb] -
                            integral[yb * (w + 1) + x0] +
                            integral[y0 * (w + 1) + x0]

                val area = (x1 - x0 + 1) * (y1 - y0 + 1)
                dst[y * w + x] = (sum / area.toLong()).toInt().coerceIn(0, 255)
            }
        }

        return dst
    }
}