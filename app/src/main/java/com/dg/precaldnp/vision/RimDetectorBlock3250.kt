@file:Suppress("SameParameterValue", "UNNECESSARY_NOT_NULL_ASSERTION")

package com.dg.precaldnp.vision

import android.graphics.PointF
import android.graphics.RectF
import android.util.Log
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * RimDetectorBlock3250 (EDGE-IN) — CENTER→OUT (INNER FIRST)
 *
 * ✅ Consume edgeMap binario ROI-local (0/255)
 * ✅ dirU8 opcional (0..3 o 255=don’t care)
 * ✅ maskU8 = máscara de ojo/ceja: mask!=0 => VETO (NO RIM)
 *
 * Regla oficial 3250:
 * - Hay UNA sola referencia de búsqueda: la línea pupilar / bridgeRow => probeYLocal.
 * - INNER y BOTTOM se buscan siempre referidos a esa misma línea oficial.
 * - La máscara NO mueve la referencia geométrica.
 * - Si un pixel cae dentro de máscara => NO VALE; se ignora/salta y se sigue buscando.
 * - Bottom arc debe ser continuo (no saltos grandes).
 * - Bottom barre CENTER→OUT dentro del dominio leftXrightX.
 */
object RimDetectorBlock3250 {

    private const val TAG = "RimDetectorBlock3250"
    private const val DBG = true

    private const val MIN_ROI_W = 120
    private const val MIN_ROI_H = 90

    private const val BAND_HALF_PX = 14
    private const val TOP_SEARCH_PAD = 10

    private const val OK_CONF_MIN = 0.55f

    private const val SCALE_MIN = 0.60f
    private const val SCALE_MAX = 1.35f
    private const val SCALE_STEP = 0.05f

    // Si te sigue agarrando OUTER, bajá W_RATIO_MAX a ~1.12
    private const val W_RATIO_MIN = 0.90f
    private const val W_RATIO_MAX = 1.25f

    // Para evitar “saltos” grandes en el arco bottom
    private const val CONT_JUMP_PX = 7

    // Para evitar “hits” espurios cerca del centro (reflejos / ruido interior)
    private const val LR_MIN_DIST_PX = 10

    private fun d(msg: String) {
        if (DBG) Log.d(TAG, msg)
    }

    // ==========================================================
    // ÚNICA SEMÁNTICA DE MÁSCARA
    // 0 = permitido (negro). !=0 = veto
    // ==========================================================
    private fun isMasked(maskU8: ByteArray?, idx: Int): Boolean =
        maskU8 != null && ((maskU8[idx].toInt() and 0xFF) != 0)

    // Alias 3250 para mantener compatibilidad de nombres sin cambiar semántica
    private fun isMaskedPx3250(maskU8: ByteArray?, idx: Int): Boolean =
        isMasked(maskU8, idx)

    // ==========================================================
    // DIRECCIÓN (dirU8) — asumimos dirección de GRADIENTE (NMS)
    // bins típicos: 0=0°, 1=45°, 2=90°, 3=135°
    // - Pared VERTICAL => gradiente HORIZONTAL => bin 0 (y diagonales toleradas)
    // - Borde HORIZONTAL => gradiente VERTICAL => bin 2 (y diagonales toleradas)
    // ==========================================================
    private fun isVertEdgeDir(d: Int): Boolean {
        val dd = d and 0xFF
        if (dd == 255) return true
        if (dd !in 0..3) return true
        return (dd == 0 || dd == 1 || dd == 3) // vertical edge: grad ~0° (+ diag)
    }

    private fun isHorzEdgeDir(d: Int): Boolean {
        val dd = d and 0xFF
        if (dd == 255) return true
        if (dd !in 0..3) return true
        return (dd == 2 || dd == 1 || dd == 3) // horizontal edge: grad ~90° (+ diag)
    }

    private data class ArcPick(
        val yMed: Int,
        val poly: List<Pair<Int, Int>>,
        val coverage: Float,
        val continuity: Float
    )

    private data class InnerSeedRow3250(
        val y: Int,
        val leftX: Int,
        val rightX: Int
    )

    private data class SideArcCandidatesLocal3250(
        val innerLeft: List<PointF>,
        val innerRight: List<PointF>,
        val outerLeft: List<PointF>,
        val outerRight: List<PointF>
    )

    // ==========================================================
    // API
    // ==========================================================
    fun detectRim(
        edgesU8: ByteArray,
        dirU8: ByteArray? = null,
        maskU8: ByteArray? = null,
        w: Int,
        h: Int,
        roiGlobal: RectF,
        midlineXpx: Float,
        browBottomYpx: Float?,
        filHboxMm: Double,
        filVboxMm: Double?,
        filOverInnerMmPerSide3250: Double,
        profile3250: RimProfile3250,
        bridgeRowYpxGlobal: Float?,
        pupilGlobal: PointF?,
        debugTag: String,
        pxPerMmGuessFace: Float? = null
    ): RimDetectPack3250? {

        var stage = "ENTER"
        fun fail(code: String, msg: String): RimDetectPack3250? {
            Log.w(TAG, "RIM[$debugTag] FAIL@$stage code=$code :: $msg")
            return null
        }

        stage = "SANITY"

        val effectiveOverF = effectiveFilOverPerSide3250(
            profile3250 = profile3250,
            filOverInnerMmPerSide = filOverInnerMmPerSide3250
        ).toFloat()

        val hboxInnerMmF = (filHboxMm.toFloat() - 2f * effectiveOverF)
            .takeIf { it.isFinite() && it > 1e-6f }
            ?: return fail(
                "HBOX",
                "HBOX inner invalid fil=$filHboxMm over=$effectiveOverF profile=$profile3250"
            )

        val vboxInnerMmF = ((filVboxMm ?: 0.0).toFloat() - 2f * effectiveOverF)
            .takeIf { it.isFinite() && it > 1e-6f }
            ?: 0f

        if (w <= 0 || h <= 0) return fail("WH", "w/h invalid $w x $h")
        if (w < MIN_ROI_W || h < MIN_ROI_H) return fail("ROI_SMALL", "ROI too small ${w}x${h}")
        if (edgesU8.size != w * h) return fail("EDGE_SIZE", "edgesU8.size=${edgesU8.size} != w*h=${w * h}")
        if (dirU8 != null && dirU8.size != w * h) return fail("DIR_SIZE", "dirU8.size=${dirU8.size} != w*h=${w * h}")
        if (maskU8 != null && maskU8.size != w * h) return fail("MASK_SIZE", "maskU8.size=${maskU8.size} != w*h=${w * h}")
        if (!midlineXpx.isFinite()) return fail("MIDLINE", "midlineXpx invalid=$midlineXpx")

        d(
            "RIM[$debugTag] profile=$profile3250 " +
                    "filH=${f1(filHboxMm.toFloat())} filV=${f1((filVboxMm ?: Double.NaN).toFloat())} " +
                    "overIn=${f2(filOverInnerMmPerSide3250.toFloat())} overEff=${f2(effectiveOverF)} " +
                    "hboxInner=${f2(hboxInnerMmF)} vboxInner=${f2(vboxInnerMmF)}"
        )

        stage = "EDGE_DENSITY"
        var nnz = 0
        for (i in edgesU8.indices) {
            if ((edgesU8[i].toInt() and 0xFF) != 0) nnz++
        }
        val dens = nnz.toFloat() / (w * h).toFloat().coerceAtLeast(1f)
        d("EDGE[$debugTag] nnz=$nnz/${w * h} dens=${"%.4f".format(Locale.US, dens)}")
        if (nnz < 50) return fail("EDGE_SPARSE", "edgeMap too sparse (nnz<50)")

        val roiLeftG = roiGlobal.left
        val roiTopG = roiGlobal.top

        // ---- PROBE + ANTI-CEJA ----
        stage = "PROBE"
        val probeYLocalRaw = run {
            val y0 = (bridgeRowYpxGlobal ?: pupilGlobal?.y ?: roiGlobal.centerY())
            (y0 - roiTopG).roundToInt()
        }.coerceIn(0, h - 1)

        val browLocal = browBottomYpx?.let { (it - roiTopG).roundToInt().coerceIn(0, h - 1) }
        val topMinAllowedY0 = if (browLocal == null) {
            0
        } else {
            val pad = clampInt((h * 0.02f).roundToInt(), 6, 18)
            (browLocal + pad).coerceIn(0, h - 1)
        }

        val topMinAllowedY =
            min(topMinAllowedY0, (probeYLocalRaw - 14).coerceIn(0, h - 1)).coerceIn(0, h - 1)

        val probeYLocal = run {
            val lo = (topMinAllowedY + 2).coerceIn(0, h - 1)
            val hi = (h - 3).coerceIn(0, h - 1)
            clampInRange(probeYLocalRaw, lo, hi)
        }

        d(
            "RIM[$debugTag] probeRaw=$probeYLocalRaw probe=$probeYLocal " +
                    "topMin0=$topMinAllowedY0 topMin=$topMinAllowedY browLocal=${browLocal ?: -1}"
        )

        // ---- DBG: máscara (bbox + frac) ----
        if (DBG) {
            if (maskU8 != null && maskU8.size == w * h) {
                var minX = w
                var minY = h
                var maxX = -1
                var maxY = -1
                var n = 0
                for (i in 0 until w * h) {
                    if ((maskU8[i].toInt() and 0xFF) != 0) {
                        n++
                        val x = i % w
                        val y = i / w
                        if (x < minX) minX = x
                        if (y < minY) minY = y
                        if (x > maxX) maxX = x
                        if (y > maxY) maxY = y
                    }
                }
                val frac = n.toFloat() / (w.toFloat() * h.toFloat()).coerceAtLeast(1f)
                if (n > 0) {
                    Log.d(TAG, "MASK[$debugTag] n=$n frac=${f3(frac)} bbox=($minX,$minY)-($maxX,$maxY)")
                } else {
                    Log.d(TAG, "MASK[$debugTag] n=0 frac=${f3(frac)} bbox=none")
                }
            } else {
                Log.d(TAG, "MASK[$debugTag] noneOrBadSize size=${maskU8?.size ?: -1} expected=${w * h}")
            }
        }

        // ---- MIDLINE + SIDE ----
        stage = "MIDLINE"
        val midLocalX = (midlineXpx - roiLeftG)
        val sideSign = when {
            pupilGlobal != null -> if (pupilGlobal.x < midlineXpx) -1f else +1f   // OD=-1, OI=+1
            else -> if (roiGlobal.centerX() < midlineXpx) -1f else +1f
        }
        val nasalAtLeft = sideSign > 0f // OI => nasal a la izquierda (hacia midline)

// ---- pxGuessBase ----
        stage = "PX_GUESS"
        val pxGuessBase = pxPerMmGuessFace ?: run {
            val approx = (w.toFloat() / hboxInnerMmF) * 0.82f
            approx.coerceIn(3.5f, 8.0f)
        }
        if (!pxGuessBase.isFinite() || pxGuessBase <= 0f) {
            return fail("PX_GUESS", "pxGuessBase invalid=$pxGuessBase")
        }

// ==========================================================
// Search over scales
// ==========================================================
        stage = "SCALES"
        val scales = buildScales(SCALE_MIN, SCALE_MAX, SCALE_STEP)
        if (scales.isEmpty()) return fail("SCALES", "empty scales")

        var bestConf = -1f
        var bestLeft = -1
        var bestRight = -1
        var bestTop = -1
        var bestBottom = -1
        var bestRefY = -1
        var bestSeedX = -1
        var bestScale = 1.0f
        var bestBottomPoly: List<Pair<Int, Int>> = emptyList()

        var bestPartialConf = -1f
        var bestPartialLeft = -1
        var bestPartialRight = -1
        var bestPartialTop = -1
        var bestPartialBottom = -1
        var bestPartialRefY = -1
        var bestPartialSeedX = -1
        var bestPartialScale = Float.NaN
        var bestPartialBottomPoly: List<Pair<Int, Int>> = emptyList()
        var bestPartialBottomEstimated = false

        for (scale in scales) {
            val pxGuessThis = (pxGuessBase * scale).coerceIn(3.5f, 8.0f)
            val expWpx = (hboxInnerMmF * pxGuessThis).coerceAtLeast(80f)
            val gapPx = max(18f, 0.10f * expWpx)

            val seamXLocal = midLocalX.coerceIn(0f, (w - 1).toFloat())
            val seedExpected = seamXLocal + sideSign * (gapPx + 0.50f * expWpx)

            val seedXLocal = seedExpected
                .roundToInt()
                .coerceIn(0, w - 1)

            val expHpxGuess =
                if (vboxInnerMmF > 1e-3f) (vboxInnerMmF * pxGuessThis) else (0.72f * expWpx)

            val maxDist = (0.85f * expWpx).roundToInt().coerceAtLeast(60).coerceAtMost(w - 1)
            val distStart = max(LR_MIN_DIST_PX, (0.10f * expWpx).roundToInt())

            if (DBG) {
                Log.d(
                    TAG,
                    "RIMDBG[$debugTag] profile=$profile3250 S=${f2(scale)} " +
                            "seed=($seedXLocal,$probeYLocal) yRef=$probeYLocal " +
                            "seedExpX=${f1(seedExpected)} seamX=${f1(seamXLocal)} side=${f1(sideSign)} " +
                            "expW=${f1(expWpx)} gap=${f1(gapPx)} expHGuess=${f1(expHpxGuess)} pxG=${f2(pxGuessThis)} " +
                            "distStart=$distStart maxDist=$maxDist"
                )
            }

            val lr = findInnerWallsAtY3250(
                edgesU8 = edgesU8,
                dirU8 = dirU8,
                maskU8 = maskU8,
                w = w,
                h = h,
                seedX = seedXLocal,
                yRef = probeYLocal,
                bandHalf = BAND_HALF_PX,
                minDist = distStart,
                maxDist = maxDist
            )

            if (lr == null) {
                if (DBG) {
                    Log.d(
                        TAG,
                        "RIMDBG[$debugTag] S=${f2(scale)} LR_FAIL yRef=$probeYLocal " +
                                "seedX=$seedXLocal probe=$probeYLocal"
                    )
                }
                continue
            }

            val a = lr.first
            val b = lr.second
            val innerW = (b - a)

            if (DBG) {
                val y0h = (probeYLocal - BAND_HALF_PX).coerceIn(0, h - 1)
                val y1h = (probeYLocal + BAND_HALF_PX).coerceIn(0, h - 1)

                var hitsL = 0
                for (yy in y0h..y1h) {
                    val idx = yy * w + a
                    if ((edgesU8[idx].toInt() and 0xFF) != 0 && !isMaskedPx3250(maskU8, idx)) {
                        val ddir = if (dirU8 == null) 255 else (dirU8[idx].toInt() and 0xFF)
                        if (isVertEdgeDir(ddir)) hitsL++
                    }
                }
                var hitsR = 0
                for (yy in y0h..y1h) {
                    val idx = yy * w + b
                    if ((edgesU8[idx].toInt() and 0xFF) != 0 && !isMaskedPx3250(maskU8, idx)) {
                        val ddir = if (dirU8 == null) 255 else (dirU8[idx].toInt() and 0xFF)
                        if (isVertEdgeDir(ddir)) hitsR++
                    }
                }

                val ratioWdbg = innerW.toFloat() / expWpx
                Log.d(
                    TAG,
                    "RIMDBG[$debugTag] S=${f2(scale)} LR_OK a=$a b=$b innerW=$innerW " +
                            "ratioW=${f3(ratioWdbg)} yRef=$probeYLocal " +
                            "dL=${abs(a - seedXLocal)} hitsL=$hitsL dR=${abs(b - seedXLocal)} hitsR=$hitsR"
                )
            }

            if (innerW < 60) {
                if (DBG) Log.d(TAG, "RIMDBG[$debugTag] S=${f2(scale)} DROP innerW<60 innerW=$innerW")
                continue
            }

            // ratioW: NO cortar candidates -> penalidad
            val ratioW = innerW.toFloat() / expWpx
            val ratioPenalty = if (ratioW !in W_RATIO_MIN..W_RATIO_MAX) {
                if (DBG) {
                    Log.d(
                        TAG,
                        "RIMDBG[$debugTag] S=${f2(scale)} RATIO_PEN ratioW=${f3(ratioW)} " +
                                "expW=${f1(expWpx)} innerW=$innerW"
                    )
                }
                (1f - 1.1f * abs(1f - ratioW)).coerceIn(0.15f, 0.95f)
            } else {
                1f
            }

            val pxPerMmX = (innerW.toFloat() / hboxInnerMmF).takeIf { it.isFinite() && it > 0f } ?: run {
                if (DBG) {
                    Log.d(
                        TAG,
                        "RIMDBG[$debugTag] S=${f2(scale)} DROP pxPerMmX invalid innerW=$innerW hboxInner=$hboxInnerMmF"
                    )
                }
                continue
            }

            val bottomPick = pickBottomInsideOut(
                edgesU8 = edgesU8,
                dirU8 = dirU8,
                maskU8 = maskU8,
                w = w,
                h = h,
                leftX = a,
                rightX = b,
                ySeed = probeYLocal,
                yMax = (h - 1),
                topMinAllowedY = topMinAllowedY,
                stepX = 4,
                minHits = 28,
                minCoverage = 0.50f,
                contJumpPx = CONT_JUMP_PX
            )

            if (bottomPick == null) {
                if (DBG) {
                    Log.d(
                        TAG,
                        "RIMDBG[$debugTag] S=${f2(scale)} BOTTOM_FAIL a=$a b=$b " +
                                "ySeed=$probeYLocal topMin=$topMinAllowedY"
                    )
                }

                // ---------------------------------------------
                // FALLBACK PARCIAL: LR_OK pero sin bottom
                // ---------------------------------------------
                val yCapPartial = (probeYLocal - TOP_SEARCH_PAD).coerceIn(0, h - 1)
                val topYPartial = (topMinAllowedY + 2).coerceAtMost(yCapPartial)

                val bottomYPartial = if (expHpxGuess.isFinite() && expHpxGuess > 60f) {
                    (topYPartial + expHpxGuess.roundToInt()).coerceIn(topYPartial + 8, h - 1)
                } else {
                    (probeYLocal + (0.70f * expWpx).roundToInt()).coerceIn(topYPartial + 8, h - 1)
                }
                fun clamp01(v: Float): Float = v.coerceIn(0f, 1f)

                if (bottomYPartial > topYPartial) {
                    val partialConf = (
                            0.55f * ratioPenalty +
                                    0.25f * clamp01((innerW.toFloat() / expWpx).coerceIn(0.70f, 1.30f)) +
                                    0.20f * clamp01((innerW - 60f) / 180f)
                            ).coerceIn(0f, 0.79f)

                    if (partialConf > bestPartialConf) {
                        bestPartialConf = partialConf
                        bestPartialLeft = a
                        bestPartialRight = b
                        bestPartialTop = topYPartial
                        bestPartialBottom = bottomYPartial
                        bestPartialRefY = probeYLocal
                        bestPartialSeedX = seedXLocal
                        bestPartialScale = scale
                        bestPartialBottomPoly = emptyList()
                        bestPartialBottomEstimated = true
                    }

                    if (DBG) {
                        Log.d(
                            TAG,
                            "RIMDBG[$debugTag] S=${f2(scale)} PARTIAL_OK " +
                                    "a=$a b=$b innerW=$innerW ratioW=${f3(ratioW)} " +
                                    "top=$topYPartial botEst=$bottomYPartial conf=${f3(partialConf)}"
                        )
                    }
                }

                continue
            } else {
                if (DBG) {
                    Log.d(
                        TAG,
                        "RIMDBG[$debugTag] S=${f2(scale)} BOTTOM_OK yMed=${bottomPick.yMed} " +
                                "hits=${bottomPick.poly.size} cov=${f3(bottomPick.coverage)} " +
                                "cont=${f3(bottomPick.continuity)}"
                    )
                }
            }

            val bottomY = bottomPick.yMed

            val yCap = (probeYLocal - TOP_SEARCH_PAD).coerceIn(0, h - 1)
            val expHpx = if (vboxInnerMmF > 1e-3f) (vboxInnerMmF * pxPerMmX) else Float.NaN
            val topY = if (expHpx.isFinite() && expHpx > 60f) {
                (bottomY - expHpx).roundToInt().coerceIn(topMinAllowedY, yCap)
            } else {
                (topMinAllowedY + 2).coerceAtMost(yCap)
            }

            val innerH = (bottomY - topY).coerceAtLeast(1)
            val minH0 = max(70, (h * 0.18f).roundToInt())
            val maxH0 = (h * 0.92f).roundToInt()
            if (innerH !in minH0..maxH0) {
                if (DBG) {
                    Log.d(
                        TAG,
                        "RIMDBG[$debugTag] S=${f2(scale)} DROP innerH=$innerH " +
                                "range=[$minH0..$maxH0] topY=$topY bottomY=$bottomY"
                    )
                }
                continue
            }

            val confCoverage = norm013250(bottomPick.coverage, 0.55f, 0.95f)
            val confContinuity = norm013250(bottomPick.continuity, 0.65f, 0.98f)
            val confSamples = norm013250(bottomPick.poly.size.toFloat(), 28f, 60f)

            var conf = (
                    0.55f * confCoverage +
                            0.35f * confContinuity +
                            0.10f * confSamples
                    ).coerceIn(0f, 1f)

            conf *= (0.90f + 0.10f * ratioPenalty).coerceIn(0.90f, 1.00f)
            conf = conf.coerceIn(0f, 1f)

            if (DBG) {
                Log.d(
                    TAG,
                    "RIMCONF[$debugTag] S=${f2(scale)} " +
                            "cov=${f3(bottomPick.coverage)} cont=${f3(bottomPick.continuity)} " +
                            "n=${bottomPick.poly.size} " +
                            "cCov=${f3(confCoverage)} cCont=${f3(confContinuity)} cN=${f3(confSamples)} " +
                            "ratioPen=${f3(ratioPenalty)} conf=${f3(conf)}"
                )
            }

            if (conf > bestConf) {
                bestConf = conf
                bestLeft = a
                bestRight = b
                bestTop = topY
                bestBottom = bottomY
                bestScale = scale
                bestBottomPoly = bottomPick.poly
                bestRefY = probeYLocal
                bestSeedX = seedXLocal
            }
        }

        stage = "BEST_FINAL"

// ----------------------------------------------------------
// si no hubo candidato normal, intentar candidato parcial
// ----------------------------------------------------------
        if (bestConf < 0f) {
            if (
                bestPartialConf >= 0f &&
                bestPartialLeft >= 0 &&
                bestPartialRight >= 0 &&
                bestPartialTop >= 0 &&
                bestPartialBottom > bestPartialTop &&
                bestPartialRefY >= 0 &&
                bestPartialSeedX >= 0
            ) {
                d(
                    "RIM[$debugTag] BEST_PARTIAL profile=$profile3250 conf=${f3(bestPartialConf)} " +
                            "L/R=$bestPartialLeft/$bestPartialRight " +
                            "T/B=$bestPartialTop/$bestPartialBottom " +
                            "scale=${f2(bestPartialScale)} refY=$bestPartialRefY seedX=$bestPartialSeedX"
                )

                val innerWpx = (bestPartialRight - bestPartialLeft).toFloat().coerceAtLeast(1f)
                val pxPerMmXFinal = (innerWpx / hboxInnerMmF).takeIf { it.isFinite() && it > 0f }
                    ?: return fail("PXMM_PARTIAL", "pxPerMmX invalid")

                val nasalG = (if (nasalAtLeft) bestPartialLeft.toFloat() else bestPartialRight.toFloat()) + roiLeftG
                val templeG = (if (nasalAtLeft) bestPartialRight.toFloat() else bestPartialLeft.toFloat()) + roiLeftG

                val innerL = min(nasalG, templeG)
                val innerR = max(nasalG, templeG)

                val refYGlobal = bestPartialRefY.toFloat() + roiTopG
                val seedXGlobal = bestPartialSeedX.toFloat() + roiLeftG

                val res = RimDetectionResult(
                    ok = true,
                    confidence = bestPartialConf,
                    roiPx = RectF(roiLeftG, roiTopG, roiLeftG + w.toFloat(), roiTopG + h.toFloat()),

                    probeYpx = probeYLocal.toFloat() + roiTopG,
                    topYpx = bestPartialTop.toFloat() + roiTopG,
                    bottomYpx = bestPartialBottom.toFloat() + roiTopG,

                    innerLeftXpx = innerL,
                    innerRightXpx = innerR,

                    nasalInnerPolylinePx = null,
                    templeInnerPolylinePx = null,
                    nasalOuterPolylinePx = null,
                    templeOuterPolylinePx = null,

                    nasalInnerXpx = nasalG,
                    templeInnerXpx = templeG,

                    innerWidthPx = (innerR - innerL),
                    heightPx = (bestPartialBottom - bestPartialTop).toFloat(),

                    wallsYpx = refYGlobal,
                    seedXpx = seedXGlobal,

                    topPolylinePx = null,
                    bottomPolylinePx = null,
                    innerArcPolylinePx = null,
                    outerLeftXpx = null,
                    outerRightXpx = null,
                    rimThicknessPx = null
                )

                return RimDetectPack3250(
                    result = res,
                    edges = edgesU8,
                    w = w,
                    h = h
                )
            }

            return fail("NO_CAND", "no candidate survived")
        }

        if (bestConf < OK_CONF_MIN) return fail("LOW_CONF", "bestConf=${f3(bestConf)} < $OK_CONF_MIN")
        if (bestLeft < 0 || bestRight < 0 || bestBottom < 0 || bestTop < 0) {
            return fail("BEST_INV", "best coords invalid")
        }
        if (bestRefY < 0 || bestSeedX < 0) {
            return fail("BEST_INV2", "bestRefY/bestSeedX invalid refY=$bestRefY seed=$bestSeedX")
        }

        val innerWpx = (bestRight - bestLeft).toFloat().coerceAtLeast(1f)
        val pxPerMmXFinal = (innerWpx / hboxInnerMmF).takeIf { it.isFinite() && it > 0f }
            ?: return fail("PXMM", "pxPerMmX invalid")

        d(
            "RIM[$debugTag] BEST profile=$profile3250 conf=${f3(bestConf)} " +
                    "L/R=$bestLeft/$bestRight W=${f1(innerWpx)} pxPerMmX=${f3(pxPerMmXFinal)} " +
                    "T/B=$bestTop/$bestBottom scale=${f2(bestScale)} " +
                    "refY=$bestRefY seedX=$bestSeedX"
        )

        val nasalG = (if (nasalAtLeft) bestLeft.toFloat() else bestRight.toFloat()) + roiLeftG
        val templeG = (if (nasalAtLeft) bestRight.toFloat() else bestLeft.toFloat()) + roiLeftG

// innerLeft/Right = SOLO min/max
        val innerL = min(nasalG, templeG)
        val innerR = max(nasalG, templeG)

        // ----------------------------------------------------------
        // arcos candidatos laterales
        // ----------------------------------------------------------
        val minOuterGapPx = max(3, (0.60f * pxPerMmXFinal).roundToInt())
        val maxOuterGapPx = max(minOuterGapPx + 2, (4.50f * pxPerMmXFinal).roundToInt())

        val sideArcLocal = buildSideArcCandidates3250(
            edgesU8 = edgesU8,
            dirU8 = dirU8,
            maskU8 = maskU8,
            w = w,
            h = h,
            seedXInside = bestSeedX,
            yRef = bestRefY,
            topY = bestTop,
            bottomY = bestBottom,
            minOuterGapPx = minOuterGapPx,
            maxOuterGapPx = maxOuterGapPx,
            profile3250 = profile3250
        )

        val innerLeftPolyG = localPolylineToGlobal3250(sideArcLocal.innerLeft, roiLeftG, roiTopG)
        val innerRightPolyG = localPolylineToGlobal3250(sideArcLocal.innerRight, roiLeftG, roiTopG)
        val outerLeftPolyG = localPolylineToGlobal3250(sideArcLocal.outerLeft, roiLeftG, roiTopG)
        val outerRightPolyG = localPolylineToGlobal3250(sideArcLocal.outerRight, roiLeftG, roiTopG)

        val nasalInnerPolyG = if (nasalAtLeft) innerLeftPolyG else innerRightPolyG
        val templeInnerPolyG = if (nasalAtLeft) innerRightPolyG else innerLeftPolyG
        val nasalOuterPolyG = if (nasalAtLeft) outerLeftPolyG else outerRightPolyG
        val templeOuterPolyG = if (nasalAtLeft) outerRightPolyG else outerLeftPolyG

        val bottomPolyGlobal = if (bestBottomPoly.isNotEmpty()) {
            bestBottomPoly
                .map { (x, y) -> PointF(x.toFloat() + roiLeftG, y.toFloat() + roiTopG) }
                .sortedBy { it.x }
        } else {
            null
        }

        val refYGlobal = bestRefY.toFloat() + roiTopG
        val seedXGlobal = bestSeedX.toFloat() + roiLeftG

        if (DBG) {
            Log.d(
                TAG,
                "RIMARC[$debugTag] profile=$profile3250 sideArcs " +
                        "nIn=${nasalInnerPolyG?.size ?: 0} tIn=${templeInnerPolyG?.size ?: 0} " +
                        "nOut=${nasalOuterPolyG?.size ?: 0} tOut=${templeOuterPolyG?.size ?: 0} " +
                        "top=$bestTop bot=$bestBottom seedX=$bestSeedX"
            )
        }

        val nasalInnerLocal = if (nasalAtLeft) sideArcLocal.innerLeft else sideArcLocal.innerRight
        val templeInnerLocal = if (nasalAtLeft) sideArcLocal.innerRight else sideArcLocal.innerLeft

        val innerArcLocal = buildOfficialInnerArc3250(
            bottomLocal = bestBottomPoly.map { (x, y) -> PointF(x.toFloat(), y.toFloat()) },
            nasalLocal = nasalInnerLocal,
            templeLocal = templeInnerLocal,
            maxJoinDistPx = 14f,
            maxStepDistPx = 12f,
            minRunPts = 6
        )

        val innerArcGlobal =
            if (innerArcLocal.isNotEmpty()) localPolylineToGlobal3250(innerArcLocal, roiLeftG, roiTopG)
            else null

        val res = RimDetectionResult(
            ok = true,
            confidence = bestConf,
            roiPx = RectF(roiLeftG, roiTopG, roiLeftG + w.toFloat(), roiTopG + h.toFloat()),

            probeYpx = probeYLocal.toFloat() + roiTopG,
            topYpx = bestTop.toFloat() + roiTopG,
            bottomYpx = bestBottom.toFloat() + roiTopG,

            innerLeftXpx = innerL,
            innerRightXpx = innerR,

            nasalInnerPolylinePx = nasalInnerPolyG,
            templeInnerPolylinePx = templeInnerPolyG,
            nasalOuterPolylinePx = nasalOuterPolyG,
            templeOuterPolylinePx = templeOuterPolyG,

            nasalInnerXpx = nasalG,
            templeInnerXpx = templeG,

            innerWidthPx = (innerR - innerL),
            heightPx = (bestBottom - bestTop).toFloat(),

            // mantenemos el campo por compatibilidad; ahora representa la fila oficial de referencia
            wallsYpx = refYGlobal,
            seedXpx = seedXGlobal,

            topPolylinePx = null,
            bottomPolylinePx = bottomPolyGlobal,
            innerArcPolylinePx = innerArcGlobal,
            outerLeftXpx = null,
            outerRightXpx = null,
            rimThicknessPx = null
        )

        return RimDetectPack3250(
            result = res,
            edges = edgesU8,
            w = w,
            h = h
        )
    }

    /**
     * Mirror fallback
     *
     * Refleja el resultado geométrico de un ojo al otro lado usando la midline global.
     * NO recalcula detección, NO recalcula px/mm, NO reinterpreta filOver/profile:
     * solo espeja la geometría ya resuelta.
     */
    fun mirrorFromOtherEye3250(
        primary: RimDetectionResult,
        targetEdgesU8: ByteArray,
        w: Int,
        h: Int,
        targetRoiGlobal: RectF,
        midlineXpx: Float,
        debugTag: String,
        srcTag: String,
        confScale: Float = 0.60f
    ): RimDetectPack3250? {

        fun fail(msg: String): RimDetectPack3250? {
            Log.w(TAG, "RIM[$debugTag] MIRROR FAIL: $msg")
            return null
        }

        if (w <= 0 || h <= 0) return fail("w/h invalid $w x $h")
        if (targetEdgesU8.size != w * h) {
            return fail("edgesU8.size=${targetEdgesU8.size} != w*h=${w * h}")
        }
        if (!midlineXpx.isFinite()) return fail("midline invalid=$midlineXpx")

        val roiLeft = targetRoiGlobal.left
        val roiTop = targetRoiGlobal.top
        val roiRight = roiLeft + (w - 1).toFloat()
        val roiBottom = roiTop + (h - 1).toFloat()

        fun mirrorX(x: Float): Float = 2f * midlineXpx - x
        fun clampX(x: Float): Float = x.coerceIn(roiLeft + 1f, roiRight - 1f)
        fun clampY(y: Float): Float = y.coerceIn(roiTop + 1f, roiBottom - 1f)

        fun mirrorPolyY(poly: List<PointF>?): List<PointF>? =
            poly
                ?.map { p -> PointF(clampX(mirrorX(p.x)), clampY(p.y)) }
                ?.sortedBy { it.y }

        fun mirrorPolyX(poly: List<PointF>?): List<PointF>? =
            poly
                ?.map { p -> PointF(clampX(mirrorX(p.x)), clampY(p.y)) }
                ?.sortedBy { it.x }

        // ---------- mirror SEMANTIC endpoints ----------
        var nasal = clampX(mirrorX(primary.nasalInnerXpx))
        var temple = clampX(mirrorX(primary.templeInnerXpx))

        // Re-clasificación por seguridad: nasal debe quedar más cerca de la midline que temple.
        if (abs(temple - midlineXpx) < abs(nasal - midlineXpx)) {
            val tmp = nasal
            nasal = temple
            temple = tmp
        }

        // innerLeft/Right = SOLO min/max
        val innerL = min(nasal, temple)
        val innerR = max(nasal, temple)

        val innerW = (innerR - innerL)
        if (!innerW.isFinite() || innerW < 60f) {
            return fail("inner width too small after mirror: $innerW")
        }

        val probeY = clampY(primary.probeYpx)
        val topY = clampY(primary.topYpx)
        val bottomY = clampY(primary.bottomYpx)
        val heightPx = (bottomY - topY).coerceAtLeast(1f)

        // bottom poly: mirror + clamp + orden por X
        val bottomPoly = mirrorPolyX(primary.bottomPolylinePx)

        // mirror de arcos candidatos laterales
        val nasalInnerPoly = mirrorPolyY(primary.nasalInnerPolylinePx)
        val templeInnerPoly = mirrorPolyY(primary.templeInnerPolylinePx)
        val nasalOuterPoly = mirrorPolyY(primary.nasalOuterPolylinePx)
        val templeOuterPoly = mirrorPolyY(primary.templeOuterPolylinePx)

        val innerArcPolylinePx =
            buildOfficialInnerArc3250(
                bottomLocal = bottomPoly ?: emptyList(),
                nasalLocal = nasalInnerPoly ?: emptyList(),
                templeLocal = templeInnerPoly ?: emptyList(),
                maxJoinDistPx = 14f,
                maxStepDistPx = 12f,
                minRunPts = 6
            ).takeIf { it.isNotEmpty() }

        // anchors
        val wallsY = clampY(primary.wallsYpx)
        val seedX = clampX(mirrorX(primary.seedXpx))

        val conf = (primary.confidence * confScale).coerceIn(0f, 0.99f)

        Log.w(
            TAG,
            "RIM[$debugTag] MIRROR_FROM_$srcTag conf=${f3(conf)} " +
                    "innerL/R=${f1(innerL)}/${f1(innerR)} W=${f1(innerW)} " +
                    "wallsY=${f1(wallsY)} seedX=${f1(seedX)} midX=${f1(midlineXpx)} " +
                    "nIn=${nasalInnerPoly?.size ?: 0} tIn=${templeInnerPoly?.size ?: 0} " +
                    "nOut=${nasalOuterPoly?.size ?: 0} tOut=${templeOuterPoly?.size ?: 0}"
        )

        val res = RimDetectionResult(
            ok = true,
            confidence = conf,
            roiPx = targetRoiGlobal,

            probeYpx = probeY,
            topYpx = topY,
            bottomYpx = bottomY,

            innerLeftXpx = innerL,
            innerRightXpx = innerR,

            nasalInnerXpx = nasal,
            templeInnerXpx = temple,

            heightPx = heightPx,
            innerWidthPx = innerW,

            topPolylinePx = null,
            bottomPolylinePx = bottomPoly,

            wallsYpx = wallsY,
            seedXpx = seedX,

            outerLeftXpx = null,
            outerRightXpx = null,
            nasalOuterXpx = null,
            templeOuterXpx = null,
            rimThicknessPx = null,
            outerWidthPx = null,

            innerArcPolylinePx = innerArcPolylinePx,
            nasalInnerPolylinePx = nasalInnerPoly,
            templeInnerPolylinePx = templeInnerPoly,
            nasalOuterPolylinePx = nasalOuterPoly,
            templeOuterPolylinePx = templeOuterPoly
        )

        return RimDetectPack3250(
            result = res,
            edges = targetEdgesU8,
            w = w,
            h = h
        )
    }

    /**
     * Bottom: contrato congruente con el detector
     * - MISMA referencia vertical oficial: ySeed
     * - DOMINIO horizontal del aro: leftXrightX
     * - BARRIDO CENTER→OUT
     * - MÁSCARA = veto local (se ignora/salta)
     * - Continuidad guiada por prevY desde el ancla
     */
    /**
     * Bottom: contrato congruente con el detector
     * - MISMA referencia vertical oficial: ySeed
     * - DOMINIO horizontal del aro: leftX..rightX
     * - BARRIDO CENTER→OUT
     * - MÁSCARA = veto local (se ignora/salta)
     * - Continuidad guiada por prevY desde el ancla
     *
     * Ajuste aplicado:
     * - seed más permisiva
     * - bottom acepta edge no perfectamente horizontal
     * - soporte horizontal ahora da bonus, no veto absoluto
     * - menor castigo por expectedY
     * - fallback extra con ventana más amplia antes de declarar miss
     */
    private fun pickBottomInsideOut(
        edgesU8: ByteArray,
        dirU8: ByteArray?,
        maskU8: ByteArray?,
        w: Int,
        h: Int,
        leftX: Int,
        rightX: Int,
        ySeed: Int,
        yMax: Int,
        topMinAllowedY: Int,
        stepX: Int,
        minHits: Int,
        minCoverage: Float,
        contJumpPx: Int,
        expectedBottomY: Int? = null,
        expectedTolPx: Int = 42
    ): ArcPick? {

        val xa = (leftX + 8).coerceIn(0, w - 1)
        val xb = (rightX - 8).coerceIn(0, w - 1)
        if (xb - xa < 50) return null

        val yLo = topMinAllowedY.coerceIn(0, h - 1)
        val yHi = yMax.coerceIn(0, h - 1)
        if (yHi - yLo < 10) return null

        val xMid = ((leftX + rightX) / 2).coerceIn(xa, xb)
        val ySeed0 = ySeed.coerceIn(yLo, yHi)

        fun findSeedAtX(x: Int): Int {
            val y0Soft = expectedBottomY?.let { (it - expectedTolPx).coerceIn(yLo, yHi) } ?: yLo
            val y1Soft = expectedBottomY?.let { (it + expectedTolPx).coerceIn(yLo, yHi) } ?: yHi

            var y = findBestBottomEdgeInWindow3250(
                edgesU8 = edgesU8,
                dirU8 = dirU8,
                maskU8 = maskU8,
                w = w,
                x = x,
                y0 = max(ySeed0, y0Soft),
                y1 = y1Soft,
                expectedY = expectedBottomY ?: ySeed0
            )

            if (y < 0) {
                y = findBestBottomEdgeInWindow3250(
                    edgesU8 = edgesU8,
                    dirU8 = dirU8,
                    maskU8 = maskU8,
                    w = w,
                    x = x,
                    y0 = ySeed0,
                    y1 = yHi,
                    expectedY = expectedBottomY ?: ySeed0
                )
            }

            // fallback extra: seed permisiva en ventana más amplia
            if (y < 0) {
                val y0Wide = (ySeed0 - contJumpPx * 3).coerceIn(yLo, yHi)
                val y1Wide = (ySeed0 + contJumpPx * 5).coerceIn(yLo, yHi)
                y = findBestBottomEdgeInWindow3250(
                    edgesU8 = edgesU8,
                    dirU8 = dirU8,
                    maskU8 = maskU8,
                    w = w,
                    x = x,
                    y0 = y0Wide,
                    y1 = y1Wide,
                    expectedY = expectedBottomY ?: ySeed0
                )
            }

            return y
        }

        data class SeedBottom(val x: Int, val y: Int)

        var seedBottom: SeedBottom? = null

        run {
            val y0 = findSeedAtX(xMid)
            if (y0 >= 0) {
                seedBottom = SeedBottom(xMid, y0)
                return@run
            }

            var d = stepX
            while (xMid - d >= xa || xMid + d <= xb) {
                val xl = xMid - d
                if (xl in xa..xb) {
                    val yl = findSeedAtX(xl)
                    if (yl >= 0) {
                        seedBottom = SeedBottom(xl, yl)
                        return@run
                    }
                }

                val xr = xMid + d
                if (xr in xa..xb) {
                    val yr = findSeedAtX(xr)
                    if (yr >= 0) {
                        seedBottom = SeedBottom(xr, yr)
                        return@run
                    }
                }

                d += stepX
            }
        }

        val seedBottomX = seedBottom?.x ?: return null
        val seedBottomY = seedBottom.y

        fun traceHalf(startX: Int, stepSign: Int): List<Pair<Int, Int>> {
            val out = ArrayList<Pair<Int, Int>>()
            var prevY = seedBottomY
            var x = startX
            var missCount = 0

            while (x in xa..xb) {
                val winPad = (contJumpPx + missCount * 2).coerceAtMost(contJumpPx * 3)
                val yWin0 = (prevY - winPad).coerceIn(yLo, yHi)
                val yWin1 = (prevY + winPad).coerceIn(yLo, yHi)

                var bestY = findBestBottomEdgeInWindow3250(
                    edgesU8 = edgesU8,
                    dirU8 = dirU8,
                    maskU8 = maskU8,
                    w = w,
                    x = x,
                    y0 = yWin0,
                    y1 = yWin1,
                    expectedY = prevY
                )

                if (bestY < 0 && expectedBottomY != null) {
                    val ySoft0 = (expectedBottomY - expectedTolPx).coerceIn(yLo, yHi)
                    val ySoft1 = (expectedBottomY + expectedTolPx).coerceIn(yLo, yHi)
                    bestY = findBestBottomEdgeInWindow3250(
                        edgesU8 = edgesU8,
                        dirU8 = dirU8,
                        maskU8 = maskU8,
                        w = w,
                        x = x,
                        y0 = ySoft0,
                        y1 = ySoft1,
                        expectedY = expectedBottomY
                    )
                }

                // fallback extra: ampliar ventana alrededor de prevY antes de declarar miss
                if (bestY < 0) {
                    val yWide0 = (prevY - winPad * 2).coerceIn(yLo, yHi)
                    val yWide1 = (prevY + winPad * 2).coerceIn(yLo, yHi)
                    bestY = findBestBottomEdgeInWindow3250(
                        edgesU8 = edgesU8,
                        dirU8 = dirU8,
                        maskU8 = maskU8,
                        w = w,
                        x = x,
                        y0 = yWide0,
                        y1 = yWide1,
                        expectedY = prevY
                    )
                }

                if (bestY >= 0) {
                    out.add(x to bestY)
                    prevY = bestY
                    missCount = 0
                } else {
                    out.add(x to -1)
                    missCount++
                }

                x += stepSign * stepX
            }

            return out
        }

        val leftHalf = traceHalf(seedBottomX, -1).drop(1)
        val rightHalf = traceHalf(seedBottomX, +1)

        val polyRaw = ArrayList<Pair<Int, Int>>(leftHalf.size + rightHalf.size)
        polyRaw.addAll(leftHalf.asReversed())
        polyRaw.addAll(rightHalf)

        val polyFilled = fillSmallBottomGaps3250(
            poly = polyRaw,
            stepX = stepX,
            maxGapCols = 2,
            maxInterpJumpPx = contJumpPx * 2
        )

        val polySmooth = smoothBottomPolyline3250(polyFilled)

        val poly = keepBestBottomRunByXStrict3250(
            poly = polySmooth,
            maxJumpX = stepX * 2,
            maxJumpY = contJumpPx * 2,
            minRunPoints = minHits,
            maxTotalSecondaryDriftPx = 120
        )

        if (poly.size < minHits) return null

        val xSpan = (poly.last().first - poly.first().first).coerceAtLeast(0)
        val expectedSamples = (xSpan / stepX + 1).coerceAtLeast(1)
        val coverage = poly.size.toFloat() / expectedSamples.toFloat()

        var contSamples = 0
        var contHits = 0
        for (i in 1 until poly.size) {
            contSamples++
            if (abs(poly[i].second - poly[i - 1].second) <= contJumpPx * 2) {
                contHits++
            }
        }
        val continuity = if (contSamples > 0) {
            contHits.toFloat() / contSamples.toFloat()
        } else {
            0f
        }

        if (coverage < minCoverage) return null

        val ys = poly.map { it.second }.sorted()
        val yMed = ys[ys.size / 2]

        return ArcPick(
            yMed = yMed,
            poly = poly,
            coverage = coverage,
            continuity = continuity
        )
    }

    private fun findBestBottomEdgeInWindow3250(
        edgesU8: ByteArray,
        dirU8: ByteArray?,
        maskU8: ByteArray?,
        w: Int,
        x: Int,
        y0: Int,
        y1: Int,
        expectedY: Int
    ): Int {
        if (w <= 0) return -1
        val h = edgesU8.size / w
        if (h <= 0) return -1
        if (x !in 0 until w) return -1

        val lo = min(y0, y1).coerceIn(0, h - 1)
        val hi = max(y0, y1).coerceIn(0, h - 1)

        var bestY = -1
        var bestScore = Int.MIN_VALUE

        for (y in lo..hi) {
            val idx = y * w + x
            val e = edgesU8[idx].toInt() and 0xFF
            if (e == 0) continue
            if (isMasked(maskU8, idx)) continue

            val d = if (dirU8 == null) 255 else (dirU8[idx].toInt() and 0xFF)

            val dirScore = when {
                dirU8 == null -> 100
                isHorzEdgeDir(d) -> 100
                else -> 45
            }

            val support = horizontalSupportAt3250(
                edgesU8 = edgesU8,
                dirU8 = dirU8,
                maskU8 = maskU8,
                w = w,
                h = h,
                x = x,
                y = y,
                halfX = 2,
                halfY = 1
            )
            if (support <= 0) continue

            val score =
                support * 100 +
                        dirScore -
                        abs(y - expectedY) * 6 +
                        (y - lo) * 2

            if (score > bestScore) {
                bestScore = score
                bestY = y
            }
        }

        return bestY
    }

    private fun horizontalSupportAt3250(
        edgesU8: ByteArray,
        dirU8: ByteArray?,
        maskU8: ByteArray?,
        w: Int,
        h: Int,
        x: Int,
        y: Int,
        halfX: Int = 2,
        halfY: Int = 1
    ): Int {
        if (x !in 0 until w || y !in 0 until h) return 0

        val x0 = (x - halfX).coerceIn(0, w - 1)
        val x1 = (x + halfX).coerceIn(0, w - 1)
        val y0 = (y - halfY).coerceIn(0, h - 1)
        val y1 = (y + halfY).coerceIn(0, h - 1)

        var s = 0
        for (yy in y0..y1) {
            val base = yy * w
            for (xx in x0..x1) {
                val idx = base + xx
                if (isMasked(maskU8, idx)) continue
                if ((edgesU8[idx].toInt() and 0xFF) == 0) continue

                val d = if (dirU8 == null) 255 else (dirU8[idx].toInt() and 0xFF)

                // soporte base por presencia de edge
                s += if (yy == y) 2 else 1

                // bonus por orientación horizontal
                if (dirU8 == null || isHorzEdgeDir(d)) {
                    s += if (yy == y) 2 else 1
                }
            }
        }
        return s
    }

    private fun fillSmallBottomGaps3250(
        poly: List<Pair<Int, Int>>,
        stepX: Int,
        maxGapCols: Int,
        maxInterpJumpPx: Int
    ): List<Pair<Int, Int>> {
        if (poly.isEmpty()) return emptyList()

        val out = poly.toMutableList()
        var i = 0

        while (i < out.size) {
            if (out[i].second >= 0) {
                i++
                continue
            }

            val start = i
            while (i < out.size && out[i].second < 0) i++
            val end = i - 1

            val gapLen = end - start + 1
            val leftIdx = start - 1
            val rightIdx = i

            if (
                gapLen <= maxGapCols &&
                leftIdx >= 0 &&
                rightIdx < out.size &&
                out[leftIdx].second >= 0 &&
                out[rightIdx].second >= 0
            ) {
                val yL = out[leftIdx].second
                val yR = out[rightIdx].second
                val xL = out[leftIdx].first
                val xR = out[rightIdx].first

                val dxCols = ((xR - xL) / stepX).coerceAtLeast(1)
                val dy = abs(yR - yL)

                if (dy <= maxInterpJumpPx * dxCols) {
                    for (k in 0 until gapLen) {
                        val a = (k + 1).toFloat() / (gapLen + 1).toFloat()
                        val yFill = ((1f - a) * yL + a * yR).roundToInt()
                        out[start + k] = out[start + k].first to yFill
                    }
                }
            }
        }

        return out.filter { it.second >= 0 }
    }

    private fun smoothBottomPolyline3250(
        poly: List<Pair<Int, Int>>
    ): List<Pair<Int, Int>> {
        if (poly.size < 3) return poly

        val n = poly.size
        val med = IntArray(n)

        for (i in 0 until n) {
            val ys = ArrayList<Int>(5)
            for (k in -2..2) {
                val j = (i + k).coerceIn(0, n - 1)
                ys.add(poly[j].second)
            }
            ys.sort()
            med[i] = ys[ys.size / 2]
        }

        val out = ArrayList<Pair<Int, Int>>(n)
        for (i in 0 until n) {
            var sum = 0
            var cnt = 0
            for (k in -2..2) {
                val j = (i + k).coerceIn(0, n - 1)
                sum += med[j]
                cnt++
            }
            out.add(poly[i].first to (sum / cnt))
        }

        return out
    }

    private fun buildScales(minS: Float, maxS: Float, step: Float): FloatArray {
        if (!minS.isFinite() || !maxS.isFinite() || !step.isFinite()) return floatArrayOf()
        if (step <= 0f) return floatArrayOf()
        val out = ArrayList<Float>(32)
        var s = minS
        while (s <= maxS + 1e-6f) {
            out += s
            s += step
        }
        return out.toFloatArray()
    }

    private fun clampInt(v: Int, lo: Int, hi: Int): Int = max(lo, min(hi, v))
    private fun clampInRange(v: Int, a: Int, b: Int): Int = v.coerceIn(min(a, b), max(a, b))

    /**
     * Busca inner L/R en una banda vertical alrededor de yRef (la referencia oficial),
     * desde seedX hacia afuera, saltando máscara.
     *
     * - mantenemos el mismo contrato Pair<Int,Int>
     * - LR exige evidencia vertical (dirU8)
     * - la máscara veta puntos/soportes, pero NO mueve la referencia
     */
    private fun findInnerWallsAtY3250(
        edgesU8: ByteArray,
        dirU8: ByteArray? = null,
        maskU8: ByteArray?,
        w: Int,
        h: Int,
        seedX: Int,
        yRef: Int,
        bandHalf: Int,
        minDist: Int,
        maxDist: Int
    ): Pair<Int, Int>? {
        val ys = ArrayList<Int>(2 * bandHalf + 1)
        for (dy in -bandHalf..bandHalf) {
            ys.add((yRef + dy).coerceIn(0, h - 1))
        }

        val leftHits = ArrayList<Int>()
        val rightHits = ArrayList<Int>()

        // No arrancar lejos, porque si no te salteás el inner real.
        val innerMinStep = minDist.coerceAtLeast(1)

        fun scanInner(dx: Int, y: Int): Int? {
            val xStart = seedX.coerceIn(0, w - 1)

            for (step in innerMinStep..maxDist) {
                val x = xStart + dx * step
                if (x !in 0 until w) break

                val idx = y * w + x
                if (isMaskedPx3250(maskU8, idx)) continue
                if ((edgesU8[idx].toInt() and 0xFF) == 0) continue

                val support = verticalSupportAt3250(
                    edgesU8 = edgesU8,
                    dirU8 = dirU8,
                    maskU8 = maskU8,
                    w = w,
                    h = h,
                    x = x,
                    y = y,
                    halfY = 2
                )
                if (support < 2) continue

                val d = if (dirU8 == null) 255 else (dirU8[idx].toInt() and 0xFF)
                val dirGood = isVertEdgeDir(d)
                if (!dirGood) continue

                return x
            }

            return null
        }
        for (y in ys) {
            scanInner(dx = -1, y = y)?.let { leftHits.add(it) }
            scanInner(dx = +1, y = y)?.let { rightHits.add(it) }
        }
        if (leftHits.isEmpty() || rightHits.isEmpty()) return null

        val left = medianInt3250(leftHits)
        val right = medianInt3250(rightHits)

        if (left >= right) return null
        if (abs(right - left) < 40) return null

        return left to right
    }

    private fun verticalSupportAt3250(
        edgesU8: ByteArray,
        dirU8: ByteArray?,
        maskU8: ByteArray?,
        w: Int,
        h: Int,
        x: Int,
        y: Int,
        halfY: Int = 2
    ): Int {
        if (x !in 0 until w || y !in 0 until h) return 0
        var s = 0
        val y0 = (y - halfY).coerceIn(0, h - 1)
        val y1 = (y + halfY).coerceIn(0, h - 1)
        for (yy in y0..y1) {
            val idx = yy * w + x
            if (isMaskedPx3250(maskU8, idx)) continue
            if ((edgesU8[idx].toInt() and 0xFF) == 0) continue
            val d = if (dirU8 == null) 255 else (dirU8[idx].toInt() and 0xFF)
            if (isVertEdgeDir(d)) s++
        }
        return s
    }
    private fun buildSideArcCandidates3250(
        edgesU8: ByteArray,
        dirU8: ByteArray? = null,
        maskU8: ByteArray?,
        w: Int,
        h: Int,
        seedXInside: Int,
        yRef: Int,
        topY: Int,
        bottomY: Int,
        minOuterGapPx: Int,
        maxOuterGapPx: Int,
        profile3250: RimProfile3250,
        stepY: Int = 3
    ): SideArcCandidatesLocal3250 {

        val y0Inner = (topY + 6).coerceIn(0, h - 1)
        val y1Inner = (bottomY - 10).coerceIn(0, h - 1)

        if (y1Inner <= y0Inner) {
            return SideArcCandidatesLocal3250(
                innerLeft = emptyList(),
                innerRight = emptyList(),
                outerLeft = emptyList(),
                outerRight = emptyList()
            )
        }

        val seedRow = buildInnerSeedRow3250(
            edgesU8 = edgesU8,
            dirU8 = dirU8,
            maskU8 = maskU8,
            w = w,
            h = h,
            seedXInside = seedXInside,
            yRef = yRef.coerceIn(y0Inner, y1Inner),
            minDist = 4,
            maxDist = ((w * 0.42f).roundToInt()).coerceAtLeast(24),
            bandHalf = 3
        ) ?: return SideArcCandidatesLocal3250(
            innerLeft = emptyList(),
            innerRight = emptyList(),
            outerLeft = emptyList(),
            outerRight = emptyList()
        )

        val innerLeftRaw = traceInnerPolylineFromSeedRow3250(
            edgesU8 = edgesU8,
            dirU8 = dirU8,
            maskU8 = maskU8,
            w = w,
            h = h,
            seedX = seedRow.leftX,
            seedY = seedRow.y,
            centerInsideSeedX = seedXInside,
            sideSign = -1,
            y0 = y0Inner,
            y1 = y1Inner,
            stepY = stepY
        )

        val innerRightRaw = traceInnerPolylineFromSeedRow3250(
            edgesU8 = edgesU8,
            dirU8 = dirU8,
            maskU8 = maskU8,
            w = w,
            h = h,
            seedX = seedRow.rightX,
            seedY = seedRow.y,
            centerInsideSeedX = seedXInside,
            sideSign = +1,
            y0 = y0Inner,
            y1 = y1Inner,
            stepY = stepY
        )

        val maxJumpY = max(stepY * 2, 6).toFloat()

        val innerLeftSmoothed = smoothSidePolyline3250(
            fillSmallSideGaps3250(
                pts = innerLeftRaw,
                stepY = stepY,
                maxGapRows = 1,          // antes 2
                maxInterpJumpPx = 10f    // antes 18f
            )
        )

        val innerRightSmoothed = smoothSidePolyline3250(
            fillSmallSideGaps3250(
                pts = innerRightRaw,
                stepY = stepY,
                maxGapRows = 1,
                maxInterpJumpPx = 10f
            )
        )

        val innerLeft = keepBestContinuousRunByYStrict3250(
            pts = innerLeftSmoothed,
            maxJumpX = 9f,
            maxJumpY = maxJumpY,
            minRunPoints = 10,
            maxTotalSecondaryDriftPx = 22f
        )

        val innerRight = keepBestContinuousRunByYStrict3250(
            pts = innerRightSmoothed,
            maxJumpX = 9f,
            maxJumpY = maxJumpY,
            minRunPoints = 10,
            maxTotalSecondaryDriftPx = 22f
        )

        val outerLeftRaw = if (profile3250 == RimProfile3250.PERFORADO || innerLeft.isEmpty()) {
            emptyList()
        } else {
            traceOuterFromInnerGuide3250(
                edgesU8 = edgesU8,
                dirU8 = dirU8,
                maskU8 = maskU8,
                w = w,
                h = h,
                innerGuide = innerLeft,
                y0 = y0Inner,
                y1 = y1Inner,
                stepY = stepY,
                outwardSign = -1,
                minOuterGapPx = minOuterGapPx,
                maxOuterGapPx = maxOuterGapPx,
                profile3250 = profile3250
            )
        }

        val outerRightRaw = if (profile3250 == RimProfile3250.PERFORADO || innerRight.isEmpty()) {
            emptyList()
        } else {
            traceOuterFromInnerGuide3250(
                edgesU8 = edgesU8,
                dirU8 = dirU8,
                maskU8 = maskU8,
                w = w,
                h = h,
                innerGuide = innerRight,
                y0 = y0Inner,
                y1 = y1Inner,
                stepY = stepY,
                outwardSign = +1,
                minOuterGapPx = minOuterGapPx,
                maxOuterGapPx = maxOuterGapPx,
                profile3250 = profile3250
            )
        }

        val outerLeft = keepBestContinuousRunByYStrict3250(
            pts = smoothSidePolyline3250(
                fillSmallSideGaps3250(
                    pts = outerLeftRaw,
                    stepY = stepY,
                    maxGapRows = 1,
                    maxInterpJumpPx = 12f
                )
            ),
            maxJumpX = 12f,
            maxJumpY = maxJumpY,
            minRunPoints = if (profile3250 == RimProfile3250.FULL_RIM) 6 else 4,
            maxTotalSecondaryDriftPx = 28f
        )

        val outerRight = keepBestContinuousRunByYStrict3250(
            pts = smoothSidePolyline3250(
                fillSmallSideGaps3250(
                    pts = outerRightRaw,
                    stepY = stepY,
                    maxGapRows = 1,
                    maxInterpJumpPx = 12f
                )
            ),
            maxJumpX = 12f,
            maxJumpY = maxJumpY,
            minRunPoints = if (profile3250 == RimProfile3250.FULL_RIM) 6 else 4,
            maxTotalSecondaryDriftPx = 28f
        )

        return SideArcCandidatesLocal3250(
            innerLeft = innerLeft,
            innerRight = innerRight,
            outerLeft = outerLeft,
            outerRight = outerRight
        )
    }

    private fun keepBestContinuousRunByYStrict3250(
        pts: List<PointF>,
        maxJumpX: Float,
        maxJumpY: Float,
        minRunPoints: Int,
        maxTotalSecondaryDriftPx: Float
    ): List<PointF> {
        if (pts.isEmpty()) return emptyList()
        if (pts.size == 1) return if (minRunPoints <= 1) pts else emptyList()

        val sorted = pts.sortedWith(compareBy<PointF> { it.y }.thenBy { it.x })

        data class Run(val pts: MutableList<PointF> = mutableListOf()) {
            fun ySpan(): Float =
                if (pts.size >= 2) pts.last().y - pts.first().y else 0f

            fun xSpan(): Float {
                if (pts.isEmpty()) return 0f
                var minX = Float.POSITIVE_INFINITY
                var maxX = Float.NEGATIVE_INFINITY
                for (p in pts) {
                    if (p.x < minX) minX = p.x
                    if (p.x > maxX) maxX = p.x
                }
                return maxX - minX
            }

            // serrucho lateral acumulado mientras avanzo sobre Y
            fun totalAbsDx(): Float {
                if (pts.size < 2) return 0f
                var acc = 0f
                for (i in 1 until pts.size) {
                    acc += abs(pts[i].x - pts[i - 1].x)
                }
                return acc
            }

            fun score(): Float {
                return pts.size * 100f +
                        ySpan() * 2f -
                        totalAbsDx() * 1.5f -
                        xSpan() * 0.5f
            }
        }

        val runs = ArrayList<Run>()
        var current = Run()
        current.pts.add(sorted.first())

        for (i in 1 until sorted.size) {
            val prev = sorted[i - 1]
            val now = sorted[i]

            val dy = now.y - prev.y
            val dx = abs(now.x - prev.x)

            val contiguous =
                dy in 0f..maxJumpY &&
                        dx <= maxJumpX

            if (contiguous) {
                current.pts.add(now)
            } else {
                runs.add(current)
                current = Run()
                current.pts.add(now)
            }
        }
        runs.add(current)

        val filtered = runs.filter { run ->
            run.pts.size >= minRunPoints &&
                    run.ySpan() > 0f &&
                    run.totalAbsDx() <= maxTotalSecondaryDriftPx
        }

        val best = filtered.maxByOrNull { it.score() } ?: return emptyList()
        return best.pts
    }
    private fun buildInnerSeedRow3250(
        edgesU8: ByteArray,
        dirU8: ByteArray? = null,
        maskU8: ByteArray?,
        w: Int,
        h: Int,
        seedXInside: Int,
        yRef: Int,
        minDist: Int,
        maxDist: Int,
        bandHalf: Int
    ): InnerSeedRow3250? {
        val ySeed = yRef.coerceIn(0, h - 1)

        val pair = findInnerWallsAtY3250(
            edgesU8 = edgesU8,
            dirU8 = dirU8,
            maskU8 = maskU8,
            w = w,
            h = h,
            seedX = seedXInside,
            yRef = ySeed,
            bandHalf = bandHalf,
            minDist = minDist,
            maxDist = maxDist
        ) ?: return null

        return InnerSeedRow3250(
            y = ySeed,
            leftX = pair.first,
            rightX = pair.second
        )
    }

    private fun traceInnerPolylineFromSeedRow3250(
        edgesU8: ByteArray,
        dirU8: ByteArray? = null,
        maskU8: ByteArray?,
        w: Int,
        h: Int,
        seedX: Int,
        seedY: Int,
        centerInsideSeedX: Int,
        sideSign: Int, // left=-1, right=+1
        y0: Int,
        y1: Int,
        stepY: Int
    ): List<PointF> {

        val up = traceInnerHalf3250(
            edgesU8 = edgesU8,
            dirU8 = dirU8,
            maskU8 = maskU8,
            w = w,
            h = h,
            startX = seedX,
            centerInsideSeedX = centerInsideSeedX,
            sideSign = sideSign,
            yStart = seedY - stepY,
            yEndInclusive = y0,
            yStep = -stepY
        )

        val down = traceInnerHalf3250(
            edgesU8 = edgesU8,
            dirU8 = dirU8,
            maskU8 = maskU8,
            w = w,
            h = h,
            startX = seedX,
            centerInsideSeedX = centerInsideSeedX,
            sideSign = sideSign,
            yStart = seedY + stepY,
            yEndInclusive = y1,
            yStep = +stepY
        )

        val out = ArrayList<PointF>(up.size + 1 + down.size)
        out.addAll(up.reversed())
        out.add(PointF(seedX.toFloat(), seedY.toFloat()))
        out.addAll(down)
        return out.sortedBy { it.y }
    }

private fun traceInnerHalf3250(
    edgesU8: ByteArray,
    dirU8: ByteArray? = null,
    maskU8: ByteArray?,
    w: Int,
    h: Int,
    startX: Int,
    centerInsideSeedX: Int,
    sideSign: Int,
    yStart: Int,
    yEndInclusive: Int,
    yStep: Int
): List<PointF> {

    val out = ArrayList<PointF>()
    var prevX = startX.coerceIn(0, w - 1)
    var prevDx = 0
    var missCount = 0

    var y = yStart
    while (if (yStep > 0) y <= yEndInclusive else y >= yEndInclusive) {

        val expectedX = (prevX + prevDx).coerceIn(0, w - 1)
        val radius = when (missCount) {
            0 -> 10
            1 -> 14
            else -> 18
        }

        val xFound = searchNearestInnerAroundPrevX3250(
            edgesU8 = edgesU8,
            dirU8 = dirU8,
            maskU8 = maskU8,
            w = w,
            h = h,
            y = y,
            prevX = expectedX,
            centerInsideSeedX = centerInsideSeedX,
            sideSign = sideSign,
            searchRadiusPx = radius
        )

        if (xFound != null) {
            val dxNow = xFound - prevX

            if (abs(dxNow) > 12) {
                missCount++
                if (missCount > 1) break
            } else {
                out.add(PointF(xFound.toFloat(), y.toFloat()))
                prevDx = dxNow
                prevX = xFound
                missCount = 0
            }
        } else {
            missCount++
            if (missCount > 1) break
        }

        y += yStep
    }

    return out
}
    private fun searchNearestInnerAroundPrevX3250(
        edgesU8: ByteArray,
        dirU8: ByteArray? = null,
        maskU8: ByteArray?,
        w: Int,
        h: Int,
        y: Int,
        prevX: Int,
        centerInsideSeedX: Int,
        sideSign: Int, // left=-1, right=+1
        searchRadiusPx: Int
    ): Int? {

        data class Cand(val x: Int, val score: Int)

        fun onCorrectSide(x: Int): Boolean {
            return if (sideSign < 0) x < centerInsideSeedX else x > centerInsideSeedX
        }

        fun buildCandidate(x: Int, strict: Boolean): Cand? {
            if (x !in 0 until w) return null
            if (!onCorrectSide(x)) return null

            val idx = y * w + x
            if (isMaskedPx3250(maskU8, idx)) return null
            if ((edgesU8[idx].toInt() and 0xFF) == 0) return null

            val support = verticalSupportAt3250(
                edgesU8 = edgesU8,
                dirU8 = dirU8,
                maskU8 = maskU8,
                w = w,
                h = h,
                x = x,
                y = y,
                halfY = if (strict) 2 else 1
            )
            if (support < if (strict) 2 else 1) return null

            val dirGood = if (dirU8 == null) true else {
                val d = dirU8[idx].toInt() and 0xFF
                isVertEdgeDir(d)
            }
            if (strict && !dirGood) return null

            val dist = abs(x - prevX)
            val score = (if (dirGood) 1000 else 0) + support * 100 - dist * 10
            return Cand(x = x, score = score)
        }

        fun search(strict: Boolean): Int? {
            var best: Cand? = null

            for (d in 0..searchRadiusPx) {
                if (d == 0) {
                    val c = buildCandidate(prevX, strict)
                    if (c != null && (best == null || c.score > best!!.score)) best = c
                } else {
                    val x1 = prevX - d
                    val x2 = prevX + d

                    val c1 = buildCandidate(x1, strict)
                    if (c1 != null && (best == null || c1.score > best!!.score)) best = c1

                    val c2 = buildCandidate(x2, strict)
                    if (c2 != null && (best == null || c2.score > best!!.score)) best = c2
                }
            }

            return best?.x
        }

        return search(strict = true)
    }

    private fun traceOuterFromInnerGuide3250(
        edgesU8: ByteArray,
        dirU8: ByteArray? = null,
        maskU8: ByteArray?,
        w: Int,
        h: Int,
        innerGuide: List<PointF>,
        y0: Int,
        y1: Int,
        stepY: Int,
        outwardSign: Int,
        minOuterGapPx: Int,
        maxOuterGapPx: Int,
        profile3250: RimProfile3250
    ): List<PointF> {
        if (innerGuide.isEmpty()) return emptyList()
        if (profile3250 == RimProfile3250.PERFORADO) return emptyList()

        val raw = ArrayList<PointF>()
        var prevGap: Int? = null

        var y = y1
        while (y >= y0) {
            val innerPt = innerGuide.minByOrNull { abs(it.y - y.toFloat()) }
            val innerX = innerPt?.x?.roundToInt()

            if (innerX != null) {
                val xOuter = searchOuterFromInner3250(
                    edgesU8 = edgesU8,
                    dirU8 = dirU8,
                    maskU8 = maskU8,
                    w = w,
                    h = h,
                    y = y,
                    innerX = innerX,
                    outwardSign = outwardSign,
                    prevGapPx = prevGap,
                    minOuterGapPx = minOuterGapPx,
                    maxOuterGapPx = maxOuterGapPx,
                    profile3250 = profile3250
                )

                if (xOuter != null) {
                    raw.add(PointF(xOuter.toFloat(), y.toFloat()))
                    prevGap = abs(xOuter - innerX)
                }
            }

            y -= stepY
        }

        return raw.sortedBy { it.y }
    }
    private fun searchOuterFromInner3250(
        edgesU8: ByteArray,
        dirU8: ByteArray?,
        maskU8: ByteArray?,
        w: Int,
        h: Int,
        y: Int,
        innerX: Int,
        outwardSign: Int, // left=-1, right=+1
        prevGapPx: Int?,
        minOuterGapPx: Int,
        maxOuterGapPx: Int,
        profile3250: RimProfile3250,
        minSupport: Int = 2
    ): Int? {
        if (profile3250 == RimProfile3250.PERFORADO) return null

        val gapTol = when (profile3250) {
            RimProfile3250.FULL_RIM -> max(3, ((prevGapPx ?: minOuterGapPx) * 0.45f).toInt())
            RimProfile3250.RANURADO -> max(4, ((prevGapPx ?: minOuterGapPx) * 0.85f).toInt())
            RimProfile3250.PERFORADO -> return null
        }

        fun isValidGap(gap: Int): Boolean {
            if (gap !in minOuterGapPx..maxOuterGapPx) return false

            val x = innerX + outwardSign * gap
            if (x !in 0 until w) return false

            val idx = y * w + x
            if (isMaskedPx3250(maskU8, idx)) return false
            if ((edgesU8[idx].toInt() and 0xFF) == 0) return false

            if (dirU8 != null) {
                val d = dirU8[idx].toInt() and 0xFF
                if (!isVertEdgeDir(d)) return false
            }

            val support = verticalSupportAt3250(
                edgesU8 = edgesU8,
                dirU8 = dirU8,
                maskU8 = maskU8,
                w = w,
                h = h,
                x = x,
                y = y,
                halfY = 2
            )
            return support >= minSupport
        }

        // 1) continuidad de gap si ya veníamos siguiendo uno
        if (prevGapPx != null && prevGapPx > 0) {
            val g0 = max(minOuterGapPx, prevGapPx - gapTol)
            val g1 = min(maxOuterGapPx, prevGapPx + gapTol)

            for (gap in g0..g1) {
                if (isValidGap(gap)) {
                    return innerX + outwardSign * gap
                }
            }
        }

        // 2) sino, primer edge válido desde inner hacia afuera
        for (gap in minOuterGapPx..maxOuterGapPx) {
            if (isValidGap(gap)) {
                return innerX + outwardSign * gap
            }
        }

        return null
    }

    private fun fillSmallSideGaps3250(
        pts: List<PointF>,
        stepY: Int,
        maxGapRows: Int,
        maxInterpJumpPx: Float
    ): List<PointF> {
        if (pts.isEmpty()) return emptyList()
        if (pts.size == 1) return pts

        val sorted = pts.sortedBy { it.y }
        val out = ArrayList<PointF>()
        out.add(sorted.first())

        for (i in 1 until sorted.size) {
            val a = sorted[i - 1]
            val b = sorted[i]

            val dy = (b.y - a.y).roundToInt()
            val rowsMissing = dy / stepY - 1

            if (
                rowsMissing in 1..maxGapRows &&
                abs(b.x - a.x) <= maxInterpJumpPx
            ) {
                for (k in 1..rowsMissing) {
                    val t = k.toFloat() / (rowsMissing + 1).toFloat()
                    val x = (1f - t) * a.x + t * b.x
                    val y = a.y + stepY * k
                    out.add(PointF(x, y))
                }
            }

            out.add(b)
        }

        return out
    }

    private fun smoothSidePolyline3250(
        pts: List<PointF>
    ): List<PointF> {
        if (pts.size < 3) return pts

        val sorted = pts.sortedBy { it.y }
        val n = sorted.size
        val medX = FloatArray(n)

        for (i in 0 until n) {
            val xs = ArrayList<Float>(5)
            for (k in -2..2) {
                val j = (i + k).coerceIn(0, n - 1)
                xs.add(sorted[j].x)
            }
            xs.sort()
            medX[i] = xs[xs.size / 2]
        }

        val out = ArrayList<PointF>(n)
        for (i in 0 until n) {
            var sum = 0f
            var cnt = 0
            for (k in -2..2) {
                val j = (i + k).coerceIn(0, n - 1)
                sum += medX[j]
                cnt++
            }
            out.add(PointF(sum / cnt.toFloat(), sorted[i].y))
        }

        return out
    }
    private fun localPolylineToGlobal3250(
        pts: List<PointF>,
        roiLeft: Float,
        roiTop: Float
    ): List<PointF>? {
        if (pts.isEmpty()) return null
        return pts.map { p -> PointF(p.x + roiLeft, p.y + roiTop) }
    }

    private fun dist3250(a: PointF, b: PointF): Float {
        val dx = a.x - b.x
        val dy = a.y - b.y
        return kotlin.math.sqrt(dx * dx + dy * dy)
    }

    private fun takeContiguousRunFromBottom3250(
        sidePts: List<PointF>,
        bottomAnchor: PointF,
        maxJoinDistPx: Float = 14f,
        maxStepDistPx: Float = 12f,
        minRunPts: Int = 6
    ): List<PointF> {
        if (sidePts.size < minRunPts) return emptyList()

        // bottom -> top
        val pts = sidePts.sortedByDescending { it.y }

        val startIdx = pts.indices.minByOrNull { i -> dist3250(pts[i], bottomAnchor) } ?: return emptyList()
        if (dist3250(pts[startIdx], bottomAnchor) > maxJoinDistPx) return emptyList()

        val out = ArrayList<PointF>()
        var prev = pts[startIdx]
        out.add(prev)

        for (i in (startIdx + 1) until pts.size) {
            val p = pts[i]

            val dyUp = prev.y - p.y
            if (dyUp <= 0f) break
            if (dyUp > maxStepDistPx) break
            if (dist3250(prev, p) > maxStepDistPx) break

            out.add(p)
            prev = p
        }

        return if (out.size >= minRunPts) out.reversed() else emptyList() // top -> bottom
    }

    private fun nearestIdxOnPolyline3250(poly: List<PointF>, p: PointF): Int {
        return poly.indices.minByOrNull { i -> dist3250(poly[i], p) } ?: 0
    }

    private fun buildOfficialInnerArc3250(
        bottomLocal: List<PointF>,
        nasalLocal: List<PointF>,
        templeLocal: List<PointF>,
        maxJoinDistPx: Float = 14f,
        maxStepDistPx: Float = 12f,
        minRunPts: Int = 6
    ): List<PointF> {
        if (bottomLocal.size < 3) return emptyList()

        val bottom = bottomLocal.sortedBy { it.x }
        val leftBottom = bottom.first()
        val rightBottom = bottom.last()

        val nasalBottomRef =
            if (nasalLocal.isNotEmpty() &&
                dist3250(nasalLocal.maxBy { it.y }, leftBottom) <= dist3250(nasalLocal.maxBy { it.y }, rightBottom)
            ) {
                leftBottom
            } else {
                rightBottom
            }

        val templeBottomRef =
            if (templeLocal.isNotEmpty() &&
                dist3250(templeLocal.maxBy { it.y }, leftBottom) <= dist3250(templeLocal.maxBy { it.y }, rightBottom)
            ) {
                leftBottom
            } else {
                rightBottom
            }

        val nasalRun = takeContiguousRunFromBottom3250(
            sidePts = nasalLocal,
            bottomAnchor = nasalBottomRef,
            maxJoinDistPx = maxJoinDistPx,
            maxStepDistPx = maxStepDistPx,
            minRunPts = minRunPts
        )

        val templeRun = takeContiguousRunFromBottom3250(
            sidePts = templeLocal,
            bottomAnchor = templeBottomRef,
            maxJoinDistPx = maxJoinDistPx,
            maxStepDistPx = maxStepDistPx,
            minRunPts = minRunPts
        )

        if (nasalRun.isEmpty() || templeRun.isEmpty()) return emptyList()

        val nasalBottomPt = nasalRun.last()
        val templeBottomPt = templeRun.last()

        val i0 = nearestIdxOnPolyline3250(bottom, nasalBottomPt)
        val i1 = nearestIdxOnPolyline3250(bottom, templeBottomPt)

        val bottomSegment =
            if (i0 <= i1) {
                bottom.subList(i0, i1 + 1)
            } else {
                bottom.subList(i1, i0 + 1).reversed()
            }

        val out = ArrayList<PointF>(nasalRun.size + bottomSegment.size + templeRun.size)

        out.addAll(nasalRun)                        // top -> bottom
        out.addAll(bottomSegment.drop(1))          // nasal -> temple
        out.addAll(templeRun.asReversed().drop(1)) // bottom -> top

        return out
    }

    private fun norm013250(v: Float, lo: Float, hi: Float): Float {
        if (hi <= lo) return 0f
        return ((v - lo) / (hi - lo)).coerceIn(0f, 1f)
    }

    private fun medianInt3250(v: List<Int>): Int {
    if (v.isEmpty()) return 0
    val s = v.sorted()
    val m = s.size / 2
    return if (s.size % 2 == 1) s[m] else ((s[m - 1] + s[m]) / 2)
}
    private fun keepBestBottomRunByXStrict3250(
        poly: List<Pair<Int, Int>>,
        maxJumpX: Int,
        maxJumpY: Int,
        minRunPoints: Int,
        maxTotalSecondaryDriftPx: Int
    ): List<Pair<Int, Int>> {
        if (poly.isEmpty()) return emptyList()
        if (poly.size == 1) return if (minRunPoints <= 1) poly else emptyList()

        val sorted = poly
            .filter { it.second >= 0 }
            .sortedWith(compareBy<Pair<Int, Int>> { it.first }.thenBy { it.second })

        if (sorted.isEmpty()) return emptyList()
        if (sorted.size == 1) return if (minRunPoints <= 1) sorted else emptyList()

        data class Run(val pts: MutableList<Pair<Int, Int>> = mutableListOf()) {
            fun xSpan(): Int =
                if (pts.size >= 2) pts.last().first - pts.first().first else 0

            fun ySpan(): Int {
                if (pts.isEmpty()) return 0
                var minY = Int.MAX_VALUE
                var maxY = Int.MIN_VALUE
                for ((_, y) in pts) {
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y
                }
                return maxY - minY
            }

            fun totalAbsDy(): Int {
                if (pts.size < 2) return 0
                var acc = 0
                for (i in 1 until pts.size) {
                    acc += abs(pts[i].second - pts[i - 1].second)
                }
                return acc
            }

            fun score(): Float {
                return pts.size * 100f +
                        xSpan() * 2f -
                        totalAbsDy() * 1.5f -
                        ySpan() * 0.5f
            }
        }

        val runs = ArrayList<Run>()
        var current = Run()
        current.pts.add(sorted.first())

        for (i in 1 until sorted.size) {
            val prev = sorted[i - 1]
            val now = sorted[i]

            val dx = now.first - prev.first
            val dy = abs(now.second - prev.second)

            val contiguous =
                dx in 0..maxJumpX &&
                        dy <= maxJumpY

            if (contiguous) {
                current.pts.add(now)
            } else {
                runs.add(current)
                current = Run()
                current.pts.add(now)
            }
        }
        runs.add(current)

        val filtered = runs.filter { run ->
            run.pts.size >= minRunPoints &&
                    run.xSpan() > 0 &&
                    run.totalAbsDy() <= maxTotalSecondaryDriftPx
        }

        val best = filtered.maxByOrNull { it.score() } ?: return emptyList()
        return best.pts
    }

    private fun f1(x: Float): String = String.format(Locale.US, "%.1f", x)
    private fun f2(x: Float): String = String.format(Locale.US, "%.2f", x)
    private fun f3(x: Float): String = String.format(Locale.US, "%.3f", x)
}
