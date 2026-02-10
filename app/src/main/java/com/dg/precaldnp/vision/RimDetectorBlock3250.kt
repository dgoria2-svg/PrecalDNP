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
 * Definición (TU regla):
 * - INNER siempre se detecta desde un seed (centro) hacia afuera.
 * - Si el edge cae dentro de máscara => NO VALE (se salta y se sigue buscando).
 * - Bottom arc debe ser continuo (no saltos grandes).
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

    private fun d(msg: String) { if (DBG) Log.d(TAG, msg) }

    // ✅ 0 = permitido (negro). !=0 = veto
    private fun isMasked(maskU8: ByteArray?, idx: Int): Boolean =
        maskU8 != null && ((maskU8[idx].toInt() and 0xFF) != 0)

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
        bridgeRowYpxGlobal: Float?,
        pupilGlobal: PointF?,
        debugTag: String,
        pxPerMmGuessFace: Float? = null,
        seedXpxGlobal3250: Float? = null
    ): RimDetectPack3250? {

    var stage = "ENTER"
    fun fail(code: String, msg: String): RimDetectPack3250? {
        Log.w(TAG, "RIM[$debugTag] FAIL@$stage code=$code :: $msg")
        return null
    }

    stage = "SANITY"
    val hboxMmF = filHboxMm.toFloat()
    if (!hboxMmF.isFinite() || hboxMmF <= 1e-6f) return fail("HBOX", "HBOX invalid=$hboxMmF")
    val vboxMmF = (filVboxMm ?: 0.0).toFloat()

    if (w <= 0 || h <= 0) return fail("WH", "w/h invalid $w x $h")
    if (w < MIN_ROI_W || h < MIN_ROI_H) return fail("ROI_SMALL", "ROI too small ${w}x${h}")
    if (edgesU8.size != w * h) return fail(
        "EDGE_SIZE",
        "edgesU8.size=${edgesU8.size} != w*h=${w * h}"
    )
    if (dirU8 != null && dirU8.size != w * h) return fail(
        "DIR_SIZE",
        "dirU8.size=${dirU8.size} != w*h=${w * h}"
    )
    if (maskU8 != null && maskU8.size != w * h) return fail(
        "MASK_SIZE",
        "maskU8.size=${maskU8.size} != w*h=${w * h}"
    )

    stage = "EDGE_DENSITY"
    var nnz = 0
    for (i in edgesU8.indices) if ((edgesU8[i].toInt() and 0xFF) != 0) nnz++
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
    val topMinAllowedY0 = if (browLocal == null) 0 else {
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

    d("RIM[$debugTag] probeRaw=$probeYLocalRaw probe=$probeYLocal topMin0=$topMinAllowedY0 topMin=$topMinAllowedY browLocal=${browLocal ?: -1}")

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
            if (n > 0) Log.d(
                TAG,
                "MASK[$debugTag] n=$n frac=${f3(frac)} bbox=($minX,$minY)-($maxX,$maxY)"
            )
            else Log.d(TAG, "MASK[$debugTag] n=0 frac=${f3(frac)} bbox=none")
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
        val approx = (w.toFloat() / hboxMmF) * 0.82f
        approx.coerceIn(3.5f, 8.0f)
    }
    if (!pxGuessBase.isFinite() || pxGuessBase <= 0f) return fail(
        "PX_GUESS",
        "pxGuessBase invalid=$pxGuessBase"
    )

    // ---- seed anchor local (opcional) ----
    val seedAnchorLocal: Int? = seedXpxGlobal3250?.let { sxG ->
        (sxG - roiLeftG).roundToInt().coerceIn(0, w - 1)
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
    var bestWallsY = -1
    var bestSeedX = -1
    var bestScale = 1.0f
    var bestBottomPoly: List<Pair<Int, Int>> = emptyList()

    for (scale in scales) {
        val pxGuessThis = (pxGuessBase * scale).coerceIn(3.5f, 8.0f)
        val expWpx = (hboxMmF * pxGuessThis).coerceAtLeast(80f)
        val gapPx = max(18f, 0.10f * expWpx)

        val seedExpected = (midLocalX + sideSign * (gapPx + 0.50f * expWpx))
        val seedXLocal =
            (seedAnchorLocal?.toFloat() ?: seedExpected).roundToInt().coerceIn(0, w - 1)

        val expHpxGuess = if (vboxMmF > 1e-3f) (vboxMmF * pxGuessThis) else (0.72f * expWpx)
        val yWalls = (probeYLocal + 0.35f * expHpxGuess).roundToInt().coerceIn(0, h - 1)

        val maxDist = (0.85f * expWpx).roundToInt().coerceAtLeast(60).coerceAtMost(w - 1)
        val distStart = max(LR_MIN_DIST_PX, (0.10f * expWpx).roundToInt())

        if (DBG) {
            val seedIdx = probeYLocal * w + seedXLocal
            val seedMask =
                if (maskU8 != null && maskU8.size == w * h) (maskU8[seedIdx].toInt() and 0xFF) else -1

            val wallsIdx = yWalls * w + seedXLocal
            val wallsMask =
                if (maskU8 != null && maskU8.size == w * h) (maskU8[wallsIdx].toInt() and 0xFF) else -1

            val x0 = (seedXLocal - 120).coerceIn(0, w - 1)
            val x1 = (seedXLocal + 120).coerceIn(0, w - 1)
            val y0m = (yWalls - BAND_HALF_PX).coerceIn(0, h - 1)
            val y1m = (yWalls + BAND_HALF_PX).coerceIn(0, h - 1)
            var mN = 0
            var mT = 0
            if (maskU8 != null && maskU8.size == w * h) {
                for (yy in y0m..y1m) {
                    val base = yy * w
                    for (xx in x0..x1) {
                        mT++
                        val idx = base + xx
                        if ((maskU8[idx].toInt() and 0xFF) != 0) mN++
                    }
                }
            }
            val maskWinFrac = if (mT > 0) (mN.toFloat() / mT.toFloat()) else 0f

            Log.d(
                TAG,
                "RIMDBG[$debugTag] S=${f2(scale)} seed=($seedXLocal,$probeYLocal) yWalls=$yWalls " +
                        "seedExpX=${f1(seedExpected)} midLocalX=${f1(midLocalX)} side=${f1(sideSign)} " +
                        "expW=${f1(expWpx)} gap=${f1(gapPx)} expHGuess=${f1(expHpxGuess)} pxG=${
                            f2(
                                pxGuessThis
                            )
                        } " +
                        "distStart=$distStart maxDist=$maxDist seedMask=$seedMask wallsMask=$wallsMask maskWinFrac=${
                            f3(
                                maskWinFrac
                            )
                        }"
            )
        }

        // ==========================================================
        // LR: medir en PROBE (o salir de máscara) y NO hard-kill por máscara
        // ==========================================================
        val yLR0 = dropOutOfMaskDown3250(maskU8, w, h, seedXLocal, probeYLocal)
        val lr0 = findInnerWallsAtYSkipMask3250(
            edgesU8 = edgesU8,
            maskU8 = maskU8,
            w = w,
            h = h,
            seedX = seedXLocal,
            yRef = yLR0,
            bandHalf = BAND_HALF_PX,
            minDist = distStart,
            maxDist = maxDist
        )

        val yLR1 = if (lr0 == null) dropOutOfMaskDown3250(maskU8, w, h, seedXLocal, yWalls) else -1
        val lr1 = if (lr0 == null) findInnerWallsAtYSkipMask3250(
            edgesU8 = edgesU8,
            maskU8 = maskU8,
            w = w,
            h = h,
            seedX = seedXLocal,
            yRef = yLR1,
            bandHalf = BAND_HALF_PX,
            minDist = distStart,
            maxDist = maxDist
        ) else null

        val lr = lr0 ?: lr1
        val yLRused = if (lr0 != null) yLR0 else yLR1

        if (lr == null) {
            if (DBG) Log.d(
                TAG,
                "RIMDBG[$debugTag] S=${f2(scale)} LR_FAIL yLRused=$yLRused seedX=$seedXLocal (probe=$probeYLocal yWalls=$yWalls)"
            )
            continue
        }

        val a = lr.first
        val b = lr.second
        val innerW = (b - a)

        if (DBG) {
            val y0h = (yLRused - BAND_HALF_PX).coerceIn(0, h - 1)
            val y1h = (yLRused + BAND_HALF_PX).coerceIn(0, h - 1)

            var hitsL = 0
            for (yy in y0h..y1h) {
                val idx = yy * w + a
                if ((edgesU8[idx].toInt() and 0xFF) != 0 && !isMasked(maskU8, idx)) {
                    val ddir = if (dirU8 == null) 255 else (dirU8[idx].toInt() and 0xFF)
                    if (isVertEdgeDir(ddir)) hitsL++
                }
            }
            var hitsR = 0
            for (yy in y0h..y1h) {
                val idx = yy * w + b
                if ((edgesU8[idx].toInt() and 0xFF) != 0 && !isMasked(maskU8, idx)) {
                    val ddir = if (dirU8 == null) 255 else (dirU8[idx].toInt() and 0xFF)
                    if (isVertEdgeDir(ddir)) hitsR++
                }
            }

            val ratioWdbg = innerW.toFloat() / expWpx
            Log.d(
                TAG,
                "RIMDBG[$debugTag] S=${f2(scale)} LR_OK a=$a b=$b innerW=$innerW ratioW=${
                    f3(
                        ratioWdbg
                    )
                } yLR=$yLRused " +
                        "dL=${abs(a - seedXLocal)} hitsL=$hitsL dR=${abs(b - seedXLocal)} hitsR=$hitsR"
            )
        }

        if (innerW < 60) {
            if (DBG) Log.d(TAG, "RIMDBG[$debugTag] S=${f2(scale)} DROP innerW<60 innerW=$innerW")
            continue
        }

        // ratioW: NO cortar candidates -> penalidad (evita NO_CAND)
        val ratioW = innerW.toFloat() / expWpx
        val ratioPenalty =
            if (ratioW !in W_RATIO_MIN..W_RATIO_MAX) {
                if (DBG) Log.d(
                    TAG,
                    "RIMDBG[$debugTag] S=${f2(scale)} RATIO_PEN ratioW=${f3(ratioW)} expW=${
                        f1(expWpx)
                    } innerW=$innerW"
                )
                (1f - 1.1f * abs(1f - ratioW)).coerceIn(0.15f, 0.95f)
            } else 1f

        val pxPerMmX = (innerW.toFloat() / hboxMmF).takeIf { it.isFinite() && it > 0f } ?: run {
            if (DBG) Log.d(
                TAG,
                "RIMDBG[$debugTag] S=${f2(scale)} DROP pxPerMmX invalid innerW=$innerW hbox=$hboxMmF"
            )
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
            ySeed = probeYLocal + 2,
            yMax = (h - 1),
            topMinAllowedY = topMinAllowedY,
            stepX = 4,
            minHits = 28,
            minCoverage = 0.50f,
            contJumpPx = CONT_JUMP_PX
        )

        if (bottomPick == null) {
            if (DBG) Log.d(
                TAG,
                "RIMDBG[$debugTag] S=${f2(scale)} BOTTOM_FAIL a=$a b=$b ySeed=${probeYLocal + 2} topMin=$topMinAllowedY"
            )
            continue
        } else {
            if (DBG) Log.d(
                TAG,
                "RIMDBG[$debugTag] S=${f2(scale)} BOTTOM_OK yMed=${bottomPick.yMed} hits=${bottomPick.poly.size} " +
                        "cov=${f3(bottomPick.coverage)} cont=${f3(bottomPick.continuity)}"
            )
        }

        val bottomY = bottomPick.yMed

        val yCap = (probeYLocal - TOP_SEARCH_PAD).coerceIn(0, h - 1)
        val expHpx = if (vboxMmF > 1e-3f) (vboxMmF * pxPerMmX) else Float.NaN
        val topY = if (expHpx.isFinite() && expHpx > 60f) {
            (bottomY - expHpx).roundToInt().coerceIn(topMinAllowedY, yCap)
        } else {
            (topMinAllowedY + 2).coerceAtMost(yCap)
        }

        val innerH = (bottomY - topY).coerceAtLeast(1)
        val minH0 = max(70, (h * 0.18f).roundToInt())
        val maxH0 = (h * 0.92f).roundToInt()
        if (innerH !in minH0..maxH0) {
            if (DBG) Log.d(
                TAG,
                "RIMDBG[$debugTag] S=${f2(scale)} DROP innerH=$innerH range=[$minH0..$maxH0] topY=$topY bottomY=$bottomY"
            )
            continue
        }

        var conf = 1.0f
        conf += 0.25f * (bottomPick.coverage - 0.55f)
        conf += 0.15f * (bottomPick.continuity - 0.65f)
        conf *= ratioPenalty
        conf = conf.coerceIn(0f, 1f)

        if (conf > bestConf) {
            bestConf = conf
            bestLeft = a
            bestRight = b
            bestTop = topY
            bestBottom = bottomY
            bestScale = scale
            bestBottomPoly = bottomPick.poly

            // ✅ anchors: guardar el y donde MEDIMOS inner (yLRused), no yWalls
            bestWallsY = yLRused
            bestSeedX = seedXLocal
        }
    }

    stage = "BEST_FINAL"
    if (bestConf < 0f) return fail("NO_CAND", "no candidate survived")
    if (bestConf < OK_CONF_MIN) return fail("LOW_CONF", "bestConf=${f3(bestConf)} < $OK_CONF_MIN")
    if (bestLeft < 0 || bestRight < 0 || bestBottom < 0 || bestTop < 0) return fail(
        "BEST_INV",
        "best coords invalid"
    )
    if (bestWallsY < 0 || bestSeedX < 0) return fail(
        "BEST_INV2",
        "bestWallsY/bestSeedX invalid walls=$bestWallsY seed=$bestSeedX"
    )

    val innerWpx = (bestRight - bestLeft).toFloat().coerceAtLeast(1f)
    val pxPerMmXFinal = (innerWpx / hboxMmF).takeIf { it.isFinite() && it > 0f } ?: return fail(
        "PXMM",
        "pxPerMmX invalid"
    )

    d(
        "RIM[$debugTag] BEST conf=${f3(bestConf)} " +
                "L/R=$bestLeft/$bestRight W=${f1(innerWpx)} pxPerMmX=${f3(pxPerMmXFinal)} " +
                "T/B=$bestTop/$bestBottom scale=${f2(bestScale)} " +
                "wallsY=$bestWallsY seedX=$bestSeedX"
    )
    val nasalG  = (if (nasalAtLeft) bestLeft.toFloat()  else bestRight.toFloat()) + roiLeftG
    val templeG = (if (nasalAtLeft) bestRight.toFloat() else bestLeft.toFloat())  + roiLeftG

// innerLeft/Right = SOLO min/max (sin semántica)
    val innerL = min(nasalG, templeG)
    val innerR = max(nasalG, templeG)

    val bottomPolyGlobal = if (bestBottomPoly.isNotEmpty()) {
        bestBottomPoly
            .map { (x, y) -> PointF(x.toFloat() + roiLeftG, y.toFloat() + roiTopG) }
            .sortedBy { it.x }
    } else null

    val wallsYGlobal = bestWallsY.toFloat() + roiTopG
    val seedXGlobal  = bestSeedX.toFloat()  + roiLeftG

    val res = RimDetectionResult(
        ok = true,
        confidence = bestConf,
        roiPx = RectF(roiLeftG, roiTopG, roiLeftG + w.toFloat(), roiTopG + h.toFloat()),

        probeYpx  = probeYLocal.toFloat() + roiTopG,
        topYpx    = bestTop.toFloat()     + roiTopG,
        bottomYpx = bestBottom.toFloat()  + roiTopG,

        innerLeftXpx  = innerL,
        innerRightXpx = innerR,

        // ✅ semántica REAL
        nasalInnerXpx  = nasalG,
        templeInnerXpx = templeG,

        innerWidthPx = (innerR - innerL),
        heightPx     = (bestBottom - bestTop).toFloat(),

        // ✅ anchors
        wallsYpx = wallsYGlobal,
        seedXpx  = seedXGlobal,

        topPolylinePx = null,
        bottomPolylinePx = bottomPolyGlobal,

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
        if (targetEdgesU8.size != w * h) return fail("edgesU8.size=${targetEdgesU8.size} != w*h=${w * h}")

        val roiLeft = targetRoiGlobal.left
        val roiTop = targetRoiGlobal.top
        val roiRight = roiLeft + (w - 1).toFloat()
        val roiBottom = roiTop + (h - 1).toFloat()

        fun mirrorX(x: Float): Float = 2f * midlineXpx - x
        fun clampX(x: Float): Float = x.coerceIn(roiLeft + 1f, roiRight - 1f)
        fun clampY(y: Float): Float = y.coerceIn(roiTop + 1f, roiBottom - 1f)

// ---------- inner endpoints (NO left/right semantics) ----------
        // ---------- mirror SEMANTIC endpoints ----------
        var nasal = clampX(mirrorX(primary.nasalInnerXpx))
        var temple = clampX(mirrorX(primary.templeInnerXpx))

// re-clasificación por seguridad (por clamp/ROI)
        if (abs(temple - midlineXpx) < abs(nasal - midlineXpx)) {
            val tmp = nasal; nasal = temple; temple = tmp
        }

// innerLeft/Right = SOLO min/max para ancho/debug
        val innerL = min(nasal, temple)
        val innerR = max(nasal, temple)

        val innerW = (innerR - innerL)
        if (innerW < 60f) return fail("inner width too small after mirror: $innerW")

// nasal/temple (por cercanía a midline) — usando endpoints reales

        val probeY = clampY(primary.probeYpx)
        val topY = clampY(primary.topYpx)
        val bottomY = clampY(primary.bottomYpx)
        val heightPx = (bottomY - topY).coerceAtLeast(1f)
        // bottom poly (mirror + clamp + orden)
        val bottomPoly = primary.bottomPolylinePx
            ?.map { p -> PointF(clampX(mirrorX(p.x)), clampY(p.y)) }
            ?.sortedBy { it.x }

        // nuevos anchors (si faltan, fallback limpio)
        val wallsY = clampY(primary.wallsYpx)
        val seedX = clampX(mirrorX(primary.seedXpx))

        val conf = (primary.confidence * confScale).coerceIn(0f, 0.99f)
        Log.w(
            TAG,
            "RIM[$debugTag] MIRROR_FROM_$srcTag conf=${f3(conf)} " +
                    "innerL/R=${f1(innerL)}/${f1(innerR)} W=${f1(innerW)} " +
                    "wallsY=${f1(wallsY)} seedX=${f1(seedX)} midX=${f1(midlineXpx)}"
        )
        val res = RimDetectionResult(
            ok = true,
            confidence = conf,
            roiPx = targetRoiGlobal, // ✅ más directo

            probeYpx = probeY,
            topYpx = topY,
            bottomYpx = bottomY,

            // innerLeft/Right = SOLO min/max
            innerLeftXpx = innerL,
            innerRightXpx = innerR,

            // ✅ semántica real
            nasalInnerXpx = nasal,
            templeInnerXpx = temple,

            innerWidthPx = innerW,
            heightPx = heightPx,

            wallsYpx = wallsY,
            seedXpx = seedX,

            topPolylinePx = null,
            bottomPolylinePx = bottomPoly,

            outerLeftXpx = null,
            outerRightXpx = null,
            rimThicknessPx = null
        )
        return RimDetectPack3250(result = res, edges = targetEdgesU8, w = w, h = h)
    }

    private data class ArcPick(val yMed: Int, val poly: List<Pair<Int, Int>>, val coverage: Float, val continuity: Float)

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
        contJumpPx: Int
    ): ArcPick? {

        val xa = (leftX + 8).coerceIn(0, w - 1)
        val xb = (rightX - 8).coerceIn(0, w - 1)
        if (xb - xa < 50) return null

        val yStart = ySeed.coerceIn(topMinAllowedY.coerceIn(0, h - 1), h - 1)
        val yHi = yMax.coerceIn(0, h - 1)
        if (yHi - yStart < 10) return null

        val poly = ArrayList<Pair<Int, Int>>((xb - xa) / stepX + 4)
        var prevY: Int? = null
        var contSamples = 0
        var contHits = 0

        var x = xa
        while (x <= xb) {
            val yWin0 = prevY?.let { (it - contJumpPx).coerceIn(yStart, yHi) } ?: yStart
            val yWin1 = prevY?.let { (it + contJumpPx).coerceIn(yStart, yHi) } ?: yHi

            var bestY = findFirstBottomEdge(edgesU8, dirU8, maskU8, w, x, yWin0, yWin1)
            if (bestY < 0 && prevY != null) {
                val y2 = findFirstBottomEdge(edgesU8, dirU8, maskU8, w, x, yStart, yHi)
                if (y2 >= 0 && abs(y2 - prevY!!) <= contJumpPx * 2) bestY = y2
            }

            if (bestY >= 0) {
                poly.add(x to bestY)
                if (prevY != null) {
                    contSamples++
                    if (abs(bestY - prevY) <= contJumpPx) contHits++
                }
                prevY = bestY
            }
            x += stepX
        }

        val hits = poly.size
        val samples = ((xb - xa) / stepX) + 1
        if (hits < minHits) return null

        val coverage = hits.toFloat() / samples.toFloat().coerceAtLeast(1f)
        if (coverage < minCoverage) return null

        val yMed = medianInt(poly.map { it.second })
        val continuity = if (contSamples > 0) contHits.toFloat() / contSamples.toFloat() else 0f
        return ArcPick(yMed = yMed, poly = poly, coverage = coverage, continuity = continuity)
    }

    // ✅ “adentro→afuera” hacia abajo: primer edge válido (no masked) en [y0..y1]
    private fun findFirstBottomEdge(
        edgesU8: ByteArray,
        dirU8: ByteArray?,
        maskU8: ByteArray?,
        w: Int,
        x: Int,
        y0: Int,
        y1: Int
    ): Int {
        val lo = min(y0, y1)
        val hi = max(y0, y1)
        for (y in lo..hi) {
            val idx = y * w + x
            val e = edgesU8[idx].toInt() and 0xFF
            if (e == 0) continue
            if (isMasked(maskU8, idx)) continue
            val d = if (dirU8 == null) 255 else (dirU8[idx].toInt() and 0xFF)
            if (!isHorzEdgeDir(d)) continue
            return y
        }
        return -1
    }

    private fun buildScales(minS: Float, maxS: Float, step: Float): FloatArray {
        if (!minS.isFinite() || !maxS.isFinite() || !step.isFinite()) return floatArrayOf()
        if (step <= 0f) return floatArrayOf()
        val out = ArrayList<Float>(32)
        var s = minS
        while (s <= maxS + 1e-6f) { out += s; s += step }
        return out.toFloatArray()
    }

    private fun clampInt(v: Int, lo: Int, hi: Int): Int = max(lo, min(hi, v))
    private fun clampInRange(v: Int, a: Int, b: Int): Int = v.coerceIn(min(a, b), max(a, b))

    private fun medianInt(list: List<Int>): Int {
        val a = list.toIntArray()
        a.sort()
        return a[a.size / 2]
    }
    private fun isMaskedPx3250(maskU8: ByteArray?, idx: Int): Boolean {
        return maskU8 != null && (maskU8[idx].toInt() and 0xFF) != 0
    }

    /** Raycast 4-dir (dx,dy enteros). La máscara NO corta: solo salta píxeles. */
    private fun scanFirstEdgeSkipMask3250(
        edgesU8: ByteArray,
        maskU8: ByteArray?,
        w: Int,
        h: Int,
        x0: Int,
        y0: Int,
        dx: Int,
        dy: Int,
        minStep: Int,
        maxStep: Int
    ): Pair<Int, Int>? {
        var x = x0
        var y = y0
        var step = 0
        while (step <= maxStep) {
            if (step >= minStep) {
                if (x !in 0 until w || y !in 0 until h) return null
                val idx = y * w + x

                // ✅ máscara = “transparent hole” (NO wall)
                if (!isMaskedPx3250(maskU8, idx)) {
                    if ((edgesU8[idx].toInt() and 0xFF) != 0) return x to y
                }
            }
            x += dx
            y += dy
            step++
        }
        return null
    }

    /**
     * Busca inner L/R en una banda vertical alrededor de yRef (ej: probeY),
     * desde seedX hacia afuera, saltando máscara.
     * Devuelve mediana (robusto a ruido y curvatura).
     */
    private fun findInnerWallsAtYSkipMask3250(
        edgesU8: ByteArray,
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
            val y = (yRef + dy).coerceIn(0, h - 1)
            ys.add(y)
        }

        val leftHits = ArrayList<Int>()
        val rightHits = ArrayList<Int>()

        for (y in ys) {
            // left
            scanFirstEdgeSkipMask3250(
                edgesU8, maskU8, w, h,
                x0 = seedX, y0 = y,
                dx = -1, dy = 0,
                minStep = minDist, maxStep = maxDist
            )?.let { (x, _) -> leftHits.add(x) }

            // right
            scanFirstEdgeSkipMask3250(
                edgesU8, maskU8, w, h,
                x0 = seedX, y0 = y,
                dx = +1, dy = 0,
                minStep = minDist, maxStep = maxDist
            )?.let { (x, _) -> rightHits.add(x) }
        }

        if (leftHits.isEmpty() || rightHits.isEmpty()) return null
        leftHits.sort()
        rightHits.sort()
        val l = leftHits[leftHits.size / 2]
        val r = rightHits[rightHits.size / 2]
        if (r <= l) return null
        return l to r
    }

    /** Si el seedY cae dentro de máscara, baja hasta salir (sin hardcode de pad). */
    private fun dropOutOfMaskDown3250(
        maskU8: ByteArray?,
        w: Int,
        h: Int,
        x: Int,
        yStart: Int,
        maxDown: Int = 120
    ): Int {
        if (maskU8 == null) return yStart.coerceIn(0, h - 1)
        var y = yStart.coerceIn(0, h - 1)
        var k = 0
        while (k < maxDown && y < h - 1) {
            val idx = y * w + x.coerceIn(0, w - 1)
            if (!isMaskedPx3250(maskU8, idx)) break
            y++
            k++
        }
        return y
    }
    private fun f1(x: Float): String = String.format(Locale.US, "%.1f", x)
    private fun f2(x: Float): String = String.format(Locale.US, "%.2f", x)
    private fun f3(x: Float): String = String.format(Locale.US, "%.3f", x)
}
