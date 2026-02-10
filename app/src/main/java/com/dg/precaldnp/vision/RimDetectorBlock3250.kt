@file:Suppress("SameParameterValue")

package com.dg.precaldnp.vision

import android.graphics.PointF
import android.graphics.RectF
import android.util.Log
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * RimDetectorBlock3250 (EDGE-IN)
 *
 * ✅ Este detector YA NO construye edges.
 * ✅ Recibe un edgeMap binario ROI-local (0/255) “DIVINE”.
 * ✅ Opcionalmente recibe dirU8 cuantizado (0..3) para filtrar orientaciones.
 * ✅ Devuelve RimDetectPack3250 con el MISMO edgesU8 para que ArcFit consuma la misma verdad.
 *
 * Contrato:
 * - edgesU8.size == w*h
 * - edgesU8 binario (0 o 255) preferentemente
 * - coords del detector son ROI-local; el resultado se devuelve en coords GLOBAL usando roiGlobal.left/top
 */
object RimDetectorBlock3250 {

    private const val TAG = "RimDetectorBlock3250"

    private const val MIN_ROI_W = 120
    private const val MIN_ROI_H = 90

    private const val BAND_HALF_PX = 14
    private const val TOP_SEARCH_PAD = 10
    private const val BOTTOM_SEARCH_PAD = 20

    private const val LR_TOL_FRAC = 0.32f
    private const val LR_TOL_MIN_PX = 30
    private const val LR_TOL_MAX_PX = 180

    private const val PEAK_THR_FRAC = 0.22f
    private const val LINE_THR_FRAC = 0.22f

    private const val OK_CONF_MIN = 0.55f

    private const val SCALE_MIN = 0.75f
    private const val SCALE_MAX = 1.20f
    private const val SCALE_STEP = 0.05f

    private const val BOTTOM_POLY_STEP_X = 5
    private const val BOTTOM_POLY_BAND_Y = 10

    private const val BORDER_GUARD_PX = 16

    private const val W_RATIO_MIN = 0.90f
    private const val W_RATIO_MAX = 1.25f

    private const val RIM_T_MM_MIN = 1.0f
    private const val RIM_T_MM_MAX = 9.0f
    private const val RIM_T_MM_FALLBACK = 2.5f

    fun detectRim(
        edgesU8: ByteArray,
        dirU8: ByteArray? = null, // puede ser null
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

        fun fail(msg: String): RimDetectPack3250? {
            Log.w(TAG, "RIM[$debugTag] FAIL: $msg")
            return null
        }

        // ---- sanity ----
        val hboxMmF = filHboxMm.toFloat()
        if (!hboxMmF.isFinite() || hboxMmF <= 1e-6f) return fail("HBOX invalid=$hboxMmF")
        val vboxMmF = (filVboxMm ?: 0.0).toFloat()

        if (w <= 0 || h <= 0) return fail("w/h invalid $w x $h")
        if (w < MIN_ROI_W || h < MIN_ROI_H) return fail("ROI too small ${w}x${h}")
        if (edgesU8.size != w * h) return fail("edgesU8.size=${edgesU8.size} != w*h=${w*h}")
        if (dirU8 != null && dirU8.size != w * h) return fail("dirU8.size=${dirU8.size} != w*h=${w*h}")

        // ---- edge density contract ----
        run {
            var nnz = 0
            for (b in edgesU8) if ((b.toInt() and 0xFF) != 0) nnz++
            val dens = nnz.toFloat() / (w * h).toFloat().coerceAtLeast(1f)
            Log.d(TAG, "EDGE[$debugTag] nnz=$nnz/${w*h} dens=${"%.4f".format(dens)} (EDGE-IN)")
            if (nnz < 50) return fail("edgeMap too sparse (nnz<50)")
        }

        // ---- ROI offsets ----
        val roiLeftG = roiGlobal.left
        val roiTopG = roiGlobal.top
        val roiRightG = roiLeftG + (w - 1).toFloat()

        // ---- PROBE + ANTI-CEJA (solo geometría) ----
        val probeYLocalRaw: Int = run {
            val y0 = (bridgeRowYpxGlobal ?: pupilGlobal?.y ?: roiGlobal.centerY())
            (y0 - roiTopG).roundToInt()
        }.coerceIn(0, h - 1)

        val browLocal: Int? = browBottomYpx?.let { (it - roiTopG).roundToInt() }

        val topMinAllowedY0: Int =
            if (browLocal == null) 0
            else {
                val pad = clampInt3250((h * 0.02f).roundToInt(), 6, 18)
                (browLocal.coerceIn(0, h - 1) + pad).coerceIn(0, h - 1)
            }

        // anti-ceja fuerte: nunca permitir topMin debajo del probe
        val topMinAllowedY: Int = min(
            topMinAllowedY0,
            (probeYLocalRaw - 10).coerceIn(0, h - 1)
        ).coerceIn(0, h - 1)

        val probeYLocal: Int = run {
            val lo = (topMinAllowedY + 2).coerceIn(0, h - 1)
            val hi = (h - 3).coerceIn(0, h - 1)
            clampInRange3250(probeYLocalRaw, lo, hi)
        }

        Log.d(
            TAG,
            "RIM[$debugTag] browLocal=${browLocal ?: -1} topMin0=$topMinAllowedY0 topMin=$topMinAllowedY probeRaw=$probeYLocalRaw probe=$probeYLocal"
        )

        // ---- midline local + nasal side ----
        val midlineLocalX = (midlineXpx - roiLeftG).coerceIn(0f, (w - 1).toFloat())

        val dL = abs(roiLeftG - midlineXpx)
        val dR = abs(roiRightG - midlineXpx)
        val nasalAtLeft = dL <= dR
        val sideSign = if (nasalAtLeft) +1f else -1f

        Log.d(
            TAG,
            "RIM[$debugTag] midlineG=${"%.1f".format(midlineXpx)} midLocal=${"%.1f".format(midlineLocalX)} nasalAtLeft=$nasalAtLeft sideSign=$sideSign"
        )

        // ---- pxGuessBase (expectativa geométrica) ----
        val pxGuessBase: Float = pxPerMmGuessFace ?: run {
            val approx = (w.toFloat() / hboxMmF) * 0.82f
            approx.coerceIn(4.0f, 7.0f)
        }
        if (!pxGuessBase.isFinite() || pxGuessBase <= 0f) return fail("pxGuessBase invalid=$pxGuessBase")

        // ---- accessors edges + dir ----
        // Asumimos dirU8 cuantizado: 0,1,2,3 ≈ 0°,45°,90°,135°
        fun isVertDir(d: Int): Boolean = (d == 0 || d == 1 || d == 3 || d !in 0..3)
        fun isHorzDir(d: Int): Boolean = (d == 2 || d == 1 || d == 3 || d !in 0..3)

        fun eAt(x: Int, y: Int): Int {
            if (x !in 0 until w || y !in 0 until h) return 0
            return edgesU8[y * w + x].toInt() and 0xFF
        }

        fun dAt(x: Int, y: Int): Int {
            if (dirU8 == null) return 255
            if (x !in 0 until w || y !in 0 until h) return 255
            return dirU8[y * w + x].toInt() and 0xFF
        }

        val vAt: (Int, Int) -> Float = { x, y ->
            val e = eAt(x, y)
            if (e == 0) 0f
            else if (dirU8 == null) e.toFloat()
            else {
                val d = dAt(x, y)
                if (isVertDir(d)) e.toFloat() else 0f
            }
        }

        val hAt: (Int, Int) -> Float = { x, y ->
            val e = eAt(x, y)
            if (e == 0) 0f
            else if (dirU8 == null) e.toFloat()
            else {
                val d = dAt(x, y)
                if (isHorzDir(d)) e.toFloat() else 0f
            }
        }

        val seedAnchorLocal: Int? = seedXpxGlobal3250?.let { sxG ->
            (sxG - roiLeftG).roundToInt().coerceIn(0, w - 1)
        }

        // ---- rowSum (para fallback por picos / thickness refine) ----
        val (rowSumProbe, maxVProbe) = buildVRowSumAtY3250(
            w = w,
            h = h,
            yCenter = probeYLocal,
            bandHalf = BAND_HALF_PX,
            yMin = topMinAllowedY,
            vAt = vAt
        )
        if (maxVProbe <= 1e-6f) return fail("maxVProbe<=eps")
        val thrProbe = maxVProbe * PEAK_THR_FRAC

        // ---- expected positions (base) para thickness ----
        val expWpxHint = (hboxMmF * pxGuessBase).coerceAtLeast(80f)
        val gapPxHint = max(18f, 0.10f * expWpxHint)
        val seedExpectedHint = (midlineLocalX + sideSign * (gapPxHint + 0.50f * expWpxHint))
        val seedUsedBase = (seedAnchorLocal?.toFloat() ?: seedExpectedHint)
            .roundToInt()
            .coerceIn(0, w - 1)

        val cxBase = seedUsedBase.toFloat()
        val expLeftBase = cxBase - 0.5f * expWpxHint
        val expRightBase = cxBase + 0.5f * expWpxHint
        val xNasalExpectedBase = if (nasalAtLeft) expLeftBase else expRightBase
        val xTempleExpectedBase = if (nasalAtLeft) expRightBase else expLeftBase

        val tolBase = clampInt3250(
            (expWpxHint * LR_TOL_FRAC).roundToInt(),
            LR_TOL_MIN_PX,
            LR_TOL_MAX_PX
        )

        // ---- thickness estimation (usa edges verticales) ----
        val tPxEst: Float = estimateRimThicknessPx3250(
            w = w,
            h = h,
            topMinAllowedY = topMinAllowedY,
            probeY = probeYLocal,
            vAt = vAt,
            xNasalExpected = xNasalExpectedBase,
            xTempleExpected = xTempleExpectedBase,
            nasalAtLeft = nasalAtLeft,
            tolPx = tolBase,
            pxGuessForMm = pxGuessBase
        )
        Log.d(TAG, "RIM[$debugTag] tPxEst=${"%.1f".format(tPxEst)} (pxGuessBase=${"%.2f".format(pxGuessBase)})")

        val scales = buildScales3250(SCALE_MIN, SCALE_MAX, SCALE_STEP)
        if (scales.isEmpty()) return fail("no scales")

        var bestConf = -1f
        var bestLeft = -1
        var bestRight = -1
        var bestTop = -1
        var bestBottom = -1
        var bestScale = 1.0f
        var bestExpW = 0f
        var bestSeedX = -1
        var bestCovB = 0f
        var bestCovT = 0f
        var bestPolyCov = 0f
        var bestPoly: List<Pair<Int, Int>> = emptyList()
        var bestTilt = 0f

        var bestOuterLeft: Int? = null
        var bestOuterRight: Int? = null
        var bestTPxMed: Float = Float.NaN
        var bestMarksInner: List<Pair<Int, Int>> = emptyList()
        var bestMarksOuter: List<Pair<Int, Int>> = emptyList()

        for (scale in scales) {
            if (!scale.isFinite() || scale <= 0f) continue

            val pxGuessThis = (pxGuessBase * scale).coerceIn(3.5f, 8.0f)
            val expWpx = (hboxMmF * pxGuessThis).coerceAtLeast(80f)
            if (!expWpx.isFinite()) continue

            val gapPx = max(18f, 0.10f * expWpx)
            val seedExpected = (midlineLocalX + sideSign * (gapPx + 0.50f * expWpx))
            val seedXLocalUsed: Int = (seedAnchorLocal?.toFloat() ?: seedExpected)
                .roundToInt()
                .coerceIn(0, w - 1)

            // 1) Preferido: scan band desde seed (inner/outer + marks)
            val scan = scanBandInnerOuterFromSeed3250(
                w = w,
                h = h,
                probeY = probeYLocal,
                bandHalf = BAND_HALF_PX,
                topMinAllowedY = topMinAllowedY,
                seedX = seedXLocalUsed,
                nasalAtLeft = nasalAtLeft,
                pxGuessThis = pxGuessThis,
                vAt = vAt
            )

            // 2) Fallback: peaks en rowSum + refine thickness
            val lrFromPeaks: Pair<Int, Int>? = if (scan == null) {
                val lrRaw = findLeftRightAtProbeFromRowSum3250(
                    w = w,
                    rowSum = rowSumProbe,
                    thr = thrProbe,
                    expWpx = expWpx,
                    xSeedLocal = seedXLocalUsed,
                    midLocal = midlineLocalX,
                    sideSign = sideSign
                ) ?: null

                lrRaw?.let {
                    refineInnerEdgesWithThickness3250(
                        w = w,
                        rowSum = rowSumProbe,
                        thr = thrProbe,
                        expWpx = expWpx,
                        leftRaw = it.first,
                        rightRaw = it.second,
                        tPx = tPxEst
                    )
                }
            } else null

            val leftX: Int
            val rightX: Int

            if (scan != null) {
                leftX = scan.innerLeft
                rightX = scan.innerRight
            } else if (lrFromPeaks != null) {
                leftX = lrFromPeaks.first
                rightX = lrFromPeaks.second
            } else {
                continue
            }

            val innerW = rightX - leftX
            if (innerW < 60) continue

            val ratioW = innerW.toFloat() / expWpx
            if (ratioW !in W_RATIO_MIN..W_RATIO_MAX) continue

            // 1) BOTTOM (ancho manda) - horizontal edges
            val bottomY = findHorizontalLineY3250(
                w = w,
                h = h,
                yFrom = min(h - 1, probeYLocal + BOTTOM_SEARCH_PAD),
                yTo = h - 1,
                xFrom = leftX + 8,
                xTo = rightX - 8,
                scoreAt = hAt
            ) ?: continue

            // px/mmX oficial por ancho
            val pxPerMmXcand = (innerW.toFloat() / hboxMmF).takeIf { it.isFinite() && it > 0f } ?: continue

            // altura esperada desde ancho (solo para top + validación suave)
            val expHpxFromWidth: Float? =
                if (vboxMmF > 1e-3f) (vboxMmF * pxPerMmXcand) else null

            // 2) TOP: preferir bottom - expH; fallback scan si no hay vbox
            val topY: Int = run {
                val yCap = (probeYLocal - TOP_SEARCH_PAD).coerceIn(0, h - 1)

                if (expHpxFromWidth != null && expHpxFromWidth.isFinite() && expHpxFromWidth > 60f) {
                    val yPred = (bottomY - expHpxFromWidth).roundToInt()
                    yPred.coerceIn(topMinAllowedY, yCap)
                } else {
                    findHorizontalLineY3250(
                        w = w,
                        h = h,
                        yFrom = topMinAllowedY,
                        yTo = yCap,
                        xFrom = leftX + 8,
                        xTo = rightX - 8,
                        scoreAt = hAt
                    ) ?: topMinAllowedY
                }
            }

            val innerH = (bottomY - topY).coerceAtLeast(1)
            val minH0 = max(70, (h * 0.18f).roundToInt())
            val maxH0 = (h * 0.92f).roundToInt()
            if (innerH !in minH0..maxH0) continue

            // validación altura SOLO para outliers
            var errHRel = 0f
            if (expHpxFromWidth != null && expHpxFromWidth.isFinite() && expHpxFromWidth > 60f) {
                errHRel = (abs(innerH - expHpxFromWidth) / expHpxFromWidth).coerceIn(0f, 9f)
                if (errHRel > 0.65f) continue
            }

            val covB = lineCoverage3250(w, h, bottomY, leftX + 6, rightX - 6, hAt)
            val covT = lineCoverage3250(w, h, topY, leftX + 6, rightX - 6, hAt)

            val errW = abs(innerW - expWpx) / expWpx

            val polyTmp = buildBottomPolylineLocal3250(
                w = w,
                h = h,
                leftX = leftX,
                rightX = rightX,
                bottomY = bottomY,
                topMinAllowedY = topMinAllowedY,
                probeY = probeYLocal,
                hAt = hAt
            )

            val polyCov = if (polyTmp.isNotEmpty()) computeBottomPolylineCoverage3250(polyTmp, hAt) else 0f
            val tilt = if (polyTmp.isNotEmpty()) fitPolylineTiltDeg3250(polyTmp) else 0f

            // CONF: ancho manda. altura penaliza suave.
            var conf = 1.0f
            conf -= (1.35f * errW)

            if (errHRel > 0f) {
                val h01 = (errHRel / 0.35f).coerceIn(0f, 1f)
                conf -= (0.35f * h01)
            }

            conf += (0.30f * (polyCov - 0.55f))
            conf += (0.10f * (covB - 0.60f))
            conf += (0.02f * (covT - 0.60f))

            if (polyCov < 0.22f) conf *= 0.55f
            conf = conf.coerceIn(0f, 1f)

            if (conf > bestConf) {
                bestConf = conf
                bestLeft = leftX
                bestRight = rightX
                bestTop = topY
                bestBottom = bottomY
                bestScale = scale
                bestExpW = expWpx
                bestSeedX = seedXLocalUsed
                bestCovB = covB
                bestCovT = covT
                bestPolyCov = polyCov
                bestPoly = polyTmp
                bestTilt = tilt

                bestOuterLeft = scan?.outerLeft
                bestOuterRight = scan?.outerRight
                bestTPxMed = scan?.tPxMed ?: Float.NaN
                bestMarksInner = scan?.marksInner ?: emptyList()
                bestMarksOuter = scan?.marksOuter ?: emptyList()
            }
        }

        if (bestConf < 0f || bestLeft < 0 || bestRight < 0 || bestBottom < 0) {
            return fail("no best candidate")
        }

        val innerWpx = (bestRight - bestLeft).toFloat().coerceAtLeast(1f)
        val pxPerMmX = (innerWpx / hboxMmF).takeIf { it.isFinite() && it > 0f } ?: return fail("pxPerMmX invalid")

        Log.d(
            TAG,
            "RIM[$debugTag] BEST conf=${"%.3f".format(bestConf)} " +
                    "L=$bestLeft R=$bestRight W=${"%.1f".format(innerWpx)} expW=${"%.1f".format(bestExpW)} " +
                    "top=$bestTop bot=$bestBottom covB=${"%.2f".format(bestCovB)} polyCov=${"%.2f".format(bestPolyCov)} " +
                    "pxPerMmX=${"%.3f".format(pxPerMmX)} scale=${"%.2f".format(bestScale)} seedX=$bestSeedX tilt=${"%.2f".format(bestTilt)} " +
                    "scan tPxMed=${if (bestTPxMed.isFinite()) "%.1f".format(bestTPxMed) else "NaN"} " +
                    "outerLR=${bestOuterLeft ?: -1}/${bestOuterRight ?: -1} marksIn=${bestMarksInner.size} marksOut=${bestMarksOuter.size}"
        )

        val leftLocal = bestLeft.toFloat()
        val rightLocal = bestRight.toFloat()

        val nasalLocal = if (nasalAtLeft) leftLocal else rightLocal
        val templeLocal = if (nasalAtLeft) rightLocal else leftLocal

        val bottomPolyGlobal: List<PointF>? =
            if (bestPoly.isNotEmpty()) bestPoly.map { (x, y) -> PointF(x.toFloat() + roiLeftG, y.toFloat() + roiTopG) }
            else null

        // (Opcional) si querés ver marks en debug overlay, loguealos afuera usando bestMarksInner/Outer.
        // Acá NO inventamos campos nuevos en RimDetectionResult.

        val res = RimDetectionResult(
            ok = bestConf >= OK_CONF_MIN,
            confidence = bestConf,
            roiPx = RectF(roiLeftG, roiTopG, roiLeftG + w, roiTopG + h),
            probeYpx = probeYLocal.toFloat() + roiTopG,
            topYpx = bestTop.toFloat() + roiTopG,
            bottomYpx = bestBottom.toFloat() + roiTopG,
            innerLeftXpx = leftLocal + roiLeftG,
            innerRightXpx = rightLocal + roiLeftG,
            nasalInnerXpx = nasalLocal + roiLeftG,
            templeInnerXpx = templeLocal + roiLeftG,
            heightPx = (bestBottom - bestTop).toFloat(),
            innerWidthPx = (rightLocal - leftLocal).toFloat(),
            topPolylinePx = null,
            bottomPolylinePx = bottomPolyGlobal
        )

        // 🔥 devolvemos el MISMO edgeMap que entró
        return RimDetectPack3250(
            result = res,
            edges = edgesU8,
            w = w,
            h = h
        )
    }
    /**
     * Mirror fallback: si un ojo falla y el otro dio RimDetectionResult válido,
     * generamos una estimación espejada para el ojo faltante usando midline global.
     *
     * IMPORTANTE:
     * - usa el roiGlobal/w/h DEL OJO TARGET (el que falló)
     * - devuelve el edgeMap del ojo TARGET (para ArcFit: misma verdad visual)
     * - penaliza confidence (fallback)
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
        if (targetEdgesU8.size != w * h) return fail("edgesU8.size=${targetEdgesU8.size} != w*h=${w*h}")

        val roiLeft = targetRoiGlobal.left
        val roiTop = targetRoiGlobal.top
        val roiRight = roiLeft + (w - 1).toFloat()
        val roiBottom = roiTop + (h - 1).toFloat()

        fun mirrorX(x: Float): Float = 2f * midlineXpx - x

        fun clampX(x: Float): Float =
            x.coerceIn(roiLeft + 1f, roiRight - 1f)

        fun clampY(y: Float): Float =
            y.coerceIn(roiTop + 1f, roiBottom - 1f)

        // inner L/R: espejado cruzado para que quede ordenado (L < R)
        var innerL = clampX(mirrorX(primary.innerRightXpx))
        var innerR = clampX(mirrorX(primary.innerLeftXpx))
        if (innerL > innerR) {
            val tmp = innerL; innerL = innerR; innerR = tmp
        }
        if (innerR - innerL < 60f) {
            // guardrail (si el ROI target es raro/estrecho, no inventamos)
            return fail("inner width too small after mirror: ${(innerR - innerL)}")
        }

        // nasal/temple: espejar ambos y reasignar por cercanía a midline
        val a = clampX(mirrorX(primary.nasalInnerXpx))
        val b = clampX(mirrorX(primary.templeInnerXpx))
        val nasal = if (abs(a - midlineXpx) <= abs(b - midlineXpx)) a else b
        val temple = if (nasal == a) b else a

        val bottomPoly = primary.bottomPolylinePx?.map { p ->
            PointF(clampX(mirrorX(p.x)), clampY(p.y))
        }

        val conf = (primary.confidence * confScale).coerceIn(0f, 0.99f)

        Log.w(
            TAG,
            "RIM[$debugTag] MIRROR_FROM_$srcTag conf=${"%.3f".format(conf)} " +
                    "innerL/R=${"%.1f".format(innerL)}/${"%.1f".format(innerR)} midX=${"%.1f".format(midlineXpx)}"
        )

        val res = RimDetectionResult(
            ok = true, // fallback: queremos que el pipeline siga
            confidence = conf,
            roiPx = RectF(roiLeft, roiTop, roiLeft + w, roiTop + h),
            probeYpx = clampY(primary.probeYpx),
            topYpx = clampY(primary.topYpx),
            bottomYpx = clampY(primary.bottomYpx),
            innerLeftXpx = innerL,
            innerRightXpx = innerR,
            nasalInnerXpx = nasal,
            templeInnerXpx = temple,
            heightPx = primary.heightPx,
            innerWidthPx = primary.innerWidthPx,
            topPolylinePx = null,
            bottomPolylinePx = bottomPoly
        )

        return RimDetectPack3250(
            result = res,
            edges = targetEdgesU8, // 🔥 edgeMap del ojo target (no el del primary)
            w = w,
            h = h
        )
    }


    // ==========================================================
    // Row “energía” en probe (SUMA) porque edgeMap es binario
    // ==========================================================
    private fun buildVRowSumAtY3250(
        w: Int,
        h: Int,
        yCenter: Int,
        bandHalf: Int,
        yMin: Int,
        vAt: (Int, Int) -> Float
    ): Pair<FloatArray, Float> {
        val out = FloatArray(w)
        val y0 = (yCenter - bandHalf).coerceIn(yMin.coerceIn(0, h - 1), h - 1)
        val y1 = (yCenter + bandHalf).coerceIn(yMin.coerceIn(0, h - 1), h - 1)

        var maxV = 0f
        for (x in 0 until w) {
            var s = 0f
            for (yy in y0..y1) s += vAt(x, yy)
            out[x] = s
            if (s.isFinite() && s > maxV) maxV = s
        }
        return Pair(out, maxV)
    }

    // ==========================================================
    // Left/Right desde rowSum (picos) con guardrails
    // ==========================================================
    private fun findLeftRightAtProbeFromRowSum3250(
        w: Int,
        rowSum: FloatArray,
        thr: Float,
        expWpx: Float,
        xSeedLocal: Int,
        midLocal: Float,
        sideSign: Float
    ): Pair<Int, Int>? {
        if (w <= 0 || rowSum.size != w) return null
        if (!expWpx.isFinite() || expWpx <= 1f) return null
        if (!thr.isFinite() || thr <= 0f) return null

        val dirNasal = if (sideSign >= 0f) 1 else -1
        val midX = midLocal.roundToInt().coerceIn(0, w - 1)

        val xMin = BORDER_GUARD_PX.coerceIn(0, w - 1)
        val xMax = (w - 1 - BORDER_GUARD_PX).coerceAtLeast(xMin)

        val gapPx = max(18f, 0.10f * expWpx)
        val expTempleX = (midX + dirNasal * (gapPx + expWpx)).roundToInt()
        val templeMargin = clampInt3250((0.25f * expWpx).roundToInt(), 24, 140)

        val templeStart = (expTempleX + dirNasal * templeMargin).coerceIn(xMin, xMax)

        val nasal0 = scanFirstPeakDir3250(sig = rowSum, start = midX, dir = dirNasal, stop = templeStart, thr = thr)
        val temple0 = scanFirstPeakDir3250(sig = rowSum, start = templeStart, dir = -dirNasal, stop = midX, thr = thr)

        if (nasal0 >= 0 && temple0 >= 0) {
            val a = min(nasal0, temple0)
            val b = max(nasal0, temple0)
            val wPx = (b - a).toFloat()
            val ratio = wPx / expWpx
            if (ratio in 0.85f..1.15f && (b - a) >= 60) {
                val aRef = bestPeakNearXInRange3250(
                    sig = rowSum,
                    x0 = (a - 14).coerceIn(xMin, xMax),
                    x1 = (a + 14).coerceIn(xMin, xMax),
                    thr = thr,
                    xExpected = a
                )
                val bRef = bestPeakNearXInRange3250(
                    sig = rowSum,
                    x0 = (b - 14).coerceIn(xMin, xMax),
                    x1 = (b + 14).coerceIn(xMin, xMax),
                    thr = thr,
                    xExpected = b
                )
                if (aRef >= 0 && bRef >= 0 && bRef - aRef >= 60) return Pair(aRef, bRef)
            }
        }

        // fallback seed
        val cx = xSeedLocal.coerceIn(xMin, xMax).toFloat()
        val expLeft = cx - expWpx * 0.5f
        val expRight = cx + expWpx * 0.5f

        val tol = clampInt3250(
            (expWpx * LR_TOL_FRAC).roundToInt(),
            LR_TOL_MIN_PX,
            LR_TOL_MAX_PX
        )

        val leftRange0 = (expLeft - tol).roundToInt().coerceIn(xMin, xMax)
        val leftRange1 = (expLeft + tol).roundToInt().coerceIn(xMin, xMax)
        val rightRange0 = (expRight - tol).roundToInt().coerceIn(xMin, xMax)
        val rightRange1 = (expRight + tol).roundToInt().coerceIn(xMin, xMax)

        val leftPeak = bestPeakNearXInRange3250(rowSum, leftRange0, leftRange1, thr, expLeft.roundToInt())
        val rightPeak = bestPeakNearXInRange3250(rowSum, rightRange0, rightRange1, thr, expRight.roundToInt())

        if (leftPeak < 0 || rightPeak < 0) return null
        if (rightPeak - leftPeak < 60) return null
        return Pair(leftPeak, rightPeak)
    }

    // ==========================================================
    // Refinado: elegir INNER con thickness
    // ==========================================================
    private fun refineInnerEdgesWithThickness3250(
        w: Int,
        rowSum: FloatArray,
        thr: Float,
        expWpx: Float,
        leftRaw: Int,
        rightRaw: Int,
        tPx: Float
    ): Pair<Int, Int> {
        if (w <= 0 || rowSum.size != w) return Pair(leftRaw, rightRaw)
        if (!tPx.isFinite() || tPx < 2f) return Pair(leftRaw, rightRaw)

        val xMin = BORDER_GUARD_PX.coerceIn(0, w - 1)
        val xMax = (w - 1 - BORDER_GUARD_PX).coerceAtLeast(xMin)

        val left0 = leftRaw.coerceIn(xMin, xMax)
        val right0 = rightRaw.coerceIn(xMin, xMax)
        if (right0 - left0 < 60) return Pair(left0, right0)

        val xCenter = 0.5f * (left0 + right0)
        val tol = max(6, (tPx * 0.30f).roundToInt()).coerceIn(4, 40)
        val thrSupport = thr * 0.80f

        fun supportStrengthNear(xTarget: Float): Float {
            val xt = xTarget.roundToInt()
            val a = (xt - tol).coerceIn(xMin, xMax)
            val b = (xt + tol).coerceIn(xMin, xMax)
            var m = 0f
            for (x in a..b) {
                val v = rowSum[x]
                if (v.isFinite() && v > m) m = v
            }
            return m
        }

        fun inwardDir(x: Int): Int = if (x.toFloat() < xCenter) 1 else -1

        val inL = inwardDir(left0)
        val inR = inwardDir(right0)

        val sL_in = supportStrengthNear(left0 + inL * tPx)
        val sL_out = supportStrengthNear(left0 - inL * tPx)
        val sR_in = supportStrengthNear(right0 + inR * tPx)
        val sR_out = supportStrengthNear(right0 - inR * tPx)

        data class Combo(val lShiftIn: Boolean, val rShiftIn: Boolean)
        val combos = arrayOf(
            Combo(lShiftIn = false, rShiftIn = false),
            Combo(lShiftIn = true, rShiftIn = false),
            Combo(lShiftIn = false, rShiftIn = true),
            Combo(lShiftIn = true, rShiftIn = true)
        )

        fun apply(c: Combo): Pair<Int, Int>? {
            val l = if (c.lShiftIn) (left0 + inL * tPx).roundToInt() else left0
            val r = if (c.rShiftIn) (right0 + inR * tPx).roundToInt() else right0
            val ll = l.coerceIn(xMin, xMax)
            val rr = r.coerceIn(xMin, xMax)
            if (rr - ll < 60) return null
            return Pair(ll, rr)
        }

        fun penalty(c: Combo, lr: Pair<Int, Int>): Float {
            val width = (lr.second - lr.first).toFloat()
            val errW = abs(width - expWpx) / expWpx
            var pen = errW
            if (c.lShiftIn) { if (sL_in < thrSupport) pen += 0.55f } else { if (sL_out < thrSupport) pen += 0.12f }
            if (c.rShiftIn) { if (sR_in < thrSupport) pen += 0.55f } else { if (sR_out < thrSupport) pen += 0.12f }
            return pen
        }

        var best: Pair<Int, Int>? = null
        var bestPen = Float.POSITIVE_INFINITY
        for (c in combos) {
            val lr = apply(c) ?: continue
            val p = penalty(c, lr)
            if (p < bestPen) { bestPen = p; best = lr }
        }
        return best ?: Pair(left0, right0)
    }

    // ==========================================================
    // Thickness px desde edges
    // ==========================================================
    private fun estimateRimThicknessPx3250(
        w: Int,
        h: Int,
        topMinAllowedY: Int,
        probeY: Int,
        vAt: (Int, Int) -> Float,
        xNasalExpected: Float,
        xTempleExpected: Float,
        nasalAtLeft: Boolean,
        tolPx: Int,
        pxGuessForMm: Float
    ): Float {
        val fallback = (pxGuessForMm * RIM_T_MM_FALLBACK).coerceIn(3f, 80f)
        if (w <= 0 || h <= 0) return fallback
        if (!pxGuessForMm.isFinite() || pxGuessForMm <= 0f) return fallback

        val tMinPx = (pxGuessForMm * RIM_T_MM_MIN).roundToInt().coerceIn(2, 120)
        val tMaxPx = (pxGuessForMm * RIM_T_MM_MAX).roundToInt().coerceIn(tMinPx + 2, 160)

        val dirInsideNasal = if (nasalAtLeft) +1 else -1
        val dirInsideTemple = -dirInsideNasal

        val yOffsets = intArrayOf(-8, -4, 0, +4, +8, +14, +20)
        val candidates = FloatArray(64)
        var n = 0

        fun nearestEdgeAtY(y: Int, xExpected: Float, tol: Int): Int? {
            val xMin = BORDER_GUARD_PX.coerceIn(0, w - 1)
            val xMax = (w - 1 - BORDER_GUARD_PX).coerceAtLeast(xMin)
            val xc = xExpected.roundToInt().coerceIn(xMin, xMax)
            val lo = (xc - tol).coerceIn(xMin, xMax)
            val hi = (xc + tol).coerceIn(xMin, xMax)
            var bestX: Int? = null
            var bestD = Int.MAX_VALUE
            for (x in lo..hi) {
                if (vAt(x, y) <= 0f) continue
                val d = abs(x - xc)
                if (d < bestD) { bestD = d; bestX = x }
            }
            return bestX
        }

        fun findSecondEdge(y: Int, x0: Int, dirInside: Int, tMin: Int, tMax: Int): Int? {
            val a = x0 + dirInside * tMin
            val b = x0 + dirInside * tMax
            val lo = min(a, b).coerceIn(0, w - 1)
            val hi = max(a, b).coerceIn(0, w - 1)
            return if (dirInside > 0) {
                for (x in lo..hi) if (vAt(x, y) > 0f) return x
                null
            } else {
                for (x in hi downTo lo) if (vAt(x, y) > 0f) return x
                null
            }
        }

        for (dy in yOffsets) {
            val yC = (probeY + dy).coerceIn(0, h - 1)
            if (yC < topMinAllowedY + 2) continue

            val xN0 = nearestEdgeAtY(yC, xNasalExpected, tolPx) ?: continue
            val xN1 = findSecondEdge(yC, xN0, dirInsideNasal, tMinPx, tMaxPx)
            if (xN1 != null && n < candidates.size) candidates[n++] = abs(xN1 - xN0).toFloat()

            val xT0 = nearestEdgeAtY(yC, xTempleExpected, tolPx) ?: continue
            val xT1 = findSecondEdge(yC, xT0, dirInsideTemple, tMinPx, tMaxPx)
            if (xT1 != null && n < candidates.size) candidates[n++] = abs(xT1 - xT0).toFloat()
        }

        if (n < 3) return fallback
        val arr = candidates.copyOfRange(0, n)
        arr.sort()
        return arr[n / 2].coerceIn(3f, 120f)
    }

    // ==========================================================
    // Scan peaks helpers
    // ==========================================================
    private fun scanFirstPeakDir3250(sig: FloatArray, start: Int, dir: Int, stop: Int, thr: Float): Int {
        if (sig.isEmpty()) return -1
        if (dir != 1 && dir != -1) return -1
        val n = sig.size
        var x = start.coerceIn(0, n - 1)
        val end = stop.coerceIn(0, n - 1)

        fun isPeak(i: Int): Boolean {
            val v = sig[i]
            if (!v.isFinite() || v < thr) return false
            val vL = if (i > 0) sig[i - 1] else v
            val vR = if (i < n - 1) sig[i + 1] else v
            return v >= vL && v >= vR
        }

        if (dir > 0) {
            while (x <= end) { if (isPeak(x)) return x; x++ }
        } else {
            while (x >= end) { if (isPeak(x)) return x; x-- }
        }
        return -1
    }

    private fun bestPeakNearXInRange3250(sig: FloatArray, x0: Int, x1: Int, thr: Float, xExpected: Int): Int {
        if (sig.isEmpty()) return -1
        if (!thr.isFinite()) return -1

        val lo = min(x0, x1).coerceIn(0, sig.lastIndex)
        val hi = max(x0, x1).coerceIn(0, sig.lastIndex)
        if (lo > hi) return -1

        var bestX = -1
        var bestDist = Int.MAX_VALUE
        var bestV = -1f

        for (x in lo..hi) {
            val v = sig[x]
            if (!v.isFinite() || v < thr) continue
            val vL = if (x > 0) sig[x - 1] else v
            val vR = if (x < sig.lastIndex) sig[x + 1] else v
            if (v < vL || v < vR) continue

            val d = abs(x - xExpected)
            if (d < bestDist || (d == bestDist && v > bestV)) {
                bestDist = d
                bestV = v
                bestX = x
            }
        }
        return bestX
    }

    // ==========================================================
    // Horizontal line search + coverage + bottom polyline
    // ==========================================================
    private fun findHorizontalLineY3250(
        w: Int, h: Int,
        yFrom: Int, yTo: Int,
        xFrom: Int, xTo: Int,
        scoreAt: (Int, Int) -> Float
    ): Int? {
        if (w <= 0 || h <= 0) return null

        val ya = min(yFrom, yTo).coerceIn(0, h - 1)
        val yb = max(yFrom, yTo).coerceIn(0, h - 1)
        val xa = xFrom.coerceIn(0, w - 1)
        val xb = xTo.coerceIn(0, w - 1)
        if (xb - xa < 40) return null
        if (yb < ya) return null

        var bestY = -1
        var bestS = -1f
        for (y in ya..yb) {
            var s = 0f
            for (x in xa..xb) s += scoreAt(x, y)
            val mean = s / (xb - xa + 1)
            if (mean.isFinite() && mean > bestS) {
                bestS = mean
                bestY = y
            }
        }
        return if (bestY >= 0) bestY else null
    }

    private fun lineCoverage3250(
        w: Int, h: Int,
        y: Int, xFrom: Int, xTo: Int,
        scoreAt: (Int, Int) -> Float
    ): Float {
        if (w <= 0 || h <= 0) return 0f
        val yy = y.coerceIn(0, h - 1)
        val xa = xFrom.coerceIn(0, w - 1)
        val xb = xTo.coerceIn(0, w - 1)
        if (xb <= xa) return 0f

        var maxV = 0f
        for (x in xa..xb) {
            val v = scoreAt(x, yy)
            if (v.isFinite() && v > maxV) maxV = v
        }
        if (maxV <= 1e-6f) return 0f

        val thr = maxV * LINE_THR_FRAC
        var hits = 0
        val nn = (xb - xa + 1)
        for (x in xa..xb) if (scoreAt(x, yy) >= thr) hits++
        return hits.toFloat() / nn.toFloat()
    }

    private fun buildBottomPolylineLocal3250(
        w: Int, h: Int,
        leftX: Int, rightX: Int,
        bottomY: Int,
        topMinAllowedY: Int,
        probeY: Int,
        hAt: (Int, Int) -> Float
    ): List<Pair<Int, Int>> {
        if (w <= 0 || h <= 0) return emptyList()
        val xa = (leftX + 6).coerceIn(0, w - 1)
        val xb = (rightX - 6).coerceIn(0, w - 1)
        if (xb - xa < 50) return emptyList()

        val yCenter = bottomY.coerceIn(0, h - 1)
        val yLo0 = (yCenter - BOTTOM_POLY_BAND_Y).coerceIn(0, h - 1)
        val yHi0 = (yCenter + BOTTOM_POLY_BAND_Y).coerceIn(0, h - 1)

        val yHardLo = max(topMinAllowedY, probeY - 1).coerceIn(0, h - 1)
        val yLo = max(yLo0, yHardLo).coerceIn(0, h - 1)
        val yHi = max(yLo, yHi0).coerceIn(0, h - 1)

        val out = ArrayList<Pair<Int, Int>>((xb - xa) / BOTTOM_POLY_STEP_X + 4)
        var x = xa
        while (x <= xb) {
            var bestY = -1
            var bestV = 0f
            for (y in yLo..yHi) {
                val v = hAt(x, y)
                if (v.isFinite() && v > bestV) { bestV = v; bestY = y }
            }
            if (bestY >= 0) out.add(Pair(x, bestY))
            x += BOTTOM_POLY_STEP_X
        }

        // median smooth
        if (out.size >= 7) {
            val ys = out.map { it.second }.toIntArray()
            val sm = IntArray(out.size)
            for (i in out.indices) {
                val a = max(0, i - 2)
                val b = min(out.lastIndex, i + 2)
                val win = IntArray(b - a + 1)
                for (k in a..b) win[k - a] = ys[k]
                win.sort()
                sm[i] = win[win.size / 2]
            }
            for (i in out.indices) out[i] = Pair(out[i].first, sm[i])
        }

        return out
    }

    private fun computeBottomPolylineCoverage3250(
        poly: List<Pair<Int, Int>>,
        hAt: (Int, Int) -> Float
    ): Float {
        if (poly.isEmpty()) return 0f
        var maxV = 0f
        for ((x, y) in poly) {
            val v = hAt(x, y)
            if (v.isFinite() && v > maxV) maxV = v
        }
        if (maxV <= 1e-6f) return 0f
        val thr = maxV * 0.35f

        var hits = 0
        for ((x, y) in poly) if (hAt(x, y) >= thr) hits++
        return hits.toFloat() / poly.size.toFloat()
    }

    private fun fitPolylineTiltDeg3250(poly: List<Pair<Int, Int>>): Float {
        if (poly.size < 8) return 0f
        val xs = poly.map { it.first }.sorted()
        val xMin = xs[xs.size / 5]
        val xMax = xs[xs.size * 4 / 5]

        var n = 0
        var sumX = 0.0
        var sumY = 0.0
        var sumXX = 0.0
        var sumXY = 0.0

        for ((x, y) in poly) {
            if (x !in xMin..xMax) continue
            val xf = x.toDouble()
            val yf = y.toDouble()
            n++
            sumX += xf
            sumY += yf
            sumXX += xf * xf
            sumXY += xf * yf
        }
        if (n < 6) return 0f

        val meanX = sumX / n
        val meanY = sumY / n
        val varX = sumXX / n - meanX * meanX
        if (abs(varX) < 1e-9) return 0f
        val covXY = sumXY / n - meanX * meanY
        val slope = covXY / varX
        val angRad = atan2(slope, 1.0)
        return (angRad * 180.0 / Math.PI).toFloat()
    }

    // ==========================================================
    // Utils
    // ==========================================================
    private fun buildScales3250(minS: Float, maxS: Float, step: Float): FloatArray {
        if (!minS.isFinite() || !maxS.isFinite() || !step.isFinite()) return floatArrayOf()
        if (step <= 0f) return floatArrayOf()
        val out = ArrayList<Float>(32)
        var s = minS
        while (s <= maxS + 1e-6f) {
            if (s.isFinite() && s > 0f) out += s
            s += step
        }
        return out.toFloatArray()
    }

    private fun clampInt3250(v: Int, lo: Int, hi: Int): Int = max(lo, min(hi, v))

    private fun clampInRange3250(v: Int, a: Int, b: Int): Int {
        val lo = min(a, b)
        val hi = max(a, b)
        return v.coerceIn(lo, hi)
    }

    // ==========================================================
    // Scan band inner/outer desde seed (nuevo)
    // ==========================================================
    private data class ScanBand3250(
        val innerLeft: Int,
        val innerRight: Int,
        val outerLeft: Int?,
        val outerRight: Int?,
        val tPxMed: Float?,
        val marksInner: List<Pair<Int, Int>>, // (x,y) ROI-local
        val marksOuter: List<Pair<Int, Int>>  // (x,y) ROI-local
    )

    private fun medianInt3250(list: List<Int>): Int {
        val a = list.toIntArray()
        a.sort()
        return a[a.size / 2]
    }

    private fun medianFloat3250(list: List<Float>): Float {
        val a = list.toFloatArray()
        a.sort()
        return a[a.size / 2]
    }

    private fun scanFirstEdgeX3250(
        y: Int,
        xStart: Int,
        step: Int,
        xMin: Int,
        xMax: Int,
        vAt: (Int, Int) -> Float
    ): Int? {
        var x = xStart.coerceIn(xMin, xMax)
        while (x in xMin..xMax) {
            if (vAt(x, y) > 0f) return x
            x += step
        }
        return null
    }

    private fun scanSecondEdgeX3250(
        y: Int,
        xInner: Int,
        step: Int,
        tMinPx: Int,
        tMaxPx: Int,
        xMin: Int,
        xMax: Int,
        vAt: (Int, Int) -> Float
    ): Int? {
        val a = (xInner + step * tMinPx).coerceIn(xMin, xMax)
        val b = (xInner + step * tMaxPx).coerceIn(xMin, xMax)
        val lo = min(a, b)
        val hi = max(a, b)

        return if (step > 0) {
            for (x in lo..hi) if (vAt(x, y) > 0f) return x
            null
        } else {
            for (x in hi downTo lo) if (vAt(x, y) > 0f) return x
            null
        }
    }

    private fun scanBandInnerOuterFromSeed3250(
        w: Int,
        h: Int,
        probeY: Int,
        bandHalf: Int,
        topMinAllowedY: Int,
        seedX: Int,
        nasalAtLeft: Boolean,
        pxGuessThis: Float,
        vAt: (Int, Int) -> Float
    ): ScanBand3250? {

        val xMin = BORDER_GUARD_PX.coerceIn(0, w - 1)
        val xMax = (w - 1 - BORDER_GUARD_PX).coerceAtLeast(xMin)

        val y0 = (probeY - bandHalf).coerceIn(topMinAllowedY.coerceIn(0, h - 1), h - 2)
        val y1 = (probeY + bandHalf).coerceIn(topMinAllowedY.coerceIn(0, h - 1), h - 2)
        if (y1 < y0) return null

        val tMinPx = (pxGuessThis * RIM_T_MM_MIN).roundToInt().coerceIn(2, 120)
        val tMaxPx = (pxGuessThis * RIM_T_MM_MAX).roundToInt().coerceIn(tMinPx + 2, 160)

        val stepN = if (nasalAtLeft) -1 else +1
        val stepT = -stepN

        val innN = ArrayList<Int>()
        val innT = ArrayList<Int>()
        val outN = ArrayList<Int>()
        val outT = ArrayList<Int>()
        val tAll = ArrayList<Float>()

        val marksInner = ArrayList<Pair<Int, Int>>(64)
        val marksOuter = ArrayList<Pair<Int, Int>>(64)

        val stepY = 2
        for (y in y0..y1 step stepY) {
            val xInnN = scanFirstEdgeX3250(y, seedX, stepN, xMin, xMax, vAt)
            if (xInnN != null) {
                innN.add(xInnN); marksInner.add(xInnN to y)
                val xOutN = scanSecondEdgeX3250(y, xInnN, stepN, tMinPx, tMaxPx, xMin, xMax, vAt)
                if (xOutN != null) {
                    outN.add(xOutN); marksOuter.add(xOutN to y)
                    tAll.add(abs(xOutN - xInnN).toFloat())
                }
            }

            val xInnT = scanFirstEdgeX3250(y, seedX, stepT, xMin, xMax, vAt)
            if (xInnT != null) {
                innT.add(xInnT); marksInner.add(xInnT to y)
                val xOutT = scanSecondEdgeX3250(y, xInnT, stepT, tMinPx, tMaxPx, xMin, xMax, vAt)
                if (xOutT != null) {
                    outT.add(xOutT); marksOuter.add(xOutT to y)
                    tAll.add(abs(xOutT - xInnT).toFloat())
                }
            }
        }

        if (innN.size < 3 || innT.size < 3) return null

        val innerN = medianInt3250(innN)
        val innerT = medianInt3250(innT)

        val innerLeft = if (nasalAtLeft) innerN else innerT
        val innerRight = if (nasalAtLeft) innerT else innerN
        if (innerRight - innerLeft < 60) return null

        val outerLeft: Int? =
            if (outN.size >= 3 && outT.size >= 3) {
                val oN = medianInt3250(outN)
                val oT = medianInt3250(outT)
                if (nasalAtLeft) min(oN, oT) else min(oT, oN)
            } else null

        val outerRight: Int? =
            if (outN.size >= 3 && outT.size >= 3) {
                val oN = medianInt3250(outN)
                val oT = medianInt3250(outT)
                if (nasalAtLeft) max(oN, oT) else max(oT, oN)
            } else null

        val tMed: Float? = if (tAll.size >= 4) medianFloat3250(tAll) else null

        return ScanBand3250(
            innerLeft = innerLeft,
            innerRight = innerRight,
            outerLeft = outerLeft,
            outerRight = outerRight,
            tPxMed = tMed,
            marksInner = marksInner,
            marksOuter = marksOuter
        )
    }
}
