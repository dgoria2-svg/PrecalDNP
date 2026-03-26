package com.dg.precaldnp.vision

import android.graphics.PointF
import android.graphics.RectF
import android.util.Log
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

object ArcFitAdapter3250 {

    private const val TAG = "ArcFitAdapter3250"
    private const val BOTTOM_DEG = 270.0

    data class PxMmPack3250(
        val pxPerMmX: Double,
        val pxPerMmY: Double,
        val pxPerMmOfficial: Double,
        val relErrXY: Double,
        val spanXpxModel: Double,
        val spanYpxModel: Double
    )

    data class InnerAtY3250(
        val yGlobal: Float,
        val innerLeftXGlobal: Float,
        val innerRightXGlobal: Float,
        val nasalXGlobal: Float,
        val templeXGlobal: Float
    )

    data class ArcWindow3250(
        val fromDeg: Double,
        val toDeg: Double,
        val lenDeg: Double,
        val expandedByDeg: Double
    )

    data class ArcFitContract3250(
        val roiPxGlobal: RectF,
        val edgesRoiU8: ByteArray,
        val maskRoiU8: ByteArray?,
        val wRoi: Int,
        val hRoi: Int,
        val eyeSideSign3250: Int,
        val profile3250: RimProfile3250,
        val originPxRoi: PointF,
        val pxPerMmInitGuess: Double,

        // =========================
        // VERDAD DEL DETECTOR
        // =========================
        val bottomAnchorYpxGlobal: Float,
        val bottomAnchorYpxRoi: Float,

        val detectorWallsYpxGlobal: Float,
        val detectorWallsYpxRoi: Float,

        val nasalXAtWallsRoi: Float?,
        val templeXAtWallsRoi: Float?,
        val pxPerMmObservedAtWalls: Double?,

        // =========================
        // VERDAD FACIAL / METRICS
        // =========================
        val bridgeRowYpxGlobal: Float,
        val bridgeRowYpxRoi: Float,

        // soporte geométrico / debug
        val bottomPolylinePxRoi: List<PointF>,
        val nasalInnerPolylinePxRoi: List<PointF>,
        val templeInnerPolylinePxRoi: List<PointF>,
        val nasalOuterPolylinePxRoi: List<PointF>?,
        val templeOuterPolylinePxRoi: List<PointF>?,
        val bottomArcWindow: ArcWindow3250?,

        val bottomConfidence: Float,
        val lateralConfidence: Float,
        val contractConfidence: Float
    )

    fun arcFitFromRimPack3250(
        filPtsMm: List<PointF>,
        filHboxMm: Double,
        filVboxMm: Double,
        filOverInnerMmPerSide3250: Double,
        maskU8Roi: ByteArray? = null,
        pack: RimDetectPack3250,
        seed: RimArcSeed3250.ArcSeed3250,
        eyeSideSign3250: Int,
        bridgeRowYpxGlobal: Float,
        profile3250: RimProfile3250,
        pxPerMmObservedOverride: Double? = null,
        bottomAnchorYpxRoiOverride: Float? = null,
        anchorArcExpandDeg: Double = 18.0,
        iters: Int = 2
    ): FilPlaceFromArc3250.Fit3250? {

        val r = pack.result
        if (!r.ok) return null

        val contract = buildArcFitContract3250(
            pack = pack,
            seed = seed,
            eyeSideSign3250 = eyeSideSign3250,
            profile3250 = profile3250,
            bridgeRowYpxGlobal = bridgeRowYpxGlobal,
            filHboxMm = filHboxMm,
            filOverInnerMmPerSide3250 = filOverInnerMmPerSide3250,
            maskU8Roi = maskU8Roi,
            anchorArcExpandDeg = anchorArcExpandDeg
        ) ?: return null

        val pxObs = pxPerMmObservedOverride
            ?.takeIf { it.isFinite() && it > 1e-6 }
            ?: contract.pxPerMmObservedAtWalls

        val pxInit = contract.pxPerMmInitGuess
            .takeIf { it.isFinite() && it > 1e-6 }
            ?: pxObs
            ?: 5.0

        val bottomAnchorYpxRoi = bottomAnchorYpxRoiOverride
            ?.takeIf { it.isFinite() }
            ?: contract.bottomAnchorYpxRoi

        Log.d(
            TAG,
            "CONTRACT side=$eyeSideSign3250 profile=$profile3250 " +
                    "bottomAnchorY=${f2(contract.bottomAnchorYpxRoi)} " +
                    "wallsY=${f2(contract.detectorWallsYpxRoi)} " +
                    "bridgeY=${f2(contract.bridgeRowYpxRoi)} " +
                    "xAtWalls(n/t)=(${f2(contract.nasalXAtWallsRoi)},${f2(contract.templeXAtWallsRoi)}) " +
                    "pxObs=${f4(pxObs)} bottomConf=${f3(contract.bottomConfidence)} " +
                    "latConf=${f3(contract.lateralConfidence)} contractConf=${f3(contract.contractConfidence)} " +
                    "arcWin=${contract.bottomArcWindow?.let { "${f1(it.fromDeg)}..${f1(it.toDeg)} len=${f1(it.lenDeg)}" } ?: "null"}"
        )

        val fit = FilPlaceFromArc3250.placeFilByArc3250(
            filPtsMm = filPtsMm,
            filHboxMm = filHboxMm,
            filVboxMm = filVboxMm,
            edgesu8 = contract.edgesRoiU8,
            w = contract.wRoi,
            h = contract.hRoi,
            maskU8 = contract.maskRoiU8,
            originPxRoi = contract.originPxRoi,
            eyeSideSign3250 = contract.eyeSideSign3250,
            pxPerMmInitGuess = pxInit,
            pxPerMmObserved = pxObs,
            pxPerMmFixed = null,
            filOverInnerMmPerSide3250 = filOverInnerMmPerSide3250,
            allowedArcFromDeg = contract.bottomArcWindow?.fromDeg,
            allowedArcToDeg = contract.bottomArcWindow?.toDeg,
            excludeDegFrom = if (contract.bottomArcWindow == null) 35.0 else 0.0,
            excludeDegTo = if (contract.bottomArcWindow == null) 145.0 else 0.0,
            stepDeg = 2.0,
            iters = iters.coerceIn(1, 3),
            rSearchRelLo = 0.70,
            rSearchRelHi = 1.35,
            bottomAnchorYpxRoi = bottomAnchorYpxRoi,

            // ✅ guías reales del detector
            bottomGuidePxRoi = contract.bottomPolylinePxRoi,
            detectorNasalGuidePxRoi = contract.nasalInnerPolylinePxRoi,
            detectorTempleGuidePxRoi = contract.templeInnerPolylinePxRoi,

            rimProfile3250 = contract.profile3250,
            bridgeRowYpxGlobal = contract.bridgeRowYpxGlobal,
            bridgeRowYpxRoi = contract.bridgeRowYpxRoi
        ) ?: return null

        // ============================================================
        // VALIDACIÓN HARD: contra la fila oficial del detector
        // ============================================================
        val innerAtWalls = innerAtRowFromFit3250(
            fit = fit,
            roiGlobal = contract.roiPxGlobal,
            rowYpxGlobal = contract.detectorWallsYpxGlobal,
            eyeSideSign3250 = contract.eyeSideSign3250,
            midlineXpxGlobal = null
        )

        if (innerAtWalls != null) {
            val spanFitPx =
                (innerAtWalls.innerRightXGlobal - innerAtWalls.innerLeftXGlobal).toDouble()

            val spanObsPx =
                if (contract.nasalXAtWallsRoi != null && contract.templeXAtWallsRoi != null) {
                    abs(contract.templeXAtWallsRoi - contract.nasalXAtWallsRoi).toDouble()
                } else {
                    Double.NaN
                }

            val relSpanErr =
                if (spanFitPx.isFinite() && spanObsPx.isFinite() && spanObsPx > 1e-6) {
                    abs(spanFitPx - spanObsPx) / spanObsPx
                } else {
                    Double.NaN
                }

            Log.d(
                TAG,
                "POSTFIT_DET side=$eyeSideSign3250 profile=$profile3250 " +
                        "wallsY=${f2(innerAtWalls.yGlobal)} " +
                        "innerL=${f2(innerAtWalls.innerLeftXGlobal)} " +
                        "innerR=${f2(innerAtWalls.innerRightXGlobal)} " +
                        "nasal=${f2(innerAtWalls.nasalXGlobal)} " +
                        "temple=${f2(innerAtWalls.templeXGlobal)} " +
                        "spanFit=${f4(spanFitPx)} spanObs=${f4(spanObsPx)} relErr=${f4(relSpanErr)}"
            )
        } else {
            Log.w(
                TAG,
                "POSTFIT_DET side=$eyeSideSign3250 profile=$profile3250 " +
                        "innerAtWalls=NULL wallsY=${f2(contract.detectorWallsYpxGlobal)}"
            )
        }

        // ============================================================
        // VALIDACIÓN SOFT: bridge row facial
        // ============================================================
        val innerAtBridge = innerAtBridgeRowFromFit3250(
            fit = fit,
            roiGlobal = contract.roiPxGlobal,
            bridgeRowYpxGlobal = contract.bridgeRowYpxGlobal,
            eyeSideSign3250 = contract.eyeSideSign3250,
            midlineXpxGlobal = null
        )

        if (innerAtBridge != null) {
            Log.d(
                TAG,
                "POSTFIT_BRIDGE side=$eyeSideSign3250 profile=$profile3250 " +
                        "bridgeY=${f2(innerAtBridge.yGlobal)} " +
                        "innerL=${f2(innerAtBridge.innerLeftXGlobal)} " +
                        "innerR=${f2(innerAtBridge.innerRightXGlobal)} " +
                        "nasal=${f2(innerAtBridge.nasalXGlobal)} " +
                        "temple=${f2(innerAtBridge.templeXGlobal)}"
            )
        } else {
            Log.w(
                TAG,
                "POSTFIT_BRIDGE side=$eyeSideSign3250 profile=$profile3250 " +
                        "innerAtBridge=NULL bridgeY=${f2(contract.bridgeRowYpxGlobal)}"
            )
        }

        return fit
    }

    fun buildArcFitContract3250(
        pack: RimDetectPack3250,
        seed: RimArcSeed3250.ArcSeed3250,
        eyeSideSign3250: Int,
        profile3250: RimProfile3250,
        bridgeRowYpxGlobal: Float,
        filHboxMm: Double,
        filOverInnerMmPerSide3250: Double,
        maskU8Roi: ByteArray? = null,
        anchorArcExpandDeg: Double = 18.0
    ): ArcFitContract3250? {

        val r = pack.result
        if (!r.ok) return null

        val roi = r.roiPx

        val bridgeRowYpxRoi = bridgeRowYpxGlobal - roi.top
        if (!bridgeRowYpxRoi.isFinite()) return null

        val effectiveOver = effectiveFilOverPerSide3250(profile3250, filOverInnerMmPerSide3250)
        val hboxInnerMm = (filHboxMm - 2.0 * effectiveOver).coerceAtLeast(10.0)

        val bottomPolylinePxRoi = localPolylineToRoi3250(r.bottomPolylinePx, roi)
        val nasalInnerPolylinePxRoi = localPolylineToRoi3250(r.nasalInnerPolylinePx, roi)
        val templeInnerPolylinePxRoi = localPolylineToRoi3250(r.templeInnerPolylinePx, roi)
        val nasalOuterPolylinePxRoi = localPolylineToRoi3250Nullable(r.nasalOuterPolylinePx, roi)
        val templeOuterPolylinePxRoi = localPolylineToRoi3250Nullable(r.templeOuterPolylinePx, roi)

        // ============================================================
        // VERDAD DEL DETECTOR
        // ============================================================
        val detectorWallsYpxRoi = r.wallsYpx.takeIf { it.isFinite() } ?: bridgeRowYpxRoi
        val detectorWallsYpxGlobal = detectorWallsYpxRoi + roi.top

        val bottomAnchorYpxRoi = r.bottomYpx.takeIf { it.isFinite() }
            ?: bottomPolylinePxRoi.maxOfOrNull { it.y }
            ?: return null
        val bottomAnchorYpxGlobal = bottomAnchorYpxRoi + roi.top

        val nasalXAtWallsRoi = r.nasalInnerXpx.takeIf { it.isFinite() }
            ?: intersectPolylineWithHLineX3250(
                pts = nasalInnerPolylinePxRoi,
                y = detectorWallsYpxRoi
            )

        val templeXAtWallsRoi = r.templeInnerXpx.takeIf { it.isFinite() }
            ?: intersectPolylineWithHLineX3250(
                pts = templeInnerPolylinePxRoi,
                y = detectorWallsYpxRoi
            )

        val observedSpanPx = r.innerWidthPx
            .takeIf { it.isFinite() && it > 1e-6f }
            ?.toDouble()
            ?: run {
                if (nasalXAtWallsRoi != null && templeXAtWallsRoi != null) {
                    abs(templeXAtWallsRoi - nasalXAtWallsRoi).toDouble()
                } else {
                    Double.NaN
                }
            }

        val pxPerMmObservedAtWalls = observedSpanPx
            .takeIf { it.isFinite() && it > 1e-6 }
            ?.div(hboxInnerMm)
            ?.takeIf { it.isFinite() && it > 1e-6 }

        val bottomArcWindow = buildBottomArcWindowFromPolylineRoi3250(
            bottomPolylineRoi = bottomPolylinePxRoi,
            originPxRoi = seed.originPxRoi,
            eyeSideSign3250 = eyeSideSign3250,
            expandDeg = anchorArcExpandDeg
        )

        val bottomConfidence = computeBottomConfidence3250(
            bottomPolylineRoi = bottomPolylinePxRoi,
            wRoi = pack.w
        )

        val lateralConfidence = computeLateralConfidence3250(
            nasalInnerPolylineRoi = nasalInnerPolylinePxRoi,
            templeInnerPolylineRoi = templeInnerPolylinePxRoi,
            nasalXAtAnchorRoi = nasalXAtWallsRoi,
            templeXAtAnchorRoi = templeXAtWallsRoi,
            anchorYpxRoi = detectorWallsYpxRoi,
            hRoi = pack.h
        )

        val contractConfidence =
            (0.45f * bottomConfidence + 0.55f * lateralConfidence).coerceIn(0f, 1f)

        return ArcFitContract3250(
            roiPxGlobal = roi,
            edgesRoiU8 = pack.edges,
            maskRoiU8 = maskU8Roi,
            wRoi = pack.w,
            hRoi = pack.h,
            eyeSideSign3250 = eyeSideSign3250,
            profile3250 = profile3250,
            originPxRoi = seed.originPxRoi,
            pxPerMmInitGuess = seed.pxPerMmInitGuess,

            bottomAnchorYpxGlobal = bottomAnchorYpxGlobal,
            bottomAnchorYpxRoi = bottomAnchorYpxRoi,

            detectorWallsYpxGlobal = detectorWallsYpxGlobal,
            detectorWallsYpxRoi = detectorWallsYpxRoi,

            nasalXAtWallsRoi = nasalXAtWallsRoi,
            templeXAtWallsRoi = templeXAtWallsRoi,
            pxPerMmObservedAtWalls = pxPerMmObservedAtWalls,

            bridgeRowYpxGlobal = bridgeRowYpxGlobal,
            bridgeRowYpxRoi = bridgeRowYpxRoi,

            bottomPolylinePxRoi = bottomPolylinePxRoi,
            nasalInnerPolylinePxRoi = nasalInnerPolylinePxRoi,
            templeInnerPolylinePxRoi = templeInnerPolylinePxRoi,
            nasalOuterPolylinePxRoi = nasalOuterPolylinePxRoi,
            templeOuterPolylinePxRoi = templeOuterPolylinePxRoi,
            bottomArcWindow = bottomArcWindow,
            bottomConfidence = bottomConfidence,
            lateralConfidence = lateralConfidence,
            contractConfidence = contractConfidence
        )
    }

    fun pxMmFromFit3250(
        fit: FilPlaceFromArc3250.Fit3250,
        filHboxMm: Double,
        filVboxMm: Double,
        filOverInnerMmPerSide3250: Double,
        profile3250: RimProfile3250,
        tolRelXY: Double = 0.10
    ): PxMmPack3250? {

        val effectiveOver = effectiveFilOverPerSide3250(
            profile3250,
            filOverInnerMmPerSide3250
        )

        val overClamped = effectiveOver.coerceIn(0.0, 2.0)

        val hboxInnerMm = (filHboxMm - 2.0 * overClamped).coerceAtLeast(10.0)
        val vboxInnerMm = (filVboxMm - 2.0 * overClamped).coerceAtLeast(10.0)

        if (!hboxInnerMm.isFinite() || !vboxInnerMm.isFinite()) return null
        if (fit.placedPxRoi.isEmpty()) return null

        val theta = Math.toRadians(fit.rotDeg)
        val c = cos(theta)
        val s = sin(theta)

        var minX = Double.POSITIVE_INFINITY
        var maxX = Double.NEGATIVE_INFINITY
        var minY = Double.POSITIVE_INFINITY
        var maxY = Double.NEGATIVE_INFINITY

        for (p in fit.placedPxRoi) {
            val dx = (p.x - fit.originPxRoi.x).toDouble()
            val dyUp = (-(p.y - fit.originPxRoi.y)).toDouble()

            val xModel = c * dx + s * dyUp
            val yModel = -s * dx + c * dyUp

            minX = min(minX, xModel)
            maxX = max(maxX, xModel)
            minY = min(minY, yModel)
            maxY = max(maxY, yModel)
        }

        val spanX = (maxX - minX).takeIf { it.isFinite() && it > 1e-6 } ?: return null
        val spanY = (maxY - minY).takeIf { it.isFinite() && it > 1e-6 } ?: return null

        val pxX = spanX / hboxInnerMm
        val pxY = spanY / vboxInnerMm
        if (!pxX.isFinite() || !pxY.isFinite() || pxX <= 1e-6 || pxY <= 1e-6) return null

        val relErr = abs(pxY - pxX) / pxX
        val pxOfficial = if (relErr <= tolRelXY) 0.5 * (pxX + pxY) else pxX

        return PxMmPack3250(
            pxPerMmX = pxX,
            pxPerMmY = pxY,
            pxPerMmOfficial = pxOfficial,
            relErrXY = relErr,
            spanXpxModel = spanX,
            spanYpxModel = spanY
        )
    }

    fun innerAtBridgeRowFromFit3250(
        fit: FilPlaceFromArc3250.Fit3250,
        roiGlobal: RectF,
        bridgeRowYpxGlobal: Float,
        eyeSideSign3250: Int,
        midlineXpxGlobal: Float? = null
    ): InnerAtY3250? {
        return innerAtRowFromFit3250(
            fit = fit,
            roiGlobal = roiGlobal,
            rowYpxGlobal = bridgeRowYpxGlobal,
            eyeSideSign3250 = eyeSideSign3250,
            midlineXpxGlobal = midlineXpxGlobal
        )
    }

    fun innerAtRowFromFit3250(
        fit: FilPlaceFromArc3250.Fit3250,
        roiGlobal: RectF,
        rowYpxGlobal: Float,
        eyeSideSign3250: Int,
        midlineXpxGlobal: Float? = null
    ): InnerAtY3250? {

        if (fit.placedPxRoi.isEmpty()) return null

        val yRoi = rowYpxGlobal - roiGlobal.top
        if (!yRoi.isFinite()) return null

        val xs = intersectPolyWithHLineXs3250(fit.placedPxRoi, yRoi)
        if (xs.size < 2) return null

        val xL = xs.first()
        val xR = xs.last()

        val xLGlobal = xL + roiGlobal.left
        val xRGlobal = xR + roiGlobal.left

        val nasal = run {
            if (midlineXpxGlobal != null && midlineXpxGlobal.isFinite()) {
                val dL = abs(xLGlobal - midlineXpxGlobal)
                val dR = abs(xRGlobal - midlineXpxGlobal)
                if (dL <= dR) xLGlobal else xRGlobal
            } else {
                if (eyeSideSign3250 < 0) xRGlobal else xLGlobal
            }
        }

        val temple = if (nasal == xLGlobal) xRGlobal else xLGlobal

        return InnerAtY3250(
            yGlobal = rowYpxGlobal,
            innerLeftXGlobal = xLGlobal,
            innerRightXGlobal = xRGlobal,
            nasalXGlobal = nasal,
            templeXGlobal = temple
        )
    }

    private fun computeBottomConfidence3250(
        bottomPolylineRoi: List<PointF>,
        wRoi: Int
    ): Float {
        if (bottomPolylineRoi.size < 2 || wRoi <= 0) return 0f

        val sorted = bottomPolylineRoi.sortedBy { it.x }
        val xSpan = (sorted.last().x - sorted.first().x).coerceAtLeast(0f)
        val spanFrac = (xSpan / wRoi.toFloat()).coerceIn(0f, 1f)
        val nScore = (bottomPolylineRoi.size / 48f).coerceIn(0f, 1f)

        return (0.70f * spanFrac + 0.30f * nScore).coerceIn(0f, 1f)
    }

    private fun computeLateralConfidence3250(
        nasalInnerPolylineRoi: List<PointF>,
        templeInnerPolylineRoi: List<PointF>,
        nasalXAtAnchorRoi: Float?,
        templeXAtAnchorRoi: Float?,
        anchorYpxRoi: Float,
        hRoi: Int
    ): Float {
        if (hRoi <= 0) return 0f

        val nScore = polylineAnchorSupport3250(
            pts = nasalInnerPolylineRoi,
            anchorY = anchorYpxRoi,
            hRoi = hRoi
        )

        val tScore = polylineAnchorSupport3250(
            pts = templeInnerPolylineRoi,
            anchorY = anchorYpxRoi,
            hRoi = hRoi
        )

        val xCrossScore = when {
            nasalXAtAnchorRoi != null && templeXAtAnchorRoi != null -> 1f
            nasalXAtAnchorRoi != null || templeXAtAnchorRoi != null -> 0.5f
            else -> 0f
        }

        return (0.40f * nScore + 0.40f * tScore + 0.20f * xCrossScore).coerceIn(0f, 1f)
    }

    private fun polylineAnchorSupport3250(
        pts: List<PointF>,
        anchorY: Float,
        hRoi: Int
    ): Float {
        if (pts.size < 2 || hRoi <= 0) return 0f

        val sorted = pts.sortedBy { it.y }
        val ySpan = (sorted.last().y - sorted.first().y).coerceAtLeast(0f)
        val spanFrac = (ySpan / hRoi.toFloat()).coerceIn(0f, 1f)

        val yMin = sorted.first().y
        val yMax = sorted.last().y
        val crosses = anchorY in yMin..yMax

        val nScore = (pts.size / 36f).coerceIn(0f, 1f)
        val crossScore = if (crosses) 1f else 0f

        return (0.40f * spanFrac + 0.30f * nScore + 0.30f * crossScore).coerceIn(0f, 1f)
    }

    private fun localPolylineToRoi3250(
        ptsGlobal: List<PointF>?,
        roiGlobal: RectF
    ): List<PointF> {
        if (ptsGlobal.isNullOrEmpty()) return emptyList()
        return ptsGlobal.map { p -> PointF(p.x - roiGlobal.left, p.y - roiGlobal.top) }
    }

    private fun localPolylineToRoi3250Nullable(
        ptsGlobal: List<PointF>?,
        roiGlobal: RectF
    ): List<PointF>? {
        if (ptsGlobal.isNullOrEmpty()) return null
        return ptsGlobal.map { p -> PointF(p.x - roiGlobal.left, p.y - roiGlobal.top) }
    }

    private fun intersectPolylineWithHLineX3250(
        pts: List<PointF>,
        y: Float
    ): Float? {
        if (pts.size < 2 || !y.isFinite()) return null

        var bestX: Float? = null
        var bestDy = Float.MAX_VALUE

        for (i in 1 until pts.size) {
            val a = pts[i - 1]
            val b = pts[i]

            val y0 = a.y
            val y1 = b.y
            val crosses = (y0 <= y && y <= y1) || (y1 <= y && y <= y0)
            if (!crosses) continue

            val dySeg = y1 - y0
            if (abs(dySeg) < 1e-6f) {
                val yc = 0.5f * (y0 + y1)
                val d = abs(yc - y)
                if (d < bestDy) {
                    bestDy = d
                    bestX = 0.5f * (a.x + b.x)
                }
                continue
            }

            val t = (y - y0) / dySeg
            if (t < 0f || t > 1f) continue

            val x = a.x + t * (b.x - a.x)
            if (0f <= bestDy) {
                bestDy = 0f
                bestX = x
            }
        }

        return bestX
    }

    private fun intersectPolyWithHLineXs3250(
        poly: List<PointF>,
        y: Float
    ): FloatArray {
        if (poly.size < 3) return floatArrayOf()

        val xs = ArrayList<Float>(8)
        val n = poly.size

        fun addX(x: Float) {
            if (x.isFinite()) xs.add(x)
        }

        for (i in 0 until n) {
            val a = poly[i]
            val b = poly[(i + 1) % n]
            val y0 = a.y
            val y1 = b.y

            val crosses = (y0 <= y && y < y1) || (y1 <= y && y < y0)
            if (!crosses) continue

            val dy = y1 - y0
            if (abs(dy) < 1e-6f) continue

            val t = (y - y0) / dy
            val x = a.x + t * (b.x - a.x)
            addX(x)
        }

        if (xs.size < 2) return floatArrayOf()

        xs.sort()

        val out = ArrayList<Float>(xs.size)
        var last = Float.NaN
        for (x in xs) {
            if (!last.isFinite() || abs(x - last) > 0.75f) {
                out.add(x)
                last = x
            }
        }

        if (out.size < 2) return floatArrayOf()
        return floatArrayOf(out.first(), out.last())
    }

    private fun buildBottomArcWindowFromPolylineRoi3250(
        bottomPolylineRoi: List<PointF>,
        originPxRoi: PointF,
        eyeSideSign3250: Int,
        expandDeg: Double,
        minLenDeg: Double = 40.0,
        gapSplitDeg: Double = 10.0,
        maxDistToBottomDeg: Double = 55.0,
        maxLenDeg: Double = 200.0
    ): ArcWindow3250? {

        if (bottomPolylineRoi.size < 3) return null

        val degs = ArrayList<Double>(bottomPolylineRoi.size)
        for (p in bottomPolylineRoi) {
            val d = degFromPointRoi3250(
                qRoi = p,
                originRoi = originPxRoi,
                eyeSideSign3250 = eyeSideSign3250
            )
            val du = if (d < 180.0) d + 360.0 else d
            degs.add(du)
        }

        if (degs.size < 3) return null
        degs.sort()

        data class Cl(val a: Double, val b: Double, val n: Int)

        val clusters = ArrayList<Cl>()
        var a0 = degs[0]
        var prev = degs[0]
        var n0 = 1

        for (i in 1 until degs.size) {
            val cur = degs[i]
            if (cur - prev > gapSplitDeg) {
                clusters.add(Cl(a0, prev, n0))
                a0 = cur
                n0 = 1
            } else {
                n0++
            }
            prev = cur
        }
        clusters.add(Cl(a0, prev, n0))

        if (clusters.isEmpty()) return null

        fun distToInterval(x: Double, lo: Double, hi: Double): Double =
            when {
                x < lo -> lo - x
                x > hi -> x - hi
                else -> 0.0
            }

        var best: Cl? = null
        var bestScore = Double.NEGATIVE_INFINITY

        for (c in clusters) {
            val contains = BOTTOM_DEG in c.a..c.b
            val dist = distToInterval(BOTTOM_DEG, c.a, c.b)
            val score = (if (contains) 1e6 else 0.0) + (c.n * 1000.0) - (dist * 10.0)
            if (score > bestScore) {
                bestScore = score
                best = c
            }
        }

        val c0 = best ?: return null
        val containsBottom = BOTTOM_DEG in c0.a..c0.b
        val distBottom = distToInterval(BOTTOM_DEG, c0.a, c0.b)

        var fromU: Double
        var toU: Double

        if (!containsBottom && distBottom > maxDistToBottomDeg) {
            val minLen = minLenDeg.coerceIn(20.0, 140.0)
            val half = 0.5 * minLen
            fromU = BOTTOM_DEG - half
            toU = BOTTOM_DEG + half
        } else {
            fromU = c0.a
            toU = c0.b
        }

        val minLen = minLenDeg.coerceIn(20.0, 140.0)
        var lenU = (toU - fromU).coerceAtLeast(1e-6)

        if (lenU < minLen) {
            val mid = if (BOTTOM_DEG in fromU..toU) 0.5 * (fromU + toU) else BOTTOM_DEG
            val half = 0.5 * minLen
            fromU = mid - half
            toU = mid + half
            lenU = toU - fromU
        }

        if (lenU > maxLenDeg) {
            val half = 0.5 * maxLenDeg
            fromU = BOTTOM_DEG - half
            toU = BOTTOM_DEG + half
        }

        val exp = expandDeg.coerceIn(0.0, 40.0)
        fromU -= exp
        toU += exp

        val from = normDeg3250(fromU)
        val to = normDeg3250(toU)

        if (!containsOnArc3250(BOTTOM_DEG, from, to)) {
            val half = 0.5 * minLen
            val f = normDeg3250(BOTTOM_DEG - half - exp)
            val t = normDeg3250(BOTTOM_DEG + half + exp)
            if (!containsOnArc3250(BOTTOM_DEG, f, t)) return null
            return ArcWindow3250(
                fromDeg = f,
                toDeg = t,
                lenDeg = arcLen3250(f, t),
                expandedByDeg = exp
            )
        }

        return ArcWindow3250(
            fromDeg = from,
            toDeg = to,
            lenDeg = arcLen3250(from, to),
            expandedByDeg = exp
        )
    }

    private fun normDeg3250(d: Double): Double {
        val a = d % 360.0
        return if (a < 0.0) a + 360.0 else a
    }

    private fun containsOnArc3250(
        target: Double,
        from: Double,
        to: Double
    ): Boolean {
        val t = normDeg3250(target)
        val a = normDeg3250(from)
        val b = normDeg3250(to)
        return if (a <= b) {
            t in a..b
        } else {
            t >= a || t <= b
        }
    }

    private fun arcLen3250(
        from: Double,
        to: Double
    ): Double {
        val a = normDeg3250(from)
        val b = normDeg3250(to)
        return (b - a + 360.0) % 360.0
    }

    private fun degFromPointRoi3250(
        qRoi: PointF,
        originRoi: PointF,
        eyeSideSign3250: Int
    ): Double {
        val eyeSide = if (eyeSideSign3250 < 0) -1 else +1
        val nasalUxFixed = (-eyeSide).toDouble()

        val dx = (qRoi.x - originRoi.x).toDouble()
        val dyUp = (-(qRoi.y - originRoi.y)).toDouble()

        val xNasal = dx * nasalUxFixed
        val ang = Math.toDegrees(atan2(dyUp, xNasal))
        return normDeg3250(ang)
    }

    private fun f1(v: Double?): String =
        if (v == null || !v.isFinite()) "NaN" else "%.1f".format(v)

    private fun f2(v: Float?): String =
        if (v == null || !v.isFinite()) "NaN" else "%.2f".format(v)

    private fun f3(v: Float?): String =
        if (v == null || !v.isFinite()) "NaN" else "%.3f".format(v)

    private fun f4(v: Double?): String =
        if (v == null || !v.isFinite()) "NaN" else "%.4f".format(v)
}