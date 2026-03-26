@file:Suppress("SameParameterValue")

package com.dg.precaldnp.vision

import android.graphics.PointF
import android.util.Log
import com.dg.precaldnp.util.Fmt3250.fmt2
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

enum class ScalePolicy3250 {
    LOCK_TO_OBSERVED,   // usa la escala observada/detector y no la toca
    CLAMP_TO_OBSERVED   // deja mover un poco la escala, pero la acota
}

/**
 * FilPlaceFromArc3250
 *
 * Ajuste por arcos del FIL contra edge map binario del ROI.
 *
 * Regla de Oro (EDGE->ARCFIT):
 * - edgesu8 debe ser binario 0/255.
 * - ArcFit no inventa intensidades, trabaja edge/no-edge.
 *
 * Regla 3250 (lado):
 * - OD en FOTO está a la IZQUIERDA de la midline => eyeSideSign=-1
 * - OI en FOTO está a la DERECHA de la midline => eyeSideSign=+1
 * - nasal siempre apunta hacia la midline => nasalUx = -eyeSideSign
 *
 * Política geométrica:
 * - El ajuste SIEMPRE se resuelve con una transformación GLOBAL:
 *   rotación + escala única + traslación.
 * - NUNCA se escala/libera cada R por separado.
 * - El encastre se calcula contra el INNER del FIL.
 * - El resultado final que se devuelve/dibuja es el OUTER del FIL
 *   con la MISMA transformación global hallada sobre el INNER.
 */
object FilPlaceFromArc3250 {

    private const val TAG = "FILFIT_ARC3250"

    data class Fit3250(
        val placedPxRoi: List<PointF>,   // OUTER final colocado
        val pxPerMmFace: Double,
        val rotDeg: Double,
        val originPxRoi: PointF,
        val rmsPx: Double,
        val usedSamples: Int,

        // pass-through para validación/debug posterior
        val rimProfile3250: RimProfile3250? = null,
        val bridgeRowYpxGlobal: Float? = null,
        val bridgeRowYpxRoi: Float? = null
    )

    /**
     * placeFilByArc3250
     *
     * Ventanas angulares:
     * - allowedArcFromDeg/allowedArcToDeg: ventana oficial -> si vienen, mandan.
     * - includeDegFrom/includeDegTo: compat legacy -> se usa solo si allowedArc* es null.
     * - excludeDegFrom/excludeDegTo: fallback para evitar arco superior cuando no hay include/allowed.
     */
    fun placeFilByArc3250(
        filPtsMm: List<PointF>,
        filHboxMm: Double,
        filVboxMm: Double,
        edgesu8: ByteArray,
        w: Int,
        h: Int,
        maskU8: ByteArray? = null,
        originPxRoi: PointF,
        eyeSideSign3250: Int,
        pxPerMmInitGuess: Double,
        pxPerMmObserved: Double? = null,
        pxPerMmFixed: Double? = null,
        filOverInnerMmPerSide3250: Double = 0.5,
        allowedArcFromDeg: Double? = null,
        allowedArcToDeg: Double? = null,
        includeDegFrom: Double? = null,
        includeDegTo: Double? = null,
        excludeDegFrom: Double = 35.0,
        excludeDegTo: Double = 145.0,
        stepDeg: Double = 2.0,
        iters: Int = 2,
        rSearchRelLo: Double = 0.70,
        rSearchRelHi: Double = 1.35,
        bottomAnchorYpxRoi: Float? = null,

        // guía oficial del bottom, ROI-local
        bottomGuidePxRoi: List<PointF>? = null,
        bottomGuideTolUpPx3250: Float = 2f,
        bottomGuideTolDownPx3250: Float = 10f,
        enforceBottomGuideIfPresent3250: Boolean = true,

        // guías del detector, ROI-local
        detectorNasalGuidePxRoi: List<PointF>? = null,
        detectorTempleGuidePxRoi: List<PointF>? = null,

        // política de escala
        scalePolicy3250: ScalePolicy3250 = ScalePolicy3250.LOCK_TO_OBSERVED,
        maxRelScaleAdj3250: Double = 0.03,

        // tolerancias de match contra guías detector
        guideNormalTolPx3250: Float = 10f,
        guideTangentialTolPx3250: Float = 24f,
        allowEdgeFallbackWhenNoGuide3250: Boolean = true,

        // pass-through
        rimProfile3250: RimProfile3250? = null,
        bridgeRowYpxGlobal: Float? = null,
        bridgeRowYpxRoi: Float? = null,

        // guardrails internos
        minRunHits3250: Int = 5,
        rmsMaxPx3250: Double = 18.0
    ): Fit3250? {

        if (filPtsMm.size < 10) {
            Log.w(TAG, "placeFilByArc3250: filPtsMm.size=${filPtsMm.size} < 10")
            return null
        }
        if (w <= 0 || h <= 0) {
            Log.w(TAG, "placeFilByArc3250: ROI invalid w=$w h=$h")
            return null
        }
        if (!originPxRoi.x.isFinite() || !originPxRoi.y.isFinite()) {
            Log.w(TAG, "placeFilByArc3250: origin invalid x=${originPxRoi.x} y=${originPxRoi.y}")
            return null
        }
        if (!pxPerMmInitGuess.isFinite() || pxPerMmInitGuess <= 1e-6) {
            Log.w(TAG, "placeFilByArc3250: pxPerMmInitGuess invalid=$pxPerMmInitGuess")
            return null
        }

        val total = w * h
        if (edgesu8.size != total) {
            Log.w(TAG, "edgeMap size mismatch size=${edgesu8.size} w*h=$total")
            return null
        }
        val mU8 = maskU8?.takeIf { it.size == total }

        run {
            var nnz = 0
            var mx = 0
            var nonBinary = 0
            var masked = 0

            for (i in 0 until total) {
                val isM = (mU8 != null && ((mU8[i].toInt() and 0xFF) != 0))
                if (isM) {
                    masked++
                    continue
                }

                val v = (edgesu8[i].toInt() and 0xFF)
                if (v != 0) nnz++
                if (v > mx) mx = v
                if (v != 0 && v != 255) nonBinary++
            }

            val eff = (total - masked).coerceAtLeast(1)
            val frac = nnz.toDouble() / eff.toDouble()
            Log.d(
                TAG,
                "edgeMap density (eff): nnz=$nnz/$eff frac=${"%.5f".format(frac)} " +
                        "max=$mx nonBinary=$nonBinary masked=$masked"
            )

            if (frac < 0.00015) {
                Log.w(TAG, "edgeMap demasiado ralo (eff): frac=${"%.5f".format(frac)} (min=0.00015)")
                return null
            }
        }

        val bottomGuideLocal =
            if (enforceBottomGuideIfPresent3250) normalizeBottomGuide3250(bottomGuidePxRoi) else null

        if (bottomGuideLocal != null) {
            Log.d(
                TAG,
                "ARC-BOTTOM-GUIDE active n=${bottomGuideLocal.size} " +
                        "tolUp=$bottomGuideTolUpPx3250 tolDown=$bottomGuideTolDownPx3250"
            )
        }

        val sideGuidesLocal = normalizeGuideCloud3250(
            detectorNasalGuidePxRoi,
            detectorTempleGuidePxRoi
        )

        val allGuidesLocal = normalizeGuideCloud3250(
            detectorNasalGuidePxRoi,
            detectorTempleGuidePxRoi,
            bottomGuideLocal
        )

        Log.d(
            TAG,
            "ARC-GUIDES side=${sideGuidesLocal.size} all=${allGuidesLocal.size} " +
                    "scalePolicy=$scalePolicy3250 maxRelScaleAdj=$maxRelScaleAdj3250"
        )

        val profile = rimProfile3250 ?: RimProfile3250.FULL_RIM
        val effectiveOver = effectiveFilOverPerSide3250(
            profile3250 = profile,
            filOverInnerMmPerSide = filOverInnerMmPerSide3250
        )

        // ✅ modelo para FIT = INNER
        val polyFitMm = buildCenteredInnerPolyMm(
            filPtsMm = filPtsMm,
            filHboxMm = filHboxMm,
            filVboxMm = filVboxMm,
            filOverInnerMmPerSide = effectiveOver
        )
        if (polyFitMm.size < 10) {
            Log.w(TAG, "polyFitMm.size=${polyFitMm.size} < 10")
            return null
        }

        // ✅ modelo para SALIDA = OUTER original centrado
        val polyOutMm = buildCenteredOuterPolyMm(filPtsMm)
        if (polyOutMm.size < 10) {
            Log.w(TAG, "polyOutMm.size=${polyOutMm.size} < 10")
            return null
        }

        val pxFixed = pxPerMmFixed
            ?.takeIf { it.isFinite() && it > 1e-6 }
            ?.coerceIn(2.5, 20.0)

        val pxObsDbg = pxPerMmObserved
            ?.takeIf { it.isFinite() && it > 1e-6 }
            ?.coerceIn(2.5, 20.0)

        if (pxObsDbg != null) {
            val rel = abs(pxPerMmInitGuess - pxObsDbg) / pxObsDbg
            if (rel > 0.35) {
                Log.w(
                    TAG,
                    "ARC-FIT: initGuess lejos de observed (DBG). " +
                            "guess=${fmt2(pxPerMmInitGuess)} obs=${fmt2(pxObsDbg)} rel=${fmt2(rel)} (no veto)"
                )
            }
        }

        val eyeSide = when {
            eyeSideSign3250 < 0 -> -1
            eyeSideSign3250 > 0 -> +1
            else -> +1
        }
        val nasalUxFixed = (-eyeSide).toDouble()

        val effectiveWindow: Pair<Double, Double>? =
            if (allowedArcFromDeg != null && allowedArcToDeg != null) {
                allowedArcFromDeg to allowedArcToDeg
            } else if (includeDegFrom != null && includeDegTo != null) {
                includeDegFrom to includeDegTo
            } else {
                null
            }

        val hasWin = effectiveWindow != null
        val winLenDeg = effectiveWindow?.let { arcLenDeg3250(it.first, it.second) } ?: 360.0

        fun allowed(deg: Double): Boolean {
            val a = normDeg3250(deg)

            effectiveWindow?.let { win ->
                return containsOnArc3250(a, win.first, win.second)
            }

            val f = normDeg3250(excludeDegFrom)
            val t = normDeg3250(excludeDegTo)
            val exclLen = arcLenDeg3250(f, t)
            if (exclLen <= 1e-6) return true
            return !containsOnArc3250(a, f, t)
        }

        fun edgeStrengthAtN(ix: Int, iy: Int): Float {
            if (ix !in 0 until w || iy !in 0 until h) return 0f
            for (dy in -1..1) {
                val y = iy + dy
                if (y !in 0 until h) continue
                val row = y * w
                for (dx in -1..1) {
                    val x = ix + dx
                    if (x !in 0 until w) continue
                    val idx = row + x

                    if (mU8 != null && ((mU8[idx].toInt() and 0xFF) != 0)) continue
                    val v = (edgesu8[idx].toInt() and 0xFF)
                    if (v != 0) return 1f
                }
            }
            return 0f
        }

        data class PairPQ(
            val pMm: PointF,
            val qRelUp: PointF,
            val weight: Double
        )

        fun intersectRayPolyFitMm(dirUpX: Double, dirUpY: Double): PointF? {
            val n = polyFitMm.size
            var bestT = Double.POSITIVE_INFINITY
            var best: PointF? = null

            fun cross(ax: Double, ay: Double, bx: Double, by: Double) = ax * by - ay * bx

            for (i in 0 until n) {
                val a = polyFitMm[i]
                val b = polyFitMm[(i + 1) % n]
                val ax = a.x.toDouble()
                val ay = a.y.toDouble()
                val ex = (b.x - a.x).toDouble()
                val ey = (b.y - a.y).toDouble()

                val denom = cross(dirUpX, dirUpY, ex, ey)
                if (abs(denom) < 1e-12) continue

                val t = cross(ax, ay, ex, ey) / denom
                val u = cross(ax, ay, dirUpX, dirUpY) / denom

                if (t >= 0.0 && u in 0.0..1.0 && t < bestT) {
                    bestT = t
                    best = PointF((t * dirUpX).toFloat(), (t * dirUpY).toFloat())
                }
            }
            return best
        }

        data class RayHit(
            val qPx: PointF,
            val edge: Float
        )

        fun raycastEdgeBestNearPredictedRadius(
            origin: PointF,
            dirDownX: Double,
            dirDownY: Double,
            rPredPx: Double,
            enforceBottomGuideForThisRay: Boolean
        ): RayHit? {

            val ox = origin.x.toDouble()
            val oy = origin.y.toDouble()

            val t0 = (rPredPx * rSearchRelLo).coerceAtLeast(10.0)
            val t1 = (rPredPx * rSearchRelHi).coerceAtMost(min(w, h) * 0.60)

            var bestHit: RayHit? = null
            var bestScore = Double.POSITIVE_INFINITY

            var t = t0
            while (t <= t1) {
                val x = ox + t * dirDownX
                val y = oy + t * dirDownY
                val ix = x.roundToInt()
                val iy = y.roundToInt()

                if (ix !in 0 until w || iy !in 0 until h) {
                    t += 1.0
                    continue
                }

                if (enforceBottomGuideForThisRay && bottomGuideLocal != null) {
                    val okGuide = passesBottomGuide3250(
                        x = x.toFloat(),
                        y = y.toFloat(),
                        bottomGuide = bottomGuideLocal,
                        tolUpPx = bottomGuideTolUpPx3250,
                        tolDownPx = bottomGuideTolDownPx3250
                    )
                    if (!okGuide) {
                        t += 1.0
                        continue
                    }
                }

                val idx0 = iy * w + ix
                if (mU8 != null && ((mU8[idx0].toInt() and 0xFF) != 0)) {
                    t += 1.0
                    continue
                }

                val ed = edgeStrengthAtN(ix, iy)
                if (ed > 0f) {
                    val d = abs(t - rPredPx)
                    if (d < bestScore) {
                        bestScore = d
                        bestHit = RayHit(PointF(x.toFloat(), y.toFloat()), ed)
                        if (d <= 0.3) break
                    }
                }
                t += 1.0
            }
            return bestHit
        }

        data class BottomAnchorAdjust3250(
            val dy: Float,
            val clamped: Boolean,
            val wanted: Float,
            val dyMin: Float,
            val dyMax: Float
        )

        fun computeBottomAnchorAdjust3250(
            placedFit: List<PointF>,
            yAnchor: Float?
        ): BottomAnchorAdjust3250 {
            if (yAnchor == null || placedFit.isEmpty()) {
                return BottomAnchorAdjust3250(
                    dy = 0f,
                    clamped = false,
                    wanted = 0f,
                    dyMin = 0f,
                    dyMax = 0f
                )
            }

            var bottomActual = Float.NEGATIVE_INFINITY
            var minY = Float.POSITIVE_INFINITY
            var maxY = Float.NEGATIVE_INFINITY
            for (p in placedFit) {
                if (p.y > bottomActual) bottomActual = p.y
                if (p.y < minY) minY = p.y
                if (p.y > maxY) maxY = p.y
            }

            val dyWanted = yAnchor - bottomActual
            if (abs(dyWanted) < 0.25f) {
                return BottomAnchorAdjust3250(
                    dy = 0f,
                    clamped = false,
                    wanted = dyWanted,
                    dyMin = 0f,
                    dyMax = 0f
                )
            }

            val dyMin = -minY
            val dyMax = (h - 1f) - maxY
            val dy = dyWanted.coerceIn(dyMin, dyMax)

            return BottomAnchorAdjust3250(
                dy = dy,
                clamped = abs(dy - dyWanted) > 2f,
                wanted = dyWanted,
                dyMin = dyMin,
                dyMax = dyMax
            )
        }

        fun applyDyToPlaced3250(placed: MutableList<PointF>, dy: Float) {
            if (abs(dy) < 1e-6f) return
            for (p in placed) p.y += dy
        }

        data class FitInternal(
            val placedPxRoi: List<PointF>,   // OUTER final colocado
            val pxPerMm: Double,
            val rotDeg: Double,
            val originUpdated: PointF,
            val rmsPx: Double,
            val used: Int,
            val rimProfile3250: RimProfile3250?,
            val bridgeRowYpxGlobal: Float?,
            val bridgeRowYpxRoi: Float?
        ) {
            fun toPublic(): Fit3250 =
                Fit3250(
                    placedPxRoi = placedPxRoi,
                    pxPerMmFace = pxPerMm,
                    rotDeg = rotDeg,
                    originPxRoi = originUpdated,
                    rmsPx = rmsPx,
                    usedSamples = used,
                    rimProfile3250 = rimProfile3250,
                    bridgeRowYpxGlobal = bridgeRowYpxGlobal,
                    bridgeRowYpxRoi = bridgeRowYpxRoi
                )
        }

        fun solveOnce(
            origin: PointF,
            pxPerMmGuess: Double,
            thetaGuessRad: Double,
            pxPerMmFixedLocal: Double?
        ): FitInternal? {

            val cG = cos(thetaGuessRad)
            val sG = sin(thetaGuessRad)

            data class TmpHit(
                val idx: Int,
                val pMm: PointF,      // ✅ siempre sobre INNER de fit
                val qRelUp: PointF
            )

            val hits = ArrayList<TmpHit>(256)
            var blockedByBottomGuide = 0

            val st = stepDeg.takeIf { it.isFinite() && it > 1e-6 } ?: 2.0
            val allowedSteps = if (hasWin) {
                (kotlin.math.ceil(winLenDeg / st).toInt() + 1).coerceAtLeast(1)
            } else {
                Int.MAX_VALUE
            }

            val minHitsRequired = if (hasWin) {
                max(16, (0.55 * allowedSteps.toDouble()).roundToInt())
            } else {
                50
            }

            val minPairsRequired = if (hasWin) {
                max(16, (0.55 * allowedSteps.toDouble()).roundToInt())
            } else {
                50
            }

            var idx = 0
            var deg = 0.0
            while (deg < 360.0) {

                if (!allowed(deg)) {
                    idx++
                    deg += st
                    continue
                }

                val rad = Math.toRadians(deg)
                val c = cos(rad)
                val s = sin(rad)

                val dirUpModelX = c * nasalUxFixed

                val pMm = intersectRayPolyFitMm(dirUpModelX, s)
                if (pMm == null) {
                    idx++
                    deg += st
                    continue
                }

                val rMm = hypot(pMm.x.toDouble(), pMm.y.toDouble())
                val rPredPx = rMm * pxPerMmGuess

                val dirUpImgX = cG * dirUpModelX - sG * s
                val dirUpImgY = sG * dirUpModelX + cG * s

                val predX = origin.x + (rPredPx * dirUpImgX).toFloat()
                val predY = origin.y - (rPredPx * dirUpImgY).toFloat()

                val requireBottomBand =
                    bottomGuideLocal != null && isLowerArcPoint3250(pMm)

                val guidesForThisRay: List<PointF> =
                    if (requireBottomBand) {
                        if (allGuidesLocal.isNotEmpty()) allGuidesLocal else sideGuidesLocal
                    } else {
                        if (sideGuidesLocal.isNotEmpty()) sideGuidesLocal else allGuidesLocal
                    }

                val guidedHit = guidedHitNearPredicted3250(
                    predX = predX,
                    predY = predY,
                    radialDownX = dirUpImgX.toFloat(),
                    radialDownY = (-dirUpImgY).toFloat(),
                    guides = guidesForThisRay,
                    normalTolPx = guideNormalTolPx3250,
                    tangentialTolPx = guideTangentialTolPx3250,
                    bottomGuide = bottomGuideLocal,
                    bottomTolUpPx = bottomGuideTolUpPx3250,
                    bottomTolDownPx = bottomGuideTolDownPx3250,
                    requireBottomBand = requireBottomBand
                )

                val edgeHit = if (
                    guidedHit == null &&
                    guidesForThisRay.isEmpty() &&
                    allowEdgeFallbackWhenNoGuide3250
                ) {
                    raycastEdgeBestNearPredictedRadius(
                        origin = origin,
                        dirDownX = dirUpImgX,
                        dirDownY = -dirUpImgY,
                        rPredPx = rPredPx,
                        enforceBottomGuideForThisRay = requireBottomBand
                    )?.qPx
                } else {
                    null
                }

                val hitQ = guidedHit ?: edgeHit

                if (hitQ != null) {
                    val qRelUp = PointF(
                        hitQ.x - origin.x,
                        -(hitQ.y - origin.y)
                    )
                    hits.add(TmpHit(idx, pMm, qRelUp))
                } else if (requireBottomBand && bottomGuideLocal != null) {
                    blockedByBottomGuide++
                }

                idx++
                deg += st
            }

            if (bottomGuideLocal != null) {
                Log.d(TAG, "ARC-BOTTOM-GUIDE blockedRays=$blockedByBottomGuide")
            }

            if (hits.size < minHitsRequired) {
                Log.w(
                    TAG,
                    "Arc-fit: insuficientes hits=${hits.size} (min=$minHitsRequired) " +
                            "hasWin=$hasWin winLenDeg=${fmt2(winLenDeg)} stepDeg=${fmt2(st)}"
                )
                return null
            }

            val nSteps = idx.coerceAtLeast(1)
            val hitPresent = BooleanArray(nSteps)
            for (h0 in hits) {
                if (h0.idx in 0 until nSteps) hitPresent[h0.idx] = true
            }

            fun markKeepRuns(minRun: Int): BooleanArray {
                val keep = BooleanArray(nSteps)
                if (minRun <= 1) {
                    for (i in 0 until nSteps) keep[i] = hitPresent[i]
                    return keep
                }

                data class Seg(val s: Int, val e: Int)

                val segs = ArrayList<Seg>()
                var i = 0
                while (i < nSteps) {
                    if (!hitPresent[i]) {
                        i++
                        continue
                    }
                    val s0 = i
                    while (i < nSteps && hitPresent[i]) i++
                    val e0 = i - 1
                    segs.add(Seg(s0, e0))
                }

                if (segs.isEmpty()) return keep

                val first = segs.first()
                val last = segs.last()
                if (segs.size >= 2 && first.s == 0 && last.e == nSteps - 1) {
                    val mergedLen = (last.e - last.s + 1) + (first.e + 1)
                    if (mergedLen >= minRun) {
                        for (k in last.s..last.e) keep[k] = true
                        for (k in 0..first.e) keep[k] = true
                    }
                    for (seg in segs.drop(1).dropLast(1)) {
                        val len = seg.e - seg.s + 1
                        if (len >= minRun) {
                            for (k in seg.s..seg.e) keep[k] = true
                        }
                    }
                    return keep
                }

                for (seg in segs) {
                    val len = seg.e - seg.s + 1
                    if (len >= minRun) {
                        for (k in seg.s..seg.e) keep[k] = true
                    }
                }
                return keep
            }

            val keepIdx = markKeepRuns(minRunHits3250.coerceAtLeast(1))

            val pairs = ArrayList<PairPQ>(hits.size)
            for (hh in hits) {
                if (hh.idx in 0 until nSteps && keepIdx[hh.idx]) {
                    pairs.add(
                        PairPQ(
                            pMm = hh.pMm,
                            qRelUp = hh.qRelUp,
                            weight = 1.0
                        )
                    )
                }
            }

            if (pairs.size < minPairsRequired) {
                Log.w(
                    TAG,
                    "Arc-fit: pairs post-continuidad=${pairs.size} (min=$minPairsRequired) " +
                            "minRunHits=$minRunHits3250 hasWin=$hasWin winLenDeg=${fmt2(winLenDeg)}"
                )
                return null
            }

            val sumW = pairs.sumOf { it.weight }
            if (sumW <= 1e-9) return null

            val meanPx = pairs.sumOf { it.weight * it.pMm.x } / sumW
            val meanPy = pairs.sumOf { it.weight * it.pMm.y } / sumW
            val meanQx = pairs.sumOf { it.weight * it.qRelUp.x } / sumW
            val meanQy = pairs.sumOf { it.weight * it.qRelUp.y } / sumW

            var aAcc = 0.0
            var bAcc = 0.0
            for (pp in pairs) {
                val wgt = pp.weight
                val px = pp.pMm.x.toDouble() - meanPx
                val py = pp.pMm.y.toDouble() - meanPy
                val qx = pp.qRelUp.x.toDouble() - meanQx
                val qy = pp.qRelUp.y.toDouble() - meanQy
                aAcc += wgt * (px * qx + py * qy)
                bAcc += wgt * (px * qy - py * qx)
            }

            val theta = atan2(bAcc, aAcc)
            val ct = cos(theta)
            val st2 = sin(theta)

            val rotDegAbs = abs(Math.toDegrees(theta))
            if (rotDegAbs > 25.0) {
                Log.w(TAG, "Arc-fit: rotación sospechosa rotDeg=${"%.1f".format(rotDegAbs)}° (max=25°)")
                return null
            }

            val pxFixedLocalClamped = pxPerMmFixedLocal
                ?.takeIf { it.isFinite() && it > 1e-6 }
                ?.coerceIn(2.5, 20.0)

            val lockedScale = when {
                pxFixedLocalClamped != null -> pxFixedLocalClamped
                scalePolicy3250 == ScalePolicy3250.LOCK_TO_OBSERVED && pxObsDbg != null -> pxObsDbg
                else -> null
            }

            // ✅ escala única global; nunca por-R
            val sScale = if (lockedScale != null) {
                lockedScale
            } else {
                var denom = 0.0
                var numer = 0.0
                for (pp in pairs) {
                    val wgt = pp.weight
                    val px = pp.pMm.x.toDouble() - meanPx
                    val py = pp.pMm.y.toDouble() - meanPy

                    val rx = ct * px - st2 * py
                    val ry = st2 * px + ct * py
                    denom += wgt * (rx * rx + ry * ry)

                    val qx = pp.qRelUp.x.toDouble() - meanQx
                    val qy = pp.qRelUp.y.toDouble() - meanQy
                    numer += wgt * (qx * rx + qy * ry)
                }
                if (denom <= 1e-12) return null

                var sFree = numer / denom
                if (!sFree.isFinite() || sFree <= 1e-9) return null

                if (scalePolicy3250 == ScalePolicy3250.CLAMP_TO_OBSERVED && pxObsDbg != null) {
                    val maxRel = maxRelScaleAdj3250.coerceIn(0.0, 0.10)
                    val lo = pxObsDbg * (1.0 - maxRel)
                    val hi = pxObsDbg * (1.0 + maxRel)

                    val unclamped = sFree
                    sFree = sFree.coerceIn(lo, hi)

                    if (abs(unclamped - sFree) > 1e-6) {
                        Log.w(
                            TAG,
                            "ARC-SCALE CLAMP obs=${fmt2(pxObsDbg)} free=${fmt2(unclamped)} " +
                                    "used=${fmt2(sFree)} range=[${fmt2(lo)},${fmt2(hi)}]"
                        )
                    }
                }

                sFree
            }

            val txRel = meanQx - sScale * (ct * meanPx - st2 * meanPy)
            val tyRel = meanQy - sScale * (st2 * meanPx + ct * meanPy)

            var newOrigin = PointF(
                (origin.x + txRel).toFloat(),
                (origin.y - tyRel).toFloat()
            )

            // error del fit SIEMPRE sobre INNER
            var err = 0.0
            for (pp in pairs) {
                val wgt = pp.weight
                val px = pp.pMm.x.toDouble()
                val py = pp.pMm.y.toDouble()

                val predX = sScale * (ct * px - st2 * py) + txRel
                val predY = sScale * (st2 * px + ct * py) + tyRel

                val dx = predX - pp.qRelUp.x.toDouble()
                val dy = predY - pp.qRelUp.y.toDouble()
                err += wgt * (dx * dx + dy * dy)
            }

            val rms = sqrt(err / sumW)
            if (!rms.isFinite() || rms > rmsMaxPx3250) {
                Log.w(TAG, "ARC-FIT FAIL: rms=${fmt2(rms)} max=${fmt2(rmsMaxPx3250)} used=${pairs.size}")
                return null
            }

            // ✅ INNER colocado (solo para bottom-anchor y coherencia interna)
            val placedFit = ArrayList<PointF>(polyFitMm.size)
            for (p in polyFitMm) {
                val px = p.x.toDouble()
                val py = p.y.toDouble()

                val xRelUp = sScale * (ct * px - st2 * py) + txRel
                val yRelUp = sScale * (st2 * px + ct * py) + tyRel

                placedFit.add(
                    PointF(
                        (origin.x + xRelUp).toFloat(),
                        (origin.y - yRelUp).toFloat()
                    )
                )
            }

            // ✅ OUTER colocado (resultado final visible)
            val placedOut = ArrayList<PointF>(polyOutMm.size)
            for (p in polyOutMm) {
                val px = p.x.toDouble()
                val py = p.y.toDouble()

                val xRelUp = sScale * (ct * px - st2 * py) + txRel
                val yRelUp = sScale * (st2 * px + ct * py) + tyRel

                placedOut.add(
                    PointF(
                        (origin.x + xRelUp).toFloat(),
                        (origin.y - yRelUp).toFloat()
                    )
                )
            }

            // ✅ el bottom anchor se decide sobre INNER;
            // el mismo dy se aplica a INNER + OUTER + origin
            val adj = computeBottomAnchorAdjust3250(
                placedFit = placedFit,
                yAnchor = bottomAnchorYpxRoi
            )

            if (adj.clamped) {
                Log.w(
                    TAG,
                    "BottomAnchor clamp: wanted=${fmt2(adj.wanted.toDouble())} " +
                            "clamped=${fmt2(adj.dy.toDouble())} " +
                            "(min=${fmt2(adj.dyMin.toDouble())}, max=${fmt2(adj.dyMax.toDouble())})"
                )
            }

            if (abs(adj.dy) >= 0.25f) {
                applyDyToPlaced3250(placedFit, adj.dy)
                applyDyToPlaced3250(placedOut, adj.dy)
                newOrigin = PointF(newOrigin.x, newOrigin.y + adj.dy)
            }

            val allInBounds = placedOut.all { p ->
                p.x in 0f..(w - 1f) && p.y in 0f..(h - 1f)
            }
            if (!allInBounds) {
                val outCount = placedOut.count { p ->
                    p.x !in 0f..(w - 1f) || p.y !in 0f..(h - 1f)
                }
                Log.w(TAG, "Arc-fit: placed OUTER fuera del ROI. outCount=$outCount/${placedOut.size} w=$w h=$h")
                return null
            }

            if (pxObsDbg != null) {
                val rel = abs(sScale - pxObsDbg) / pxObsDbg
                Log.d(
                    TAG,
                    "GUANTE(DBG) obs=${"%.3f".format(pxObsDbg)} " +
                            "scale=${"%.3f".format(sScale)} rel=${"%.3f".format(rel)} " +
                            "rms=${"%.1f".format(rms)} used=${pairs.size} " +
                            "rotDeg=${"%.1f".format(Math.toDegrees(theta))}"
                )
            }

            val placedSpanX = computePlacedSpanXPx3250(placedOut)
            val pxFromPlacedHbox =
                if (filHboxMm > 1e-6) placedSpanX / filHboxMm else Double.NaN

            Log.d(
                TAG,
                "ARC-HBOX placedSpanX=${fmt2(placedSpanX)} " +
                        "filHboxMm=${fmt2(filHboxMm)} pxFromPlacedHbox=${fmt2(pxFromPlacedHbox)} " +
                        "scaleUsed=${fmt2(sScale)}"
            )

            return FitInternal(
                placedPxRoi = placedOut,   // ✅ devolver OUTER final
                pxPerMm = sScale,
                rotDeg = Math.toDegrees(theta),
                originUpdated = newOrigin,
                rmsPx = rms,
                used = pairs.size,
                rimProfile3250 = rimProfile3250,
                bridgeRowYpxGlobal = bridgeRowYpxGlobal,
                bridgeRowYpxRoi = bridgeRowYpxRoi ?: bottomAnchorYpxRoi
            )
        }

        var origin = PointF(originPxRoi.x, originPxRoi.y)
        var guess = pxPerMmInitGuess
        var thetaGuessRad = 0.0

        var last: FitInternal? = null

        repeat(iters.coerceIn(1, 3)) { iterNum ->
            val fit = solveOnce(
                origin = origin,
                pxPerMmGuess = guess,
                thetaGuessRad = thetaGuessRad,
                pxPerMmFixedLocal = pxFixed
            ) ?: return last?.toPublic()

            Log.d(
                TAG,
                "Arc-fit iter=$iterNum pairs=${fit.used} " +
                        "scale=${fmt2(fit.pxPerMm)} rot=${fmt2(fit.rotDeg)}° rms=${fmt2(fit.rmsPx)} " +
                        "origin=(${fmt2(fit.originUpdated.x.toDouble())},${fmt2(fit.originUpdated.y.toDouble())}) " +
                        "profile=${fit.rimProfile3250} bridgeG=${fit.bridgeRowYpxGlobal} bridgeRoi=${fit.bridgeRowYpxRoi}"
            )

            last = fit
            origin = fit.originUpdated
            thetaGuessRad = Math.toRadians(fit.rotDeg)

            guess = when {
                pxFixed != null -> pxFixed
                scalePolicy3250 == ScalePolicy3250.LOCK_TO_OBSERVED && pxObsDbg != null -> pxObsDbg
                else -> fit.pxPerMm
            }
        }

        return last?.toPublic()
    }

    private fun buildCenteredInnerPolyMm(
        filPtsMm: List<PointF>,
        filHboxMm: Double,
        filVboxMm: Double,
        filOverInnerMmPerSide: Double
    ): ArrayList<PointF> {

        if (!filHboxMm.isFinite() || filHboxMm <= 1e-9) return arrayListOf()
        if (!filVboxMm.isFinite() || filVboxMm <= 1e-9) return arrayListOf()

        var minX = Double.POSITIVE_INFINITY
        var maxX = Double.NEGATIVE_INFINITY
        var minY = Double.POSITIVE_INFINITY
        var maxY = Double.NEGATIVE_INFINITY

        for (p in filPtsMm) {
            minX = min(minX, p.x.toDouble())
            maxX = max(maxX, p.x.toDouble())
            minY = min(minY, p.y.toDouble())
            maxY = max(maxY, p.y.toDouble())
        }

        val cx = (minX + maxX) * 0.5
        val cy = (minY + maxY) * 0.5

        val innerW = (filHboxMm - 2.0 * filOverInnerMmPerSide).coerceAtLeast(filHboxMm * 0.80)
        val innerH = (filVboxMm - 2.0 * filOverInnerMmPerSide).coerceAtLeast(filVboxMm * 0.80)

        val sX = (innerW / filHboxMm).coerceIn(0.80, 1.00)
        val sY = (innerH / filVboxMm).coerceIn(0.80, 1.00)

        val poly = ArrayList<PointF>(filPtsMm.size)
        for (p in filPtsMm) {
            poly.add(
                PointF(
                    ((p.x.toDouble() - cx) * sX).toFloat(),
                    ((p.y.toDouble() - cy) * sY).toFloat()
                )
            )
        }
        return poly
    }

    // ✅ OUTER original centrado, sin shrink por filOver
    private fun buildCenteredOuterPolyMm(
        filPtsMm: List<PointF>
    ): ArrayList<PointF> {
        if (filPtsMm.isEmpty()) return arrayListOf()

        var minX = Double.POSITIVE_INFINITY
        var maxX = Double.NEGATIVE_INFINITY
        var minY = Double.POSITIVE_INFINITY
        var maxY = Double.NEGATIVE_INFINITY

        for (p in filPtsMm) {
            minX = min(minX, p.x.toDouble())
            maxX = max(maxX, p.x.toDouble())
            minY = min(minY, p.y.toDouble())
            maxY = max(maxY, p.y.toDouble())
        }

        val cx = (minX + maxX) * 0.5
        val cy = (minY + maxY) * 0.5

        val poly = ArrayList<PointF>(filPtsMm.size)
        for (p in filPtsMm) {
            poly.add(
                PointF(
                    (p.x.toDouble() - cx).toFloat(),
                    (p.y.toDouble() - cy).toFloat()
                )
            )
        }
        return poly
    }

    private fun normalizeBottomGuide3250(
        pts: List<PointF>?
    ): List<PointF>? {
        if (pts.isNullOrEmpty()) return null

        val buckets = LinkedHashMap<Int, MutableList<Float>>()
        for (p in pts) {
            if (!p.x.isFinite() || !p.y.isFinite()) continue
            val k = p.x.roundToInt()
            buckets.getOrPut(k) { ArrayList(1) }.add(p.y)
        }

        if (buckets.size < 2) return null

        return buckets.entries
            .sortedBy { it.key }
            .map { e ->
                val ys = e.value
                val y = ys.average().toFloat()
                PointF(e.key.toFloat(), y)
            }
            .takeIf { it.size >= 2 }
    }

    private fun interpYOnGuide3250(
        guide: List<PointF>,
        x: Float
    ): Float? {
        if (guide.size < 2) return null

        val pts = guide.sortedBy { it.x }

        if (x <= pts.first().x) return pts.first().y
        if (x >= pts.last().x) return pts.last().y

        for (i in 0 until pts.lastIndex) {
            val a = pts[i]
            val b = pts[i + 1]
            if (x >= a.x && x <= b.x) {
                val dx = b.x - a.x
                if (abs(dx) < 1e-3f) return a.y
                val t = (x - a.x) / dx
                return a.y + t * (b.y - a.y)
            }
        }
        return null
    }

    private fun passesBottomGuide3250(
        x: Float,
        y: Float,
        bottomGuide: List<PointF>,
        tolUpPx: Float,
        tolDownPx: Float
    ): Boolean {
        val yGuide = interpYOnGuide3250(bottomGuide, x) ?: return true
        return y >= (yGuide - tolUpPx) && y <= (yGuide + tolDownPx)
    }

    /**
     * Modelo en coords "up":
     * - y > 0 => mitad superior
     * - y < 0 => mitad inferior
     */
    private fun isLowerArcPoint3250(
        pMm: PointF
    ): Boolean = pMm.y < -0.5f

    private fun normDeg3250(d: Double): Double {
        val a = d % 360.0
        return if (a < 0) a + 360.0 else a
    }

    private fun containsOnArc3250(
        targetDeg: Double,
        fromDeg: Double,
        toDeg: Double
    ): Boolean {
        val t = normDeg3250(targetDeg)
        val a = normDeg3250(fromDeg)
        val b = normDeg3250(toDeg)
        return if (a <= b) (t in a..b) else (t >= a || t <= b)
    }

    private fun arcLenDeg3250(
        fromDeg: Double,
        toDeg: Double
    ): Double {
        val a = normDeg3250(fromDeg)
        val b = normDeg3250(toDeg)
        return (b - a + 360.0) % 360.0
    }

    private fun normalizeGuideCloud3250(vararg lists: List<PointF>?): List<PointF> {
        val out = ArrayList<PointF>(256)
        for (list in lists) {
            if (list.isNullOrEmpty()) continue
            for (p in list) {
                if (!p.x.isFinite() || !p.y.isFinite()) continue
                out.add(PointF(p.x, p.y))
            }
        }
        return out
    }

    private fun guidedHitNearPredicted3250(
        predX: Float,
        predY: Float,
        radialDownX: Float,
        radialDownY: Float,
        guides: List<PointF>,
        normalTolPx: Float,
        tangentialTolPx: Float,
        bottomGuide: List<PointF>?,
        bottomTolUpPx: Float,
        bottomTolDownPx: Float,
        requireBottomBand: Boolean
    ): PointF? {
        if (guides.isEmpty()) return null

        val nLen = sqrt(
            radialDownX * radialDownX + radialDownY * radialDownY
        ).coerceAtLeast(1e-6f)

        val nx = radialDownX / nLen
        val ny = radialDownY / nLen

        val tx = -ny
        val ty = nx

        var best: PointF? = null
        var bestScore = Float.POSITIVE_INFINITY

        for (g in guides) {
            val dx = g.x - predX
            val dy = g.y - predY

            val nDist = abs(dx * nx + dy * ny)
            val tDist = abs(dx * tx + dy * ty)

            if (nDist > normalTolPx) continue
            if (tDist > tangentialTolPx) continue

            if (requireBottomBand && bottomGuide != null) {
                val ok = passesBottomGuide3250(
                    x = g.x,
                    y = g.y,
                    bottomGuide = bottomGuide,
                    tolUpPx = bottomTolUpPx,
                    tolDownPx = bottomTolDownPx
                )
                if (!ok) continue
            }

            val score = nDist + 0.35f * tDist
            if (score < bestScore) {
                bestScore = score
                best = g
            }
        }

        return best
    }

    private fun computePlacedSpanXPx3250(pts: List<PointF>): Double {
        if (pts.isEmpty()) return Double.NaN
        var minX = Float.POSITIVE_INFINITY
        var maxX = Float.NEGATIVE_INFINITY
        for (p in pts) {
            if (p.x < minX) minX = p.x
            if (p.x > maxX) maxX = p.x
        }
        return (maxX - minX).toDouble()
    }
}