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
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * FilPlaceFromArc3250
 *
 * Ajuste por arcos (raycast) del FIL contra edgeMap binario (ROI).
 *
 * Regla de Oro (EDGE->ARCFIT):
 * - edgesGray debe ser un edgeMap binario 0/255 (DIVINE o LEGACY binario).
 * - ArcFit lo trata como booleano (edge/no-edge). No “intensidades inventadas”.
 */
object FilPlaceFromArc3250 {

    private const val TAG = "FILFIT_ARC3250"

    data class Fit3250(
        val placedPxRoi: List<PointF>,
        val pxPerMmFace: Double,
        val rotDeg: Double,
        val originPxRoi: PointF,
        val rmsPx: Double,
        val usedSamples: Int
    )

    fun placeFilByArc3250(
        filPtsMm: List<PointF>,
        filHboxMm: Double,
        filVboxMm: Double,

        edgesGray: ByteArray?,
        w: Int,
        h: Int,

        originPxRoi: PointF,
        midlineXpxRoi: Float,
        pxPerMmInitGuess: Double,

        pxPerMmObserved: Double? = null,
        pxPerMmFixed: Double? = null,
        filOverInnerMmPerSide3250: Double,
        excludeDegFrom: Double = 40.0,
        excludeDegTo: Double = 140.0,
        stepDeg: Double = 2.0,
        iters: Int = 2,
        rSearchRelLo: Double = 0.70,
        rSearchRelHi: Double = 1.35,

        bottomAnchorYpxRoi: Float? = null,

        tolScaleRel3250: Double = 0.15,
        rmsMaxPx3250: Double = 20.0,

        // Con edgeMap binario (0/255): edgeStrengthAtN devuelve 0 o 1
        thrEdge01: Float = 0.28f,

        wMin: Double = 0.10,
        rayLambda: Double = 0.85,
        edgePow: Double = 1.4
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

        val eGray = edgesGray ?: run {
            Log.w(TAG, "placeFilByArc3250: edgesGray=null")
            return null
        }

        val total = w * h
        if (eGray.size != total) {
            Log.w(TAG, "edgeMap size mismatch size=${eGray.size} w*h=$total")
            return null
        }

        // =========================
        // EdgeMap sanity + densidad
        // =========================
        run {
            var nnz = 0
            var mx = 0
            var nonBinary = 0

            for (i in 0 until total) {
                val v = (eGray[i].toInt() and 0xFF)
                if (v != 0) nnz++
                if (v > mx) mx = v
                if (v != 0 && v != 255) nonBinary++
            }

            val frac = nnz.toDouble() / total.toDouble()
            Log.d(
                TAG,
                "edgeMap density: nnz=$nnz/$total frac=${"%.5f".format(frac)} max=$mx nonBinary=$nonBinary thrEdge01=$thrEdge01"
            )

            if (frac < 0.00015) {
                Log.w(TAG, "edgeMap demasiado ralo: frac=${"%.5f".format(frac)} (min=0.00015)")
                return null
            }
        }

        val polyMm = buildCenteredInnerPolyMm(
            filPtsMm = filPtsMm,
            filHboxMm = filHboxMm,
            filVboxMm = filVboxMm,
            filOverInnerMmPerSide = filOverInnerMmPerSide3250
        )
        if (polyMm.size < 10) {
            Log.w(TAG, "polyMm.size=${polyMm.size} < 10")
            return null
        }

        // Lock duro SOLO para fixed. observed = constraint blanda.
        val pxFixed = pxPerMmFixed?.takeIf { it.isFinite() && it > 1e-6 }?.coerceIn(2.5, 20.0)
        val pxObs = pxPerMmObserved?.takeIf { it.isFinite() && it > 1e-6 }?.coerceIn(2.5, 20.0)

        if (pxObs != null) {
            val rel = abs(pxPerMmInitGuess - pxObs) / pxObs
            if (rel > max(0.25, tolScaleRel3250 * 2.0)) {
                Log.w(TAG, "ARC-FIT: initGuess lejos de observed. guess=${fmt2(pxPerMmInitGuess)} obs=${fmt2(pxObs)} rel=${fmt2(rel)}")
            }
        }
        if (pxFixed != null) {
            val rel = abs(pxPerMmInitGuess - pxFixed) / pxFixed
            if (rel > max(0.25, tolScaleRel3250 * 2.0)) {
                Log.w(TAG, "ARC-FIT: initGuess lejos de fixed. guess=${fmt2(pxPerMmInitGuess)} fixed=${fmt2(pxFixed)} rel=${fmt2(rel)}")
            }
        }

        // nasalUxFixed: si midline está a la derecha del centro ROI, este ROI es ojo izquierdo (nasal hacia +X)
        // si midline está a la izquierda del centro ROI, este ROI es ojo derecho (nasal hacia -X)
        val nasalUxFixed = if (midlineXpxRoi >= (w * 0.5f)) +1.0 else -1.0

        fun allowed(deg: Double): Boolean {
            val a = ((deg % 360.0) + 360.0) % 360.0
            return a !in excludeDegFrom..excludeDegTo
        }

        /**
         * Edge strength en vecindad 3x3, pero con edgeMap BINARIO:
         * - devuelve 1f si hay algún edge !=0 cerca
         * - si no, 0f
         */
        fun edgeStrengthAtN(ix: Int, iy: Int): Float {
            if (ix !in 0 until w || iy !in 0 until h) return 0f
            for (dy in -1..1) {
                val y = iy + dy
                if (y !in 0 until h) continue
                val row = y * w
                for (dx in -1..1) {
                    val x = ix + dx
                    if (x !in 0 until w) continue
                    val v = (eGray[row + x].toInt() and 0xFF)
                    if (v != 0) return 1f
                }
            }
            return 0f
        }

        data class PairPQ(val pMm: PointF, val qRelUp: PointF, val weight: Double)

        fun intersectRayPolyMm(dirUpX: Double, dirUpY: Double): PointF? {
            val n = polyMm.size
            var bestT = Double.POSITIVE_INFINITY
            var best: PointF? = null

            fun cross(ax: Double, ay: Double, bx: Double, by: Double) =
                ax * by - ay * bx

            for (i in 0 until n) {
                val a = polyMm[i]
                val b = polyMm[(i + 1) % n]
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

        data class RayHit(val qPx: PointF, val edge: Float)

        fun raycastEdgeBestNearPredictedRadius(
            origin: PointF,
            dirDownX: Double,
            dirDownY: Double,
            rPredPx: Double
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

                val ed = edgeStrengthAtN(ix, iy)
                if (ed >= thrEdge01) {
                    val d = abs(t - rPredPx)

                    // Con edgeMap binario: ed es 0/1 => edW es 0/1 (edgePow queda por compat).
                    val edW = ed.toDouble().pow(edgePow).coerceIn(0.0, 1.0)

                    val score = d - rayLambda * edW
                    if (score < bestScore) {
                        bestScore = score
                        bestHit = RayHit(PointF(x.toFloat(), y.toFloat()), ed)
                        if (d <= 0.3 && ed >= 0.85f) break
                    }
                }
                t += 1.0
            }
            return bestHit
        }

        fun applyBottomAnchorIfAny(placed: ArrayList<PointF>, origin: PointF): PointF {
            val yAnchor = bottomAnchorYpxRoi ?: return origin
            if (placed.isEmpty()) return origin

            var bottomActual = Float.NEGATIVE_INFINITY
            var minY = Float.POSITIVE_INFINITY
            var maxY = Float.NEGATIVE_INFINITY
            for (p in placed) {
                if (p.y > bottomActual) bottomActual = p.y
                if (p.y < minY) minY = p.y
                if (p.y > maxY) maxY = p.y
            }

            val dyWanted = (yAnchor - bottomActual)
            if (abs(dyWanted) < 0.25f) return origin

            // clamp global (no deformar por clamping punto-a-punto)
            val dyMin = -minY
            val dyMax = (h - 1f) - maxY
            val dy = dyWanted.coerceIn(dyMin, dyMax)

            if (abs(dy - dyWanted) > 2f) {
                Log.w(TAG, "BottomAnchor clamp: wanted=${fmt2(dyWanted.toDouble())} clamped=${fmt2(dy.toDouble())} (min=${fmt2(dyMin.toDouble())}, max=${fmt2(dyMax.toDouble())})")
            }

            for (p in placed) p.y += dy
            return PointF(origin.x, origin.y + dy)
        }

        data class FitInternal(
            val placedPxRoi: List<PointF>,
            val pxPerMm: Double,
            val rotDeg: Double,
            val originUpdated: PointF,
            val rmsPx: Double,
            val used: Int
        ) {
            fun toPublic() = Fit3250(
                placedPxRoi = placedPxRoi,
                pxPerMmFace = pxPerMm,
                rotDeg = rotDeg,
                originPxRoi = originUpdated,
                rmsPx = rmsPx,
                usedSamples = used
            )
        }

        fun solveOnce(
            origin: PointF,
            pxPerMmGuess: Double,
            thetaGuessRad: Double,
            pxPerMmFixedLocal: Double?,
            pxPerMmObservedLocal: Double?
        ): FitInternal? {

            val pairs = ArrayList<PairPQ>(256)

            val cG = cos(thetaGuessRad)
            val sG = sin(thetaGuessRad)

            var deg = 0.0
            while (deg < 360.0) {
                if (!allowed(deg)) {
                    deg += stepDeg
                    continue
                }

                val rad = Math.toRadians(deg)
                val c = cos(rad)
                val s = sin(rad)

                // Dirección en "modelo Up" (mm)
                val dirUpModelX = c * nasalUxFixed
                val dirUpModelY = s

                val pMm = intersectRayPolyMm(dirUpModelX, dirUpModelY)
                if (pMm == null) {
                    deg += stepDeg
                    continue
                }

                val rMm = hypot(pMm.x.toDouble(), pMm.y.toDouble())
                val rPredPx = rMm * pxPerMmGuess

                // raycast en imagen usando thetaGuess (corrige correspondencias con rotación real)
                val dirUpImgX = cG * dirUpModelX - sG * dirUpModelY
                val dirUpImgY = sG * dirUpModelX + cG * dirUpModelY
                val dirDownX = dirUpImgX
                val dirDownY = -dirUpImgY

                val hit = raycastEdgeBestNearPredictedRadius(
                    origin = origin,
                    dirDownX = dirDownX,
                    dirDownY = dirDownY,
                    rPredPx = rPredPx
                )

                if (hit != null) {
                    val qPx = hit.qPx
                    val qRelUp = PointF(
                        (qPx.x - origin.x),
                        -(qPx.y - origin.y)
                    )

                    val edW = hit.edge.toDouble().pow(edgePow).coerceIn(0.0, 1.0)
                    val wPair = (wMin + (1.0 - wMin) * edW).coerceIn(0.05, 1.0)

                    pairs.add(PairPQ(pMm, qRelUp, wPair))
                }

                deg += stepDeg
            }

            if (pairs.size < 50) {
                Log.w(TAG, "Arc-fit: insuficientes pairs=${pairs.size} (min=50)")
                return null
            }

            val sumW = pairs.sumOf { it.weight }
            if (sumW <= 1e-9) return null

            val meanPx = pairs.sumOf { it.weight * it.pMm.x } / sumW
            val meanPy = pairs.sumOf { it.weight * it.pMm.y } / sumW
            val meanQx = pairs.sumOf { it.weight * it.qRelUp.x } / sumW
            val meanQy = pairs.sumOf { it.weight * it.qRelUp.y } / sumW

            var A = 0.0
            var B = 0.0

            for (pp in pairs) {
                val wgt = pp.weight

                val px = pp.pMm.x.toDouble() - meanPx
                val py = pp.pMm.y.toDouble() - meanPy
                val qx = pp.qRelUp.x.toDouble() - meanQx
                val qy = pp.qRelUp.y.toDouble() - meanQy

                A += wgt * (px * qx + py * qy)
                B += wgt * (px * qy - py * qx)
            }

            val theta = atan2(B, A)
            val ct = cos(theta)
            val st = sin(theta)

            val rotDegAbs = abs(Math.toDegrees(theta))
            if (rotDegAbs > 25.0) {
                Log.w(TAG, "Arc-fit: rotación sospechosa rotDeg=${"%.1f".format(rotDegAbs)}° (max=25°)")
                return null
            }

            // Lock duro SOLO fixed
            val pxFixedLocalClamped = pxPerMmFixedLocal?.takeIf { it.isFinite() && it > 1e-6 }?.coerceIn(2.5, 20.0)

            val sScale = if (pxFixedLocalClamped != null) {
                pxFixedLocalClamped
            } else {
                var denom = 0.0
                var numer = 0.0

                for (pp in pairs) {
                    val wgt = pp.weight

                    val px = pp.pMm.x.toDouble() - meanPx
                    val py = pp.pMm.y.toDouble() - meanPy

                    val rx = ct * px - st * py
                    val ry = st * px + ct * py
                    denom += wgt * (rx * rx + ry * ry)

                    val qx = pp.qRelUp.x.toDouble() - meanQx
                    val qy = pp.qRelUp.y.toDouble() - meanQy
                    numer += wgt * (qx * rx + qy * ry)
                }

                if (denom <= 1e-12) return null

                val sFree = numer / denom
                if (!sFree.isFinite() || sFree <= 1e-9) return null
                sFree
            }

            val txRel = meanQx - sScale * (ct * meanPx - st * meanPy)
            val tyRel = meanQy - sScale * (st * meanPx + ct * meanPy)

            var newOrigin = PointF(
                (origin.x + txRel).toFloat(),
                (origin.y - tyRel).toFloat()
            )

            var err = 0.0
            for (pp in pairs) {
                val wgt = pp.weight
                val px = pp.pMm.x.toDouble()
                val py = pp.pMm.y.toDouble()

                val predX = sScale * (ct * px - st * py) + txRel
                val predY = sScale * (st * px + ct * py) + tyRel

                val dx = predX - pp.qRelUp.x.toDouble()
                val dy = predY - pp.qRelUp.y.toDouble()

                err += wgt * (dx * dx + dy * dy)
            }
            val rms = sqrt(err / sumW)

            // Validación RMS estricta si hay fixed
            if (pxFixedLocalClamped != null) {
                if (!rms.isFinite() || rms > rmsMaxPx3250) {
                    Log.w(TAG, "ARC-FIT FAIL: fixedScale rms=${fmt2(rms)} max=${fmt2(rmsMaxPx3250)} fixedPxMm=${fmt2(pxFixedLocalClamped)} used=${pairs.size}")
                    return null
                }
            }

            val placed = ArrayList<PointF>(polyMm.size)
            for (p in polyMm) {
                val px = p.x.toDouble()
                val py = p.y.toDouble()

                val xRelUp = sScale * (ct * px - st * py) + txRel
                val yRelUp = sScale * (st * px + ct * py) + tyRel

                placed.add(
                    PointF(
                        (origin.x + xRelUp).toFloat(),
                        (origin.y - yRelUp).toFloat()
                    )
                )
            }

            newOrigin = applyBottomAnchorIfAny(placed, newOrigin)

            val allInBounds = placed.all { p ->
                p.x in 0f..(w - 1f) && p.y in 0f..(h - 1f)
            }
            if (!allInBounds) {
                val outCount = placed.count { p ->
                    p.x !in 0f..(w - 1f) || p.y !in 0f..(h - 1f)
                }
                Log.w(TAG, "Arc-fit: placed fuera del ROI. outCount=$outCount/${placed.size} w=$w h=$h")
                return null
            }

            // Constraint blanda contra observed (si existe)
            if (pxPerMmObservedLocal != null && pxPerMmObservedLocal.isFinite() && pxPerMmObservedLocal > 1e-6) {
                val obs = pxPerMmObservedLocal.coerceIn(2.5, 20.0)
                val rel = abs(sScale - obs) / obs
                val okScale = rel <= tolScaleRel3250
                val okRms = rms.isFinite() && rms <= rmsMaxPx3250

                Log.d(
                    TAG,
                    "GUANTE obs=${"%.3f".format(obs)} scale=${"%.3f".format(sScale)} " +
                            "rel=${"%.3f".format(rel)} rms=${"%.1f".format(rms)} " +
                            "okScale=$okScale okRms=$okRms used=${pairs.size} rotDeg=${"%.1f".format(Math.toDegrees(theta))} thetaGuessDeg=${"%.1f".format(Math.toDegrees(thetaGuessRad))}"
                )

                if (!okScale) {
                    Log.w(TAG, "Arc-fit REJECT: escala vs observed fuera de tol. rel=${fmt2(rel)} > tol=${fmt2(tolScaleRel3250)}")
                    return null
                }
                if (!okRms) {
                    Log.w(TAG, "Arc-fit REJECT: RMS muy alto. rms=${fmt2(rms)} > max=${fmt2(rmsMaxPx3250)}")
                    return null
                }
            }

            return FitInternal(
                placedPxRoi = placed,
                pxPerMm = sScale,
                rotDeg = Math.toDegrees(theta),
                originUpdated = newOrigin,
                rmsPx = rms,
                used = pairs.size
            )
        }

        var origin = PointF(originPxRoi.x, originPxRoi.y)
        var guess = pxPerMmInitGuess
        var thetaGuessRad = 0.0

        var last: FitInternal? = null

        val lockFixed = (pxFixed != null)

        repeat(iters.coerceIn(1, 3)) { iterNum ->
            val fit = solveOnce(
                origin = origin,
                pxPerMmGuess = guess,
                thetaGuessRad = thetaGuessRad,
                pxPerMmFixedLocal = pxFixed,
                pxPerMmObservedLocal = pxObs
            ) ?: return last?.toPublic()

            Log.d(
                TAG,
                "Arc-fit iter=$iterNum pairs=${fit.used} scale=${fmt2(fit.pxPerMm)} " +
                        "rot=${fmt2(fit.rotDeg)}° rms=${fmt2(fit.rmsPx)} " +
                        "origin=(${fmt2(fit.originUpdated.x.toDouble())},${fmt2(fit.originUpdated.y.toDouble())})"
            )

            last = fit
            origin = fit.originUpdated
            thetaGuessRad = Math.toRadians(fit.rotDeg)

            guess = if (!lockFixed) {
                fit.pxPerMm
            } else {
                pxFixed
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

        // Shrink X/Y separado usando HBOX/VBOX (evita deformación)
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
}
