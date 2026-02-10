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

/**
 * FilPlaceFromArc3250
 *
 * Ajuste por arcos (raycast) del FIL contra edgeMap binario (ROI).
 *
 * Regla de Oro (EDGE->ARCFIT):
 * - edgesGray debe ser un edgeMap binario 0/255 (DIVINE o LEGACY binario).
 * - ArcFit lo trata como booleano (edge/no-edge). No “intensidades inventadas”.
 *
 * Regla 3250 (lado):
 * - OD (ojo derecho) en la FOTO está a la IZQUIERDA de la midline => eyeSideSign=-1
 * - OI (ojo izquierdo) en la FOTO está a la DERECHA de la midline => eyeSideSign=+1
 * - nasal siempre apunta hacia la midline => nasalUx = -eyeSideSign
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

    /**
     * placeFilByArc3250
     *
     * Ventanas angulares:
     * - allowedArcFromDeg/allowedArcToDeg: ventana “oficial” (p.ej. bottom arc) -> si vienen, mandan.
     * - includeDegFrom/includeDegTo: ventana candidata (legacy / compat) -> se usa SOLO si allowedArc* es null.
     * - excludeDegFrom/excludeDegTo: fallback para NO muestrear arco superior (si no hay ventana include/allowed).
     */
    fun placeFilByArc3250(
        filPtsMm: List<PointF>,
        filHboxMm: Double,
        filVboxMm: Double,

        // ✅ COMPAT: si te lo pasan desde afuera, ahora SE USA (si allowedArc* no viene)
        includeDegFrom: Double? = null,
        includeDegTo: Double? = null,

        edgesGray: ByteArray?,
        w: Int,
        h: Int,

        originPxRoi: PointF,

        // -1 = OD (xEye < midline), +1 = OI (xEye > midline)
        eyeSideSign3250: Int,

        pxPerMmInitGuess: Double,

        // ✅ NUEVO (ROI-local, mismo w*h)
        maskU8: ByteArray? = null,

        // ✅ ventana “oficial” (wrap OK). Si viene, manda sobre includeDeg*
        allowedArcFromDeg: Double? = null,
        allowedArcToDeg: Double? = null,

        // ⚠️ observed NO se usa como veto (porque puede ser arco parcial). Se ignora para reject.
        pxPerMmObserved: Double? = null,

        pxPerMmFixed: Double? = null,
        filOverInnerMmPerSide3250: Double,

        // fallback si no hay ventana allowed/include
        excludeDegFrom: Double = 70.0,
        excludeDegTo: Double = 120.0,

        stepDeg: Double = 2.0,

        // continuidad mínima de hits (para descartar puntos sueltos)
        minRunHits3250: Int = 4,

        iters: Int = 2,
        rSearchRelLo: Double = 0.70,
        rSearchRelHi: Double = 1.35,

        bottomAnchorYpxRoi: Float? = null,

        // guardrails
        rmsMaxPx3250: Double = 20.0
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
        val mU8 = maskU8?.takeIf { it.size == total }

        run {
            var nnz = 0
            var mx = 0
            var nonBinary = 0
            var masked = 0

            for (i in 0 until total) {
                val isM = (mU8 != null && ((mU8[i].toInt() and 0xFF) != 0))
                if (isM) { masked++; continue }

                val v = (eGray[i].toInt() and 0xFF)
                if (v != 0) nnz++
                if (v > mx) mx = v
                if (v != 0 && v != 255) nonBinary++
            }

            val eff = (total - masked).coerceAtLeast(1)
            val frac = nnz.toDouble() / eff.toDouble()
            Log.d(TAG, "edgeMap density (eff): nnz=$nnz/$eff frac=${"%.5f".format(frac)} max=$mx nonBinary=$nonBinary masked=$masked")

            if (frac < 0.00015) {
                Log.w(TAG, "edgeMap demasiado ralo (eff): frac=${"%.5f".format(frac)} (min=0.00015)")
                return null
            }
        }
        // =========================
        // EdgeMap sanity + densidad
        // =========================

        // Modelo inner centrado (tu lógica actual)
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

        // Lock duro SOLO para fixed.
        val pxFixed = pxPerMmFixed?.takeIf { it.isFinite() && it > 1e-6 }?.coerceIn(2.5, 20.0)

        // observed: solo debug, NO veto
        val pxObsDbg = pxPerMmObserved?.takeIf { it.isFinite() && it > 1e-6 }?.coerceIn(2.5, 20.0)
        if (pxObsDbg != null) {
            val rel = abs(pxPerMmInitGuess - pxObsDbg) / pxObsDbg
            if (rel > 0.35) {
                Log.w(TAG, "ARC-FIT: initGuess lejos de observed (DBG). guess=${fmt2(pxPerMmInitGuess)} obs=${fmt2(pxObsDbg)} rel=${fmt2(rel)} (no veto)")
            }
        }

        // nasal siempre apunta a la midline => nasalUx = -eyeSideSign
        val eyeSide = when {
            eyeSideSign3250 < 0 -> -1
            eyeSideSign3250 > 0 -> +1
            else -> +1
        }
        val nasalUxFixed = (-eyeSide).toDouble()

        // ------------------------------------------------------------
        // Ventana angular efectiva (precedencia):
        // allowedArc* > includeDeg* > null (fallback exclude)
        // ------------------------------------------------------------
        val winFromDeg = allowedArcFromDeg ?: includeDegFrom
        val winToDeg = allowedArcToDeg ?: includeDegTo
        val hasWin = (winFromDeg != null && winToDeg != null)
        val winLenDeg = if (hasWin) arcLenDeg3250(winFromDeg, winToDeg) else 360.0

        fun allowed(deg: Double): Boolean {
            val a = normDeg3250(deg)

            // ✅ si tengo ventana (allowed/include): SOLO muestreo ahí
            if (hasWin) return containsOnArc3250(a, winFromDeg, winToDeg)

            // fallback: excluye arco superior (wrap OK)
            val f = normDeg3250(excludeDegFrom)
            val t = normDeg3250(excludeDegTo)
            val exclLen = arcLenDeg3250(f, t)
            if (exclLen <= 1e-6) return true
            return !containsOnArc3250(a, f, t)
        }

        /**
         * Edge strength en vecindad 3x3, con edgeMap BINARIO:
         * - devuelve 1f si hay algún edge !=0 cerca
         * - si no, 0f
         *
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
                    val idx = row + x

                    if (mU8 != null && ((mU8[idx].toInt() and 0xFF) != 0)) continue // ✅ veto
                    val v = (eGray[idx].toInt() and 0xFF)
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

            fun cross(ax: Double, ay: Double, bx: Double, by: Double) = ax * by - ay * bx

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

                // (opcional pero recomendado) bounds rápido
                if (ix !in 0 until w || iy !in 0 until h) {
                    t += 1.0
                    continue
                }

                // ✅ ACÁ va tu snippet (veto por máscara en el punto central)
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
            pxPerMmFixedLocal: Double?
        ): FitInternal? {

            val cG = cos(thetaGuessRad)
            val sG = sin(thetaGuessRad)

            data class TmpHit(val idx: Int, val pMm: PointF, val qRelUp: PointF)

            val hits = ArrayList<TmpHit>(256)

            // mínimos dinámicos si hay ventana
            val st = stepDeg.takeIf { it.isFinite() && it > 1e-6 } ?: 2.0
            val allowedSteps = if (hasWin) (kotlin.math.ceil(winLenDeg / st).toInt() + 1).coerceAtLeast(1) else Int.MAX_VALUE

            val minHitsRequired = if (hasWin) max(16, (0.55 * allowedSteps.toDouble()).roundToInt()) else 50
            val minPairsRequired = if (hasWin) max(16, (0.55 * allowedSteps.toDouble()).roundToInt()) else 50

            // sampleo angular
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

                // Dirección en "modelo Up" (mm)
                val dirUpModelX = c * nasalUxFixed

                val pMm = intersectRayPolyMm(dirUpModelX, s)
                if (pMm == null) {
                    idx++
                    deg += st
                    continue
                }

                val rMm = hypot(pMm.x.toDouble(), pMm.y.toDouble())
                val rPredPx = rMm * pxPerMmGuess

                // raycast en imagen usando thetaGuess
                val dirUpImgX = cG * dirUpModelX - sG * s
                val dirUpImgY = sG * dirUpModelX + cG * s
                val dirDownY = -dirUpImgY

                val hit = raycastEdgeBestNearPredictedRadius(
                    origin = origin,
                    dirDownX = dirUpImgX,
                    dirDownY = dirDownY,
                    rPredPx = rPredPx
                )

                if (hit != null) {
                    val qPx = hit.qPx
                    val qRelUp = PointF(
                        (qPx.x - origin.x),
                        -(qPx.y - origin.y)
                    )
                    hits.add(TmpHit(idx, pMm, qRelUp))
                }

                idx++
                deg += st
            }

            if (hits.size < minHitsRequired) {
                Log.w(
                    TAG,
                    "Arc-fit: insuficientes hits=${hits.size} (min=$minHitsRequired) hasWin=$hasWin winLenDeg=${fmt2(winLenDeg)} stepDeg=${fmt2(st)}"
                )
                return null
            }

            // ========= Continuidad angular (runs) =========
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

                data class Seg(val s: Int, val e: Int) // inclusive
                val segs = ArrayList<Seg>()

                var i = 0
                while (i < nSteps) {
                    if (!hitPresent[i]) { i++; continue }
                    val s0 = i
                    while (i < nSteps && hitPresent[i]) i++
                    val e0 = i - 1
                    segs.add(Seg(s0, e0))
                }

                if (segs.isEmpty()) return keep

                // merge wrap
                val first = segs.first()
                val last = segs.last()
                if (segs.size >= 2 && first.s == 0 && last.e == nSteps - 1) {
                    val mergedLen = (last.e - last.s + 1) + (first.e - 0 + 1)
                    if (mergedLen >= minRun) {
                        for (k in last.s..last.e) keep[k] = true
                        for (k in 0..first.e) keep[k] = true
                    }
                    for (seg in segs.drop(1).dropLast(1)) {
                        val len = seg.e - seg.s + 1
                        if (len >= minRun) for (k in seg.s..seg.e) keep[k] = true
                    }
                    return keep
                }

                for (seg in segs) {
                    val len = seg.e - seg.s + 1
                    if (len >= minRun) for (k in seg.s..seg.e) keep[k] = true
                }
                return keep
            }

            val keepIdx = markKeepRuns(minRunHits3250.coerceAtLeast(1))
            val pairs = ArrayList<PairPQ>(hits.size)
            for (hh in hits) {
                if (hh.idx in 0 until nSteps && keepIdx[hh.idx]) {
                    pairs.add(PairPQ(hh.pMm, hh.qRelUp, 1.0))
                }
            }

            if (pairs.size < minPairsRequired) {
                Log.w(TAG, "Arc-fit: pairs post-continuidad=${pairs.size} (min=$minPairsRequired) minRunHits=$minRunHits3250 hasWin=$hasWin winLenDeg=${fmt2(winLenDeg)}")
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
            val st2 = sin(theta)

            val rotDegAbs = abs(Math.toDegrees(theta))
            if (rotDegAbs > 25.0) {
                Log.w(TAG, "Arc-fit: rotación sospechosa rotDeg=${"%.1f".format(rotDegAbs)}° (max=25°)")
                return null
            }

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

                    val rx = ct * px - st2 * py
                    val ry = st2 * px + ct * py
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

            val txRel = meanQx - sScale * (ct * meanPx - st2 * meanPy)
            val tyRel = meanQy - sScale * (st2 * meanPx + ct * meanPy)

            var newOrigin = PointF(
                (origin.x + txRel).toFloat(),
                (origin.y - tyRel).toFloat()
            )

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

            val placed = ArrayList<PointF>(polyMm.size)
            for (p in polyMm) {
                val px = p.x.toDouble()
                val py = p.y.toDouble()

                val xRelUp = sScale * (ct * px - st2 * py) + txRel
                val yRelUp = sScale * (st2 * px + ct * py) + tyRel

                placed.add(
                    PointF(
                        (origin.x + xRelUp).toFloat(),
                        (origin.y - yRelUp).toFloat()
                    )
                )
            }

            newOrigin = applyBottomAnchorIfAny(placed, newOrigin)

            val allInBounds = placed.all { p -> p.x in 0f..(w - 1f) && p.y in 0f..(h - 1f) }
            if (!allInBounds) {
                val outCount = placed.count { p -> p.x !in 0f..(w - 1f) || p.y !in 0f..(h - 1f) }
                Log.w(TAG, "Arc-fit: placed fuera del ROI. outCount=$outCount/${placed.size} w=$w h=$h")
                return null
            }

            // debug (sin veto por observed)
            if (pxObsDbg != null) {
                val rel = abs(sScale - pxObsDbg) / pxObsDbg
                Log.d(TAG, "GUANTE(DBG) obs=${"%.3f".format(pxObsDbg)} scale=${"%.3f".format(sScale)} rel=${"%.3f".format(rel)} rms=${"%.1f".format(rms)} used=${pairs.size} rotDeg=${"%.1f".format(Math.toDegrees(theta))}")
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
                pxPerMmFixedLocal = pxFixed
            ) ?: return last?.toPublic()

            Log.d(TAG, "Arc-fit iter=$iterNum pairs=${fit.used} scale=${fmt2(fit.pxPerMm)} rot=${fmt2(fit.rotDeg)}° rms=${fmt2(fit.rmsPx)} origin=(${fmt2(fit.originUpdated.x.toDouble())},${fmt2(fit.originUpdated.y.toDouble())})")

            last = fit
            origin = fit.originUpdated
            thetaGuessRad = Math.toRadians(fit.rotDeg)
            guess = if (!lockFixed) fit.pxPerMm else pxFixed
        }

        return last?.toPublic()
    }

    // ------------------------------------------------------------
    // Modelo inner centrado (tu lógica)
    // ------------------------------------------------------------
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

    // ------------------------------------------------------------
    // Helpers angulares UNICOS (sin duplicación)
    // ------------------------------------------------------------
    private fun normDeg3250(d: Double): Double {
        val a = d % 360.0
        return if (a < 0) a + 360.0 else a
    }

    private fun containsOnArc3250(targetDeg: Double, fromDeg: Double, toDeg: Double): Boolean {
        val t = normDeg3250(targetDeg)
        val a = normDeg3250(fromDeg)
        val b = normDeg3250(toDeg)
        return if (a <= b) (t in a..b) else (t >= a || t <= b) // wrap
    }

    private fun arcLenDeg3250(fromDeg: Double, toDeg: Double): Double {
        val a = normDeg3250(fromDeg)
        val b = normDeg3250(toDeg)
        return (b - a + 360.0) % 360.0
    }
}
