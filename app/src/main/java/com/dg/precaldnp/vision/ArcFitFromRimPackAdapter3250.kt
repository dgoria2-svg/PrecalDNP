package com.dg.precaldnp.vision

import android.graphics.PointF
import android.graphics.RectF
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * RimDetectPack3250 (EDGE-IN) -> FilPlaceFromArc3250 (ARC-FIT)
 *
 * ✅ Ancla por arco inferior usando bottomPolylinePx -> ventana angular (deg modelo).
 * ✅ Deriva px/mm oficial desde el FIL calzado (spanX/spanY en ejes del modelo) con helper.
 * ✅ Extrae inner en una línea Y (bridgeRow) con helper.
 *
 * Consume (del pack):
 * - pack.edges / pack.w / pack.h  (edgeMap ROI-local binario 0/255)
 * - pack.result.roiPx             (GLOBAL)
 * - pack.result.bottomYpx         (GLOBAL)
 * - pack.result.innerWidthPx      (seed/observed opcional)
 * - pack.result.bottomPolylinePx  (GLOBAL)  <-- para arco inferior
 *
 * Consume (del seed):
 * - seed.originPxRoi + seed.pxPerMmInitGuess
 */
object ArcFitAdapter3250 {

    private const val BOTTOM_DEG = 270.0

    // ------------------------------------------------------------
    // Outputs auxiliares (escala + inner en bridgeRow)
    // ------------------------------------------------------------
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

    // ------------------------------------------------------------
    // Entry
    // ------------------------------------------------------------
    fun arcFitFromRimPack3250(
        filPtsMm: List<PointF>,
        filHboxMm: Double,
        filVboxMm: Double,
        filOverInnerMmPerSide3250: Double,
        maskU8Roi: ByteArray? = null,
        pack: RimDetectPack3250,
        seed: RimArcSeed3250.ArcSeed3250,

        // -1 OD, +1 OI
        eyeSideSign3250: Int,

        // si ya tenés px/mm “seed/observed” (por ancho), pasalo acá
        pxPerMmObservedOverride: Double? = null,

        // si ya calculaste bottomAnchorYpxRoi afuera, pasalo acá
        bottomAnchorYpxRoiOverride: Float? = null,

        // ventana del arco inferior: cuánto expandimos (deg) si está muy corto
        anchorArcExpandDeg: Double = 18.0,

        iters: Int = 2
    ): FilPlaceFromArc3250.Fit3250? {

        val r = pack.result
        if (r.ok != true) return null

        val over = filOverInnerMmPerSide3250.takeIf { it.isFinite() } ?: 0.5
        val overClamped = over.coerceIn(0.0, 2.0)

        // HBOX/VBOX inner consistentes con buildCenteredInnerPolyMm()
        val hboxInnerMm = (filHboxMm - 2.0 * overClamped).coerceAtLeast(10.0)

        // observed “blando”: por innerWidth (detector) o override
        val pxObsFromInner = run {
            val iw = r.innerWidthPx.toDouble()
            val v = (iw / hboxInnerMm)
            v.takeIf { it.isFinite() && it > 1e-6 }
        }

        val pxObs = pxPerMmObservedOverride
            ?.takeIf { it.isFinite() && it > 1e-6 }
            ?: pxObsFromInner

        val guess = seed.pxPerMmInitGuess
            .takeIf { it.isFinite() && it > 1e-6 }
            ?: pxObs
            ?: 5.0

        // bottomYpx GLOBAL -> ROI-local
        val bottomAnchorYpxRoi: Float? =
            bottomAnchorYpxRoiOverride
                ?.takeIf { it.isFinite() }
                ?: r.bottomYpx
                    .takeIf { it.isFinite() }
                    ?.let { byG -> (byG - r.roiPx.top).toFloat() }

        // ------------------------------------------------------------
        // Ventana angular del ARCO INFERIOR usando bottomPolylinePx
        // (deg modelo de ArcFit: 0 nasal, 90 arriba, 180 temporal, 270 abajo)
        // ------------------------------------------------------------
        val arcWin = buildBottomArcWindowFromPolyline3250(
            bottomPolylineGlobal = r.bottomPolylinePx,
            roiGlobal = r.roiPx,
            originPxRoi = seed.originPxRoi,
            eyeSideSign3250 = eyeSideSign3250,
            expandDeg = anchorArcExpandDeg
        )

        // Fallback si no hay polyline: excluimos el arco superior grande
        val fallbackExcludeFrom = 35.0
        val fallbackExcludeTo = 145.0

        // ------------------------------------------------------------
        // ARC-FIT
        // ------------------------------------------------------------
        return FilPlaceFromArc3250.placeFilByArc3250(
            filPtsMm = filPtsMm,
            filHboxMm = filHboxMm,
            filVboxMm = filVboxMm,

            edgesGray = pack.edges,
            w = pack.w,
            h = pack.h,
            maskU8 = maskU8Roi,

            originPxRoi = seed.originPxRoi,
            eyeSideSign3250 = eyeSideSign3250,

            pxPerMmInitGuess = guess,
            pxPerMmObserved = pxObs,           // guante blando
            pxPerMmFixed = null,               // sin lock duro acá
            filOverInnerMmPerSide3250 = overClamped,

            // ✅ si tenemos arco inferior: usamos allowedArcFrom/To
            allowedArcFromDeg = arcWin?.fromDeg,
            allowedArcToDeg = arcWin?.toDeg,

            // fallback (si no hay ventana de arco)
            excludeDegFrom = if (arcWin == null) fallbackExcludeFrom else 0.0,
            excludeDegTo = if (arcWin == null) fallbackExcludeTo else 0.0,

            stepDeg = 2.0,

            iters = iters.coerceIn(1, 3),
            rSearchRelLo = 0.70,
            rSearchRelHi = 1.35,

            bottomAnchorYpxRoi = bottomAnchorYpxRoi
        )
    }

    // ============================================================
    // 2) px/mm OFICIAL DESDE FIT (FIL colocado) en ejes del MODELO
    // ============================================================
    fun pxMmFromFit3250(
        fit: FilPlaceFromArc3250.Fit3250,
        filHboxMm: Double,
        filVboxMm: Double,
        filOverInnerMmPerSide3250: Double,
        tolRelXY: Double = 0.10
    ): PxMmPack3250? {

        val over = filOverInnerMmPerSide3250.takeIf { it.isFinite() } ?: 0.5
        val overClamped = over.coerceIn(0.0, 2.0)

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

        // Convertimos cada punto ROI (x,y down) a Up-rel, y lo “desrotamos” a ejes de modelo:
        // qRelUp = (dx, dyUp), luego qModel = R(-theta)*qRelUp
        for (p in fit.placedPxRoi) {
            val dx = (p.x - fit.originPxRoi.x).toDouble()
            val dyUp = (-(p.y - fit.originPxRoi.y)).toDouble()

            val xModel = c * dx + s * dyUp
            val yModel = -s * dx + c * dyUp

            minX = min(minX, xModel); maxX = max(maxX, xModel)
            minY = min(minY, yModel); maxY = max(maxY, yModel)
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

    // ============================================================
    // 3) INNER en la línea bridgeRow (GLOBAL) usando el FIL colocado
    // ============================================================
    fun innerAtBridgeRowFromFit3250(
        fit: FilPlaceFromArc3250.Fit3250,
        roiGlobal: RectF,
        bridgeRowYpxGlobal: Float,
        eyeSideSign3250: Int,
        midlineXpxGlobal: Float? = null
    ): InnerAtY3250? {

        if (fit.placedPxRoi.isEmpty()) return null

        val yRoi = (bridgeRowYpxGlobal - roiGlobal.top)
        if (!yRoi.isFinite()) return null

        val xs = intersectPolyWithHLineXs3250(fit.placedPxRoi, yRoi)
        if (xs.size < 2) return null

        val xL = xs.first()
        val xR = xs.last()

        val xL_g = (xL + roiGlobal.left)
        val xR_g = (xR + roiGlobal.left)

        // nasal = hacia midline
        val nasal = run {
            if (midlineXpxGlobal != null && midlineXpxGlobal.isFinite()) {
                val dL = abs(xL_g - midlineXpxGlobal)
                val dR = abs(xR_g - midlineXpxGlobal)
                if (dL <= dR) xL_g else xR_g
            } else {
                // fallback por lado: OD(-1) nasal es el de la derecha (hacia midline),
                // OI(+1) nasal es el de la izquierda.
                if (eyeSideSign3250 < 0) xR_g else xL_g
            }
        }

        val temple = if (nasal == xL_g) xR_g else xL_g

        return InnerAtY3250(
            yGlobal = bridgeRowYpxGlobal,
            innerLeftXGlobal = xL_g,
            innerRightXGlobal = xR_g,
            nasalXGlobal = nasal,
            templeXGlobal = temple
        )
    }

    // ============================================================
    // Helpers geom
    // ============================================================
    private fun intersectPolyWithHLineXs3250(poly: List<PointF>, y: Float): FloatArray {
        if (poly.size < 3) return floatArrayOf()

        val xs = ArrayList<Float>(8)
        val n = poly.size

        fun addX(x: Float) {
            if (!x.isFinite()) return
            xs.add(x)
        }

        for (i in 0 until n) {
            val a = poly[i]
            val b = poly[(i + 1) % n]
            val y0 = a.y
            val y1 = b.y

            // cruce half-open para evitar duplicados en vértices
            val crosses = (y0 <= y && y < y1) || (y1 <= y && y < y0)
            if (!crosses) continue

            val dy = (y1 - y0)
            if (abs(dy) < 1e-6f) continue

            val t = (y - y0) / dy
            val x = a.x + t * (b.x - a.x)
            addX(x)
        }

        if (xs.size < 2) return floatArrayOf()

        xs.sort()
        // colapsa casi-duplicados
        val out = ArrayList<Float>(xs.size)
        var last = Float.NaN
        for (x in xs) {
            if (!last.isFinite() || abs(x - last) > 0.75f) {
                out.add(x); last = x
            }
        }
        if (out.size < 2) return floatArrayOf()
        return floatArrayOf(out.first(), out.last())
    }

    // ------------------------------------------------------------
    // Ventana angular por arco inferior desde bottomPolyline
    // ------------------------------------------------------------
    private fun buildBottomArcWindowFromPolyline3250(
        bottomPolylineGlobal: List<PointF>?,
        roiGlobal: RectF,
        originPxRoi: PointF,
        eyeSideSign3250: Int,
        expandDeg: Double,
        minLenDeg: Double = 40.0,     // mínimo arco permitido
        gapSplitDeg: Double = 10.0,   // gap para cortar “nichos”
        maxDistToBottomDeg: Double = 55.0, // si el cluster no contiene 270 y está lejos → fallback
        maxLenDeg: Double = 200.0     // cap para no abrir ventanas gigantes
    ): ArcWindow3250? {

        val poly = bottomPolylineGlobal ?: return null
        if (poly.size < 3) return null

        // 1) poly GLOBAL -> ROI + deg modelo (unwrapped)
        val degs = ArrayList<Double>(poly.size)
        for (p in poly) {
            val qRoi = PointF(p.x - roiGlobal.left, p.y - roiGlobal.top)
            val d = degFromPointRoi3250(qRoi, originPxRoi, eyeSideSign3250)

            // unwrap alrededor del wrap 0°: lo que cae 0..90 lo llevamos a 360..450
            val du = if (d < 180.0) d + 360.0 else d
            degs.add(du)
        }
        if (degs.size < 3) return null
        degs.sort()

        // 2) clusters por continuidad (nichos)
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

        // 3) elegir cluster: contiene 270 primero; si no, el más cercano (con soporte)
        var best: Cl? = null
        var bestScore = Double.NEGATIVE_INFINITY
        for (c in clusters) {
            val contains = (BOTTOM_DEG in c.a..c.b)
            val dist = distToInterval(BOTTOM_DEG, c.a, c.b)
            val score = (if (contains) 1e6 else 0.0) + (c.n * 1000.0) - (dist * 10.0)
            if (score > bestScore) {
                bestScore = score
                best = c
            }
        }
        val c0 = best ?: return null

        val containsBottom = (BOTTOM_DEG in c0.a..c0.b)
        val distBottom = distToInterval(BOTTOM_DEG, c0.a, c0.b)

        // 4) ventana base (unwrapped)
        var fromU: Double
        var toU: Double

        if (!containsBottom && distBottom > maxDistToBottomDeg) {
            // fallback: ventana explícita centrada en 270
            val minLen = minLenDeg.coerceIn(20.0, 140.0)
            val half = 0.5 * minLen
            fromU = BOTTOM_DEG - half
            toU = BOTTOM_DEG + half
        } else {
            fromU = c0.a
            toU = c0.b
        }

        // 5) asegurar mínimo largo (pero SIN perder el bottom)
        val minLen = minLenDeg.coerceIn(20.0, 140.0)
        var lenU = (toU - fromU).coerceAtLeast(1e-6)
        if (lenU < minLen) {
            val mid = if (BOTTOM_DEG in fromU..toU) 0.5 * (fromU + toU) else BOTTOM_DEG
            val half = 0.5 * minLen
            fromU = mid - half
            toU = mid + half
            lenU = (toU - fromU)
        }

        // 6) cap para no abrir ventanas gigantes
        if (lenU > maxLenDeg) {
            val mid = BOTTOM_DEG
            val half = 0.5 * maxLenDeg
            fromU = mid - half
            toU = mid + half
            lenU = (toU - fromU)
        }

        // 7) expand extra (tu knob)
        val exp = expandDeg.coerceIn(0.0, 40.0)
        fromU -= exp
        toU += exp

        // 8) volver a [0..360)
        val from = normDeg3250(fromU)
        val to = normDeg3250(toU)

        // 9) validación dura: tiene que incluir 270 en el arco from->to (con wrap)
        if (!containsOnArc3250(BOTTOM_DEG, from, to)) {
            // fallback último recurso: ventana explícita alrededor de 270
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
        return if (a < 0) a + 360.0 else a
    }

    private fun containsOnArc3250(target: Double, from: Double, to: Double): Boolean {
        val t = normDeg3250(target)
        val a = normDeg3250(from)
        val b = normDeg3250(to)
        return if (a <= b) (t in a..b) else (t >= a || t <= b) // wrap
    }

    private fun arcLen3250(from: Double, to: Double): Double {
        val a = normDeg3250(from)
        val b = normDeg3250(to)
        return (b - a + 360.0) % 360.0
    }

    /**
     * Convierte un punto ROI (x,y down) a “deg del modelo” de ArcFit:
     * - Y se pasa a Up (dyUp = -(y-origin.y))
     * - X se pasa a “eje nasal” usando nasalUxFixed = -eyeSideSign
     */
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
}
