package com.dg.precaldnp.vision

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.util.ArrayDeque
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * RefCalibrator75 (3250) - Plantilla A4: 4 marcadores (cuadrado+cuadradito) + círculo Ø100 + barra 100mm.
 *
 * Estado corregido:
 * - La referencia real es Ø100 + barra 100 mm (el sufijo 75 quedó legacy).
 * - Se rectifica primero por cuadrados.
 * - El círculo se detecta con Hough + sweep radial con espesor.
 * - La barra valida/refina X, pero el círculo sigue siendo la verdad principal.
 * - La orientación de página sale primero del quad (si existe), no de la barra.
 * - En stills, por defecto NO usa hold temporal global.
 */
object RefCalibrator75 {

    data class Ref75(
        val pxPerMm: Float,
        val cx: Float,
        val cy: Float,
        val rPx: Float,
        val thetaPageDeg: Float,
        val barMidX: Float?,
        val barMidY: Float?,
        val barLenMm: Float?,
        val coverageCircle: Float,
        val sharpness: Float,
        val confidence: Float,
        val pxPerMmX: Float,
        val pxPerMmY: Float
    )

    private const val TAG = "RefCalibrator75"

    private const val MARKERS_CENTER_SQUARE_MM = 150.0
    private const val BAR_LEN_MM = 100.0
    private const val CIRCLE_DIAM_MM = 100.0

    private const val RECT_PX_PER_MM_NOM = 10.0

    private const val DETECT_MAX_W = 1600
    private const val TOPK_PER_QUAD = 2
    private const val CORNER_FRAC = 0.42

    private const val CENTER_X0 = 0.16
    private const val CENTER_X1 = 0.84
    private const val CENTER_Y0 = 0.12
    private const val CENTER_Y1 = 0.95

    private const val GEOM_RATIO_SOFT = 0.35
    private const val CENTER_SOFT = 1.25
    private const val SIZE_SOFT = 0.90

    // ---------------------- VALIDACIÓN/ESTABILIDAD px/mm ----------------------

    private const val BAND_FRAC = 0.12f
    private const val ANGLE_STEP_DEG = 3

    // Gate mínimo del círculo
    private const val MIN_COV_SWEEP = 0.85
    private const val MAX_MAD_REL = 0.030
    private const val MAX_DIFF_HS = 0.14

    // ESPESOR: si el círculo es una línea “real”, Canny suele dar 2 bordes.
    private const val THICK_MIN_PX = 2.0
    private const val THICK_MAX_PX = 18.0
    private const val MIN_THICK_HIT_FRAC = 0.25

    // Consistencia barra vs círculo
    private const val MAX_BAR_CIRCLE_REL_ERR = 0.08f

    // Estabilidad temporal (solo si useTemporalHold = true)
    private const val REF_MAX_SAMPLES = 7
    private const val MAX_JUMP_REL = 0.08

    private val refRadiusSamples = ArrayDeque<Double>(REF_MAX_SAMPLES)
    private val refCxSamples = ArrayDeque<Double>(REF_MAX_SAMPLES)
    private val refCySamples = ArrayDeque<Double>(REF_MAX_SAMPLES)

    fun clearTemporalState() {
        refRadiusSamples.clear()
        refCxSamples.clear()
        refCySamples.clear()
    }

    fun detect(bmp: Bitmap, useTemporalHold: Boolean = false): Ref75? {
        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "OpenCV initLocal() = false")
            return null
        }

        val rgbaFull = Mat(bmp.height, bmp.width, CvType.CV_8UC4)
        Utils.bitmapToMat(bmp, rgbaFull)

        val rect = tryRectifyWithSquares3250(rgbaFull)
        val rgbaWork = rect?.rgbaRect ?: rgbaFull
        val hInv = rect?.Hinv

        val wWork = rgbaWork.cols()
        val hWork = rgbaWork.rows()

        val roi = Rect(
            (CENTER_X0 * wWork).roundToInt().coerceIn(0, wWork - 2),
            (CENTER_Y0 * hWork).roundToInt().coerceIn(0, hWork - 2),
            ((CENTER_X1 - CENTER_X0) * wWork).roundToInt().coerceAtLeast(2).coerceAtMost(wWork),
            ((CENTER_Y1 - CENTER_Y0) * hWork).roundToInt().coerceAtLeast(2).coerceAtMost(hWork)
        ).let { r ->
            val x0 = r.x.coerceIn(0, wWork - 2)
            val y0 = r.y.coerceIn(0, hWork - 2)
            val ww = r.width.coerceIn(2, wWork - x0)
            val hh = r.height.coerceIn(2, hWork - y0)
            Rect(x0, y0, ww, hh)
        }

        val rgbaRoi = rgbaWork.submat(roi)

        val gray = Mat()
        Imgproc.cvtColor(rgbaRoi, gray, Imgproc.COLOR_RGBA2GRAY)

        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        clahe.apply(gray, gray)
        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)

        val edges = Mat()
        Imgproc.Canny(gray, edges, 50.0, 150.0)

        val circles = Mat()
        Imgproc.HoughCircles(
            gray,
            circles,
            Imgproc.HOUGH_GRADIENT,
            1.5,
            (gray.rows() / 8).toDouble(),
            120.0,
            30.0,
            (0.35 * min(gray.cols(), gray.rows())).toInt(),
            (0.70 * min(gray.cols(), gray.rows())).toInt()
        )

        if (circles.empty()) {
            safeRelease(rgbaRoi, circles, edges, gray)
            rect?.release()
            rgbaFull.release()
            return null
        }

        val edgeU8 = matToU8(edges)

        val circleCand = selectBestCircleCandidate(
            circles = circles,
            edgeU8 = edgeU8,
            w = edges.cols(),
            h = edges.rows()
        )

        if (circleCand == null) {
            safeRelease(rgbaRoi, circles, edges, gray)
            rect?.release()
            rgbaFull.release()
            return null
        }

        val cxDetRoi = circleCand.cx
        val cyDetRoi = circleCand.cy
        val rH = circleCand.rH
        val rS = circleCand.sweep.rMed
        val covS = circleCand.covS
        val madRel = circleCand.madRel
        val diffHS = circleCand.diffHS

        val prevStable = if (useTemporalHold) medianOrNull(refRadiusSamples) else null
        val prevCx = if (useTemporalHold) medianOrNull(refCxSamples) else null
        val prevCy = if (useTemporalHold) medianOrNull(refCySamples) else null
        val jumpRel =
            if (prevStable == null || rS.isNaN() || prevStable <= 1e-6) 0.0
            else abs(rS - prevStable) / prevStable

        var q = 0
        var rStable: Double? = null
        var cxStable: Double? = null
        var cyStable: Double? = null

        if (circleCand.ok) {
            if (!useTemporalHold) {
                rStable = rS
                cxStable = cxDetRoi
                cyStable = cyDetRoi
                q = 1
            } else {
                if (prevStable == null || jumpRel <= MAX_JUMP_REL) {
                    refRadiusSamples.addLast(rS)
                    while (refRadiusSamples.size > REF_MAX_SAMPLES) refRadiusSamples.removeFirst()
                    rStable = medianOrNull(refRadiusSamples)

                    refCxSamples.addLast(cxDetRoi)
                    refCySamples.addLast(cyDetRoi)
                    while (refCxSamples.size > REF_MAX_SAMPLES) refCxSamples.removeFirst()
                    while (refCySamples.size > REF_MAX_SAMPLES) refCySamples.removeFirst()
                    cxStable = medianOrNull(refCxSamples)
                    cyStable = medianOrNull(refCySamples)
                    q = 1
                } else {
                    rStable = prevStable
                    cxStable = prevCx
                    cyStable = prevCy
                    q = 0
                }
            }
        } else if (useTemporalHold) {
            rStable = prevStable
            cxStable = prevCx
            cyStable = prevCy
            q = 0
        }

        if (rStable == null || cxStable == null || cyStable == null) {
            Log.d(
                TAG,
                "PXM[REJ] rH=%.2f rS=%.2f rSt=---- cx=---- cy=---- covS=%.3f madRel=%.3f diffHS=%.3f thickMed=%.2f thickFrac=%.2f sym=%.3f candScore=%.3f"
                    .format(
                        rH,
                        rS,
                        covS,
                        madRel,
                        diffHS,
                        circleCand.sweep.thickMed,
                        circleCand.sweep.thickHitFrac,
                        circleCand.sweep.sym,
                        circleCand.score
                    )
            )
            safeRelease(rgbaRoi, circles, edges, gray)
            rect?.release()
            rgbaFull.release()
            return null
        }

        val covE = circleEdgeCoverageBand(edges, cxStable, cyStable, rStable, halfThickPx = 4)
        val pxPerMmCircle = (2.0 * rStable / CIRCLE_DIAM_MM).toFloat()

        Log.d(
            TAG,
            "PXM[%s] rH=%.2f rS=%.2f rSt=%.2f covS=%.3f covE=%.3f madRel=%.3f diffHS=%.3f thickMed=%.2f thickFrac=%.2f sym=%.3f jump=%.3f pxmmCircle=%.4f quadQ=%.3f"
                .format(
                    if (q == 1) "OK" else "HOLD",
                    rH,
                    rS,
                    rStable,
                    covS,
                    covE,
                    madRel,
                    diffHS,
                    circleCand.sweep.thickMed,
                    circleCand.sweep.thickHitFrac,
                    circleCand.sweep.sym,
                    jumpRel,
                    pxPerMmCircle,
                    rect?.quadQuality ?: 0f
                )
        )

        val sharp = laplacianVar(gray)

        // ---------------------- Barra 100 mm ----------------------
        val bar = detectBar(edges, cxStable, cyStable, rStable)
        val barMidXDetRoi = bar?.midX
        val barMidYDetRoi = bar?.midY

        val circleCenterWork = Point(cxStable + roi.x, cyStable + roi.y)
        val circlePxOnXAxisWork = Point(circleCenterWork.x + rStable, circleCenterWork.y)
        val circlePxOnYAxisWork = Point(circleCenterWork.x, circleCenterWork.y + rStable)

        val barX1Work = bar?.x1?.let { it + roi.x }
        val barY1Work = bar?.y1?.let { it + roi.y }
        val barX2Work = bar?.x2?.let { it + roi.x }
        val barY2Work = bar?.y2?.let { it + roi.y }

        val mapped = mapOutputsToOriginal(
            hInv = hInv,
            circleCenterWork = circleCenterWork,
            circlePxOnXAxisWork = circlePxOnXAxisWork,
            circlePxOnYAxisWork = circlePxOnYAxisWork,
            barX1Work = barX1Work,
            barY1Work = barY1Work,
            barX2Work = barX2Work,
            barY2Work = barY2Work,
            pxPerMmCircleRect = pxPerMmCircle
        )

        val pxPerMmBarOut =
            if (mapped.barLenPxOut != null && mapped.barLenPxOut > 1.0)
                (mapped.barLenPxOut / BAR_LEN_MM).toFloat()
            else null

        val barLenMmByCircle =
            if (mapped.barLenPxOut != null && mapped.pxPerMmCircleXOut > 0f)
                (mapped.barLenPxOut / mapped.pxPerMmCircleXOut).toFloat()
            else null

        val relBarVsCircle =
            if (pxPerMmBarOut != null && mapped.pxPerMmCircleXOut > 0f)
                abs(pxPerMmBarOut - mapped.pxPerMmCircleXOut) / mapped.pxPerMmCircleXOut
            else Float.NaN

        val barCoherent =
            pxPerMmBarOut != null &&
                    !relBarVsCircle.isNaN() &&
                    relBarVsCircle <= MAX_BAR_CIRCLE_REL_ERR

        val pxPerMmXOut = pxPerMmBarOut ?: mapped.pxPerMmCircleXOut
        val pxPerMmYOut = mapped.pxPerMmCircleYOut
        val pxPerMmOfficial =
            if (barCoherent && pxPerMmBarOut != null) {
                0.5f * (mapped.pxPerMmCircleXOut + pxPerMmBarOut)
            } else {
                mapped.pxPerMmCircleXOut
            }

        val thetaBarOut =
            if (mapped.barX1Out != null && mapped.barY1Out != null && mapped.barX2Out != null && mapped.barY2Out != null) {
                Math.toDegrees(
                    atan2(
                        mapped.barY2Out - mapped.barY1Out,
                        mapped.barX2Out - mapped.barX1Out
                    )
                ).toFloat()
            } else {
                Float.NaN
            }

        val thetaOut = when {
            rect != null -> rect.thetaPageDeg
            !thetaBarOut.isNaN() -> thetaBarOut
            else -> 0f
        }

        val confCircle = covE.coerceIn(0.0, 1.0)
        val confBar =
            if (barLenMmByCircle != null) {
                (1.0 - abs(barLenMmByCircle - BAR_LEN_MM) / BAR_LEN_MM).coerceIn(0.0, 1.0)
            } else {
                0.0
            }
        val confRect = rect?.quadQuality?.toDouble() ?: 0.0

        val conf =
            if (rect != null) {
                (0.55 * confCircle + 0.20 * confBar + 0.25 * confRect).toFloat()
            } else {
                (0.72 * confCircle + 0.28 * confBar).toFloat()
            }

        Log.d(
            TAG,
            "PXM_OUT[%s] cx=%.1f cy=%.1f rPx=%.2f pxmm=%.4f pxmmX=%.4f pxmmY=%.4f barMmByCircle=%s relBar=%.4f theta=%.2f"
                .format(
                    if (q == 1) "OK" else "HOLD",
                    mapped.cxOut,
                    mapped.cyOut,
                    mapped.rOut,
                    pxPerMmOfficial,
                    pxPerMmXOut,
                    pxPerMmYOut,
                    barLenMmByCircle?.let { "%.2f".format(it) } ?: "----",
                    if (relBarVsCircle.isNaN()) -1f else relBarVsCircle,
                    thetaOut
                )
        )

        safeRelease(rgbaRoi, circles, edges, gray)
        rect?.release()
        rgbaFull.release()

        return Ref75(
            cx = mapped.cxOut,
            cy = mapped.cyOut,
            rPx = mapped.rOut,
            thetaPageDeg = thetaOut,
            barMidX = mapped.barMidXOut,
            barMidY = mapped.barMidYOut,
            barLenMm = barLenMmByCircle,
            coverageCircle = covE.toFloat(),
            sharpness = sharp,
            confidence = conf,
            pxPerMm = pxPerMmOfficial,
            pxPerMmX = pxPerMmXOut,
            pxPerMmY = pxPerMmYOut
        )
    }

    // ---------------------------------------------------------------------------------------------
    // Rectificación por 4 cuadrados
    // ---------------------------------------------------------------------------------------------

    private data class RectifyPack3250(
        val rgbaRect: Mat,
        val H: Mat,
        val Hinv: Mat,
        val quadQuality: Float,
        val thetaPageDeg: Float
    ) {
        fun release() {
            rgbaRect.release()
            H.release()
            Hinv.release()
        }
    }

    private data class Quint(
        val a: Float,
        val b: Float,
        val c: Float,
        val d: Float?,
        val e: Float?
    )

    private data class SquareCand(
        val centerF: Point,
        val centerS: Point,
        val areaS: Double,
        val maxCos: Double,
        val nestedScore: Double,
        val score: Double
    )

    private data class QuadPick(
        val tl: SquareCand,
        val tr: SquareCand,
        val br: SquareCand,
        val bl: SquareCand,
        val score: Double,
        val quality: Double
    )

    private fun tryRectifyWithSquares3250(rgbaFull: Mat): RectifyPack3250? {
        val wFull = rgbaFull.cols()
        val hFull = rgbaFull.rows()
        val c0F = Point(wFull / 2.0, hFull / 2.0)

        val scale = if (wFull > DETECT_MAX_W) DETECT_MAX_W.toDouble() / wFull.toDouble() else 1.0
        val wS = (wFull * scale).roundToInt().coerceAtLeast(64)
        val hS = (hFull * scale).roundToInt().coerceAtLeast(64)

        val rgbaS = Mat()
        if (scale != 1.0) {
            Imgproc.resize(
                rgbaFull,
                rgbaS,
                Size(wS.toDouble(), hS.toDouble()),
                0.0,
                0.0,
                Imgproc.INTER_AREA
            )
        } else {
            rgbaFull.copyTo(rgbaS)
        }

        val grayS = Mat()
        Imgproc.cvtColor(rgbaS, grayS, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.GaussianBlur(grayS, grayS, Size(5.0, 5.0), 0.0)

        val binS = Mat()
        Imgproc.threshold(grayS, binS, 0.0, 255.0, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU)

        val k = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        Imgproc.morphologyEx(binS, binS, Imgproc.MORPH_CLOSE, k, Point(-1.0, -1.0), 1)

        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(binS, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        if (contours.isEmpty()) {
            safeRelease(rgbaS, grayS, binS, hierarchy, k)
            contours.forEach { it.release() }
            return null
        }

        val imgArea = wS.toDouble() * hS.toDouble()
        val minArea = max(500.0, imgArea * 0.00006)
        val maxArea = min(imgArea * 0.06, imgArea)

        val cornerW = (CORNER_FRAC * wS).roundToInt()
        val cornerH = (CORNER_FRAC * hS).roundToInt()

        fun inCornerRoi(p: Point): Boolean {
            val x = p.x
            val y = p.y
            val tl = (x < cornerW && y < cornerH)
            val tr = (x >= (wS - cornerW) && y < cornerH)
            val bl = (x < cornerW && y >= (hS - cornerH))
            val br = (x >= (wS - cornerW) && y >= (hS - cornerH))
            return tl || tr || bl || br
        }

        val cands = ArrayList<SquareCand>(64)

        for (cnt in contours) {
            val area = Imgproc.contourArea(cnt)
            if (area < minArea || area > maxArea) continue

            val cnt2f = MatOfPoint2f(*cnt.toArray())
            val peri = Imgproc.arcLength(cnt2f, true)
            if (peri <= 1.0) {
                cnt2f.release()
                continue
            }

            val approx2f = MatOfPoint2f()
            Imgproc.approxPolyDP(cnt2f, approx2f, 0.02 * peri, true)
            val pts = approx2f.toArray()
            if (pts.size != 4) {
                cnt2f.release()
                approx2f.release()
                continue
            }

            val approx = MatOfPoint(*pts)
            if (!Imgproc.isContourConvex(approx)) {
                cnt2f.release()
                approx2f.release()
                approx.release()
                continue
            }

            val maxCos = maxAngleCos(pts)
            if (maxCos > 0.38) {
                cnt2f.release()
                approx2f.release()
                approx.release()
                continue
            }

            val rect = Imgproc.boundingRect(approx)
            val ar = rect.width.toDouble() / rect.height.toDouble()
            if (ar < 0.70 || ar > 1.30) {
                cnt2f.release()
                approx2f.release()
                approx.release()
                continue
            }

            val m = Imgproc.moments(approx)
            val m00 = m.m00
            if (abs(m00) < 1e-6) {
                cnt2f.release()
                approx2f.release()
                approx.release()
                continue
            }

            val centerS = Point(m.m10 / m00, m.m01 / m00)
            if (!inCornerRoi(centerS)) {
                cnt2f.release()
                approx2f.release()
                approx.release()
                continue
            }

            val nested = nestedMarkerScore(binS, rect)
            if (nested < 0.20) {
                cnt2f.release()
                approx2f.release()
                approx.release()
                continue
            }

            val centerF =
                if (scale != 1.0) Point(centerS.x / scale, centerS.y / scale)
                else Point(centerS.x, centerS.y)

            val dx = centerS.x - (wS / 2.0)
            val dy = centerS.y - (hS / 2.0)
            val dNorm = (hypot(dx, dy) / hypot(wS / 2.0, hS / 2.0)).coerceIn(0.0, 1.0)
            val cornerBoost = 0.55 + 0.45 * dNorm

            val base = area * (1.0 - maxCos).coerceAtLeast(0.0)
            val score = base * cornerBoost * (0.35 + 0.65 * nested)

            cands.add(
                SquareCand(
                    centerF = centerF,
                    centerS = centerS,
                    areaS = area,
                    maxCos = maxCos,
                    nestedScore = nested,
                    score = score
                )
            )

            cnt2f.release()
            approx2f.release()
            approx.release()
        }

        contours.forEach { it.release() }
        hierarchy.release()
        k.release()
        binS.release()
        grayS.release()
        rgbaS.release()

        if (cands.size < 4) return null

        val tlList = cands.filter { it.centerS.x < cornerW && it.centerS.y < cornerH }
            .sortedByDescending { it.score }
            .take(TOPK_PER_QUAD)
        val trList = cands.filter { it.centerS.x >= (wS - cornerW) && it.centerS.y < cornerH }
            .sortedByDescending { it.score }
            .take(TOPK_PER_QUAD)
        val blList = cands.filter { it.centerS.x < cornerW && it.centerS.y >= (hS - cornerH) }
            .sortedByDescending { it.score }
            .take(TOPK_PER_QUAD)
        val brList = cands.filter { it.centerS.x >= (wS - cornerW) && it.centerS.y >= (hS - cornerH) }
            .sortedByDescending { it.score }
            .take(TOPK_PER_QUAD)

        if (tlList.isEmpty() || trList.isEmpty() || blList.isEmpty() || brList.isEmpty()) return null

        var best: QuadPick? = null
        val minDimF = min(wFull.toDouble(), hFull.toDouble())

        for (tl in tlList) for (tr in trList) for (br in brList) for (bl in blList) {
            val wTop = dist(tl.centerF, tr.centerF)
            val wBot = dist(bl.centerF, br.centerF)
            val hLeft = dist(tl.centerF, bl.centerF)
            val hRight = dist(tr.centerF, br.centerF)
            val d1 = dist(tl.centerF, br.centerF)
            val d2 = dist(tr.centerF, bl.centerF)

            if (wTop < 10 || wBot < 10 || hLeft < 10 || hRight < 10) continue

            val ratioW = wTop / wBot
            val ratioH = hLeft / hRight
            val ratioD = d1 / d2

            val geomPenalty = abs(ratioW - 1.0) + abs(ratioH - 1.0) + abs(ratioD - 1.0)

            val cxQ = (tl.centerF.x + tr.centerF.x + br.centerF.x + bl.centerF.x) / 4.0
            val cyQ = (tl.centerF.y + tr.centerF.y + br.centerF.y + bl.centerF.y) / 4.0
            val dCenter = hypot(cxQ - c0F.x, cyQ - c0F.y) / hypot(c0F.x, c0F.y)

            val sideMean = (wTop + wBot + hLeft + hRight) / 4.0
            val sideFrac = (sideMean / minDimF).coerceIn(0.0, 2.0)

            val pGeom = 1.0 / (1.0 + GEOM_RATIO_SOFT * 4.0 * geomPenalty)
            val pCenter = 1.0 / (1.0 + CENTER_SOFT * max(0.0, dCenter - 0.06))
            val pSize = 1.0 / (1.0 + SIZE_SOFT * max(0.0, abs(sideFrac - 0.55)))

            val quality = (0.45 * pGeom + 0.25 * pCenter + 0.30 * pSize).coerceIn(0.0, 1.0)
            val base = tl.score + tr.score + br.score + bl.score
            val score = base * quality

            val pick = QuadPick(tl, tr, br, bl, score, quality)
            if (best == null || pick.score > best.score) best = pick
        }

        val pick = best ?: return null

        val srcPts = MatOfPoint2f(pick.tl.centerF, pick.tr.centerF, pick.br.centerF, pick.bl.centerF)

        val sidePx = MARKERS_CENTER_SQUARE_MM * RECT_PX_PER_MM_NOM
        val dstPts = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(sidePx, 0.0),
            Point(sidePx, sidePx),
            Point(0.0, sidePx)
        )

        val H = Imgproc.getPerspectiveTransform(srcPts, dstPts)
        val Hinv = H.inv()

        val outW = sidePx.toInt().coerceAtLeast(64)
        val outH = sidePx.toInt().coerceAtLeast(64)

        val rgbaRect = Mat(outH, outW, rgbaFull.type())
        Imgproc.warpPerspective(
            rgbaFull,
            rgbaRect,
            H,
            Size(outW.toDouble(), outH.toDouble()),
            Imgproc.INTER_LINEAR,
            Core.BORDER_CONSTANT,
            Scalar(255.0, 255.0, 255.0, 255.0)
        )

        val thetaPageDeg = Math.toDegrees(
            atan2(
                pick.tr.centerF.y - pick.tl.centerF.y,
                pick.tr.centerF.x - pick.tl.centerF.x
            )
        ).toFloat()

        srcPts.release()
        dstPts.release()

        return RectifyPack3250(
            rgbaRect = rgbaRect,
            H = H,
            Hinv = Hinv,
            quadQuality = pick.quality.toFloat(),
            thetaPageDeg = thetaPageDeg
        )
    }

    private fun nestedMarkerScore(binInv: Mat, rect: Rect): Double {
        val w = binInv.cols()
        val h = binInv.rows()

        val x0 = rect.x.coerceIn(0, w - 2)
        val y0 = rect.y.coerceIn(0, h - 2)
        val rw = rect.width.coerceIn(2, w - x0)
        val rh = rect.height.coerceIn(2, h - y0)

        val r = Rect(x0, y0, rw, rh)
        val roi = binInv.submat(r)

        val bw = max(2, (0.16 * rw).roundToInt())
        val bh = max(2, (0.16 * rh).roundToInt())

        val outer = roi.submat(Rect(0, 0, rw, rh))
        val outerMean = Core.mean(outer).`val`[0]

        val midX = bw
        val midY = bh
        val midW = (rw - 2 * bw).coerceAtLeast(2)
        val midH = (rh - 2 * bh).coerceAtLeast(2)
        val mid = roi.submat(Rect(midX, midY, midW, midH))
        val midMean = Core.mean(mid).`val`[0]

        val cx0 = (0.40 * rw).roundToInt().coerceIn(0, rw - 2)
        val cy0 = (0.40 * rh).roundToInt().coerceIn(0, rh - 2)
        val cw = (0.20 * rw).roundToInt().coerceIn(2, rw - cx0)
        val ch = (0.20 * rh).roundToInt().coerceIn(2, rh - cy0)
        val center = roi.submat(Rect(cx0, cy0, cw, ch))
        val centerMean = Core.mean(center).`val`[0]

        outer.release()
        mid.release()
        center.release()
        roi.release()

        val borderOk = ((outerMean - 90.0) / 165.0).coerceIn(0.0, 1.0)
        val holeOk = ((140.0 - midMean) / 140.0).coerceIn(0.0, 1.0)
        val dotOk = ((centerMean - 80.0) / 175.0).coerceIn(0.0, 1.0)

        return (0.40 * borderOk + 0.35 * holeOk + 0.25 * dotOk).coerceIn(0.0, 1.0)
    }

    // ---------------------------------------------------------------------------------------------
    // Helpers círculo/barra/nitidez
    // ---------------------------------------------------------------------------------------------

    private data class CircleCand(
        val cx: Double,
        val cy: Double,
        val rH: Double,
        val sweep: SweepPack,
        val covS: Double,
        val madRel: Double,
        val diffHS: Double,
        val centerPenalty: Double,
        val ok: Boolean,
        val score: Double
    )

    private fun selectBestCircleCandidate(
        circles: Mat,
        edgeU8: ByteArray,
        w: Int,
        h: Int
    ): CircleCand? {
        val expCx = w * 0.5
        val expCy = h * 0.46
        val normD = hypot(w * 0.5, h * 0.5).coerceAtLeast(1.0)

        var best: CircleCand? = null

        for (i in 0 until circles.cols()) {
            val c = circles.get(0, i) ?: continue
            if (c.size < 3) continue

            val cx = c[0]
            val cy = c[1]
            val rH = c[2]

            val sweep = sweepRadiusWithThickness(
                edges = edgeU8,
                w = w,
                h = h,
                cx = cx.toFloat(),
                cy = cy.toFloat(),
                r0 = rH.toFloat(),
                bandFrac = BAND_FRAC,
                angleStepDeg = ANGLE_STEP_DEG
            )

            val rS = sweep.rMed
            val covS = sweep.coverage
            val madRel = if (rS.isNaN() || rS <= 1e-6) 1.0 else (sweep.mad / rS)
            val diffHS = if (rS.isNaN() || rH <= 1e-6) 9.0 else abs(rH - rS) / rH
            val centerPenalty = hypot(cx - expCx, cy - expCy) / normD

            val thickOk =
                !sweep.thickMed.isNaN() && sweep.thickMed in THICK_MIN_PX..THICK_MAX_PX

            val ok =
                (!rS.isNaN()) &&
                        (covS >= MIN_COV_SWEEP) &&
                        (madRel <= MAX_MAD_REL) &&
                        (diffHS <= MAX_DIFF_HS) &&
                        (sweep.thickHitFrac >= MIN_THICK_HIT_FRAC) &&
                        thickOk

            val score =
                (if (ok) 2.0 else 0.0) +
                        (3.0 * covS) -
                        (2.0 * madRel) -
                        (1.5 * diffHS) +
                        (1.0 * sweep.thickHitFrac) +
                        (if (thickOk) 0.8 else -0.8) -
                        (0.6 * centerPenalty) -
                        (0.25 * sweep.sym)

            val cand = CircleCand(
                cx = cx,
                cy = cy,
                rH = rH,
                sweep = sweep,
                covS = covS,
                madRel = madRel,
                diffHS = diffHS,
                centerPenalty = centerPenalty,
                ok = ok,
                score = score
            )

            if (best == null || cand.score > best!!.score) {
                best = cand
            }
        }

        return best
    }

    private fun circleEdgeCoverageBand(
        edges: Mat,
        cx: Double,
        cy: Double,
        r: Double,
        halfThickPx: Int
    ): Double {
        val n = 360
        var hit = 0
        val cols = edges.cols()
        val rows = edges.rows()

        for (k in 0 until n) {
            val th = 2.0 * Math.PI * k / n
            val dx = cos(th)
            val dy = sin(th)

            var ok = false
            for (t in -halfThickPx..halfThickPx) {
                val rr = r + t
                val x = (cx + rr * dx).roundToInt().coerceIn(0, cols - 1)
                val y = (cy + rr * dy).roundToInt().coerceIn(0, rows - 1)
                if ((edges.get(y, x)?.get(0)?.toInt() ?: 0) > 0) {
                    ok = true
                    break
                }
            }
            if (ok) hit++
        }
        return hit.toDouble() / n.toDouble()
    }

    private fun laplacianVar(gray: Mat): Float {
        val lap = Mat()
        Imgproc.Laplacian(gray, lap, CvType.CV_64F)
        val mean = MatOfDouble()
        val std = MatOfDouble()
        Core.meanStdDev(lap, mean, std)
        val sigma = std.toArray()[0]
        lap.release()
        mean.release()
        std.release()
        return (sigma * sigma).toFloat()
    }

    private data class BarSeg(val x1: Double, val y1: Double, val x2: Double, val y2: Double) {
        val midX: Double get() = (x1 + x2) * 0.5
        val midY: Double get() = (y1 + y2) * 0.5
        val len: Double get() = hypot(x2 - x1, y2 - y1)
    }

    private fun detectBar(edges: Mat, cx: Double, cy: Double, r: Double): BarSeg? {
        val cols = edges.cols()
        val rows = edges.rows()

        val x0 = (cx - 1.10 * r).roundToInt().coerceIn(0, cols - 2)
        val x1 = (cx + 1.10 * r).roundToInt().coerceIn(x0 + 1, cols - 1)
        val y0 = (cy + 0.18 * r).roundToInt().coerceIn(0, rows - 2)
        val y1 = (cy + 1.15 * r).roundToInt().coerceIn(y0 + 1, rows - 1)

        val roi = Rect(
            x0,
            y0,
            (x1 - x0).coerceAtLeast(2),
            (y1 - y0).coerceAtLeast(2)
        )

        val band = edges.submat(roi)
        val lines = Mat()

        Imgproc.HoughLinesP(
            band,
            lines,
            1.0,
            Math.PI / 180.0,
            45,
            max(30.0, r * 0.55),
            24.0
        )

        if (lines.empty()) {
            band.release()
            lines.release()
            return null
        }

        var best: BarSeg? = null
        var bestScore = Double.NEGATIVE_INFINITY

        for (i in 0 until lines.rows()) {
            val l = lines.get(i, 0) ?: continue

            val x1g = l[0] + roi.x
            val y1g = l[1] + roi.y
            val x2g = l[2] + roi.x
            val y2g = l[3] + roi.y

            val dx = x2g - x1g
            val dy = y2g - y1g
            val len = hypot(dx, dy)
            if (len <= 1.0) continue

            val mx = (x1g + x2g) / 2.0
            val my = (y1g + y2g) / 2.0

            val horizOk = abs(dy) <= 0.10 * len
            val belowOk = my >= (cy + 0.24 * r)
            val radialOk = hypot(mx - cx, my - cy) in (0.55 * r)..(1.80 * r)

            val centerAlign = 1.0 - min(1.0, abs(mx - cx) / max(1.0, 0.90 * r))
            val lenFit = 1.0 - min(1.0, abs((len / max(1.0, 2.0 * r)) - 1.0))

            val wHoriz = if (horizOk) 1.0 else 0.35
            val wBelow = if (belowOk) 1.0 else 0.25
            val wRadial = if (radialOk) 1.0 else 0.45

            val score =
                len *
                        wHoriz *
                        wBelow *
                        wRadial *
                        (0.45 + 0.55 * centerAlign) *
                        (0.40 + 0.60 * lenFit)

            if (score > bestScore) {
                bestScore = score
                best = BarSeg(x1g, y1g, x2g, y2g)
            }
        }

        band.release()
        lines.release()
        return best
    }

    // ---------------------------------------------------------------------------------------------
    // Sweep con espesor (inner/outer)
    // ---------------------------------------------------------------------------------------------

    private data class SweepPack(
        val rMed: Double,
        val coverage: Double,
        val sym: Double,
        val mad: Double,
        val thickMed: Double,
        val thickHitFrac: Double
    )

    private fun sweepRadiusWithThickness(
        edges: ByteArray,
        w: Int,
        h: Int,
        cx: Float,
        cy: Float,
        r0: Float,
        bandFrac: Float,
        angleStepDeg: Int
    ): SweepPack {

        val maxR = min(w, h).toFloat()
        val bandPx = (r0 * bandFrac).coerceIn(12f, 260f)
        val dMax = bandPx.roundToInt()

        val radiiC = ArrayList<Float>(360 / angleStepDeg + 8)
        val thick = ArrayList<Float>(360 / angleStepDeg + 8)

        var hits = 0
        var thickHits = 0
        var total = 0

        val keyAngles = intArrayOf(0, 45, 90, 135, 180, 225, 270, 315)
        val keyRC = FloatArray(keyAngles.size) { Float.NaN }

        fun at(x: Int, y: Int): Boolean {
            return (x in 0 until w && y in 0 until h) &&
                    ((edges[y * w + x].toInt() and 0xFF) != 0)
        }

        fun sampleAtAngleCenter(aDeg: Int): Pair<Float, Float> {
            val rad = Math.toRadians(aDeg.toDouble())
            val dx = cos(rad).toFloat()
            val dy = sin(rad).toFloat()

            var rIn = Float.NaN
            var rOut = Float.NaN

            for (d in 0..dMax) {
                val rA = r0 - d
                if (rA < 1f || rA > maxR) continue
                val x = (cx + dx * rA).toInt()
                val y = (cy + dy * rA).toInt()
                if (at(x, y)) {
                    rIn = rA
                    break
                }
            }

            for (d in 0..dMax) {
                val rB = r0 + d
                if (rB < 1f || rB > maxR) continue
                val x = (cx + dx * rB).toInt()
                val y = (cy + dy * rB).toInt()
                if (at(x, y)) {
                    rOut = rB
                    break
                }
            }

            return when {
                !rIn.isNaN() && !rOut.isNaN() -> {
                    val rc = 0.5f * (rIn + rOut)
                    val t = abs(rOut - rIn)
                    Pair(rc, t)
                }
                !rIn.isNaN() -> Pair(rIn, Float.NaN)
                !rOut.isNaN() -> Pair(rOut, Float.NaN)
                else -> Pair(Float.NaN, Float.NaN)
            }
        }

        for (a in 0 until 360 step angleStepDeg) {
            total++
            val (rc, t) = sampleAtAngleCenter(a)
            if (!rc.isNaN()) {
                hits++
                radiiC.add(rc)
                if (!t.isNaN()) {
                    thickHits++
                    thick.add(t)
                }
            }
        }

        for (i in keyAngles.indices) {
            val (rc, _) = sampleAtAngleCenter(keyAngles[i])
            keyRC[i] = rc
        }

        if (radiiC.isEmpty()) {
            return SweepPack(
                Double.NaN,
                0.0,
                1.0,
                Double.POSITIVE_INFINITY,
                Double.NaN,
                0.0
            )
        }

        radiiC.sort()
        val rMed = radiiC[radiiC.size / 2].toDouble()

        val absDev = DoubleArray(radiiC.size)
        for (i in radiiC.indices) {
            absDev[i] = abs(radiiC[i].toDouble() - rMed)
        }
        absDev.sort()
        val mad = absDev[absDev.size / 2]

        fun rel(a: Float, b: Float): Double {
            val den = max(1e-6f, (a + b) * 0.5f)
            return (abs(a - b) / den).toDouble()
        }

        var sym = 0.0
        var pairs = 0
        val pairsIdx = arrayOf(0 to 4, 1 to 5, 2 to 6, 3 to 7)
        for (p in pairsIdx) {
            val a = keyRC[p.first]
            val b = keyRC[p.second]
            if (!a.isNaN() && !b.isNaN()) {
                sym = max(sym, rel(a, b))
                pairs++
            }
        }
        if (pairs == 0) sym = 1.0

        val cov = hits.toDouble() / total.toDouble()

        val thickMed =
            if (thick.isNotEmpty()) {
                thick.sort()
                thick[thick.size / 2].toDouble()
            } else {
                Double.NaN
            }

        val thickFrac =
            if (hits > 0) thickHits.toDouble() / hits.toDouble()
            else 0.0

        return SweepPack(
            rMed = rMed,
            coverage = cov.coerceIn(0.0, 1.0),
            sym = sym.coerceIn(0.0, 1.0),
            mad = mad,
            thickMed = thickMed,
            thickHitFrac = thickFrac.coerceIn(0.0, 1.0)
        )
    }

    // ---------------------------------------------------------------------------------------------
    // OpenCV / geom helpers
    // ---------------------------------------------------------------------------------------------

    private data class MappedOutput(
        val cxOut: Float,
        val cyOut: Float,
        val rOut: Float,
        val pxPerMmCircleXOut: Float,
        val pxPerMmCircleYOut: Float,
        val barX1Out: Double?,
        val barY1Out: Double?,
        val barX2Out: Double?,
        val barY2Out: Double?,
        val barMidXOut: Float?,
        val barMidYOut: Float?,
        val barLenPxOut: Double?
    )

    private fun mapOutputsToOriginal(
        hInv: Mat?,
        circleCenterWork: Point,
        circlePxOnXAxisWork: Point,
        circlePxOnYAxisWork: Point,
        barX1Work: Double?,
        barY1Work: Double?,
        barX2Work: Double?,
        barY2Work: Double?,
        pxPerMmCircleRect: Float
    ): MappedOutput {

        return if (hInv != null) {
            val cOrig = mapPoint(hInv, circleCenterWork.x, circleCenterWork.y)
            val pXOrig = mapPoint(hInv, circlePxOnXAxisWork.x, circlePxOnXAxisWork.y)
            val pYOrig = mapPoint(hInv, circlePxOnYAxisWork.x, circlePxOnYAxisWork.y)

            val rX = dist(cOrig, pXOrig)
            val rY = dist(cOrig, pYOrig)
            val rOut = ((rX + rY) * 0.5).toFloat()

            val pxPerMmCircleXOut = (2.0 * rX / CIRCLE_DIAM_MM).toFloat()
            val pxPerMmCircleYOut = (2.0 * rY / CIRCLE_DIAM_MM).toFloat()

            val bp1 =
                if (barX1Work != null && barY1Work != null) mapPoint(hInv, barX1Work, barY1Work) else null
            val bp2 =
                if (barX2Work != null && barY2Work != null) mapPoint(hInv, barX2Work, barY2Work) else null

            val barLenPxOut =
                if (bp1 != null && bp2 != null) dist(bp1, bp2)
                else null

            val barMidXOut =
                if (bp1 != null && bp2 != null) ((bp1.x + bp2.x) * 0.5).toFloat()
                else null
            val barMidYOut =
                if (bp1 != null && bp2 != null) ((bp1.y + bp2.y) * 0.5).toFloat()
                else null

            MappedOutput(
                cxOut = cOrig.x.toFloat(),
                cyOut = cOrig.y.toFloat(),
                rOut = rOut,
                pxPerMmCircleXOut = pxPerMmCircleXOut,
                pxPerMmCircleYOut = pxPerMmCircleYOut,
                barX1Out = bp1?.x,
                barY1Out = bp1?.y,
                barX2Out = bp2?.x,
                barY2Out = bp2?.y,
                barMidXOut = barMidXOut,
                barMidYOut = barMidYOut,
                barLenPxOut = barLenPxOut
            )
        } else {
            val rOut = dist(circleCenterWork, circlePxOnXAxisWork).toFloat()

            val barLenPxOut =
                if (barX1Work != null && barY1Work != null && barX2Work != null && barY2Work != null) {
                    hypot(barX2Work - barX1Work, barY2Work - barY1Work)
                } else {
                    null
                }

            MappedOutput(
                cxOut = circleCenterWork.x.toFloat(),
                cyOut = circleCenterWork.y.toFloat(),
                rOut = rOut,
                pxPerMmCircleXOut = pxPerMmCircleRect,
                pxPerMmCircleYOut = pxPerMmCircleRect,
                barX1Out = barX1Work,
                barY1Out = barY1Work,
                barX2Out = barX2Work,
                barY2Out = barY2Work,
                barMidXOut = if (barX1Work != null && barX2Work != null) ((barX1Work + barX2Work) * 0.5).toFloat() else null,
                barMidYOut = if (barY1Work != null && barY2Work != null) ((barY1Work + barY2Work) * 0.5).toFloat() else null,
                barLenPxOut = barLenPxOut
            )
        }
    }

    private fun mapPoint(H: Mat, x: Double, y: Double): Point {
        val src = MatOfPoint2f(Point(x, y))
        val dst = MatOfPoint2f()
        Core.perspectiveTransform(src, dst, H)
        val p = dst.toArray()[0]
        src.release()
        dst.release()
        return p
    }

    private fun dist(a: Point, b: Point): Double = hypot(a.x - b.x, a.y - b.y)

    private fun maxAngleCos(pts: Array<Point>): Double {
        val c = Point(
            (pts[0].x + pts[1].x + pts[2].x + pts[3].x) / 4.0,
            (pts[0].y + pts[1].y + pts[2].y + pts[3].y) / 4.0
        )
        val ordered = pts.sortedBy { atan2(it.y - c.y, it.x - c.x) }.toTypedArray()

        fun cosAngle(a: Point, b: Point, c0: Point): Double {
            val abx = a.x - b.x
            val aby = a.y - b.y
            val cbx = c0.x - b.x
            val cby = c0.y - b.y
            val dot = abx * cbx + aby * cby
            val lab = hypot(abx, aby)
            val lcb = hypot(cbx, cby)
            if (lab < 1e-6 || lcb < 1e-6) return 1.0
            return abs(dot / (lab * lcb))
        }

        var maxCos = 0.0
        for (i in 0 until 4) {
            val a = ordered[(i + 3) % 4]
            val b = ordered[i]
            val c1 = ordered[(i + 1) % 4]
            val cs = cosAngle(a, b, c1)
            if (cs > maxCos) maxCos = cs
        }
        return maxCos
    }

    private fun matToU8(m: Mat): ByteArray {
        val n = (m.total() * m.channels()).toInt()
        val out = ByteArray(n)
        if (m.isContinuous) {
            m.get(0, 0, out)
        } else {
            val tmp = m.clone()
            tmp.get(0, 0, out)
            tmp.release()
        }
        return out
    }

    private fun medianOrNull(dq: ArrayDeque<Double>): Double? {
        if (dq.isEmpty()) return null
        val arr = DoubleArray(dq.size)
        var i = 0
        for (v in dq) {
            arr[i++] = v
        }
        arr.sort()
        return arr[arr.size / 2]
    }

    private fun safeRelease(vararg mats: Mat?) {
        for (m in mats) {
            try {
                m?.release()
            } catch (_: Exception) {
            }
        }
    }
}