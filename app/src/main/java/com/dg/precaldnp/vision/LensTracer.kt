package com.dg.precaldnp.vision

import android.graphics.Bitmap
import android.graphics.PointF
import android.util.Log
import com.dg.precaldnp.model.PrecalMetrics
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin

object LensTracer {

    private const val TAG = "Precal/LensTracer"

    private const val N_RADIALES = 800
    private const val OVER_SAMPLING = 3
    private const val N_RADIALES_FINE = N_RADIALES * OVER_SAMPLING

    private const val RADIAL_STEP_PX = 0.8

    private const val SMOOTH_WIN_MEDIAN = 5
    private const val SMOOTH_WIN_MEAN = 9

    private const val FALLBACK_MAX_POINTS = 2000

    data class Params(
        // El círculo impreso de tu plantilla (Ø100 por defecto)
        val circleDiamMm: Float = 100f,

        // ROI: disco interior y banda a excluir alrededor del aro
        val marginMmInside: Float = 6f,
        val ringBandMmExclude: Float = 3.0f,

        // bordes
        val cannyLow: Int = 40,
        val cannyHigh: Int = 120,

        // radial principal
        val minCoverageRadial: Float = 0.60f,

        // limpieza/continuidad
        val maxJumpMm: Float = 0.2f,

        // prepro
        val clahe: Boolean = true,
        val sharpen: Boolean = true,

        // Anti-reflejos
        val highlightThr: Double = 235.0,
        val highlightDilate: Int = 3,
        val highlightFillGray: Double = 180.0,

        // diámetro “seguro” desde donde arranca el radial
        val scanDiamMm: Float = 92f,

        // legacy / reservados
        val refineCenter: Boolean = true,
        val refineCenterSearchFrac: Float = 0.06f,
        val refineCoarseStepPx: Int = 3,
        val refineFineStepPx: Int = 1,
        val refineCoarseStepDeg: Int = 6,
        val refineFineStepDeg: Int = 3
    )

    fun trace(
        bitmap: Bitmap,
        pxPerMm: Float,
        cameraId: String,
        zoom: Float,
        refCircle: RefCalibrator75.Ref75? = null,
        params: Params = Params()
    ): PrecalMetrics? {
        val srcRgba = Mat()
        Utils.bitmapToMat(bitmap, srcRgba)

        val bgr = Mat()
        Imgproc.cvtColor(srcRgba, bgr, Imgproc.COLOR_RGBA2BGR)
        srcRgba.release()

        val out = traceBgr(
            matBgr = bgr,
            pxPerMm = pxPerMm,
            params = params,
            refCircle = refCircle
        )
        bgr.release()

        return out?.copy(cameraId = cameraId, zoom = zoom)
    }

    fun traceBgr(
        matBgr: Mat,
        pxPerMm: Float,
        params: Params = Params(),
        refCircle: RefCalibrator75.Ref75? = null
    ): PrecalMetrics? {

        val bufW = matBgr.cols()
        val bufH = matBgr.rows()

        // Si viene del calibrador pro, usamos SU escala oficial.
        val pxPerMmUsed = refCircle?.pxPerMm?.takeIf { it > 0f } ?: pxPerMm

        Log.d(
            TAG,
            "TRACE_IN pxArg=%.4f pxUse=%.4f ref(cx=%.1f cy=%.1f r=%.1f pxX=%.4f pxY=%.4f th=%.2f)"
                .format(
                    pxPerMm,
                    pxPerMmUsed,
                    refCircle?.cx ?: Float.NaN,
                    refCircle?.cy ?: Float.NaN,
                    refCircle?.rPx ?: Float.NaN,
                    refCircle?.pxPerMmX ?: Float.NaN,
                    refCircle?.pxPerMmY ?: Float.NaN,
                    refCircle?.thetaPageDeg ?: Float.NaN
                )
        )

        // -------- Prepro base (FULL) --------
        val grayFull = Mat()
        Imgproc.cvtColor(matBgr, grayFull, Imgproc.COLOR_BGR2GRAY)

        if (params.clahe) {
            val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
            clahe.apply(grayFull, grayFull)
        }

        if (params.sharpen) {
            val blur = Mat()
            Imgproc.GaussianBlur(grayFull, blur, Size(0.0, 0.0), 1.0)
            Core.addWeighted(grayFull, 1.5, blur, -0.5, 0.0, grayFull)
            blur.release()
        }

        Imgproc.GaussianBlur(grayFull, grayFull, Size(5.0, 5.0), 0.0)

        // -------- Anti-reflejos (aplanar highlights) --------
        val highlights = Mat()
        Imgproc.threshold(grayFull, highlights, params.highlightThr, 255.0, Imgproc.THRESH_BINARY)
        val k3 = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0, 3.0))
        Imgproc.morphologyEx(highlights, highlights, Imgproc.MORPH_CLOSE, k3)
        if (params.highlightDilate > 0) {
            Imgproc.dilate(highlights, highlights, k3, Point(-1.0, -1.0), params.highlightDilate)
        }
        grayFull.setTo(Scalar(params.highlightFillGray), highlights)
        Imgproc.GaussianBlur(grayFull, grayFull, Size(5.0, 5.0), 0.0)
        k3.release()
        highlights.release()

        // -------- Edges FULL (para el radial de la lente) --------
        val edgesFull = Mat()
        Imgproc.Canny(grayFull, edgesFull, params.cannyLow.toDouble(), params.cannyHigh.toDouble())
        val kClose = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0, 3.0))
        Imgproc.morphologyEx(edgesFull, edgesFull, Imgproc.MORPH_CLOSE, kClose, Point(-1.0, -1.0), 1)
        kClose.release()

        val edgesFullU8 = matToU8(edgesFull)

        // -------- Círculo de referencia: usar el del calibrador si existe --------
        val ring = detectRingCircleForRoi(
            grayFull = grayFull,
            pxPerMmFull = pxPerMmUsed,
            params = params,
            refCircle = refCircle
        )

        if (ring == null) {
            grayFull.release()
            edgesFull.release()
            return null
        }

        val refCx = ring.cxFull
        val refCy = ring.cyFull
        val detRpx = ring.detRFull
        val useRpx = ring.useRFull

        // -------- ROI interior (disco) + CAP configurable --------
        val marginPx = (params.marginMmInside * pxPerMmUsed).coerceAtLeast(2f)

        val scanRatio = (params.scanDiamMm / params.circleDiamMm).coerceIn(0.70f, 0.98f)
        val capRpx = useRpx * scanRatio

        // ROI final: min(useRpx - margen, capRpx)
        val roiR = max(8f, min(useRpx - marginPx, capRpx))

        val roiMask = Mat.zeros(bufH, bufW, CvType.CV_8UC1)
        Imgproc.circle(
            roiMask,
            Point(refCx.toDouble(), refCy.toDouble()),
            roiR.roundToInt(),
            Scalar(255.0),
            -1
        )

        // -------- Banda a excluir alrededor del aro (useRpx ± bandPx) --------
        val bandPx = (params.ringBandMmExclude * pxPerMmUsed).coerceAtLeast(2f)

        val ringOuter = Mat.zeros(bufH, bufW, CvType.CV_8UC1)
        val ringInner = Mat.zeros(bufH, bufW, CvType.CV_8UC1)

        val rOuterBand = (useRpx + bandPx).coerceAtLeast(2f)
        val rInnerBand = (useRpx - bandPx).coerceAtLeast(2f)

        Imgproc.circle(
            ringOuter,
            Point(refCx.toDouble(), refCy.toDouble()),
            rOuterBand.roundToInt(),
            Scalar(255.0),
            -1
        )
        Imgproc.circle(
            ringInner,
            Point(refCx.toDouble(), refCy.toDouble()),
            rInnerBand.roundToInt(),
            Scalar(255.0),
            -1
        )

        val ringBand = Mat()
        Core.subtract(ringOuter, ringInner, ringBand)
        Core.subtract(roiMask, ringBand, roiMask)

        ringOuter.release()
        ringInner.release()
        ringBand.release()

        // -------- Veto explícito de gráfica de plantilla --------
        val templateVeto = buildTemplateVetoMask(
            w = bufW,
            h = bufH,
            cx = refCx,
            cy = refCy,
            useRpx = useRpx,
            thetaPageDeg = refCircle?.thetaPageDeg ?: 0f
        )
        Core.subtract(roiMask, templateVeto, roiMask)
        templateVeto.release()

        // -------- Bordes en ROI (FULL) --------
        val edgesRoiU8 = edgesFullU8.copyOf()
        applyMaskU8(edgesRoiU8, bufW, bufH, roiMask)

        // -------- Radial afuera→adentro --------
        val insetStartPx = max(2.0, (1.0 * pxPerMmUsed).toDouble())
        val rOuter = (roiR.toDouble() - insetStartPx).coerceAtLeast(4.0)
        val rMin = max(4.0, rOuter * 0.35)

        val rawRFine = DoubleArray(N_RADIALES_FINE) { Double.NaN }
        val hitFine = BooleanArray(N_RADIALES_FINE)
        var hits = 0

        for (i in 0 until N_RADIALES_FINE) {
            val theta = -2.0 * Math.PI * i / N_RADIALES_FINE
            val r = scanRadialOutsideInU8(
                edgesU8 = edgesRoiU8,
                w = bufW,
                h = bufH,
                cx = refCx.toDouble(),
                cy = refCy.toDouble(),
                theta = theta,
                rOuter = rOuter,
                rMin = rMin
            )

            if (r != null) {
                rawRFine[i] = r
                hitFine[i] = true
                hits++
            } else {
                hitFine[i] = false
            }
        }

        val coverageRadial = hits.toFloat() / N_RADIALES_FINE

        val outlinePx: List<PointF> =
            if (coverageRadial >= params.minCoverageRadial) {
                val filled = fillMissingCircularLinear(rawRFine, hitFine)
                val smooth = smoothCircular(filled)

                val maxJumpPx = (params.maxJumpMm * pxPerMmUsed).coerceAtLeast(1f)
                val smoothClamped = clampCircularDeltas(smooth, maxJumpPx)

                val hitAll = BooleanArray(smoothClamped.size) { true }
                buildContinuousOutlinePolar(
                    cx = refCx,
                    cy = refCy,
                    radii = smoothClamped,
                    hitMask = hitAll,
                    pxPerMm = pxPerMmUsed,
                    maxJumpMm = max(0.05f, params.maxJumpMm)
                )
            } else {
                // Fallback por contorno (sobre edgesRoiU8 ya vetado)
                val edgesMat = Mat(bufH, bufW, CvType.CV_8UC1)
                edgesMat.put(0, 0, edgesRoiU8)

                val contours = mutableListOf<MatOfPoint>()
                val hierarchy = Mat()
                Imgproc.findContours(
                    edgesMat,
                    contours,
                    hierarchy,
                    Imgproc.RETR_EXTERNAL,
                    Imgproc.CHAIN_APPROX_NONE
                )

                hierarchy.release()
                edgesMat.release()

                var best: MatOfPoint? = null
                var bestScore = -1.0
                val roiArea = Math.PI * roiR.toDouble() * roiR.toDouble()

                for (cnt in contours) {
                    val area = Imgproc.contourArea(cnt)
                    if (area < 10.0) {
                        cnt.release()
                        continue
                    }

                    val m = Imgproc.moments(cnt)
                    val ccx = (m.m10 / (m.m00 + 1e-6))
                    val ccy = (m.m01 / (m.m00 + 1e-6))
                    val dc = hypot(ccx - refCx, ccy - refCy)
                    if (dc > roiR * 0.40) {
                        cnt.release()
                        continue
                    }

                    val cnt2f = MatOfPoint2f(*cnt.toArray())
                    val per = Imgproc.arcLength(cnt2f, true)
                    cnt2f.release()
                    if (per <= 1e-3) {
                        cnt.release()
                        continue
                    }

                    val circularity = (4.0 * Math.PI * area) / (per * per)
                    val areaRatio = area / roiArea

                    if (areaRatio !in 0.02..0.85) {
                        cnt.release()
                        continue
                    }
                    if (circularity > 0.92) {
                        cnt.release()
                        continue
                    }

                    val score = areaRatio * (1.0 - circularity)
                    if (score > bestScore) {
                        best?.release()
                        best = cnt
                        bestScore = score
                    } else {
                        cnt.release()
                    }
                }

                val pts = if (best != null) {
                    val arr = best.toArray().map { p -> PointF(p.x.toFloat(), p.y.toFloat()) }
                    best.release()
                    arr
                } else {
                    emptyList()
                }

                if (pts.isNotEmpty()) closeAndSubsample(pts) else emptyList()
            }

        // -------- Métricas de calidad (nitidez en ROI) --------
        val lap = Mat()
        Imgproc.Laplacian(grayFull, lap, CvType.CV_32F)

        val mean = MatOfDouble()
        val std = MatOfDouble()
        Core.meanStdDev(lap, mean, std, roiMask)

        val sharp = (std.toArray().firstOrNull() ?: 0.0).toFloat()

        lap.release()
        mean.release()
        std.release()

        val ok = outlinePx.isNotEmpty()
        val coverage = if (ok) max(coverageRadial, 0f) else 0f

        // -------- Downsample fino → 800 para radii --------
        val radiiPxRaw: FloatArray =
            if (ok && coverageRadial >= params.minCoverageRadial) {
                val filled = fillMissingCircularLinear(rawRFine, hitFine)
                val smooth = smoothCircular(filled)
                val maxJumpPx = (params.maxJumpMm * pxPerMmUsed).coerceAtLeast(1f)
                val smoothClamped = clampCircularDeltas(smooth, maxJumpPx)

                FloatArray(N_RADIALES) { i ->
                    val idxFine = (i.toLong() * N_RADIALES_FINE / N_RADIALES).toInt()
                    smoothClamped[idxFine.coerceIn(0, N_RADIALES_FINE - 1)].toFloat()
                }
            } else {
                floatArrayOf()
            }

        val radiiMmRaw: FloatArray =
            if (radiiPxRaw.isNotEmpty()) {
                FloatArray(radiiPxRaw.size) { i -> radiiPxRaw[i] / pxPerMmUsed }
            } else {
                floatArrayOf()
            }

        // -------- Limpieza adicional por continuidad en mm --------
        val (radiiMm, radiiPx) =
            if (radiiMmRaw.isNotEmpty() && params.maxJumpMm > 0f) {
                val cleanedMm = cleanRadiiByContinuity(radiiMmRaw, params.maxJumpMm, iterations = 2)
                val cleanedPx = FloatArray(cleanedMm.size) { i -> cleanedMm[i] * pxPerMmUsed }
                cleanedMm to cleanedPx
            } else {
                radiiMmRaw to radiiPxRaw
            }

        Log.d(
            TAG,
            "ROI Ø${params.circleDiamMm.roundToInt()}: covRadial=${"%.2f".format(coverageRadial)} hits=$hits/$N_RADIALES_FINE px/mm=${"%.2f".format(pxPerMmUsed)} detRpx=${"%.1f".format(detRpx)} useRpx=${"%.1f".format(useRpx)} scanRatio=${"%.2f".format(scanRatio)}"
        )

        roiMask.release()
        edgesFull.release()
        grayFull.release()

        return PrecalMetrics(
            radiiPx = radiiPx,
            radiiMm = radiiMm,
            cxPx = refCx,
            cyPx = refCy,
            bufW = bufW,
            bufH = bufH,
            pxPerMm = pxPerMmUsed,
            cameraId = "back",
            zoom = 1.0f,
            coverage = coverage,
            sharpness = sharp,
            confidence = when {
                coverageRadial >= 0.80f -> 0.9f
                coverageRadial >= 0.60f -> 0.7f
                else -> if (ok) 0.5f else 0f
            },
            valid = ok,
            reasonIfInvalid = if (ok) null else "No outline found inside circle ROI",
            outlineBuf = outlinePx.map { Pair(it.x, it.y) }
        )
    }

    private data class RingDet(
        val cxFull: Float,
        val cyFull: Float,
        val detRFull: Float,
        val useRFull: Float
    )

    /**
     * MISMA FOTO => el calibrador ya resolvió cx/cy/rPx en coords del bitmap.
     * Acá SOLO armamos un ROI estable, no “detectamos” nada.
     */
    private fun detectRingCircleForRoi(
        grayFull: Mat,
        pxPerMmFull: Float,
        params: Params,
        refCircle: RefCalibrator75.Ref75? = null
    ): RingDet? {
        val wFull = grayFull.cols()
        val hFull = grayFull.rows()
        if (wFull <= 8 || hFull <= 8) return null

        val expectedRFull = (0.5f * params.circleDiamMm * pxPerMmFull).coerceAtLeast(8f)

        val cx = (refCircle?.cx ?: (wFull * 0.5f)).coerceIn(0f, (wFull - 1).toFloat())
        val cy = (refCircle?.cy ?: (hFull * 0.5f)).coerceIn(0f, (hFull - 1).toFloat())

        val detR = refCircle?.rPx?.takeIf { it > 8f } ?: expectedRFull

        val maxFit = min(
            min(cx, (wFull - 1).toFloat() - cx),
            min(cy, (hFull - 1).toFloat() - cy)
        )

        // Si vino radio del calibrador pro, lo usamos como base.
        // Si no, caemos al esperado por escala.
        val seedR = if (refCircle != null) detR else expectedRFull
        val useR = min(seedR, (maxFit * 0.98f).coerceAtLeast(8f))

        Log.d(
            TAG,
            "CIRC[TRC] cx=%.1f cy=%.1f detR=%.1f expR=%.1f useR=%.1f"
                .format(cx, cy, detR, expectedRFull, useR)
        )

        return RingDet(
            cxFull = cx,
            cyFull = cy,
            detRFull = detR,
            useRFull = useR
        )
    }

    /**
     * Veto explícito de elementos impresos de plantilla.
     * - franja baja total (texto muy inferior)
     * - barra nominal debajo del círculo, rotada según thetaPageDeg
     */
    private fun buildTemplateVetoMask(
        w: Int,
        h: Int,
        cx: Float,
        cy: Float,
        useRpx: Float,
        thetaPageDeg: Float
    ): Mat {
        val veto = Mat.zeros(h, w, CvType.CV_8UC1)

        // 1) Franja MUY baja
        val yLow = (cy + useRpx * 0.93f).roundToInt().coerceIn(0, h - 1)
        Imgproc.rectangle(
            veto,
            Point(0.0, yLow.toDouble()),
            Point((w - 1).toDouble(), (h - 1).toDouble()),
            Scalar(255.0),
            -1
        )

        // 2) Barra esperada, usando la orientación de página del calibrador
        val theta = Math.toRadians(thetaPageDeg.toDouble())
        val barCx = cx
        val barCy = cy + 0.72f * useRpx
        val halfW = 0.62f * useRpx
        val halfH = 0.08f * useRpx

        val local = arrayOf(
            Point(-halfW.toDouble(), -halfH.toDouble()),
            Point( halfW.toDouble(), -halfH.toDouble()),
            Point( halfW.toDouble(),  halfH.toDouble()),
            Point(-halfW.toDouble(),  halfH.toDouble())
        )

        val pts = ArrayList<Point>(4)
        for (p in local) {
            val rx = p.x * cos(theta) - p.y * sin(theta)
            val ry = p.x * sin(theta) + p.y * cos(theta)
            pts.add(
                Point(
                    (barCx + rx).coerceIn(0.0, (w - 1).toDouble()),
                    (barCy + ry).coerceIn(0.0, (h - 1).toDouble())
                )
            )
        }

        val poly = MatOfPoint(*pts.toTypedArray())
        Imgproc.fillConvexPoly(veto, poly, Scalar(255.0))
        poly.release()

        return veto
    }

    // =============================================================================================
    // Core helpers del tracer
    // =============================================================================================

    private fun scanRadialOutsideInU8(
        edgesU8: ByteArray,
        w: Int,
        h: Int,
        cx: Double,
        cy: Double,
        theta: Double,
        rOuter: Double,
        rMin: Double
    ): Double? {
        val ct = cos(theta)
        val st = sin(theta)

        var r = rOuter
        while (r >= rMin) {
            val x = (cx + r * ct).roundToInt()
            val y = (cy + r * st).roundToInt()

            if (x in 0 until w && y in 0 until h) {
                val idx = y * w + x
                if (idx in edgesU8.indices) {
                    if ((edgesU8[idx].toInt() and 0xFF) != 0) return r
                }
            }
            r -= RADIAL_STEP_PX
        }
        return null
    }

    private fun applyMaskU8(
        edgesU8: ByteArray,
        w: Int,
        h: Int,
        maskU8Mat: Mat
    ) {
        val mask = matToU8(maskU8Mat)

        val expected = w * h
        val n = min(min(edgesU8.size, mask.size), expected)

        for (i in 0 until n) {
            if ((mask[i].toInt() and 0xFF) == 0) {
                edgesU8[i] = 0
            }
        }

        val n2 = min(edgesU8.size, expected)
        for (i in n until n2) {
            edgesU8[i] = 0
        }
    }

    private fun fillMissingCircularLinear(r: DoubleArray, hit: BooleanArray): DoubleArray {
        val n = r.size
        if (n == 0) return r

        val hitsIdx = ArrayList<Int>()
        for (i in 0 until n) if (i < hit.size && hit[i] && r[i].isFinite()) hitsIdx.add(i)
        if (hitsIdx.isEmpty()) return r.copyOf()

        val out = r.copyOf()

        if (hitsIdx.size == 1) {
            val v = out[hitsIdx[0]]
            for (i in 0 until n) out[i] = v
            return out
        }

        hitsIdx.sort()

        for (k in hitsIdx.indices) {
            val i0 = hitsIdx[k]
            val i1 = hitsIdx[(k + 1) % hitsIdx.size]
            val r0 = out[i0]
            val r1 = out[i1]
            val d = (i1 - i0 + n) % n
            if (d == 0) continue

            for (t in 1 until d) {
                val idx = (i0 + t) % n
                val a = t.toDouble() / d.toDouble()
                out[idx] = (1.0 - a) * r0 + a * r1
            }
        }

        return out
    }

    private fun smoothCircular(values: DoubleArray): DoubleArray {
        fun wrap(a: DoubleArray, i: Int) = a[(i % a.size + a.size) % a.size]
        val n = values.size

        val med = DoubleArray(n)
        val m2 = SMOOTH_WIN_MEDIAN / 2
        for (i in 0 until n) {
            val w = DoubleArray(SMOOTH_WIN_MEDIAN) { k -> wrap(values, i + k - m2) }
            w.sort()
            med[i] = w[SMOOTH_WIN_MEDIAN / 2]
        }

        val out = DoubleArray(n)
        val a2 = SMOOTH_WIN_MEAN / 2
        for (i in 0 until n) {
            var s = 0.0
            for (k in -a2..a2) s += wrap(med, i + k)
            out[i] = s / SMOOTH_WIN_MEAN
        }
        return out
    }

    private fun clampCircularDeltas(r: DoubleArray, maxDeltaPx: Float): DoubleArray {
        val n = r.size
        if (n < 3) return r.copyOf()
        val md = max(1.0, maxDeltaPx.toDouble())

        fun clamp(v: Double, lo: Double, hi: Double) = max(lo, min(hi, v))

        val fwd = DoubleArray(n)
        fwd[0] = r[0]
        for (i in 1 until n) {
            val d = r[i] - fwd[i - 1]
            fwd[i] = fwd[i - 1] + clamp(d, -md, md)
        }

        val bwd = DoubleArray(n)
        bwd[n - 1] = r[n - 1]
        for (i in n - 2 downTo 0) {
            val d = r[i] - bwd[i + 1]
            bwd[i] = bwd[i + 1] + clamp(d, -md, md)
        }

        val out = DoubleArray(n)
        for (i in 0 until n) out[i] = (fwd[i] + bwd[i]) * 0.5
        return out
    }

    private fun closeAndSubsample(src: List<PointF>): List<PointF> {
        if (src.isEmpty()) return src
        val out = ArrayList<PointF>(min(src.size, FALLBACK_MAX_POINTS) + 1)
        if (src.size <= FALLBACK_MAX_POINTS) {
            out.addAll(src)
        } else {
            val step = src.size.toFloat() / FALLBACK_MAX_POINTS.toFloat()
            var f = 0f
            while (f < src.size && out.size < FALLBACK_MAX_POINTS) {
                out.add(src[f.toInt()])
                f += step
            }
        }
        if (out.size >= 2) {
            val a = out.first()
            val b = out.last()
            if (a.x != b.x || a.y != b.y) out.add(PointF(a.x, a.y))
        }
        return out
    }

    private fun buildContinuousOutlinePolar(
        cx: Float,
        cy: Float,
        radii: DoubleArray,
        hitMask: BooleanArray,
        pxPerMm: Float,
        maxJumpMm: Float
    ): List<PointF> {

        val n = radii.size
        if (n == 0 || pxPerMm <= 0f) return emptyList()

        val maxJumpPx = maxJumpMm * pxPerMm

        var startIdx = -1
        for (i in 0 until min(n, hitMask.size)) {
            if (hitMask[i] && radii[i].isFinite()) {
                startIdx = i
                break
            }
        }
        if (startIdx < 0) return emptyList()

        fun pointAt(idx: Int): PointF {
            val i = (idx % n + n) % n
            val th = -2.0 * Math.PI * i / n
            val r = radii[i]
            val x = cx + (r * cos(th)).toFloat()
            val y = cy + (r * sin(th)).toFloat()
            return PointF(x, y)
        }

        val selIdx = ArrayList<Int>(n + 1)
        var lastR = radii[startIdx]
        selIdx.add(startIdx)

        var idx = (startIdx + 1) % n
        var visited = 0
        val lookAhead = 8

        while (visited < n - 1) {
            if (idx < hitMask.size && hitMask[idx] && radii[idx].isFinite()) {
                val r = radii[idx]
                val dr = abs(r - lastR)

                if (dr <= maxJumpPx) {
                    selIdx.add(idx)
                    lastR = r
                    idx = (idx + 1) % n
                    visited++
                } else {
                    var found = false
                    var offset = 1
                    while (offset <= lookAhead && visited + offset < n) {
                        val j = (idx + offset) % n
                        if (j < hitMask.size && hitMask[j] && radii[j].isFinite()) {
                            val r2 = radii[j]
                            val dr2 = abs(r2 - lastR)
                            if (dr2 <= maxJumpPx) {
                                selIdx.add(j)
                                lastR = r2
                                idx = (j + 1) % n
                                visited += offset
                                found = true
                                break
                            }
                        }
                        offset++
                    }

                    if (!found) {
                        idx = (idx + 1) % n
                        visited++
                    }
                }
            } else {
                idx = (idx + 1) % n
                visited++
            }
        }

        if (selIdx.size < 2) return emptyList()

        val out = ArrayList<PointF>(selIdx.size + 1)
        for (i in selIdx) out.add(pointAt(i))

        val firstIdxSel = selIdx.first()
        val lastIdxSel = selIdx.last()
        val firstR = radii[firstIdxSel]
        val lastR2 = radii[lastIdxSel]
        val drClose = abs(lastR2 - firstR)
        val angGap = (firstIdxSel - lastIdxSel + n) % n

        if (drClose <= maxJumpPx && angGap <= lookAhead * 2 + 1) {
            val firstPt = out.first()
            out.add(PointF(firstPt.x, firstPt.y))
        }

        return out
    }

    private fun cleanRadiiByContinuity(
        radiiMm: FloatArray,
        maxJumpMm: Float,
        iterations: Int = 1
    ): FloatArray {
        val n = radiiMm.size
        if (n < 3 || maxJumpMm <= 0f) return radiiMm

        var cur = radiiMm.copyOf()
        val big = maxJumpMm * 1.5f

        fun idx(i: Int) = (i + n) % n

        repeat(iterations.coerceIn(1, 4)) {
            val out = cur.copyOf()
            for (i in 0 until n) {
                val prev = cur[idx(i - 1)]
                val curV = cur[i]
                val next = cur[idx(i + 1)]

                val dPrev = abs(curV - prev)
                val dNext = abs(curV - next)
                val dPN = abs(prev - next)

                if (dPrev > big && dNext > big && dPN <= maxJumpMm) {
                    out[i] = (prev + next) * 0.5f
                }
            }
            cur = out
        }

        return cur
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
}
