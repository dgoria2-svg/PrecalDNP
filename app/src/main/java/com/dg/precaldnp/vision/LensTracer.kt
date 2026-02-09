package com.dg.precaldnp.vision

import android.graphics.Bitmap
import android.graphics.PointF
import android.util.Log
import com.dg.precaldnp.model.PrecalMetrics
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.*

/**
 * Trazador de lente dentro del aro Ø75.
 * - Detecta el círculo de referencia (Ø75 impreso) con Hough (prioriza radio esperado).
 * - Construye ROI interior excluyendo una banda alrededor del aro (para no “tomar” el aro).
 * - Búsqueda principal: barrido radial desde afuera→adentro con suavizado anti-serrucho.
 * - Oversampling angular: se muestrea fino (p.ej. 2400 radiales) y se resume a 800.
 * - Fallback: mayor contorno útil en ROI si el radial no llega a cobertura mínima.
 */
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
        val marginMmInside: Float = 6f,
        val ringBandMmExclude: Float = 3.0f,
        val cannyLow: Int = 40,
        val cannyHigh: Int = 120,
        val minCoverageRadial: Float = 0.60f,
        val approxEpsRel: Float = 0.001f,
        val maxCircularity: Float = 0.92f,
        val minAreaRatio: Float = 0.02f,
        val maxAreaRatio: Float = 0.85f,
        val clahe: Boolean = true,
        val sharpen: Boolean = true,

        // continuidad (mm) usada para clamp / limpieza
        val maxJumpMm: Float = 0.2f,

        // Anti-reflejos
        val highlightThr: Double = 280.0,     // umbral de “casi blanco”
        val highlightDilate: Int = 3,         // cuánto agrandar la máscara
        val highlightFillGray: Double = 180.0 // qué valor meter en zonas brillantes
    )

    fun trace(
        bitmap: Bitmap,
        pxPerMm: Float,
        cameraId: String,
        zoom: Float,
        params: Params = Params()
    ): PrecalMetrics? {
        val srcRgba = Mat()
        Utils.bitmapToMat(bitmap, srcRgba)
        val bgr = Mat()
        Imgproc.cvtColor(srcRgba, bgr, Imgproc.COLOR_RGBA2BGR)
        srcRgba.release()

        val out = traceBgr(bgr, pxPerMm, params)
        bgr.release()

        return out?.copy(cameraId = cameraId, zoom = zoom)
    }

    fun traceBgr(
        matBgr: Mat,
        pxPerMm: Float,
        params: Params = Params()
    ): PrecalMetrics? {

        val bufW = matBgr.cols()
        val bufH = matBgr.rows()

        // -------- Prepro base --------
        val gray = Mat()
        Imgproc.cvtColor(matBgr, gray, Imgproc.COLOR_BGR2GRAY)

        if (params.clahe) {
            val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
            clahe.apply(gray, gray)
        }

        if (params.sharpen) {
            val blur = Mat()
            Imgproc.GaussianBlur(gray, blur, Size(0.0, 0.0), 1.0)
            Core.addWeighted(gray, 1.5, blur, -0.5, 0.0, gray)
            blur.release()
        }

        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)

        // -------- Anti-reflejos (aplanar highlights) --------
        val highlights = Mat()
        Imgproc.threshold(gray, highlights, params.highlightThr, 255.0, Imgproc.THRESH_BINARY)
        val k3 = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0, 3.0))
        Imgproc.morphologyEx(highlights, highlights, Imgproc.MORPH_CLOSE, k3)
        if (params.highlightDilate > 0) {
            Imgproc.dilate(highlights, highlights, k3, Point(-1.0, -1.0), params.highlightDilate)
        }
        // “Neutralizamos” brillo: evita bordes internos falsos y cortes del borde real
        gray.setTo(Scalar(params.highlightFillGray), highlights)
        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)
        k3.release()
        highlights.release()

        // -------- Detectar aro Ø75 --------
        val expectedR = 37.5 * pxPerMm // px
        val circles = Mat()
        Imgproc.HoughCircles(
            gray, circles, Imgproc.HOUGH_GRADIENT,
            1.2,
            min(bufW, bufH) * 0.6,
            160.0,
            38.0,
            (min(bufW, bufH) * 0.15).toInt(),
            (min(bufW, bufH) * 0.49).toInt()
        )

        if (circles.empty()) {
            gray.release(); circles.release()
            return null
        }

        var refCx = 0f
        var refCy = 0f
        var refR = 0f
        run {
            val cxImg = bufW * 0.5
            val cyImg = bufH * 0.5
            var bestScore = Double.NEGATIVE_INFINITY
            for (i in 0 until circles.cols()) {
                val c = circles.get(0, i) ?: continue
                val x = c[0]; val y = c[1]; val r = c[2]
                val dr = -abs(r - expectedR)
                val dc = -hypot(x - cxImg, y - cyImg)
                val score = dr * 0.8 + dc * 0.2
                if (score > bestScore) {
                    bestScore = score
                    refCx = x.toFloat()
                    refCy = y.toFloat()
                    refR = r.toFloat()
                }
            }
        }
        circles.release()

        // -------- ROI interior (disco) y exclusión de banda en torno al aro --------
        val marginPx = (params.marginMmInside * pxPerMm).coerceAtLeast(2f)
        val roiR = max(8f, refR - marginPx)

        val roiMask = Mat.zeros(bufH, bufW, CvType.CV_8UC1)
        Imgproc.circle(
            roiMask,
            Point(refCx.toDouble(), refCy.toDouble()),
            roiR.toInt(),
            Scalar(255.0),
            -1
        )

        val bandPx = (params.ringBandMmExclude * pxPerMm).coerceAtLeast(2f)
        val ringOuter = Mat.zeros(bufH, bufW, CvType.CV_8UC1)
        val ringInner = Mat.zeros(bufH, bufW, CvType.CV_8UC1)
        Imgproc.circle(
            ringOuter,
            Point(refCx.toDouble(), refCy.toDouble()),
            (refR + bandPx).toInt(),
            Scalar(255.0),
            -1
        )
        Imgproc.circle(
            ringInner,
            Point(refCx.toDouble(), refCy.toDouble()),
            (refR - bandPx).toInt(),
            Scalar(255.0),
            -1
        )
        val ringBand = Mat()
        Core.subtract(ringOuter, ringInner, ringBand)
        Core.subtract(roiMask, ringBand, roiMask)
        ringOuter.release(); ringInner.release(); ringBand.release()

        // -------- Bordes en ROI --------
        val edges = Mat()
        Imgproc.Canny(gray, edges, params.cannyLow.toDouble(), params.cannyHigh.toDouble())
        Core.bitwise_and(edges, roiMask, edges)

        val k = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0, 3.0))
        Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, k, Point(-1.0, -1.0), 1)
        k.release()

        // -------- Radial afuera→adentro con SOBRE-MUESTREO --------
        val rOuter = (min(roiR.toDouble(), (refR - bandPx).toDouble()) -
                max(1.5, (1.0 * pxPerMm))).coerceAtLeast(4.0)
        val rMin = roiR * 0.35

        // OJO: si no hay hit, dejamos NaN (NO rMin), para no achicar con el smoothing
        val rawRFine = DoubleArray(N_RADIALES_FINE) { Double.NaN }
        val hitFine = BooleanArray(N_RADIALES_FINE)
        var hits = 0

        for (i in 0 until N_RADIALES_FINE) {
            val theta = -2.0 * Math.PI * i / N_RADIALES_FINE
            val r = scanRadialOutsideIn(
                edgesU8 = edges,
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

        val outlinePx: List<PointF> = if (coverageRadial >= params.minCoverageRadial) {

            // 1) Relleno circular por interpolación entre hits vecinos (evita “mordidas” por NaN)
            val filled = fillMissingCircularLinear(rawRFine, hitFine)

            // 2) Suavizado circular + clamp de deltas (evita spikes por reflejos)
            val smooth = smoothCircular(filled)
            val maxJumpPx = (params.maxJumpMm * pxPerMm).coerceAtLeast(1f)
            val smoothClamped = clampCircularDeltas(smooth, maxJumpPx)

            // 3) Contorno polar (pasamos hitMask = true para usar todos los ángulos ya “sanitizados”)
            val hitAll = BooleanArray(smoothClamped.size) { true }
            buildContinuousOutlinePolar(
                cx = refCx,
                cy = refCy,
                radii = smoothClamped,
                hitMask = hitAll,
                pxPerMm = pxPerMm,
                maxJumpMm = max(0.05f, params.maxJumpMm) // guardita
            )

        } else {

            // Fallback por contorno
            val contours = mutableListOf<MatOfPoint>()
            val hierarchy = Mat()
            Imgproc.findContours(
                edges,
                contours,
                hierarchy,
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_NONE
            )
            hierarchy.release()

            var best: MatOfPoint? = null
            var bestScore = -1.0
            val roiArea = Math.PI * roiR * roiR

            for (cnt in contours) {
                val area = Imgproc.contourArea(cnt)
                if (area < 10.0) {
                    cnt.release(); continue
                }
                val m = Imgproc.moments(cnt)
                val cx = (m.m10 / (m.m00 + 1e-6))
                val cy = (m.m01 / (m.m00 + 1e-6))
                val dc = hypot(cx - refCx, cy - refCy)
                if (dc > roiR * 0.40) {
                    cnt.release(); continue
                }

                val cnt2f = MatOfPoint2f(*cnt.toArray())
                val per = Imgproc.arcLength(cnt2f, true)
                cnt2f.release()
                if (per <= 1e-3) {
                    cnt.release(); continue
                }

                val circularity = (4.0 * Math.PI * area) / (per * per)
                val areaRatio = area / roiArea
                if (areaRatio !in params.minAreaRatio.toDouble()..params.maxAreaRatio.toDouble()) {
                    cnt.release(); continue
                }
                if (circularity > params.maxCircularity) {
                    cnt.release(); continue
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

        // -------- Métricas de calidad --------
        val lap = Mat()
        Imgproc.Laplacian(gray, lap, CvType.CV_32F)
        val mean = MatOfDouble()
        val std = MatOfDouble()
        Core.meanStdDev(lap, mean, std, roiMask)
        val sharp = (std.toArray().firstOrNull() ?: 0.0).toFloat()
        lap.release(); mean.release(); std.release()

        gray.release()
        roiMask.release()
        edges.release()

        val ok = outlinePx.isNotEmpty()
        val coverage = if (ok) max(coverageRadial, 0f) else 0f

        // -------- Downsample fino → 800 para radii --------
        val radiiPxRaw: FloatArray =
            if (ok && coverageRadial >= params.minCoverageRadial) {
                // reconstruimos una serie 2400 limpia y bajamos a 800
                val filled = fillMissingCircularLinear(rawRFine, hitFine)
                val smooth = smoothCircular(filled)
                val maxJumpPx = (params.maxJumpMm * pxPerMm).coerceAtLeast(1f)
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
                FloatArray(radiiPxRaw.size) { i -> radiiPxRaw[i] / pxPerMm }
            } else {
                floatArrayOf()
            }

        // -------- Limpieza adicional por continuidad en mm (suave, 2 iteraciones) --------
        val (radiiMm, radiiPx) =
            if (radiiMmRaw.isNotEmpty() && params.maxJumpMm > 0f) {
                val cleanedMm = cleanRadiiByContinuity(radiiMmRaw, params.maxJumpMm, iterations = 2)
                val cleanedPx = FloatArray(cleanedMm.size) { i -> cleanedMm[i] * pxPerMm }
                cleanedMm to cleanedPx
            } else {
                radiiMmRaw to radiiPxRaw
            }

        Log.d(
            TAG,
            "covRadial=${"%.2f".format(coverageRadial)} hits=$hits/$N_RADIALES_FINE  px/mm=${"%.2f".format(pxPerMm)}  refR=${"%.1f".format(refR)} expR=${"%.1f".format(expectedR)}"
        )

        return PrecalMetrics(
            radiiPx = radiiPx,
            radiiMm = radiiMm,
            cxPx = refCx,
            cyPx = refCy,
            bufW = bufW,
            bufH = bufH,
            pxPerMm = pxPerMm,
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
            reasonIfInvalid = if (ok) null else "No outline found inside ring",
            outlineBuf = outlinePx.map { Pair(it.x, it.y) }
        )
    }

    // ---------- Helpers ----------

    /** Barrido radial desde rOuter hacia adentro. Devuelve r o null si no encuentra. */
    private fun scanRadialOutsideIn(
        edgesU8: Mat,
        cx: Double,
        cy: Double,
        theta: Double,
        rOuter: Double,
        rMin: Double
    ): Double? {
        val w = edgesU8.cols()
        val h = edgesU8.rows()
        val ct = cos(theta)
        val st = sin(theta)

        var r = rOuter
        while (r >= rMin) {
            val x = (cx + r * ct).roundToInt()
            val y = (cy + r * st).roundToInt()
            if (x in 0 until w && y in 0 until h) {
                val v = edgesU8.get(y, x)
                if (v != null && v.isNotEmpty() && v[0] > 0.0) return r
            }
            r -= RADIAL_STEP_PX
        }
        return null
    }

    /**
     * Rellena NaN (misses) en un array circular interpolando entre hits vecinos.
     * Si hay 1 solo hit, replica ese valor en todo el círculo.
     */
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

        // recorremos cada segmento entre hit[k] -> hit[k+1], con wrap al final
        for (k in hitsIdx.indices) {
            val i0 = hitsIdx[k]
            val i1 = hitsIdx[(k + 1) % hitsIdx.size]
            val r0 = out[i0]
            val r1 = out[i1]
            val d = (i1 - i0 + n) % n
            if (d == 0) continue

            // llenamos los internos (1..d-1)
            for (t in 1 until d) {
                val idx = (i0 + t) % n
                val a = t.toDouble() / d.toDouble()
                out[idx] = (1.0 - a) * r0 + a * r1
            }
        }

        return out
    }

    /** Suaviza en círculo: mediana seguida de promedio móvil. */
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

    /**
     * Clamp circular de deltas: limita |Ri - Ri-1| a maxDeltaPx.
     * Hace forward + backward y promedia (suaviza spikes sin achicar).
     */
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

    /** Cierra y submuestrea un contorno largo a <= FALLBACK_MAX_POINTS, manteniendo orden. */
    private fun closeAndSubsample(src: List<PointF>): List<PointF> {
        if (src.isEmpty()) return src
        val maxPoints = FALLBACK_MAX_POINTS
        val out = ArrayList<PointF>(min(src.size, maxPoints) + 1)
        if (src.size <= maxPoints) {
            out.addAll(src)
        } else {
            val step = src.size.toFloat() / maxPoints.toFloat()
            var f = 0f
            while (f < src.size && out.size < maxPoints) {
                out.add(src[f.toInt()])
                f += step
            }
        }
        if (out.size >= 2) {
            val a = out.first(); val b = out.last()
            if (a.x != b.x || a.y != b.y) out.add(PointF(a.x, a.y))
        }
        return out
    }

    /**
     * Construye el contorno a partir de radios en polar imponiendo continuidad.
     * (Ahora se usa con hitMask=true en todos los ángulos si ya “sanitizamos” radii)
     */
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

        // cierre si es coherente
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

    /**
     * Limpia picos aislados en radios (mm) por continuidad (circular), con N iteraciones.
     */
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
}
