@file:Suppress("SameParameterValue")

package com.dg.precaldnp.io

import android.graphics.PointF
import com.dg.precaldnp.model.PrecalMetrics
import java.util.Locale
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.min
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.sin

data class FilSimpleGeo(
    val hboxMm: Float,
    val vboxMm: Float,
    val fedMm: Float,
    val circMm: Float
)

/**
 * OUTLINE -> FIL (convención 3250 AS-IS)
 *
 * mm (interno):
 *  - Origen (0,0) = (cxPx, cyPx) de PrecalMetrics en coords del BUFFER
 *  - +X derecha
 *  - +Y arriba  (invertimos Y al pasar de imagen->mm)
 *
 * FIL 800 (AS-IS 3250, consistente con FilContourBuilder3250):
 *  - idx 0   => +X
 *  - idx 200 => +Y
 *  - idx 400 => -X
 *  - idx 600 => -Y
 *  - CCW: theta = 2π * idx / 800
 *
 * Este FIL representa OD "as-is". Para OI en DNP: espejo en X (tu flipOdToOi3250).
 */
object OutlineToFil {

    private const val N_SAMPLES = 800
    private const val RESAMPLE_ARC = 1600
    private const val SMOOTH_WIN_POLY = 9

    // Regularización (más suave, y sin “escalones”)
    private const val DESPIKE_WIN = 5       // mediana circular
    private const val SMOOTH_WIN_R = 7      // promedio circular
    private const val CLAMP_K = 7.0         // k * MAD(diffs)
    private const val CLAMP_MIN_HUND = 12   // mínimo clamp en 0.12 mm (hundredths)

    // RadiusBias.dat (800 floats en mm) en convención AS-IS 3250
    private var radiusBiasMm: FloatArray? = null
    private var radiusBiasMeanMm: Float = 0f

    fun setRadiusBias(biasMm: FloatArray, shiftSteps3250: Int = 0) {
        require(biasMm.size == N_SAMPLES) {
            "RadiusBias debe tener $N_SAMPLES valores, tiene ${biasMm.size}"
        }
        val b = if (shiftSteps3250 == 0) biasMm.copyOf() else rotateBiasLikePrecal(biasMm, shiftSteps3250)
        radiusBiasMm = b
        val sum = b.fold(0.0) { acc, v -> acc + v }
        radiusBiasMeanMm = (sum / b.size).toFloat()
    }

    fun buildFilFromPrecal(jobId: String, m: PrecalMetrics): String {
        require(m.valid) { "PrecalMetrics no válido: ${m.reasonIfInvalid}" }
        require(m.outlineBuf.isNotEmpty()) { "Contorno vacío en PrecalMetrics.outlineBuf" }

        val pxPerMm = m.pxPerMm.toDouble()
        require(pxPerMm > 0.0) { "pxPerMm inválido: ${m.pxPerMm}" }

        // 0) Si ya tenemos radiiMm (800) del tracer, preferimos eso (evita reintroducir ruido)
        val radiiFromMetricsHund: IntArray? =
            if (m.radiiMm.size == N_SAMPLES && m.radiiMm.all { it.isFinite() && it > 0f }) {
                IntArray(N_SAMPLES) { i -> (m.radiiMm[i] * 100f).roundToInt().coerceAtLeast(0) }
            } else null

        // 1) BUFFER(px) -> mm centrado en (0,0), Y-up
        val cxPx = m.cxPx.toDouble()
        val cyPx = m.cyPx.toDouble()
        val polyMm = m.outlineBuf.map { (xPx, yPx) ->
            val x = (xPx.toDouble() - cxPx) / pxPerMm
            val y = -(yPx.toDouble() - cyPx) / pxPerMm
            PointF(x.toFloat(), y.toFloat())
        }

        // 2) Cerrar + remuestreo + suavizado (para ray-cast)
        val closed = ensureClosed(polyMm)
        val resampled = resampleByArc(closed, RESAMPLE_ARC)
        val smoothPoly = smoothClosedPolyline(resampled, SMOOTH_WIN_POLY)

        // 3) Radii base: preferir métricas si están; si no, ray-cast robusto (sin zeros por miss)
        var radiiHund: IntArray =
            radiiFromMetricsHund ?: run {
                val rMm = radiiFromPolygonAsIs3250_Mm(smoothPoly, N_SAMPLES)  // DoubleArray con NaNs
                val filled = fillMissingCircularLinear(rMm)
                IntArray(N_SAMPLES) { i -> (filled[i] * 100.0).roundToInt().coerceAtLeast(0) }
            }

        // 4) Regularización (sin escalones)
        radiiHund = regularizeRadii(radiiHund)

        // 5) Geo consistente con radii finales
        val geo = computeGeoFromRadiiAsIs3250(radiiHund)

        // 6) Texto .FIL
        return buildString {
            appendLine("REQ=FIL")
            appendLine("JOB=\"$jobId\"")
            appendLine("STATUS=0")
            appendLine("TRCFMT=1;$N_SAMPLES;E;R;D")

            radiiHund.toList().chunked(8).forEach { chunk ->
                append("R=")
                append(chunk.joinToString(separator = ";", postfix = ";"))
                appendLine()
            }

            appendLine()
            appendLine("CIRC=%.2f;?".format(Locale.US, geo.circMm))
            appendLine("FED=%.2f;?".format(Locale.US, geo.fedMm))
            appendLine("HBOX=%.2f;?".format(Locale.US, geo.hboxMm))
            appendLine("VBOX=%.2f;?".format(Locale.US, geo.vboxMm))
            appendLine()
            appendLine("FMFR=MEDIRDNP")
            appendLine("FRAM=$jobId")
            appendLine("EYESIZ=%.2f".format(Locale.US, geo.fedMm))
            appendLine()
        }
    }

    // ---------------- helpers ----------------

    private fun ensureClosed(list: List<PointF>): List<PointF> {
        if (list.size < 2) return list
        val a = list.first()
        val b = list.last()
        return if (a.x == b.x && a.y == b.y) list else list + a
    }

    private fun resampleByArc(poly: List<PointF>, m: Int): MutableList<PointF> {
        val P = ensureClosed(poly)
        val n = P.size
        if (n < 3 || m < 3) return P.toMutableList()

        val seg = FloatArray(n - 1)
        var L = 0f
        for (i in 0 until n - 1) {
            val a = P[i]
            val b = P[i + 1]
            val d = hypot((b.x - a.x).toDouble(), (b.y - a.y).toDouble()).toFloat()
            seg[i] = d
            L += d
        }
        if (L <= 1e-6f) return P.toMutableList()

        val out = ArrayList<PointF>(m)
        val step = L / (m - 1)
        var target = 0f
        var i = 0
        var acc = 0f

        for (k in 0 until m) {
            while (i < seg.size && acc + seg[i] < target) {
                acc += seg[i]
                i++
            }
            if (i >= seg.size) {
                out.add(P.last())
                continue
            }
            val a = P[i]
            val b = P[i + 1]
            val denom = seg[i].coerceAtLeast(1e-6f)
            val t = ((target - acc) / denom).coerceIn(0f, 1f)
            out.add(PointF(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t))
            target += step
        }

        out[out.lastIndex] = out.first()
        return out
    }

    private fun smoothClosedPolyline(poly: List<PointF>, win: Int): MutableList<PointF> {
        if (win <= 1 || win % 2 == 0) return poly.toMutableList()
        val n = poly.size
        val k = win / 2
        fun idx(t: Int) = ((t % n) + n) % n

        val out = ArrayList<PointF>(n)
        for (i in 0 until n) {
            var sx = 0.0
            var sy = 0.0
            for (j in -k..k) {
                val p = poly[idx(i + j)]
                sx += p.x
                sy += p.y
            }
            out.add(PointF((sx / win).toFloat(), (sy / win).toFloat()))
        }
        out[out.lastIndex] = out.first()
        return out
    }

    /**
     * Ray casting desde origen en dirección theta (AS-IS 3250):
     *  theta = 2π*k/N, CCW, idx0 en +X.
     *
     * Importante: tomamos la intersección MÁS CERCANA (min t > 0).
     * Si no hay intersección, devolvemos NaN (NO 0).
     */
    private fun radiiFromPolygonAsIs3250_Mm(polyIn: List<PointF>, nSamples: Int): DoubleArray {
        val poly = ensureClosed(polyIn)
        require(poly.size >= 4) { "Polígono insuficiente para ray-cast (size=${poly.size})" }

        val pts = Array(poly.size) { i ->
            doubleArrayOf(poly[i].x.toDouble(), poly[i].y.toDouble())
        }

        val radiiMm = DoubleArray(nSamples) { Double.NaN }
        val eps = 1e-9
        val tMinEps = 1e-6

        for (k in 0 until nSamples) {
            val theta = 2.0 * Math.PI * k.toDouble() / nSamples.toDouble()
            val dx = cos(theta)
            val dy = sin(theta)

            var bestT = Double.POSITIVE_INFINITY

            for (i in 0 until pts.size - 1) {
                val p = pts[i]
                val q = pts[i + 1]
                val vx = q[0] - p[0]
                val vy = q[1] - p[1]
                if (abs(vx) + abs(vy) < 1e-12) continue

                val denom = dx * vy - dy * vx
                if (abs(denom) < eps) continue

                val t = (p[0] * vy - p[1] * vx) / denom
                if (t <= tMinEps) continue

                val u = (p[0] * dy - p[1] * dx) / denom
                if (u < 0.0 || u > 1.0) continue

                if (t < bestT) bestT = t
            }

            if (bestT.isFinite()) radiiMm[k] = bestT
        }

        return radiiMm
    }

    /**
     * Rellena NaN en un array circular por interpolación lineal entre hits vecinos.
     * Si hay 1 hit, replica ese valor en todo el círculo.
     */
    private fun fillMissingCircularLinear(r: DoubleArray): DoubleArray {
        val n = r.size
        if (n == 0) return r

        val hits = ArrayList<Int>()
        for (i in 0 until n) if (r[i].isFinite()) hits.add(i)
        if (hits.isEmpty()) return r.copyOf()
        if (hits.size == 1) {
            val v = r[hits[0]]
            return DoubleArray(n) { v }
        }

        hits.sort()
        val out = r.copyOf()

        for (k in hits.indices) {
            val i0 = hits[k]
            val i1 = hits[(k + 1) % hits.size]
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

    private fun computeGeoFromRadiiAsIs3250(rHund: IntArray): FilSimpleGeo {
        require(rHund.size == N_SAMPLES) { "rHund size != $N_SAMPLES" }

        var minX = Double.POSITIVE_INFINITY
        var maxX = Double.NEGATIVE_INFINITY
        var minY = Double.POSITIVE_INFINITY
        var maxY = Double.NEGATIVE_INFINITY

        var maxR = 0.0
        var circ = 0.0

        var prevX = 0.0
        var prevY = 0.0

        for (i in 0 until N_SAMPLES) {
            val rMm = rHund[i].toDouble() / 100.0
            val th = 2.0 * Math.PI * i.toDouble() / N_SAMPLES.toDouble()
            val x = rMm * cos(th)
            val y = rMm * sin(th)

            minX = min(minX, x); maxX = max(maxX, x)
            minY = min(minY, y); maxY = max(maxY, y)
            if (rMm > maxR) maxR = rMm

            if (i > 0) circ += hypot(x - prevX, y - prevY)
            prevX = x
            prevY = y
        }

        run {
            val r0 = rHund[0].toDouble() / 100.0
            val x0 = r0 * cos(0.0)
            val y0 = r0 * sin(0.0)
            circ += hypot(x0 - prevX, y0 - prevY)
        }

        return FilSimpleGeo(
            hboxMm = (maxX - minX).toFloat(),
            vboxMm = (maxY - minY).toFloat(),
            fedMm = (2.0 * maxR).toFloat(),
            circMm = circ.toFloat()
        )
    }

    // --------------- regularización ---------------

    private fun regularizeRadii(r: IntArray): IntArray {
        if (r.isEmpty()) return r

        // 1) Despike circular (mediana corta) – mata picos por reflejos
        var out = medianFilterCircular(r, DESPIKE_WIN)

        // 2) Suavizado leve
        out = smoothRadii(out, SMOOTH_WIN_R)

        // 3) Bias (si está)
        out = applyRadiusBias(out)

        // 4) Clamp robusto por diffs + cierre circular (sin escalones)
        out = clampDiffsCircular(out, CLAMP_K, CLAMP_MIN_HUND)

        // 5) Suavizado final leve
        out = smoothRadii(out, SMOOTH_WIN_R)

        return out
    }

    private fun smoothRadii(r: IntArray, win: Int): IntArray {
        if (win <= 1 || win % 2 == 0) return r
        val n = r.size
        val k = win / 2
        fun idx(t: Int) = ((t % n) + n) % n
        val out = IntArray(n)
        for (i in 0 until n) {
            var s = 0
            for (j in -k..k) s += r[idx(i + j)]
            out[i] = (s / win.toDouble()).roundToInt()
        }
        return out
    }

    private fun medianFilterCircular(r: IntArray, win: Int): IntArray {
        if (win <= 1 || win % 2 == 0) return r
        val n = r.size
        val k = win / 2
        fun idx(t: Int) = ((t % n) + n) % n
        val out = IntArray(n)
        val tmp = IntArray(win)
        for (i in 0 until n) {
            for (j in -k..k) tmp[j + k] = r[idx(i + j)]
            tmp.sort()
            out[i] = tmp[win / 2]
        }
        return out
    }

    /**
     * Clamp robusto de diffs (circular):
     * - diff[i] = r[i] - r[i-1] (con r[-1]=r[n-1])
     * - lim = max(minLim, k * MAD(|diff|))
     * - clamp cada diff a [-lim, +lim]
     * - ajusta residual para que sum(diff)=0 (cierre circular)
     * - integra para reconstruir r
     */
    private fun clampDiffsCircular(r: IntArray, k: Double, minLimHund: Int): IntArray {
        val n = r.size
        if (n < 3) return r
        val dif = IntArray(n)
        for (i in 0 until n) {
            val prev = r[(i - 1 + n) % n]
            dif[i] = r[i] - prev
        }

        val mad = medianAbs(dif).coerceAtLeast(1.0)
        val lim = max(minLimHund.toDouble(), k * mad).roundToInt()

        val dClamped = IntArray(n) { i ->
            val d = dif[i]
            when {
                d > lim -> lim
                d < -lim -> -lim
                else -> d
            }
        }

        // cierre circular: queremos sum(d)=0
        var residual = dClamped.sum()
        if (residual != 0) {
            // distribuimos residual restando/ sumando 1 hasta cerrar
            val step = if (residual > 0) -1 else 1
            residual = abs(residual)
            var idx = 0
            while (residual > 0 && idx < n * 4) {
                dClamped[idx % n] += step
                residual--
                idx++
            }
        }

        // integrar
        val out = IntArray(n)
        out[0] = r[0]
        for (i in 1 until n) {
            out[i] = out[i - 1] + dClamped[i]
        }

        // normalizar al promedio original (evita drift)
        val meanIn = r.average()
        val meanOut = out.average()
        val delta = (meanIn - meanOut).roundToInt()
        for (i in 0 until n) out[i] = (out[i] + delta).coerceAtLeast(0)

        return out
    }

    private fun medianAbs(values: IntArray): Double {
        if (values.isEmpty()) return 0.0
        val v = IntArray(values.size) { i -> kotlin.math.abs(values[i]) }
        v.sort()
        val n = v.size
        return if (n % 2 == 1) v[n / 2].toDouble()
        else (v[n / 2 - 1] + v[n / 2]) / 2.0
    }

    private fun applyRadiusBias(rHund: IntArray): IntArray {
        val bias = radiusBiasMm ?: return rHund
        if (bias.size != N_SAMPLES || rHund.size != N_SAMPLES) return rHund

        val mean = radiusBiasMeanMm
        val out = IntArray(N_SAMPLES)
        for (i in 0 until N_SAMPLES) {
            val deltaMm = bias[i] - mean
            val corr = rHund[i] - (deltaMm * 100f)
            out[i] = corr.roundToInt().coerceAtLeast(0)
        }
        return out
    }

    private fun rotateBiasLikePrecal(src: FloatArray, shiftSteps: Int): FloatArray {
        val n = src.size
        if (n == 0) return src
        val shift = ((shiftSteps % n) + n) % n
        if (shift == 0) return src.copyOf()

        val out = FloatArray(n)
        for (i in 0 until n) out[i] = src[(i - shift + n) % n]
        return out
    }
}
