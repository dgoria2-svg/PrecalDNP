package com.dg.precaldnp.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker.FaceLandmarkerOptions
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import java.io.Closeable
import kotlin.math.abs

data class DnpMeasurementResult3250(
    val leftIrisCenterPx: PointF,
    val rightIrisCenterPx: PointF,
    val distancePx: Float,
    val distanceMm: Float?,
    val hasScale3250: Boolean,
    val pxPerMmUsed3250: Float?,
    val landmarksCount: Int,
    val timestampMs: Long,

    val noseTipPx: PointF? = null,
    val mouthCenterPx: PointF? = null,
    val midlineXpx: Float? = null,

    val midlineApx: PointF? = null,
    val midlineBpx: PointF? = null,

    // borde inferior de la ceja (px, coords imagen)
    val leftBrowBottomYpx: Float? = null,
    val rightBrowBottomYpx: Float? = null,
    val maskPolysGlobal3250: List<List<PointF>>? = null
)

class IrisDnpLandmarker3250(
    context: Context,
    private val pxPerMm3250: Float? = null
) : Closeable {

    companion object {
        private const val TAG = "IrisDnpLandmarker3250"
        private const val MODEL_ASSET = "face_landmarker.task"

        private val LEFT_IRIS = intArrayOf(468, 469, 470, 471, 472)
        private val RIGHT_IRIS = intArrayOf(473, 474, 475, 476, 477)

        private val LEFT_EYE = intArrayOf(
            33, 7, 163, 144, 145, 153, 154, 155,
            133, 173, 157, 158, 159, 160, 161, 246
        )

        private val RIGHT_EYE = intArrayOf(
            263, 249, 390, 373, 374, 380, 381, 382,
            362, 398, 384, 385, 386, 387, 388, 466
        )

        private const val IDX_NOSE_TIP = 1
        private const val IDX_UPPER_LIP = 13
        private const val IDX_LOWER_LIP = 14

        private const val IDX_NOSE_BRIDGE = 168
        private const val IDX_CHIN = 152

        private val LEFT_BROW = intArrayOf(70, 63, 105, 66, 107, 55, 65, 52, 53, 46)
        private val RIGHT_BROW = intArrayOf(300, 293, 334, 296, 336, 285, 295, 282, 283, 276)

        private fun maxIdx(vararg arrays: IntArray): Int {
            var m = 0
            for (arr in arrays) for (v in arr) if (v > m) m = v
            return m
        }
    }

    private val landmarker: FaceLandmarker

    init {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath(MODEL_ASSET)
            .build()

        val options = FaceLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setNumFaces(1)
            .setMinFaceDetectionConfidence(0.6f)
            .setMinFacePresenceConfidence(0.6f)
            .setMinTrackingConfidence(0.6f)
            .build()

        landmarker = FaceLandmarker.createFromOptions(context, options)
    }

    fun estimateDnp(
        bitmap: Bitmap,
        timestampMs: Long = 0L,
        pxPerMmOverride3250: Float? = null
    ): DnpMeasurementResult3250? {

        val mpImage: MPImage = BitmapImageBuilder(bitmap).build()

        val result: FaceLandmarkerResult = try {
            landmarker.detect(mpImage)
        } catch (t: Throwable) {
            Log.e(TAG, "FaceLandmarker.detect()", t)
            return null
        }

        val faces = result.faceLandmarks()
        if (faces.isEmpty()) return null

        val landmarks = faces[0]

        val maxIdxNeeded = maxIdx(RIGHT_IRIS, LEFT_BROW, RIGHT_BROW, intArrayOf(IDX_LOWER_LIP))
        if (landmarks.size <= maxIdxNeeded) {
            Log.w(TAG, "Muy pocos landmarks (${landmarks.size})")
            return null
        }

        val w = bitmap.width.toFloat()
        val h = bitmap.height.toFloat()

        // ---------- IRIS ----------
        val leftCenterNorm = irisCenter(landmarks, LEFT_IRIS)
        val rightCenterNorm = irisCenter(landmarks, RIGHT_IRIS)

        var leftIrisPx = PointF(leftCenterNorm.x * w, leftCenterNorm.y * h)
        var rightIrisPx = PointF(rightCenterNorm.x * w, rightCenterNorm.y * h)

        // ordenar por X en imagen (izq->der)
        if (leftIrisPx.x > rightIrisPx.x) {
            val t = leftIrisPx
            leftIrisPx = rightIrisPx
            rightIrisPx = t
        }

        val dx = rightIrisPx.x - leftIrisPx.x
        val distPxH = abs(dx)

        // ---------- NARIZ / BOCA -> MIDLINE ----------
        var noseTipPx: PointF? = null
        var mouthCenterPx: PointF? = null
        var midlineXpx: Float? = null
        var midlineApx: PointF? = null
        var midlineBpx: PointF? = null

        try {
            val lmNose = landmarks[IDX_NOSE_TIP]
            val lmUp = landmarks[IDX_UPPER_LIP]
            val lmLo = landmarks[IDX_LOWER_LIP]

            val noseTip = PointF(lmNose.x() * w, lmNose.y() * h)
            val mouthC = PointF(
                ((lmUp.x() + lmLo.x()) * 0.5f) * w,
                ((lmUp.y() + lmLo.y()) * 0.5f) * h
            )

            noseTipPx = noseTip
            mouthCenterPx = mouthC

            val pts = ArrayList<PointF>(4)
            pts += noseTip
            pts += mouthC
            landmarks.getOrNull(IDX_NOSE_BRIDGE)?.let { lm -> pts += PointF(lm.x() * w, lm.y() * h) }
            landmarks.getOrNull(IDX_CHIN)?.let { lm -> pts += PointF(lm.x() * w, lm.y() * h) }

            val ab = fitLineXonY3250(pts)
            if (ab != null) {
                val (a0, b0) = ab

                // y de referencia: línea de irises (promedio)
                val irisY = (leftIrisPx.y + rightIrisPx.y) * 0.5f

                // x en ese y según fit
                var xMid = xAtY3250(a0, b0, irisY)

                // guardrail: SIEMPRE entre irises (con margen)
                val lo = leftIrisPx.x + 6f
                val hi = rightIrisPx.x - 6f
                val hasRoom = hi > lo
                if (hasRoom) xMid = xMid.coerceIn(lo, hi)

                // ✅ CLAVE: re-anclar la recta para que pase por (irisY, xMid clamped)
                val bAdj = xMid - a0 * irisY

                // sanity de pendiente: si se va a la mierda, fallback vertical
                val dxOverH = a0 * h // delta X entre y=0..h
                val slopeOk = dxOverH.isFinite() && abs(dxOverH) <= (w * 0.75f)

                if (slopeOk) {
                    val xA = xAtY3250(a0, bAdj, 0f)
                    val xB = xAtY3250(a0, bAdj, h)

                    midlineApx = PointF(xA, 0f)
                    midlineBpx = PointF(xB, h)
                    midlineXpx = xMid
                } else {
                    // fallback vertical en xMid (ya clamp entre irises)
                    midlineXpx = xMid
                    midlineApx = PointF(xMid, 0f)
                    midlineBpx = PointF(xMid, h)
                }

                Log.d(TAG, "MID3250 fit a=$a0 b0=$b0 -> bAdj=$bAdj irisY=$irisY xMid=$midlineXpx slopeOk=$slopeOk")
            } else {
                // fallback vertical
                val xMid0 = ((lmNose.x() + (lmUp.x() + lmLo.x()) * 0.5f) * 0.5f) * w
                val xMid = xMid0.coerceIn(0f, w - 1f)
                midlineXpx = xMid
                midlineApx = PointF(xMid, 0f)
                midlineBpx = PointF(xMid, h)
            }
        } catch (_: Throwable) {
            // caller hace fallback
        }

        // ---------- CEJAS ----------
        fun browBottom(indices: IntArray): Float =
            indices.maxOf { landmarks[it].y() * h }

        fun browCenterX(indices: IntArray): Float =
            indices.map { landmarks[it].x() }.average().toFloat() * w

        var lb = browBottom(LEFT_BROW)
        var rb = browBottom(RIGHT_BROW)

        if (browCenterX(LEFT_BROW) > browCenterX(RIGHT_BROW)) {
            val t = lb; lb = rb; rb = t
        }

        val margin = 6f
        val leftBrow = lb.takeIf { it < leftIrisPx.y - margin }
        val rightBrow = rb.takeIf { it < rightIrisPx.y - margin }

        // ---------- ESCALA ----------
        val scale = pxPerMmOverride3250 ?: pxPerMm3250
        val distMm = scale?.takeIf { it > 0f }?.let { distPxH / it }
        val hasScale = scale != null && scale > 0f

        val maskPolys = buildMaskPolysGlobal3250(landmarks, w, h)

        return DnpMeasurementResult3250(
            leftIrisCenterPx = leftIrisPx,
            rightIrisCenterPx = rightIrisPx,
            distancePx = distPxH,
            distanceMm = distMm,
            hasScale3250 = hasScale,
            pxPerMmUsed3250 = scale,
            landmarksCount = landmarks.size,
            timestampMs = timestampMs,
            noseTipPx = noseTipPx,
            mouthCenterPx = mouthCenterPx,
            midlineXpx = midlineXpx,
            midlineApx = midlineApx,
            midlineBpx = midlineBpx,
            maskPolysGlobal3250 = maskPolys,
            leftBrowBottomYpx = leftBrow,
            rightBrowBottomYpx = rightBrow
        )
    }

    private fun irisCenter(
        landmarks: List<NormalizedLandmark>,
        indices: IntArray
    ): PointF {
        var sx = 0f
        var sy = 0f
        for (i in indices) {
            sx += landmarks[i].x()
            sy += landmarks[i].y()
        }
        val inv = 1f / indices.size
        return PointF(sx * inv, sy * inv)
    }

    private fun buildMaskPolysGlobal3250(
        landmarks: List<NormalizedLandmark>,
        w: Float,
        h: Float
    ): List<List<PointF>> {

        fun ptsFromIdx(idx: IntArray): List<PointF> =
            idx.map {
                val lm = landmarks[it]
                PointF(lm.x() * w, lm.y() * h)
            }

        fun hull(points: List<PointF>): List<PointF> =
            FaceMaskHull3250.convexHull(points)

        val out = ArrayList<List<PointF>>(4)
        out += hull(ptsFromIdx(LEFT_EYE))
        out += hull(ptsFromIdx(RIGHT_EYE))
        out += hull(ptsFromIdx(LEFT_BROW))
        out += hull(ptsFromIdx(RIGHT_BROW))
        return out.filter { it.size >= 3 }
    }

    private fun fitLineXonY3250(points: List<PointF>): Pair<Float, Float>? {
        if (points.size < 2) return null

        var sumY = 0.0
        var sumX = 0.0
        var sumYY = 0.0
        var sumYX = 0.0

        for (p in points) {
            val y = p.y.toDouble()
            val x = p.x.toDouble()
            sumY += y
            sumX += x
            sumYY += y * y
            sumYX += y * x
        }

        val n = points.size.toDouble()
        val denom = (n * sumYY - sumY * sumY)
        if (abs(denom) < 1e-6) {
            val b = (sumX / n).toFloat()
            return 0f to b
        }

        val a = ((n * sumYX - sumY * sumX) / denom).toFloat()
        val b = ((sumX - a * sumY) / n).toFloat()
        return a to b
    }

    private fun xAtY3250(a: Float, b: Float, y: Float): Float = a * y + b

    override fun close() {
        try { landmarker.close() } catch (_: Throwable) {}
    }

    object FaceMaskHull3250 {
        fun convexHull(points: List<PointF>): List<PointF> {
            if (points.size < 3) return points

            val sorted = points.sortedWith(compareBy({ it.y }, { it.x }))
            val pivot = sorted.first()

            val byAngle = sorted.drop(1).sortedBy { p ->
                kotlin.math.atan2(p.y - pivot.y, p.x - pivot.x)
            }

            if (byAngle.isEmpty()) return listOf(pivot)

            val stack = mutableListOf(pivot, byAngle.first())

            for (i in 1 until byAngle.size) {
                val p = byAngle[i]

                while (stack.size >= 2) {
                    val top = stack[stack.size - 1]
                    val second = stack[stack.size - 2]

                    val cross = (top.x - second.x) * (p.y - second.y) -
                            (top.y - second.y) * (p.x - second.x)

                    if (cross <= 0) stack.removeAt(stack.size - 1) else break
                }

                stack.add(p)
            }

            return stack
        }
    }
}
