// vision/FilInnerRimFitter3250.kt
package com.dg.precaldnp.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.graphics.Rect
import android.graphics.RectF
import android.os.Environment
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream
import kotlin.math.*


class FilInnerRimFitter3250(
    private val context: Context,
    private val tag: String = "FilInnerRimFitter3250"
) {

    data class FilFitResult3250(
        val centerPx: PointF,
        val pxPerMm: Double,
        val rotDeg: Double,
        val outlinePx: List<PointF>,
        val score: Double,
        val inBoundsFrac: Double,
        val roiRect: Rect
    )

    fun fitInnerRimByFil3250(
        stillBmp: Bitmap,
        boxApprox: RectF,
        filPtsMm: List<PointF>,
        initPxPerMm: Double,
        initRotDeg: Double = 0.0,
        debugTag3250: String? = null
    ): FilFitResult3250? {
        if (filPtsMm.size < 40) return null

        val pad = ((boxApprox.width() * 0.22f).roundToInt()).coerceIn(30, 160)
        val l = (boxApprox.left.roundToInt() - pad).coerceIn(0, stillBmp.width - 2)
        val t = (boxApprox.top.roundToInt() - pad).coerceIn(0, stillBmp.height - 2)
        val r = (boxApprox.right.roundToInt() + pad).coerceIn(l + 2, stillBmp.width)
        val b = (boxApprox.bottom.roundToInt() + pad).coerceIn(t + 2, stillBmp.height)

        val roiW = r - l
        val roiH = b - t
        if (roiW < 80 || roiH < 80) return null

        val roiRect = Rect(l, t, r, b)
        val roiBmp = Bitmap.createBitmap(stillBmp, l, t, roiW, roiH)

        val mat = Mat()
        val gray = Mat()
        val gx = Mat()
        val gy = Mat()
        val grad = Mat()

        try {
            Utils.bitmapToMat(roiBmp, mat)
            Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2GRAY)
            Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)

            Imgproc.Sobel(gray, gx, CvType.CV_32F, 1, 0, 3)
            Imgproc.Sobel(gray, gy, CvType.CV_32F, 0, 1, 3)
            Core.magnitude(gx, gy, grad)
            Core.normalize(grad, grad, 0.0, 255.0, Core.NORM_MINMAX)

            val cols = grad.cols()
            val rows = grad.rows()
            val gradArr = FloatArray((grad.total()).toInt())
            grad.get(0, 0, gradArr)

            val initCx = (boxApprox.centerX() - l).toDouble()
            val initCy = (boxApprox.centerY() - t).toDouble()

            var bestCx = initCx
            var bestCy = initCy
            var bestS = max(initPxPerMm, 1e-6)
            var bestRot = initRotDeg
            var bestScore = Double.NEGATIVE_INFINITY
            var bestInFrac = 0.0

            fun search(
                rangeC: Int,
                stepC: Int,
                rangeSFrac: Double,
                stepSFrac: Double,
                rangeRot: Double,
                stepRot: Double,
                stepPts: Int
            ) {
                val s0 = bestS
                val cx0 = bestCx
                val cy0 = bestCy
                val rot0 = bestRot

                val sMin = (1.0 - rangeSFrac) * s0
                val sMax = (1.0 + rangeSFrac) * s0
                val sStep = max(1e-6, s0 * stepSFrac)

                var rot = rot0 - rangeRot
                while (rot <= rot0 + rangeRot + 1e-9) {
                    var s = sMin
                    while (s <= sMax + 1e-12) {
                        var dy = -rangeC
                        while (dy <= rangeC) {
                            var dx = -rangeC
                            while (dx <= rangeC) {
                                val cx = cx0 + dx
                                val cy = cy0 + dy
                                val (sum, inFrac) = score(
                                    filPtsMm,
                                    cols,
                                    rows,
                                    stepPts,
                                    gradArr,
                                    cx,
                                    cy,
                                    s,
                                    rot
                                )
                                val scPen = sum * inFrac
                                if (scPen > bestScore) {
                                    bestScore = scPen
                                    bestInFrac = inFrac
                                    bestCx = cx
                                    bestCy = cy
                                    bestS = s
                                    bestRot = rot
                                }
                                dx += stepC
                            }
                            dy += stepC
                        }
                        s += sStep
                    }
                    rot += stepRot
                }
            }

            // coarse -> fine (copiado conceptualmente del original)
            search(
                rangeC = 30,
                stepC = 3,
                rangeSFrac = 0.12,
                stepSFrac = 0.02,
                rangeRot = 6.0,
                stepRot = 2.0,
                stepPts = 2
            )
            search(
                rangeC = 12,
                stepC = 1,
                rangeSFrac = 0.05,
                stepSFrac = 0.01,
                rangeRot = 2.0,
                stepRot = 1.0,
                stepPts = 2
            )

            if (bestInFrac < 0.90) return null

            val th = Math.toRadians(bestRot)
            val c = cos(th)
            val sn = sin(th)

            val outline = ArrayList<PointF>(filPtsMm.size)
            for (p in filPtsMm) {
                val xMm = p.x.toDouble()
                val yMm0 = p.y.toDouble()

                val xrMm = (xMm * c) - (yMm0 * sn)
                val yrMm = (xMm * sn) + (yMm0 * c)

                val xPxRoi = bestS * xrMm + bestCx
                val yPxRoi = bestS * (-yrMm) + bestCy // ojo: flip Y igual que el decompiled

                outline.add(PointF((xPxRoi + l).toFloat(), (yPxRoi + t).toFloat()))
            }

            if (debugTag3250 != null) {
                saveDebugOutline3250(
                    stillBmp,
                    boxApprox,
                    outline,
                    bestS,
                    bestRot,
                    bestScore,
                    bestInFrac,
                    debugTag3250
                )
            }

            return FilFitResult3250(
                centerPx = PointF((bestCx + l).toFloat(), (bestCy + t).toFloat()),
                pxPerMm = bestS,
                rotDeg = bestRot,
                outlinePx = outline,
                score = bestScore,
                inBoundsFrac = bestInFrac,
                roiRect = roiRect
            )
        } catch (t: Throwable) {
            Log.e(tag, "fitInnerRimByFil3250 error", t)
            return null
        } finally {
            mat.release(); gray.release(); gx.release(); gy.release(); grad.release()
            roiBmp.recycle()
        }
    }

    private fun score(
        ptsMm: List<PointF>,
        cols: Int,
        rows: Int,
        stepPts: Int,
        gradArr: FloatArray,
        cx: Double,
        cy: Double,
        s: Double,
        rotDeg: Double
    ): Pair<Double, Double> {
        val th = Math.toRadians(rotDeg)
        val c = cos(th)
        val sn = sin(th)

        var sum = 0.0
        var inb = 0
        var total = 0

        var i = 0
        while (i < ptsMm.size) {
            val p = ptsMm[i]
            val xMm = p.x.toDouble()
            val yMm0 = p.y.toDouble()

            val xrMm = (xMm * c) - (yMm0 * sn)
            val yrMm = (xMm * sn) + (yMm0 * c)

            val xr = (xrMm * s) + cx
            val yr = ((-yrMm) * s) + cy

            val xi = xr.roundToInt()
            val yi = yr.roundToInt()
            total++

            if (xi in 0 until cols && yi in 0 until rows) {
                inb++
                sum += gradArr[yi * cols + xi].toDouble()
            }

            i += stepPts
        }

        val inFrac = if (total > 0) inb.toDouble() / total.toDouble() else 0.0
        return Pair(sum, inFrac)
    }
    private fun saveDebugOutline3250(
        stillBmp: Bitmap,
        box: RectF,
        outline: List<PointF>,
        s: Double,
        rot: Double,
        score: Double,
        inFrac: Double,
        debugTag: String
    ) {
        try {
            val dbg = stillBmp.copy(Bitmap.Config.ARGB_8888, true)
            val c = Canvas(dbg)

            val paintBox = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                style = Paint.Style.STROKE
                strokeWidth = 3f
                color = Color.MAGENTA
            }
            val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                style = Paint.Style.STROKE
                strokeWidth = 4f
                color = Color.GREEN
            }
            val text = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                style = Paint.Style.FILL
                textSize = 32f
                color = Color.WHITE
            }

            c.drawRect(box, paintBox)
            for (i in 1 until outline.size) {
                val a = outline[i - 1]
                val b = outline[i]
                c.drawLine(a.x, a.y, b.x, b.y, paint)
            }
            if (outline.isNotEmpty()) {
                val a0 = outline.last()
                val b0 = outline.first()
                c.drawLine(a0.x, a0.y, b0.x, b0.y, paint)
            }

            c.drawText(
                java.lang.String.format(
                    java.util.Locale.US,
                    "fit px/mm=%.4f rot=%.1f score=%.1f in=%.2f",
                    s, rot, score, inFrac
                ),
                20f, 40f, text
            )

            val base = context.getExternalFilesDir(Environment.DIRECTORY_PICTURES)
                ?: throw IllegalStateException("getExternalFilesDir returned null")

            val dir = File(base, "debug3250").apply { mkdirs() }
            val f = File(dir, "fil_fit_${debugTag}_${System.currentTimeMillis()}.jpg")

            FileOutputStream(f).use { out ->
                dbg.compress(Bitmap.CompressFormat.JPEG, 92, out)
            }

            Log.d(tag, "DEBUG saved: ${f.absolutePath}")
        } catch (t: Throwable) {
            Log.e(tag, "DEBUG save failed", t)
        }
    }
}
