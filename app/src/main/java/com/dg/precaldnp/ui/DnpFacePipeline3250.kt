package com.dg.precaldnp.ui

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.PointF
import android.graphics.RectF
import android.net.Uri
import android.os.Environment
import android.util.Log
import android.view.Surface
import androidx.annotation.StringRes
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import com.dg.precaldnp.R
import com.dg.precaldnp.model.DnpShotHint3250
import com.dg.precaldnp.model.MeasurementResult
import com.dg.precaldnp.model.ShapeTraceResult
import com.dg.precaldnp.model.WorkEye3250
import com.dg.precaldnp.vision.DebugDump3250
import com.dg.precaldnp.vision.EdgeArcfitDebugDump3250
import com.dg.precaldnp.vision.EdgeMapBuilder3250
import com.dg.precaldnp.vision.FilContourBuilder3250
import com.dg.precaldnp.vision.FilPlaceFromArc3250
import com.dg.precaldnp.vision.IrisDnpLandmarker3250
import com.dg.precaldnp.vision.PupilFrameEngine3250
import com.dg.precaldnp.vision.RimArcSeed3250
import com.dg.precaldnp.vision.RimDetectionResult
import com.dg.precaldnp.vision.RimDetectorBlock3250
import com.dg.precaldnp.vision.RimRenderSampler3250
import org.opencv.core.Rect
import java.io.ByteArrayInputStream
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt


object DnpFacePipeline3250 {

    private const val TAG = "DnpFacePipeline3250"

    fun interface StageCb3250 {
        fun onStage(@StringRes resId: Int)
    }

    data class MeasureOut3250(
        val originalUriStr: String,
        val dbgPath3250: String?,
        val metrics: DnpMetrics3250
    )

    data class IrisPack3250(
        val ok: Boolean,
        val midlineXpx: Float?,
        val midlineA: PointF?,
        val midlineB: PointF?,
        val pLeft: PointF?,
        val pRight: PointF?,
        val browLeftBottomYpx: Float?,
        val browRightBottomYpx: Float?
    )

    data class BoxesPack3250(
        val boxOdF: RectF,
        val boxOiF: RectF,
        val leftIsReal: Boolean,
        val rightIsReal: Boolean
    )

    data class PupilPick(
        val od: PointF,
        val oi: PointF,
        val estimated: Boolean,
        val src: String
    )

    data class RoiSrcPack3250(
        val roiOdRectF: RectF,
        val roiOiRectF: RectF,
        val roiOdCv: Rect,
        val roiOiCv: Rect,
        val pxPerMmGuessFace: Float,
        val bridgeRowYGlobal3250: Float,
        val midXBridgeGlobal3250: Float,
        val rimSrcBmp: Bitmap,
        val rimSrcBmpToRecycle: Bitmap?,
        val browBottomOdY: Float?,
        val browBottomOiY: Float?
    )

    data class FitPack3250(
        val pxPerMmFaceD: Double,
        val placedOdUsed: List<PointF>?,
        val placedOiUsed: List<PointF>?,
        val rimOd: RimDetectionResult?,
        val rimOi: RimDetectionResult?,
        val dbgPath3250: String?
    )

    data class Midline3250(
        val a: PointF,
        val b: PointF,
        val src: String,
        val estimated: Boolean
    ) {
        fun xAt(y: Float, w: Int): Float {
            val dy = (b.y - a.y)
            val x = if (abs(dy) < 1e-3f) {
                (a.x + b.x) * 0.5f
            } else {
                val t = (y - a.y) / dy
                a.x + t * (b.x - a.x)
            }
            return x.coerceIn(0f, (w - 1).toFloat())
        }
    }

    data class PupilsMidPack3250(
        var pupilOdDet: PointF,
        var pupilOiDet: PointF,
        val odOkReal: Boolean,
        val oiOkReal: Boolean,
        val hadBothReal: Boolean,

        val midline3250: Midline3250,
        val midXpxBridge3250: Float,
        val bridgeRowYpxGlobalForRoi: Float,

        val pupilOdForRoi: PointF,
        val pupilOiForRoi: PointF,
        val pupilPick: PupilPick
    )

    // ============================================================
    // ✅ ÚNICO CONTRATO: FULL edges U8 (0/255) desde STILL, una sola vez.
    // ============================================================

    fun measureArcFitOnlyCore3250(
        ctx: Context,
        savedUri: Uri,
        st: ShapeTraceResult,
        sp: DnpShotHint3250,
        filHboxMm: Double,
        filVboxMm: Double,
        filOdMm: FilContourBuilder3250.ContourMm3250,
        filOiMm: FilContourBuilder3250.ContourMm3250,
        irisEngine: IrisDnpLandmarker3250?,
        pupilEngine: PupilFrameEngine3250,
        filOverInnerMmPerSide: Double,
        edgeMapFullU8_3250: ByteArray? = null, // ✅ inyección opcional (debug); si null se construye.
        saveRingRoiDebug: Boolean = false,
        saveEdgeArcfitDebugToGallery3250: Boolean = false,
        stage: StageCb3250? = null
    ): MeasureOut3250? {

        val stillBmp = loadBitmapOriented(ctx, savedUri) ?: error("No se pudo leer la foto.")

        // 1) Landmarks primero (para topKill/mask)
        val iris = estimateIrisPack3250(ctx, stillBmp, irisEngine)

        // 2) Boxes (para máscara simple de ojos/cejas)
        val boxes = resolveBoxesStill3250(stillBmp, sp, filHboxMm, filVboxMm)

        // 3) topKillY + maskFull (si no hay iris => 0 y null)
        val topKillY3250 = computeTopKillYFromIris3250(iris, stillBmp.height)
        val maskFull3250 = buildMaskFullEyesAndBrows3250(
            w = stillBmp.width,
            h = stillBmp.height,
            iris = iris,
            boxes = boxes
        )

        // 4) ✅ FULL edge map U8 (0/255): UNA sola vez por foto
        val w = stillBmp.width
        val h = stillBmp.height

        val nFullL = w.toLong() * h.toLong()
        if (nFullL <= 0L || nFullL > Int.MAX_VALUE.toLong()) {
            Log.e("EDGE3250", "nFull overflow/invalid: w=$w h=$h n=$nFullL")
            // abortar o bajar resolución del still antes de edge
        }
        val nFull = nFullL.toInt()

        val edgeFullU8: ByteArray =
            edgeMapFullU8_3250?.takeIf { it.size == nFull }
                ?: EdgeMapBuilder3250.buildFullFrameEdgeMapFromBitmap3250(
                    stillBmp = stillBmp,
                    borderKillPx = 4,
                    topKillY = topKillY3250,
                    maskFull = maskFull3250,
                    debugTag = "FACE",
                    ctx = ctx,
                    debugSaveToGallery3250 = true
                )

        // 5) Pupilas + midline (solo landmarks/cascade; NO OpenCV midline)
        val pm = resolvePupilsAndMidline3250(
            stillBmp = stillBmp,
            iris = iris,
            boxes = boxes,
            pupilEngine = pupilEngine,
            isMirrored3250 = sp.isMirrored3250
        )

        // 6) ROI packs (global)
        val roiSrc = buildRoiAndRimSource3250(
            ctx = ctx,
            stillBmp = stillBmp,
            iris = iris,
            boxes = boxes,
            pm = pm,
            filHboxMm = filHboxMm,
            filVboxMm = filVboxMm,
            saveRingRoiDebug = saveRingRoiDebug
        )

        // 7) RimDetector + ArcFit consumen SOLO edgeFullU8
        val fit = detectFitAndRefine3250(
            ctx = ctx,
            stillBmp = stillBmp,
            rimSrcBmp = roiSrc.rimSrcBmp,
            roiSrc = roiSrc,
            pm = pm,
            iris = iris,
            filHboxMm = filHboxMm,
            filVboxMm = filVboxMm,
            filOdMm = filOdMm,
            filOiMm = filOiMm,
            filOverInnerMmPerSide = filOverInnerMmPerSide,
            edgeMapFullU8_3250 = edgeFullU8,
            saveEdgeArcfitDebugToGallery3250 = saveEdgeArcfitDebugToGallery3250,
            stage = stage
        )

        val metrics = DnpFaceMetrics3250.computeMetrics3250(
            stillBmp = stillBmp,
            pm = pm,
            fit = fit,
            pupilSrc = pm.pupilPick.src
        )

        try { roiSrc.rimSrcBmpToRecycle?.recycle() } catch (_: Throwable) {}

        return MeasureOut3250(
            originalUriStr = savedUri.toString(),
            dbgPath3250 = fit.dbgPath3250,
            metrics = metrics
        )
    }

    // ============================================================

    fun readTextFromUri(ctx: Context, uri: Uri): String? =
        try {
            ctx.contentResolver.openInputStream(uri)?.bufferedReader()?.use { it.readText() }
        } catch (_: Throwable) {
            null
        }

    fun surfaceRotationToDegrees3250(rotation: Int): Int = when (rotation) {
        Surface.ROTATION_0 -> 0
        Surface.ROTATION_90 -> 90
        Surface.ROTATION_180 -> 180
        Surface.ROTATION_270 -> 270
        else -> 0
    }

    fun buildBaselineShotHint3250(
        viewW: Int,
        viewH: Int,
        rotationDeg: Int,
        filHboxMm: Double,
        filVboxMm: Double
    ): DnpShotHint3250 {

        val midX = viewW * 0.5f
        val vboxMm = filVboxMm.takeIf { it.isFinite() && it > 1.0 } ?: 60.0
        val hboxMm = filHboxMm.takeIf { it.isFinite() && it > 1.0 } ?: (vboxMm * 1.20)
        val ratio = (hboxMm / vboxMm).coerceIn(0.85, 1.60).toFloat()

        val boxH = (viewH * 0.26f).coerceIn(120f, viewH * 0.60f)
        val boxW = (boxH * ratio).coerceIn(140f, viewW * 0.70f)

        val cy = (viewH * 0.40f).coerceIn(boxH * 0.6f, viewH - boxH * 0.6f)
        val cxOd = viewW * 0.35f
        val cxOi = viewW * 0.65f

        val boxOd =
            RectF(cxOd - boxW * 0.5f, cy - boxH * 0.5f, cxOd + boxW * 0.5f, cy + boxH * 0.5f)
        val boxOi =
            RectF(cxOi - boxW * 0.5f, cy - boxH * 0.5f, cxOi + boxW * 0.5f, cy + boxH * 0.5f)

        val odC = clampRectF3250(boxOd, viewW, viewH)
        val oiC = clampRectF3250(boxOi, viewW, viewH)

        val pxPerMm = (odC.height() / vboxMm.toFloat()).coerceAtLeast(0.0001f)

        return DnpShotHint3250(
            previewWidth = viewW,
            previewHeight = viewH,
            leftBoxPreview = oiC,
            rightBoxPreview = odC,
            leftBoxReal3250 = true,
            rightBoxReal3250 = true,
            midlineXPreview3250 = midX,
            midlineEstimated3250 = true,
            pxPerMmFace3250 = pxPerMm, // ✅ ya no gris
            rotationDegPreview3250 = rotationDeg,
            isMirrored3250 = false,
            workEye3250 = WorkEye3250.RIGHT_OD
        )
}

        // ============================================================

    private fun isPngBytes3250(bytes: ByteArray): Boolean {
        if (bytes.size < 8) return false
        return (bytes[0].toInt() and 0xFF) == 0x89 &&
                bytes[1].toInt() == 'P'.code &&
                bytes[2].toInt() == 'N'.code &&
                bytes[3].toInt() == 'G'.code &&
                (bytes[4].toInt() and 0xFF) == 0x0D &&
                (bytes[5].toInt() and 0xFF) == 0x0A &&
                (bytes[6].toInt() and 0xFF) == 0x1A &&
                (bytes[7].toInt() and 0xFF) == 0x0A
    }

    private fun loadBitmapOriented(ctx: Context, uri: Uri): Bitmap? {
        return try {
            val bytes = ctx.contentResolver.openInputStream(uri)?.use { it.readBytes() } ?: return null

            val opts = BitmapFactory.Options().apply {
                inPreferredConfig = Bitmap.Config.ARGB_8888
                @Suppress("DEPRECATION")
                inDither = false
                inScaled = false
            }

            val decoded = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts) ?: return null
            val bmp0 = if (decoded.config != Bitmap.Config.ARGB_8888) {
                decoded.copy(Bitmap.Config.ARGB_8888, false).also {
                    try { decoded.recycle() } catch (_: Throwable) {}
                }
            } else decoded

            // PNG (burst) => ya viene orientado, no EXIF.
            if (isPngBytes3250(bytes)) return bmp0

            val orientation = try {
                ExifInterface(ByteArrayInputStream(bytes)).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL
                )
            } catch (_: Throwable) {
                ExifInterface.ORIENTATION_NORMAL
            }

            val matrix = Matrix()
            when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
                ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
                ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            }

            if (!matrix.isIdentity) {
                val rotated = Bitmap.createBitmap(bmp0, 0, 0, bmp0.width, bmp0.height, matrix, true)
                if (rotated !== bmp0) {
                    try { bmp0.recycle() } catch (_: Throwable) {}
                }
                rotated
            } else {
                bmp0
            }
        } catch (_: Throwable) {
            null
        }
    }

    private fun clampRectF3250(r: RectF, w: Int, h: Int): RectF {
        val l = r.left.coerceIn(0f, (w - 1).toFloat())
        val t = r.top.coerceIn(0f, (h - 1).toFloat())
        val rr = r.right.coerceIn(0f, (w - 1).toFloat())
        val bb = r.bottom.coerceIn(0f, (h - 1).toFloat())
        return RectF(min(l, rr), min(t, bb), max(l, rr), max(t, bb))
    }

    private fun mapRectPreviewToStillLinearF(
        rView: RectF,
        viewW: Int,
        viewH: Int,
        stillW: Int,
        stillH: Int
    ): RectF {
        val sx = stillW.toFloat() / viewW.toFloat()
        val sy = stillH.toFloat() / viewH.toFloat()

        var l = rView.left * sx
        var t = rView.top * sy
        var r = rView.right * sx
        var b = rView.bottom * sy

        l = l.coerceIn(0f, stillW - 2f)
        t = t.coerceIn(0f, stillH - 2f)
        r = r.coerceIn(l + 1f, stillW.toFloat())
        b = b.coerceIn(t + 1f, stillH.toFloat())

        return RectF(l, t, r, b)
    }

    private fun estimateIrisPack3250(
        ctx: Context,
        stillBmp: Bitmap,
        irisEngine: IrisDnpLandmarker3250?
    ): IrisPack3250 {
        val irisRaw = try {
            irisEngine?.estimateDnp(
                stillBmp,
                System.currentTimeMillis(),
                pxPerMmOverride3250 = null
            )
        } catch (t: Throwable) {
            Log.e(TAG, "Iris error", t)
            null
        }

        return if (irisRaw != null) {
            IrisPack3250(
                ok = true,
                midlineXpx = irisRaw.midlineXpx,
                midlineA = irisRaw.midlineApx,
                midlineB = irisRaw.midlineBpx,
                pLeft = irisRaw.leftIrisCenterPx,
                pRight = irisRaw.rightIrisCenterPx,
                browLeftBottomYpx = irisRaw.leftBrowBottomYpx,
                browRightBottomYpx = irisRaw.rightBrowBottomYpx
            )
        } else {
            IrisPack3250(false, null, null, null, null, null, null, null)
        }
    }

    private fun resolveBoxesStill3250(
        stillBmp: Bitmap,
        sp: DnpShotHint3250,
        filHboxMm: Double,
        filVboxMm: Double
    ): BoxesPack3250 {

        val base = buildBaselineShotHint3250(
            sp.previewWidth,
            sp.previewHeight,
            sp.rotationDegPreview3250,
            filHboxMm,
            filVboxMm
        )

        // Contrato 3250: LEFT = OI, RIGHT = OD (ojo del paciente), independientemente de mirroring.
        val boxOiPreviewNN = sp.leftBoxPreview ?: sp.leftPupilRoiPreview ?: requireNotNull(base.leftBoxPreview)
        val boxOdPreviewNN = sp.rightBoxPreview ?: sp.rightPupilRoiPreview ?: requireNotNull(base.rightBoxPreview)

        val leftIsReal = (sp.leftBoxPreview != null && sp.leftBoxReal3250)     // OI
        val rightIsReal = (sp.rightBoxPreview != null && sp.rightBoxReal3250) // OD

        val boxOiStill0 = mapRectPreviewToStillLinearF(
            boxOiPreviewNN, sp.previewWidth, sp.previewHeight, stillBmp.width, stillBmp.height
        )
        val boxOdStill0 = mapRectPreviewToStillLinearF(
            boxOdPreviewNN, sp.previewWidth, sp.previewHeight, stillBmp.width, stillBmp.height
        )

        // Sanity: si el hint vino “cruzado”, lo corregimos SOLO si la separación es clara.
        val expectedOdLeft = !sp.isMirrored3250
        val odLeftOfOi = boxOdStill0.centerX() < boxOiStill0.centerX()
        val sep = abs(boxOdStill0.centerX() - boxOiStill0.centerX())
        val needSwap = (sep > 14f) && (odLeftOfOi != expectedOdLeft)

        val boxOdF = if (!needSwap) boxOdStill0 else boxOiStill0
        val boxOiF = if (!needSwap) boxOiStill0 else boxOdStill0

        if (needSwap) {
            Log.w(
                TAG,
                "BOXES3250: swapped by mirror sanity (mirrored=${sp.isMirrored3250}) " +
                        "sep=${"%.1f".format(sep)} odLeft=$odLeftOfOi expectedOdLeft=$expectedOdLeft"
            )
        }

        return BoxesPack3250(
            boxOdF = boxOdF,
            boxOiF = boxOiF,
            leftIsReal = leftIsReal,
            rightIsReal = rightIsReal
        )
    }

    private fun detectPupilsAssignOdOi(
        stillBitmap: Bitmap,
        boxOd: RectF,
        boxOi: RectF,
        pupilEngine: PupilFrameEngine3250
    ): PupilPick {
        val fc = pupilEngine.detectAndCheckFrame(stillBitmap)
        val raw = fc.pupils
        val p1 = raw?.left?.let { PointF(it.x.toFloat(), it.y.toFloat()) }
        val p2 = raw?.right?.let { PointF(it.x.toFloat(), it.y.toFloat()) }

        fun center(b: RectF): PointF = PointF(b.centerX(), b.centerY())

        if (p1 == null || p2 == null) return PupilPick(
            center(boxOd),
            center(boxOi),
            true,
            "cascade"
        )

        val p1InOd = boxOd.contains(p1.x, p1.y)
        val p2InOd = boxOd.contains(p2.x, p2.y)
        val p1InOi = boxOi.contains(p1.x, p1.y)
        val p2InOi = boxOi.contains(p2.x, p2.y)

        return when {
            p1InOd && p2InOi -> PupilPick(p1, p2, false, "cascade")
            p2InOd && p1InOi -> PupilPick(p2, p1, false, "cascade")
            p1InOd -> PupilPick(p1, center(boxOi), true, "cascade")
            p2InOd -> PupilPick(p2, center(boxOi), true, "cascade")
            p1InOi -> PupilPick(center(boxOd), p1, true, "cascade")
            p2InOi -> PupilPick(center(boxOd), p2, true, "cascade")
            else -> {
                val (left, right) = if (p1.x <= p2.x) p1 to p2 else p2 to p1
                PupilPick(left, right, false, "cascade")
            }
        }
    }

    private fun assignIrisToBoxes(
        p1: PointF,
        p2: PointF,
        boxOd: RectF,
        boxOi: RectF
    ): Pair<PointF, PointF> {
        val p1InOd = boxOd.contains(p1.x, p1.y)
        val p1InOi = boxOi.contains(p1.x, p1.y)
        val p2InOd = boxOd.contains(p2.x, p2.y)
        val p2InOi = boxOi.contains(p2.x, p2.y)

        fun dist2(a: PointF, b: PointF): Float {
            val dx = a.x - b.x
            val dy = a.y - b.y
            return dx * dx + dy * dy
        }

        return when {
            p1InOd && p2InOi -> p1 to p2
            p1InOi && p2InOd -> p2 to p1
            else -> {
                val cOd = PointF(boxOd.centerX(), boxOd.centerY())
                val cOi = PointF(boxOi.centerX(), boxOi.centerY())
                val cost12 = dist2(p1, cOd) + dist2(p2, cOi)
                val cost21 = dist2(p2, cOd) + dist2(p1, cOi)
                if (cost12 <= cost21) p1 to p2 else p2 to p1
            }
        }
    }

    private fun swapPupilsIfCrossedByBoxes3250(
        pOd: PointF,
        pOi: PointF,
        boxOd: RectF,
        boxOi: RectF
    ): Pair<PointF, PointF> {
        val odInOd = boxOd.contains(pOd.x, pOd.y)
        val oiInOi = boxOi.contains(pOi.x, pOi.y)
        if (odInOd && oiInOi) return pOd to pOi

        val odInOi = boxOi.contains(pOd.x, pOd.y)
        val oiInOd = boxOd.contains(pOi.x, pOi.y)

        // Swap SOLO si están claramente cruzadas.
        return if (odInOi && oiInOd) {
            Log.w(TAG, "PUPILS3250: swap SAFE by boxes (crossed)")
            pOi to pOd
        } else {
            pOd to pOi
        }
    }

    private fun resolvePupilsAndMidline3250(
        stillBmp: Bitmap,
        iris: IrisPack3250,
        boxes: BoxesPack3250,
        pupilEngine: PupilFrameEngine3250,
        isMirrored3250: Boolean
    ): PupilsMidPack3250 {

        val w = stillBmp.width
        val hF = stillBmp.height.toFloat()

        // 1) Pupilas oficiales: LANDMARKS si existen, si no cascade
        val pupilPick: PupilPick =
            if (iris.ok && iris.pLeft != null && iris.pRight != null) {
                val (od0, oi0) = assignIrisToBoxes(
                    iris.pLeft,
                    iris.pRight,
                    boxes.boxOdF,
                    boxes.boxOiF
                )
                PupilPick(od0, oi0, estimated = false, src = "iris")
            } else {
                detectPupilsAssignOdOi(stillBmp, boxes.boxOdF, boxes.boxOiF, pupilEngine)
            }

        var pupilOdDet = pupilPick.od
        var pupilOiDet = pupilPick.oi

        // 2) ÚNICO swap permitido: por boxes (cruce claro)
        run {
            val (od1, oi1) = swapPupilsIfCrossedByBoxes3250(pupilOdDet, pupilOiDet, boxes.boxOdF, boxes.boxOiF)
            pupilOdDet = od1
            pupilOiDet = oi1
        }

        val odOkReal = pupilOdDet.x.isFinite() && pupilOdDet.y.isFinite()
        val oiOkReal = pupilOiDet.x.isFinite() && pupilOiDet.y.isFinite()
        val hadBothReal = odOkReal && oiOkReal

        // 3) bridgeRowY oficial
        val bridgeRowY = when {
            hadBothReal -> ((pupilOdDet.y + pupilOiDet.y) * 0.5f)
            odOkReal -> pupilOdDet.y
            oiOkReal -> pupilOiDet.y
            else -> error("No pupila detected.")
        }.coerceIn(0f, hF)

        // 4) midX oficial: LANDMARKS si hay iris; NO usar OpenCV para midline
        val midXOfficial: Float
        val midSrc: String
        val midEstimated: Boolean

        if (iris.ok && hadBothReal) {
            val avgIrisX = ((pupilOdDet.x + pupilOiDet.x) * 0.5f).coerceIn(0f, (w - 1).toFloat())

            val dist = abs(pupilOdDet.x - pupilOiDet.x).coerceAtLeast(1f)
            val margin = max(8f, dist * 0.12f)
            val lo = min(pupilOdDet.x, pupilOiDet.x) + margin
            val hi = max(pupilOdDet.x, pupilOiDet.x) - margin

            val irisX = iris.midlineXpx
                ?.takeIf { it.isFinite() }
                ?.coerceIn(0f, (w - 1).toFloat())

            val useIrisX = (irisX != null && irisX in lo..hi)

            midXOfficial = if (useIrisX) irisX else avgIrisX
            midSrc = if (useIrisX) "irisX_betweenIrises" else "irisAvgX"
            midEstimated = !useIrisX

            if (irisX != null && !useIrisX) {
                Log.w(TAG, "MID3250: irisX=$irisX NOT between irises [${"%.1f".format(lo)}..${"%.1f".format(hi)}], using irisAvg=$avgIrisX")
            }
        } else {
            val fallback = if (hadBothReal) {
                ((pupilOdDet.x + pupilOiDet.x) * 0.5f)
            } else {
                iris.midlineXpx ?: (w * 0.5f)
            }
            midXOfficial = fallback.coerceIn(0f, (w - 1).toFloat())
            midSrc = if (hadBothReal) "pupilsAvg_fallback" else "center_fallback"
            midEstimated = true
        }

        // 5) midline vertical estable
        val midline = Midline3250(
            a = PointF(midXOfficial, 0f),
            b = PointF(midXOfficial, hF),
            src = midSrc,
            estimated = midEstimated
        )

        // 6) Para ROI: si falta un ojo, espejar respecto a midXOfficial en la MISMA fila
        val pupilRealAny = when {
            odOkReal -> pupilOdDet
            else -> pupilOiDet
        }

        val pupilOdForRoi =
            if (odOkReal) pupilOdDet else PointF(2f * midXOfficial - pupilRealAny.x, bridgeRowY)
        val pupilOiForRoi =
            if (oiOkReal) pupilOiDet else PointF(2f * midXOfficial - pupilRealAny.x, bridgeRowY)

        Log.d(
            TAG,
            "MID3250 OFFICIAL x=$midXOfficial src=$midSrc est=$midEstimated mirrored=$isMirrored3250 " +
                    "pOD=(%.1f,%.1f) pOI=(%.1f,%.1f) bridgeY=%.1f srcPupil=${pupilPick.src}".format(
                        pupilOdDet.x, pupilOdDet.y, pupilOiDet.x, pupilOiDet.y, bridgeRowY
                    )
        )

        return PupilsMidPack3250(
            pupilOdDet = pupilOdDet,
            pupilOiDet = pupilOiDet,
            odOkReal = odOkReal,
            oiOkReal = oiOkReal,
            hadBothReal = hadBothReal,
            midline3250 = midline,
            midXpxBridge3250 = midXOfficial,
            bridgeRowYpxGlobalForRoi = bridgeRowY,
            pupilOdForRoi = pupilOdForRoi,
            pupilOiForRoi = pupilOiForRoi,
            pupilPick = pupilPick
        )
    }

    // ============================================================
    // ✅ FULL edge map: topKill + mask helpers (para que compile y sea consistente)
    // ============================================================

    private fun computeTopKillYFromIris3250(iris: IrisPack3250, h: Int): Int {
        if (!iris.ok) return 0
        val y1 = iris.browLeftBottomYpx?.takeIf { it.isFinite() }
        val y2 = iris.browRightBottomYpx?.takeIf { it.isFinite() }
        val yMin = when {
            y1 != null && y2 != null -> min(y1, y2)
            y1 != null -> y1
            y2 != null -> y2
            else -> null
        } ?: return 0

        // Kill global opcional (suave). Si querés desactivarlo: retorná 0.
        val pad = 18f
        return (yMin - pad).roundToInt().coerceIn(0, h.coerceAtLeast(1) - 1)
    }

    /**
     * Máscara simple (U8): marca zonas de ojos/cejas para que EdgeMapBuilder haga flattening (anti-halo)
     * y no compute edges adentro.
     *
     * Si no hay iris/brows => null (queda sin máscara).
     */
    private fun buildMaskFullEyesAndBrows3250(
        w: Int,
        h: Int,
        iris: IrisPack3250,
        boxes: BoxesPack3250
    ): ByteArray? {
        if (!iris.ok) return null

        val browL = iris.browLeftBottomYpx?.takeIf { it.isFinite() }
        val browR = iris.browRightBottomYpx?.takeIf { it.isFinite() }
        if (browL == null && browR == null) return null

        val mask = ByteArray(w * h)

        fun fillRect(r: RectF) {
            val l = r.left.roundToInt().coerceIn(0, w - 1)
            val t = r.top.roundToInt().coerceIn(0, h - 1)
            val rr = r.right.roundToInt().coerceIn(l + 1, w)
            val bb = r.bottom.roundToInt().coerceIn(t + 1, h)
            for (yy in t until bb) {
                val off = yy * w
                for (xx in l until rr) {
                    mask[off + xx] = 1
                }
            }
        }

        fun eyeMaskFromBox(box: RectF, browBottomY: Float?) {
            val bb = browBottomY?.takeIf { it.isFinite() } ?: return
            val bw = box.width().coerceAtLeast(1f)
            val bh = box.height().coerceAtLeast(1f)

            val padX = (0.14f * bw).coerceIn(18f, 60f)
            val padY = (0.16f * bh).coerceIn(16f, 70f)

            // zona superior del box (ceja + párpado) hasta un poco debajo del browBottom
            val top = (box.top - padY).coerceIn(0f, (h - 1).toFloat())
            val bottom = (bb + 0.35f * bh).coerceIn(top + 2f, (h - 1).toFloat())

            val r = RectF(
                (box.left - padX).coerceIn(0f, (w - 1).toFloat()),
                top,
                (box.right + padX).coerceIn(0f, (w - 1).toFloat()),
                bottom
            )
            fillRect(r)
        }

        // OJO: browLeft/browRight están en coords STILL, pero “left/right” de iris no siempre coincide con OD/OI.
        // Acá solo hacemos una máscara útil; la asignación fina OD/OI la hace tu pipeline más abajo.
        // Usamos ambos browBottom para ambos boxes (robusto).
        val bbAny = listOfNotNull(browL, browR).minOrNull()
        val bbAny2 = listOfNotNull(browL, browR).maxOrNull()

        eyeMaskFromBox(boxes.boxOdF, bbAny)
        eyeMaskFromBox(boxes.boxOiF, bbAny2 ?: bbAny)

        return mask
    }

    // ============================================================

    private fun roiFromMidline40603250(
        mid: Midline3250,
        pupil: PointF,
        bmpW: Int,
        bmpH: Int,
        expWpx: Float,
        expHpx: Float,
        yRefGlobal: Float,
        pupilXFracFromMid: Float = 0.33f,
        gapPx: Float = 0f,
        widthScale: Float = 1.40f,
        heightScale: Float = 1.60f,
        pupilYFracFromTop: Float = 0.45f
    ): RectF {

        val midAtRef = mid.xAt(yRefGlobal, bmpW)
        val isLeftSide = pupil.x < midAtRef
        val distMidToPupil = abs(pupil.x - midAtRef).coerceAtLeast(12f)
        val w = bmpW.toFloat()
        val h = bmpH.toFloat()
        Log.d(
            TAG,
            "ROI3250 yRef=%.1f midAtRef=%.1f pupil=(%.1f,%.1f) isLeft=%s dist=%.1f expW=%.1f expH=%.1f"
                .format(
                    yRefGlobal, midAtRef, pupil.x, pupil.y,
                    isLeftSide, distMidToPupil, expWpx, expHpx
                )
        )

        val roiW = max(
            distMidToPupil / pupilXFracFromMid,
            (expWpx * widthScale).coerceAtLeast(expWpx + 80f)
        ).coerceIn(180f, w * 0.95f)

        val roiH = (expHpx * heightScale).coerceAtLeast(expHpx + 90f).coerceIn(180f, h * 0.90f)

        val top = (pupil.y - roiH * pupilYFracFromTop).coerceIn(0f, h - 2f)
        val bottom = (top + roiH).coerceIn(top + 2f, h)

        val gap = gapPx.coerceIn(0f, 60f)

        val midTop = mid.xAt(top, bmpW)
        val midBot = mid.xAt(bottom, bmpW)

        val left: Float
        val right: Float

        if (isLeftSide) {
            val seam = min(midTop, midBot) - gap
            right = seam.coerceIn(0f, w - 1f)
            left = (right - roiW).coerceIn(0f, right - 2f)
        } else {
            val seam = max(midTop, midBot) + gap
            left = seam.coerceIn(0f, w - 1f)
            right = (left + roiW).coerceIn(left + 2f, w)
        }
        Log.d(TAG, "ROI3250 rect L=%.1f T=%.1f R=%.1f B=%.1f".format(left, top, right, bottom))

        return RectF(left, top, right, bottom)
    }

    private fun clampRectFToBmp3250(r: RectF, w: Int, h: Int): RectF {
        val l = r.left.coerceIn(0f, (w - 2).toFloat())
        val t = r.top.coerceIn(0f, (h - 2).toFloat())
        val rr = r.right.coerceIn(l + 1f, (w - 1).toFloat())
        val bb = r.bottom.coerceIn(t + 1f, (h - 1).toFloat())
        return RectF(l, t, rr, bb)
    }

    private fun rectFToCvRect3250(r: RectF, w: Int, h: Int): Rect {
        val x = r.left.roundToInt().coerceIn(0, w - 2)
        val y = r.top.roundToInt().coerceIn(0, h - 2)
        val ww = r.width().roundToInt().coerceAtLeast(2).coerceAtMost(w - x)
        val hh = r.height().roundToInt().coerceAtLeast(2).coerceAtMost(h - y)
        return Rect(x, y, ww, hh)
    }

    private fun buildRoiAndRimSource3250(
        ctx: Context,
        stillBmp: Bitmap,
        iris: IrisPack3250,
        boxes: BoxesPack3250,
        pm: PupilsMidPack3250,
        filHboxMm: Double,
        filVboxMm: Double,
        saveRingRoiDebug: Boolean
    ): RoiSrcPack3250 {

        val browBottomOdY: Float?
        val browBottomOiY: Float?

        val bridgeY = pm.bridgeRowYpxGlobalForRoi
            .coerceIn(0f, (stillBmp.height - 1).toFloat())

        val midXBridge = pm.midXpxBridge3250
            .coerceIn(0f, (stillBmp.width - 1).toFloat())

        if (!iris.ok || iris.pLeft == null) {
            browBottomOdY = null
            browBottomOiY = null
        } else {
            val pL = iris.pLeft
            val leftIsOd = when {
                boxes.boxOdF.contains(pL.x, pL.y) && !boxes.boxOiF.contains(pL.x, pL.y) -> true
                !boxes.boxOdF.contains(pL.x, pL.y) && boxes.boxOiF.contains(pL.x, pL.y) -> false
                else -> abs(pL.x - boxes.boxOdF.centerX()) <= abs(pL.x - boxes.boxOiF.centerX())
            }
            browBottomOdY = if (leftIsOd) iris.browLeftBottomYpx else iris.browRightBottomYpx
            browBottomOiY = if (leftIsOd) iris.browRightBottomYpx else iris.browLeftBottomYpx
        }

        val distPupPx = abs(pm.pupilOiForRoi.x - pm.pupilOdForRoi.x)
        val pxPerMmGuessFace = (distPupPx / 65f).coerceIn(3.0f, 12.0f)

        val expWpx = (filHboxMm.toFloat() * pxPerMmGuessFace)
        val expHpx = (filVboxMm.toFloat() * pxPerMmGuessFace)

        Log.d(
            TAG,
            String.format(
                Locale.US,
                "MID3250 bridgeY=%.1f midXBridge=%.1f pxPerMmGuess=%.2f  pOD=(%.1f,%.1f) pOI=(%.1f,%.1f)  w=%d h=%d",
                bridgeY, midXBridge, pxPerMmGuessFace,
                pm.pupilOdForRoi.x, pm.pupilOdForRoi.y,
                pm.pupilOiForRoi.x, pm.pupilOiForRoi.y,
                stillBmp.width, stillBmp.height
            )
        )

        val rimSrcBmpToRecycle: Bitmap? = null

        val roiOD0 = roiFromMidline40603250(
            mid = pm.midline3250,
            yRefGlobal = bridgeY,
            pupil = pm.pupilOdForRoi,
            bmpW = stillBmp.width,
            bmpH = stillBmp.height,
            expWpx = expWpx,
            expHpx = expHpx
        )

        val roiOI0 = roiFromMidline40603250(
            mid = pm.midline3250,
            yRefGlobal = bridgeY,
            pupil = pm.pupilOiForRoi,
            bmpW = stillBmp.width,
            bmpH = stillBmp.height,
            expWpx = expWpx,
            expHpx = expHpx
        )

        val roiOdRectF = clampRectFToBmp3250(roiOD0, stillBmp.width, stillBmp.height)
        val roiOiRectF = clampRectFToBmp3250(roiOI0, stillBmp.width, stillBmp.height)

        if (saveRingRoiDebug) {
            cropBitmap(stillBmp, roiOdRectF)?.let { saveDebugBitmap3250(ctx, it, "roi_od") }
            cropBitmap(stillBmp, roiOiRectF)?.let { saveDebugBitmap3250(ctx, it, "roi_oi") }
        }

        val roiOdCv = rectFToCvRect3250(roiOdRectF, stillBmp.width, stillBmp.height)
        val roiOiCv = rectFToCvRect3250(roiOiRectF, stillBmp.width, stillBmp.height)

        return RoiSrcPack3250(
            roiOdRectF = roiOdRectF,
            roiOiRectF = roiOiRectF,
            roiOdCv = roiOdCv,
            roiOiCv = roiOiCv,
            pxPerMmGuessFace = pxPerMmGuessFace,
            bridgeRowYGlobal3250 = bridgeY,
            midXBridgeGlobal3250 = midXBridge,
            rimSrcBmp = stillBmp,
            rimSrcBmpToRecycle = rimSrcBmpToRecycle,
            browBottomOdY = browBottomOdY,
            browBottomOiY = browBottomOiY
        )
    }

    private fun cropBitmap(src: Bitmap, r: RectF): Bitmap? {
        val l = r.left.toInt().coerceIn(0, src.width - 2)
        val t = r.top.toInt().coerceIn(0, src.height - 2)
        val rr = r.right.toInt().coerceIn(l + 1, src.width)
        val bb = r.bottom.toInt().coerceIn(t + 1, src.height)
        return try {
            Bitmap.createBitmap(src, l, t, rr - l, bb - t)
        } catch (_: Throwable) {
            null
        }
    }

    private fun saveDebugBitmap3250(ctx: Context, bmp: Bitmap, namePrefix: String): String? {
        return try {
            val time = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())
            val dir = File(
                ctx.getExternalFilesDir(Environment.DIRECTORY_PICTURES),
                "debug3250"
            ).apply { mkdirs() }
            val f = File(dir, "${namePrefix}_$time.png")
            FileOutputStream(f).use { out -> bmp.compress(Bitmap.CompressFormat.PNG, 100, out) }
            Log.d(TAG, "DBG saved: ${f.absolutePath}")
            f.absolutePath
        } catch (t: Throwable) {
            Log.e(TAG, "DBG save error", t)
            null
        }
    }

    // ============================================================

    private fun fToLocalIntOrNull3250(vGlobal: Float?, x0: Int, limit: Int): Int? {
        val v = vGlobal ?: return null
        if (!v.isFinite()) return null
        return (v - x0.toFloat()).roundToInt().coerceIn(0, (limit - 1).coerceAtLeast(0))
    }

    private fun detectFitAndRefine3250(
        ctx: Context,
        stillBmp: Bitmap,
        rimSrcBmp: Bitmap,
        roiSrc: RoiSrcPack3250,
        pm: PupilsMidPack3250,
        iris: IrisPack3250,
        filHboxMm: Double,
        filVboxMm: Double,
        filOdMm: FilContourBuilder3250.ContourMm3250,
        filOiMm: FilContourBuilder3250.ContourMm3250,
        filOverInnerMmPerSide: Double,
        edgeMapFullU8_3250: ByteArray?,                 // ✅ ÚNICA fuente de edges (FULL)
        saveEdgeArcfitDebugToGallery3250: Boolean = false,
        stage: StageCb3250? = null
    ): FitPack3250 {

        // ✅ DIMENSIONES STILL
        val wFull = stillBmp.width
        val hFull = stillBmp.height
        val nFull = wFull * hFull

        if (edgeMapFullU8_3250 == null || edgeMapFullU8_3250.size != nFull) {
            Log.e(
                TAG,
                "EDGE_FULL missing/bad: edges=${edgeMapFullU8_3250?.size} expected=$nFull. Abort arcfit."
            )
            return FitPack3250(
                pxPerMmFaceD = roiSrc.pxPerMmGuessFace.toDouble(),
                placedOdUsed = null,
                placedOiUsed = null,
                rimOd = null,
                rimOi = null,
                dbgPath3250 = null
            )
        }

        // --------- OD ROI (GLOBAL) ----------
        val odL = roiSrc.roiOdRectF.left.toInt().coerceIn(0, wFull - 2)
        val odT = roiSrc.roiOdRectF.top.toInt().coerceIn(0, hFull - 2)
        val odR = roiSrc.roiOdRectF.right.toInt().coerceIn(odL + 1, wFull)
        val odB = roiSrc.roiOdRectF.bottom.toInt().coerceIn(odT + 1, hFull)

        val wOd = odR - odL
        val hOd = odB - odT
        val edgesOdU8 = ByteArray(wOd * hOd)

        for (yy in 0 until hOd) {
            System.arraycopy(
                edgeMapFullU8_3250, (odT + yy) * wFull + odL,
                edgesOdU8, yy * wOd,
                wOd
            )
        }
        val roiOdGlobal = RectF(odL.toFloat(), odT.toFloat(), odR.toFloat(), odB.toFloat())
        val roiOdCvCanon = Rect(odL, odT, wOd, hOd) // ✅ ROI canónico

        // --------- OI ROI (GLOBAL) ----------
        val oiL = roiSrc.roiOiRectF.left.toInt().coerceIn(0, wFull - 2)
        val oiT = roiSrc.roiOiRectF.top.toInt().coerceIn(0, hFull - 2)
        val oiR = roiSrc.roiOiRectF.right.toInt().coerceIn(oiL + 1, wFull)
        val oiB = roiSrc.roiOiRectF.bottom.toInt().coerceIn(oiT + 1, hFull)

        val wOi = oiR - oiL
        val hOi = oiB - oiT
        val edgesOiU8 = ByteArray(wOi * hOi)

        for (yy in 0 until hOi) {
            System.arraycopy(
                edgeMapFullU8_3250, (oiT + yy) * wFull + oiL,
                edgesOiU8, yy * wOi,
                wOi
            )
        }
        val roiOiGlobal = RectF(oiL.toFloat(), oiT.toFloat(), oiR.toFloat(), oiB.toFloat())
        val roiOiCvCanon = Rect(oiL, oiT, wOi, hOi) // ✅ ROI canónico

        // ---- FIL inner mm
        val over = filOverInnerMmPerSide.coerceAtLeast(1.0)
        val hboxInnerMm = (filHboxMm - 2.0 * over).coerceAtLeast(10.0)
        val vboxInnerMm = (filVboxMm - 2.0 * over).coerceAtLeast(10.0)

        Log.d(
            TAG,
            "FIL inset: overPerSideMm=$over -> innerHbox=$hboxInnerMm innerVbox=$vboxInnerMm"
        )

        stage?.onStage(R.string.dnp_stage_rim_detecting)

        // ✅ RimDetector trabaja SOBRE el recorte (ROI-local) + roiGlobal
        val detOd = RimDetectorBlock3250.detectRim(
            edgesU8 = edgesOdU8,
            w = wOd,
            h = hOd,
            roiGlobal = roiOdGlobal,
            midlineXpx = roiSrc.midXBridgeGlobal3250,
            browBottomYpx = roiSrc.browBottomOdY,
            filHboxMm = hboxInnerMm,
            filVboxMm = vboxInnerMm,
            bridgeRowYpxGlobal = roiSrc.bridgeRowYGlobal3250,
            pupilGlobal = pm.pupilOdForRoi,
            debugTag = "OD",
            pxPerMmGuessFace = roiSrc.pxPerMmGuessFace,
            seedXpxGlobal3250 = null
        )

        val detOi = RimDetectorBlock3250.detectRim(
            edgesU8 = edgesOiU8,
            w = wOi,
            h = hOi,
            roiGlobal = roiOiGlobal,
            midlineXpx = roiSrc.midXBridgeGlobal3250,
            browBottomYpx = roiSrc.browBottomOiY,
            filHboxMm = hboxInnerMm,
            filVboxMm = vboxInnerMm,
            bridgeRowYpxGlobal = roiSrc.bridgeRowYGlobal3250,
            pupilGlobal = pm.pupilOiForRoi,
            debugTag = "OI",
            pxPerMmGuessFace = roiSrc.pxPerMmGuessFace,
            seedXpxGlobal3250 = null
        )

        fun pxFrom(r: RimDetectionResult?): Double? {
            if (r?.ok != true) return null
            val w = r.innerWidthPx
            if (!w.isFinite() || w <= 1f) return null
            val pxmm = w.toDouble() / hboxInnerMm
            return pxmm.takeIf { it.isFinite() && it > 1e-6 }
        }

        val rimOd: RimDetectionResult? = detOd?.result
        val rimOi: RimDetectionResult? = detOi?.result

        val odPx = pxFrom(rimOd)
        val oiPx = pxFrom(rimOi)

        val pxPerMmOfficialRaw: Double = when {
            odPx != null && oiPx != null -> {
                val avg = (odPx + oiPx) * 0.5
                val relDiff = abs(odPx - oiPx) / avg.coerceAtLeast(1e-9)
                if (relDiff <= 0.18) avg
                else {
                    val g = roiSrc.pxPerMmGuessFace.toDouble()
                    val pick = if (abs(odPx - g) <= abs(oiPx - g)) odPx else oiPx
                    Log.w(
                        TAG,
                        "PX/MM seedFromRim[FACE]: OD/OI divergen od=$odPx oi=$oiPx relDiff=$relDiff -> pick=$pick (guess=$g)"
                    )
                    pick
                }
            }

            odPx != null -> {
                Log.w(TAG, "PX/MM seedFromRim[FACE]: using OD only = $odPx")
                odPx
            }

            oiPx != null -> {
                Log.w(TAG, "PX/MM seedFromRim[FACE]: using OI only = $oiPx")
                oiPx
            }

            else -> {
                val g = roiSrc.pxPerMmGuessFace.toDouble()
                Log.e(TAG, "PX/MM seedFromRim[FACE]: FAIL (W). Using guess=$g")
                g
            }
        }
        val seed = DnpScale3250.pxPerMmSeedFromRimWidth3250(
            rimOd = rimOd,
            rimOi = rimOi,
            hboxInnerMm = hboxInnerMm,
            pxPerMmGuessFace = roiSrc.pxPerMmGuessFace,
            tag = "FACE"
        )

        val pxPerMmOfficial = if (seed.seedPxPerMm.isFinite()) {
            seed.seedPxPerMm.toDouble()
        } else {
            roiSrc.pxPerMmGuessFace.toDouble() // fallback
        }

// si ya tenías clamp, dejalo igual
        val pxPerMmOfficialClamped = pxPerMmOfficial.coerceIn(0.5, 30.0)

        Log.d(
            TAG,
            "PX/MM seedFromRim[FACE]: od=${seed.pxPerMmOd} oi=${seed.pxPerMmOi} used=${seed.seedPxPerMm} src=${seed.src} -> usedClamped=$pxPerMmOfficialClamped"
        )


        Log.d(
            TAG,
            "PX/MM used[FACE]=$pxPerMmOfficialClamped (od=$odPx oi=$oiPx guess=${roiSrc.pxPerMmGuessFace})"
        )

        // ✅ Seeds usando EXACTAMENTE el ROI canónico (el recorte)
        val seedOd = RimArcSeed3250.seedFromPupilOrRoi3250(
            roiCv = roiOdCvCanon,
            filHboxMm = hboxInnerMm,
            filVboxMm = vboxInnerMm,
            pxPerMmGuess = pxPerMmOfficialClamped.toFloat(),
            midlineXpxGlobal = roiSrc.midXBridgeGlobal3250,
            pupilGlobal = pm.pupilOdForRoi,
            usePupil3250 = false,
            bridgeRowYpxGlobal = roiSrc.bridgeRowYGlobal3250
        )

        val seedOi = RimArcSeed3250.seedFromPupilOrRoi3250(
            roiCv = roiOiCvCanon,
            filHboxMm = hboxInnerMm,
            filVboxMm = vboxInnerMm,
            pxPerMmGuess = pxPerMmOfficialClamped.toFloat(),
            midlineXpxGlobal = roiSrc.midXBridgeGlobal3250,
            pupilGlobal = pm.pupilOiForRoi,
            usePupil3250 = false,
            bridgeRowYpxGlobal = roiSrc.bridgeRowYGlobal3250
        )

        stage?.onStage(R.string.dnp_stage_arc_fit)

        val midXGlobal = roiSrc.midXBridgeGlobal3250

        val midOdRoiX = (midXGlobal - odL.toFloat()).coerceIn(0f, (wOd - 1).toFloat())
        val midOiRoiX = (midXGlobal - oiL.toFloat()).coerceIn(0f, (wOi - 1).toFloat())

        val bottomAnchorOdYpxRoi = run {
            val yLocal =
                rimOd?.takeIf { it.ok }?.bottomYpx?.takeIf { it.isFinite() }
                    ?.let { it - odT.toFloat() }
                    ?: (seedOd.originPxRoi.y + 0.40f * (vboxInnerMm * pxPerMmOfficialClamped).toFloat())
            yLocal.coerceIn(0f, (hOd - 1).toFloat())
        }

        val bottomAnchorOiYpxRoi = run {
            val yLocal =
                rimOi?.takeIf { it.ok }?.bottomYpx?.takeIf { it.isFinite() }
                    ?.let { it - oiT.toFloat() }
                    ?: (seedOi.originPxRoi.y + 0.40f * (vboxInnerMm * pxPerMmOfficialClamped).toFloat())
            yLocal.coerceIn(0f, (hOi - 1).toFloat())
        }

        val fitArcOd = detOd?.let { pack ->
            FilPlaceFromArc3250.placeFilByArc3250(
                filPtsMm = filOdMm.ptsMm,
                filHboxMm = filHboxMm,
                filVboxMm = filVboxMm,
                edgesGray = pack.edges, // ✅ ROI-local (wOd*hOd)
                w = pack.w,
                h = pack.h,
                originPxRoi = seedOd.originPxRoi,
                midlineXpxRoi = midOdRoiX,
                pxPerMmInitGuess = seedOd.pxPerMmInitGuess,
                pxPerMmObserved = pxPerMmOfficialClamped,
                pxPerMmFixed = null,
                filOverInnerMmPerSide3250 = filOverInnerMmPerSide,
                excludeDegFrom = 89.0,
                excludeDegTo = 91.0,
                stepDeg = 2.0,
                iters = 2,
                rSearchRelLo = 0.70,
                rSearchRelHi = 1.35,
                bottomAnchorYpxRoi = bottomAnchorOdYpxRoi
            )
        }

        val fitArcOi = detOi?.let { pack ->
            FilPlaceFromArc3250.placeFilByArc3250(
                filPtsMm = filOiMm.ptsMm,
                filHboxMm = filHboxMm,
                filVboxMm = filVboxMm,
                edgesGray = pack.edges, // ✅ ROI-local
                w = pack.w,
                h = pack.h,
                originPxRoi = seedOi.originPxRoi,
                midlineXpxRoi = midOiRoiX,
                pxPerMmInitGuess = seedOi.pxPerMmInitGuess,
                pxPerMmObserved = pxPerMmOfficialClamped,
                pxPerMmFixed = null,
                filOverInnerMmPerSide3250 = filOverInnerMmPerSide,
                excludeDegFrom = 89.0,
                excludeDegTo = 91.0,
                stepDeg = 2.0,
                iters = 1,
                rSearchRelLo = 0.70,
                rSearchRelHi = 1.35,
                bottomAnchorYpxRoi = bottomAnchorOiYpxRoi
            )
        }

        // ============================================================
        // ✅ EDGE + ARC-FIT DEBUG: TODO EN ROI-LOCAL
        // ============================================================
        if (saveEdgeArcfitDebugToGallery3250) {
            detOd?.let { pack ->
                val r = pack.result
                fun xLocal(v: Float?): Int? = fToLocalIntOrNull3250(v, odL, pack.w)
                fun yLocal(v: Float?): Int? = fToLocalIntOrNull3250(v, odT, pack.h)

                val debugUri = EdgeArcfitDebugDump3250.saveEdgeWithArcfitMarksToGallery3250(
                    ctx = ctx,
                    eyeTag = "OD",
                    edgeGray = pack.edges,
                    w = pack.w,
                    h = pack.h,
                    innerLeftPx = xLocal(r.innerLeftXpx),
                    innerRightPx = xLocal(r.innerRightXpx),
                    outerNasalPx = null,
                    outerTemplePx = null,
                    topPx = yLocal(r.topYpx),
                    bottomPx = yLocal(r.bottomYpx),
                    probeYPx = yLocal(r.probeYpx),
                    placedPxRoi = fitArcOd?.placedPxRoi // 👈 puede ser null, igual guardamos
                )
                Log.d(
                    "EDGE_SAVE3250",
                    "OD debugUri=$debugUri placed=${fitArcOd?.placedPxRoi?.size ?: 0}"
                )
            }

            detOi?.let { pack ->
                val r = pack.result
                fun xLocal(v: Float?): Int? = fToLocalIntOrNull3250(v, oiL, pack.w)
                fun yLocal(v: Float?): Int? = fToLocalIntOrNull3250(v, oiT, pack.h)

                val debugUri = EdgeArcfitDebugDump3250.saveEdgeWithArcfitMarksToGallery3250(
                    ctx = ctx,
                    eyeTag = "OI",
                    edgeGray = pack.edges,
                    w = pack.w,
                    h = pack.h,
                    innerLeftPx = xLocal(r.innerLeftXpx),
                    innerRightPx = xLocal(r.innerRightXpx),
                    outerNasalPx = null,
                    outerTemplePx = null,
                    topPx = yLocal(r.topYpx),
                    bottomPx = yLocal(r.bottomYpx),
                    probeYPx = yLocal(r.probeYpx),
                    placedPxRoi = fitArcOi?.placedPxRoi
                )
                Log.d(
                    "EDGE_SAVE3250",
                    "OI debugUri=$debugUri placed=${fitArcOi?.placedPxRoi?.size ?: 0}"
                )
            }
        }

        // ✅ placed ROI-local -> GLOBAL usando ROI canónico (sumar odL/odT / oiL/oiT)
        val placedOdGlobal0 = run {
            val pts = fitArcOd?.placedPxRoi
            if (pts.isNullOrEmpty()) null
            else pts.map { p -> PointF(p.x + odL, p.y + odT) }
        }
        val placedOiGlobal0 = run {
            val pts = fitArcOi?.placedPxRoi
            if (pts.isNullOrEmpty()) null
            else pts.map { p -> PointF(p.x + oiL, p.y + oiT) }
        }

        val pupilYGlobalUsed = roiSrc.bridgeRowYGlobal3250

        stage?.onStage(R.string.dnp_stage_sampler)

        val odRef = placedOdGlobal0?.let { pts ->
            RimRenderSampler3250.refinePlacedBySampling3250(
                stillBmp = stillBmp,
                roiGlobal = roiOdGlobal,
                placedPtsGlobal = pts,
                pupilYGlobal = pupilYGlobalUsed,
                useBottomOnly = true
            )
        }
        val oiRef = placedOiGlobal0?.let { pts ->
            RimRenderSampler3250.refinePlacedBySampling3250(
                stillBmp = stillBmp,
                roiGlobal = roiOiGlobal,
                placedPtsGlobal = pts,
                pupilYGlobal = pupilYGlobalUsed,
                useBottomOnly = true
            )
        }

        val placedOdUsed = odRef?.best?.takeIf { it.score >= 0.25f }?.transformedPtsGlobal ?: placedOdGlobal0
        val placedOiUsed = oiRef?.best?.takeIf { it.score >= 0.25f }?.transformedPtsGlobal ?: placedOiGlobal0

        // dumps del detector
        detOd?.let { pack ->
            DebugDump3250.dumpEdgesOnly(ctx, pack, "OD")
            DebugDump3250.dumpRim(ctx, rimSrcBmp, pack, "OD")
        }
        detOi?.let { pack ->
            DebugDump3250.dumpEdgesOnly(ctx, pack, "OI")
            DebugDump3250.dumpRim(ctx, rimSrcBmp, pack, "OI")
        }

        stage?.onStage(R.string.dnp_stage_debug)

        val dbgBmp = buildFinalDebugBitmap3250(
            stillBmp = stillBmp,
            roiSrc = roiSrc,
            pm = pm,
            placedOdUsed = placedOdUsed,
            placedOiUsed = placedOiUsed
        )
        val dbgUri = saveDebugBitmapToCache3250(
            ctx = ctx,
            fileName = "DBG_FACE_${System.currentTimeMillis()}.png",
            bmp = dbgBmp,
            alsoCopyToGallery3250 = saveEdgeArcfitDebugToGallery3250, // tu switch
            galleryRelativePath3250 = "Pictures/PrecalDNP/DEBUG"
        )


        try { dbgBmp.recycle() } catch (_: Throwable) {}

        val dbgUriStr = dbgUri?.toString()
        Log.d(TAG, "DBG_URI_3250=$dbgUriStr")

        return FitPack3250(
            pxPerMmFaceD = pxPerMmOfficialClamped,
            placedOdUsed = placedOdUsed,
            placedOiUsed = placedOiUsed,
            rimOd = rimOd,
            rimOi = rimOi,
            dbgPath3250 = dbgUriStr
        )
    }

    private fun buildFinalDebugBitmap3250(
        stillBmp: Bitmap,
        roiSrc: RoiSrcPack3250,
        pm: PupilsMidPack3250,
        placedOdUsed: List<PointF>?,
        placedOiUsed: List<PointF>?
    ): Bitmap {
        val out = stillBmp.copy(Bitmap.Config.ARGB_8888, true)
        val c = android.graphics.Canvas(out)

        val pOd = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
            style = android.graphics.Paint.Style.STROKE
            strokeWidth = 6f
            color = 0xFFFF0000.toInt()
        }
        val pOi = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
            style = android.graphics.Paint.Style.STROKE
            strokeWidth = 6f
            color = 0xFF0000FF.toInt()
        }
        val pRoi = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
            style = android.graphics.Paint.Style.STROKE
            strokeWidth = 4f
            color = 0xFF00FF00.toInt()
        }
        val pMid = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
            style = android.graphics.Paint.Style.STROKE
            strokeWidth = 4f
            color = 0xFF00FF00.toInt()
        }
        val pPupil = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
            style = android.graphics.Paint.Style.FILL
            color = 0xFF00BCD4.toInt()
        }

        val a = pm.midline3250.a
        val b = pm.midline3250.b
        c.drawLine(a.x, a.y, b.x, b.y, pMid)

        c.drawCircle(pm.pupilOdDet.x, pm.pupilOdDet.y, 10f, pPupil)
        c.drawCircle(pm.pupilOiDet.x, pm.pupilOiDet.y, 10f, pPupil)

        c.drawRect(roiSrc.roiOdRectF, pRoi)
        c.drawRect(roiSrc.roiOiRectF, pRoi)

        fun drawPolyline(pts: List<PointF>?, paint: android.graphics.Paint) {
            if (pts.isNullOrEmpty()) return
            for (i in 1 until pts.size) {
                val aa = pts[i - 1]
                val bb = pts[i]
                c.drawLine(aa.x, aa.y, bb.x, bb.y, paint)
            }
            val first = pts.first()
            val last = pts.last()
            c.drawLine(last.x, last.y, first.x, first.y, paint)
        }

        drawPolyline(placedOdUsed, pOd)
        drawPolyline(placedOiUsed, pOi)

        return out
    }

    private fun saveDebugBitmapToCache3250(
        ctx: Context,
        fileName: String,
        bmp: Bitmap,
        alsoCopyToGallery3250: Boolean = false,
        galleryRelativePath3250: String = "Pictures/PrecalDNP/DEBUG"
    ): Uri? {

        val dbgUri: Uri? = try {
            val dir = File(ctx.cacheDir, "debug3250").apply { mkdirs() }
            val f = File(dir, fileName)
            FileOutputStream(f).use { out ->
                bmp.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
            FileProvider.getUriForFile(ctx, "${ctx.packageName}.fileprovider", f)
        } catch (t: Throwable) {
            Log.e(TAG, "saveDebugBitmapToCache3250 error", t)
            null
        }

        if (alsoCopyToGallery3250 && dbgUri != null) {
            val g = copyUriToGallery3250(
                ctx = ctx,
                srcUri = dbgUri,
                displayName = fileName,
                relativePath = galleryRelativePath3250
            )
            Log.d(TAG, "DBG_FACE copied to gallery: $g")
        }

        return dbgUri
    }

    private fun copyUriToGallery3250(
        ctx: Context,
        srcUri: Uri,
        displayName: String,
        relativePath: String
    )
    {
    }

    fun metricsToMeasurementResult3250(
        m: DnpMetrics3250,
        originalUriStr: String,
        dbgUriStr: String?
    ): MeasurementResult {

        fun d(x: Double) = if (x.isFinite()) x else Double.NaN

        return MeasurementResult(
            anchoMm = Double.NaN,
            altoMm = Double.NaN,
            diagMayorMm = Double.NaN,
            puenteMm = d(m.bridgeMm),
            dnpOdMm = d(m.dnpOdMm),
            dnpOiMm = d(m.dnpOiMm),
            altOdMm = d(m.heightOdMm),
            altOiMm = d(m.heightOiMm),
            diamUtilOdMm = d(m.diamUtilOdMm),
            diamUtilOiMm = d(m.diamUtilOiMm),
            annotatedPath = dbgUriStr ?: "",
            originalPath = originalUriStr
        )
    }
}
