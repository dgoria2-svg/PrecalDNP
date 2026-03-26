package com.dg.precaldnp.ui

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.PointF
import android.graphics.RectF
import android.net.Uri
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import androidx.annotation.StringRes
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import com.dg.precaldnp.R
import com.dg.precaldnp.model.DnpShotHint3250
import com.dg.precaldnp.model.MeasurementResult
import com.dg.precaldnp.model.ShapeTraceResult
import com.dg.precaldnp.model.WorkEye3250
import com.dg.precaldnp.vision.ArcFitAdapter3250
import com.dg.precaldnp.vision.DebugDump3250
import com.dg.precaldnp.vision.DebugDumpFace3250
import com.dg.precaldnp.vision.EdgeMapBuilder3250
import com.dg.precaldnp.vision.EyeEllipseMask3250
import com.dg.precaldnp.vision.FilContourBuilder3250
import com.dg.precaldnp.vision.IrisDnpLandmarker3250
import com.dg.precaldnp.vision.PupilFrameEngine3250
import com.dg.precaldnp.vision.RimArcSeed3250
import com.dg.precaldnp.vision.RimDetectPack3250
import com.dg.precaldnp.vision.RimDetectionResult
import com.dg.precaldnp.vision.RimDetectorBlock3250
import com.dg.precaldnp.vision.RimProfile3250
import com.dg.precaldnp.vision.RimRenderSampler3250
import com.dg.precaldnp.vision.RoiAppearanceNormalizer3250
import com.dg.precaldnp.vision.effectiveFilOverPerSide3250
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


@Suppress("SameParameterValue")
object DnpFacePipeline3250 {

    private const val TAG = "DnpFacePipeline3250"

    fun interface StageCb3250 {
        fun onStage(@StringRes resId: Int)
    }

    data class MeasureOut3250(
        val originalUriStr: String,
        val dbgPath3250: String?,
        val metrics: DnpMetrics3250,
        val finalAnnotatedUriStr3250: String? = null   // ✅ nuevo (no rompe)
    )

    data class IrisPack3250(
        val ok: Boolean,
        val midlineXpx: Float?,
        val midlineA: PointF?,
        val midlineB: PointF?,
        val pLeft: PointF?,
        val pRight: PointF?,
        val browLeftBottomYpx: Float?,
        val browRightBottomYpx: Float?,
        val eyeEllipsesGlobal3250: List<EyeEllipseMask3250>? = null
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

    fun measureArcFitOnlyCore3250(
        ctx: Context,
        savedUri: Uri,
        st: ShapeTraceResult,
        sp: DnpShotHint3250,
        filHboxMm: Double,
        filVboxMm: Double,
        filOdMm: FilContourBuilder3250.ContourMm3250,
        filOiMm: FilContourBuilder3250.ContourMm3250,
        orderId3250: String,
        irisEngine: IrisDnpLandmarker3250?,
        pupilEngine: PupilFrameEngine3250,
        filOverInnerMmPerSide: Double,
        saveRingRoiDebug: Boolean = false,
        saveEdgeArcfitDebugToGallery3250: Boolean = false,
        stage: StageCb3250? = null
    ): MeasureOut3250 {

        val stillBmp = loadBitmapOriented(ctx, savedUri) ?: error("No se pudo leer la foto.")

        // 1) Landmarks primero
        val iris = estimateIrisPack3250(ctx, stillBmp, irisEngine)

        // 2) Boxes still
        val boxes = resolveBoxesStill3250(stillBmp, sp, filHboxMm, filVboxMm)

        // 3) topKill + máscara full simple
        val topKillY3250 = computeTopKillYFromIris3250(iris, stillBmp.height)
        val maskFull3250 = buildMaskFullEyesAndBrows3250(
            w = stillBmp.width,
            h = stillBmp.height,
            iris = iris
        )

        // 4) Pupilas + midline ANTES del edge map
        val pm = resolvePupilsAndMidline3250(
            stillBmp = stillBmp,
            iris = iris,
            boxes = boxes,
            pupilEngine = pupilEngine,
            isMirrored3250 = sp.isMirrored3250
        )

        // 5) ROI packs ANTES del edge map
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
// 6) FULL edge pack UNA sola vez
        val w = stillBmp.width
        val h = stillBmp.height
        val nFullL = w.toLong() * h.toLong()
        if (nFullL <= 0L || nFullL > Int.MAX_VALUE.toLong()) {
            error("EDGE3250: still demasiado grande w=$w h=$h (n=$nFullL)")
        }

        val edgePack = EdgeMapBuilder3250.buildFullFrameEdgePackFromBitmap3250(
            stillBmp = stillBmp,
            borderKillPx = 4,
            topKillY = topKillY3250,
            maskFull = maskFull3250,
            debugTag = "FACE",
            ctx = ctx,
            debugSaveToGallery3250 = saveEdgeArcfitDebugToGallery3250,
            debugAlsoSaveGray3250 = saveEdgeArcfitDebugToGallery3250
        )

        val edgeFullU8 = edgePack.edgesU8
        val dirFullU8 = edgePack.dirU8


        // 7) RimDetector + ArcFit consumen SOLO edgeFullU8 (+ dirFullU8)
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
            edgemapfullu83250 = edgeFullU8,
            dirfullu83250 = dirFullU8,
            maskfullu83250 = maskFull3250,
            saveEdgeArcfitDebugToGallery3250 = saveEdgeArcfitDebugToGallery3250,
            stage = stage
        )

        val metrics = DnpFaceMetrics3250.computeMetrics3250(
            stillBmp = stillBmp,
            pm = pm,
            fit = fit,
            pupilSrc = pm.pupilPick.src
        )

        try {
            roiSrc.rimSrcBmpToRecycle?.recycle()
        } catch (_: Throwable) {
        }

        val finalUriStr = try {
            DnpResultsRoiRenderer3250.renderAndSaveFinalRois3250(
                ctx = ctx,
                stillBmp = stillBmp,
                pm = pm,
                fit = fit,
                metrics = metrics,
                orderId3250 = orderId3250
            )
        } catch (_: Throwable) {
            null
        }

        val finalAnnotated = finalUriStr ?: fit.dbgPath3250 ?: savedUri.toString()

        return MeasureOut3250(
            originalUriStr = savedUri.toString(),
            dbgPath3250 = fit.dbgPath3250,
            metrics = metrics,
            finalAnnotatedUriStr3250 = finalAnnotated
        )
    }
    // ============================================================

    fun readTextFromUri(ctx: Context, uri: Uri): String? =
        try {
            ctx.contentResolver.openInputStream(uri)?.bufferedReader(Charsets.UTF_8)
                ?.use { it.readText() }
        } catch (_: Throwable) {
            null
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
            val bytes =
                ctx.contentResolver.openInputStream(uri)?.use { it.readBytes() } ?: return null

            val opts = BitmapFactory.Options().apply {
                inPreferredConfig = Bitmap.Config.ARGB_8888
                @Suppress("DEPRECATION")
                inDither = false
                inScaled = false
            }

            val decoded = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts) ?: return null
            val bmp0 = if (decoded.config != Bitmap.Config.ARGB_8888) {
                decoded.copy(Bitmap.Config.ARGB_8888, false).also {
                    try {
                        decoded.recycle()
                    } catch (_: Throwable) {
                    }
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
                    try {
                        bmp0.recycle()
                    } catch (_: Throwable) {
                    }
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
        val l = r.left.coerceIn(0f, w.toFloat())
        val t = r.top.coerceIn(0f, h.toFloat())
        val rr = r.right.coerceIn(0f, w.toFloat())
        val bb = r.bottom.coerceIn(0f, h.toFloat())

        // ordenar
        val left = min(l, rr)
        val top = min(t, bb)
        val right0 = max(l, rr)
        val bottom0 = max(t, bb)

        // asegurar tamaño mínimo (right/bottom EXCLUSIVOS)
        val right = right0.coerceIn(left + 1f, w.toFloat())
        val bottom = bottom0.coerceIn(top + 1f, h.toFloat())

        return RectF(left, top, right, bottom)
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
                browRightBottomYpx = irisRaw.rightBrowBottomYpx,
                eyeEllipsesGlobal3250 = irisRaw.eyeEllipsesGlobal3250
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
        val boxOiPreviewNN =
            sp.leftBoxPreview ?: sp.leftPupilRoiPreview ?: requireNotNull(base.leftBoxPreview)
        val boxOdPreviewNN =
            sp.rightBoxPreview ?: sp.rightPupilRoiPreview ?: requireNotNull(base.rightBoxPreview)

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
            val (od1, oi1) = swapPupilsIfCrossedByBoxes3250(
                pupilOdDet,
                pupilOiDet,
                boxes.boxOdF,
                boxes.boxOiF
            )
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
                Log.w(
                    TAG,
                    "MID3250: irisX=$irisX NOT between irises [${"%.1f".format(lo)}..${
                        "%.1f".format(hi)
                    }], using irisAvg=$avgIrisX"
                )
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
            else -> return 0
        }
        val pad = 18f
        return (yMin - pad).toInt().coerceIn(0, h.coerceAtLeast(1) - 1)
    }

    /**
     * Máscara simple (U8): marca zonas de ojos/cejas para que EdgeMapBuilder haga flattening (anti-halo)
     * y no compute edges adentro.
     *
     * Si no hay iris/brows => null (queda sin máscara).
     */
    private const val MASK_ON: Byte = 0xFF.toByte()

    private fun buildMaskFullEyesAndBrows3250(
        w: Int,
        h: Int,
        iris: IrisPack3250,
    ): ByteArray? {
        if (!iris.ok) return null
        val pL = iris.pLeft
        val pR = iris.pRight

        // Si no tengo al menos centros de iris, no invento.
        if (pL == null || pR == null) return null

        val mask = ByteArray(w * h)

        fun fillEllipse(cx: Float, cy: Float, rx: Float, ry: Float) {
            if (!cx.isFinite() || !cy.isFinite() || !rx.isFinite() || !ry.isFinite()) return
            if (rx <= 2f || ry <= 2f) return

            val x0 = (cx - rx).roundToInt().coerceIn(0, w - 1)
            val x1 = (cx + rx).roundToInt().coerceIn(0, w - 1)
            val y0 = (cy - ry).roundToInt().coerceIn(0, h - 1)
            val y1 = (cy + ry).roundToInt().coerceIn(0, h - 1)

            val invRx2 = 1f / (rx * rx)
            val invRy2 = 1f / (ry * ry)

            for (yy in y0..y1) {
                val dy = (yy - cy)
                val dy2 = dy * dy * invRy2
                val off = yy * w
                for (xx in x0..x1) {
                    val dx = (xx - cx)
                    val v = (dx * dx) * invRx2 + dy2
                    if (v <= 1f) mask[off + xx] = MASK_ON
                }
            }
        }

        fun dilate(radius: Int) {
            if (radius <= 0) return
            val src = mask.copyOf()
            for (y in 0 until h) {
                val y0 = max(0, y - radius)
                val y1 = min(h - 1, y + radius)
                for (x in 0 until w) {
                    val x0 = max(0, x - radius)
                    val x1 = min(w - 1, x + radius)
                    var hit = false
                    var yy = y0
                    while (yy <= y1 && !hit) {
                        val off = yy * w
                        var xx = x0
                        while (xx <= x1) {
                            if (src[off + xx].toInt() != 0) {
                                hit = true; break
                            }
                            xx++
                        }
                        yy++
                    }
                    if (hit) mask[y * w + x] = MASK_ON
                }
            }
        }

        // Tamaños ROBUSTOS basados en distancia entre iris (px)
        val d = abs(pR.x - pL.x).coerceIn(40f, (w * 0.90f))
        val rxBase = (0.25f * d).coerceIn(40f, w * 0.26f)     // ↑ un poco
        var ryBase = (0.19f * d).coerceIn(30f, h * 0.22f)     // ↑ un poco
        // anti “tapa media imagen”
        ryBase = min(ryBase, 0.88f * rxBase)

        fun addEyeMask(c: PointF) {
            val cx = c.x.coerceIn(0f, (w - 1).toFloat())
            val cy = c.y.coerceIn(0f, (h - 1).toFloat())

            // ---- elipse principal agrandada y corrida levemente hacia arriba (asimetría “soft”) ----
            val rx = (1.12f * rxBase).coerceIn(20f, w * 0.32f)
            val ry = (1.28f * ryBase).coerceIn(18f, h * 0.28f)
            val cy2 = (cy - 0.18f * ry).coerceIn(0f, (h - 1).toFloat())

            // ojo/párpado (SIN rectángulos)
            fillEllipse(cx, cy2, rx, ry)

            // “cap” superior elíptico (tapa pestañas/ceja suave, sin cuadrado)
            val rxCap = 0.95f * rx
            val ryCap = 0.55f * ry
            val cyCap = (cy2 - 0.70f * ry).coerceIn(0f, (h - 1).toFloat())
            fillEllipse(cx, cyCap, rxCap, ryCap)
        }

        addEyeMask(pL)
        addEyeMask(pR)

        // un poquito más grande (sin pasarte)
        dilate(radius = 5)

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

        val roiH = (expHpx * heightScale + 90f).coerceIn(180f, h * 0.90f)
        val top = (pupil.y - roiH * pupilYFracFromTop).coerceIn(0f, h - 2f)
        val bottom = (top + roiH).coerceIn(top + 2f, h)

        val overlap = gapPx.coerceIn(0f, 60f)   // (si querés, después renombrás param a overlapPx)

        val midTop = mid.xAt(top, bmpW)
        val midBot = mid.xAt(bottom, bmpW)

        val seamMin = min(midTop, midBot)
        val seamMax = max(midTop, midBot)

        val left: Float
        val right: Float

        if (isLeftSide) {
            // ✅ ojo izq.: el ROI TIENE que cruzar la midline hacia la derecha
            right = (seamMax + overlap).coerceIn(2f, w)
            left = (right - roiW).coerceIn(0f, right - 2f)
        } else {
            // ✅ ojo der: el ROI TIENE que cruzar la midline hacia la izquierda
            left = (seamMin - overlap).coerceIn(0f, w - 2f)
            right = (left + roiW).coerceIn(left + 2f, w)
        }

        val nasalMargin = if (isLeftSide) (midAtRef - right) else (left - midAtRef)

        Log.d(
            TAG,
            String.format(
                Locale.US,
                "ROI3250 yRef=%.1f midAtRef=%.1f seamMin=%.1f seamMax=%.1f overlap=%.1f nasalMargin=%.1f  L=%.1f R=%.1f T=%.1f B=%.1f roiW=%.1f roiH=%.1f",
                yRefGlobal, midAtRef, seamMin, seamMax, overlap, nasalMargin,
                left, right, top, bottom, (right - left), (bottom - top)
            )
        )
        return RectF(left, top, right, bottom)
    }

    private fun clampRectFToBmp3250(r: RectF, w: Int, h: Int): RectF {
        val ww = w.toFloat()
        val hh = h.toFloat()
        val l = r.left.coerceIn(0f, ww - 2f)
        val t = r.top.coerceIn(0f, hh - 2f)
        val rr = r.right.coerceIn(l + 2f, ww)   // ✅ permite rr==w
        val bb = r.bottom.coerceIn(t + 2f, hh)  // ✅ permite bb==h
        return RectF(l, t, rr, bb)
    }

    private fun rectFToCvRect3250(r: RectF, w: Int, h: Int): Rect {
        val roi = rectFToRectCoverPx3250(r, w, h)
        val ww = roi.width().coerceAtLeast(2)
        val hh = roi.height().coerceAtLeast(2)
        return Rect(roi.left, roi.top, ww, hh)
    }

    private fun rectFToRectCoverPx3250(r: RectF, w: Int, h: Int): android.graphics.Rect {
        val l = kotlin.math.floor(r.left).toInt().coerceIn(0, w)
        val t = kotlin.math.floor(r.top).toInt().coerceIn(0, h)
        val rr = kotlin.math.ceil(r.right).toInt().coerceIn(l, w)
        val bb = kotlin.math.ceil(r.bottom).toInt().coerceIn(t, h)
        return android.graphics.Rect(l, t, rr, bb) // right/bottom EXCLUSIVOS
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
        val roi = rectFToRectCoverPx3250(r, src.width, src.height)
        val w = roi.width()
        val h = roi.height()
        if (w <= 0 || h <= 0) return null
        return try {
            Bitmap.createBitmap(src, roi.left, roi.top, w, h)
        } catch (_: Throwable) {
            null
        }
    }

    private fun saveDebugBitmap3250(ctx: Context, bmp: Bitmap, namePrefix: String): String? {
        return try {
            val time = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())
            val dir =
                File(
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
        edgemapfullu83250: ByteArray,          // FULL edges crudo (0/255)
        dirfullu83250: ByteArray,              // FULL dirU8
        maskfullu83250: ByteArray? = null,     // FULL mask marker
        saveEdgeArcfitDebugToGallery3250: Boolean = false,
        stage: StageCb3250? = null
    ): FitPack3250 {

        // ============================================================
        // FULL dims
        // ============================================================
        val wFull = stillBmp.width
        val hFull = stillBmp.height
        val nFull = wFull * hFull

        if (edgemapfullu83250.size != nFull || dirfullu83250.size != nFull) {
            Log.e(
                TAG,
                "EDGE_FULL mismatch edges=${edgemapfullu83250.size} dir=${dirfullu83250.size} N=$nFull"
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

        val maskFullCanon: ByteArray? = maskfullu83250?.takeIf { it.size == nFull }

        // ============================================================
        // TODO: reemplazar por el profile real cuando venga desde UI / flujo
        // ============================================================
        val profileOd3250 = RimProfile3250.FULL_RIM
        val profileOi3250 = RimProfile3250.FULL_RIM

        // ============================================================
        // Midline overlap (asegurar cruce nasal)
        // ============================================================
        val midX = roiSrc.midXBridgeGlobal3250
        val pxmmGuess = roiSrc.pxPerMmGuessFace
        val overlapPx = ((pxmmGuess * 25f).takeIf { it.isFinite() && it > 0f } ?: 140f)
            .coerceIn(80f, 240f)

        val midL = (midX - overlapPx).roundToInt()
        val midR = (midX + overlapPx).roundToInt()

        // cruza midline
        val odL = roiSrc.roiOdRectF.left.roundToInt().coerceIn(0, wFull - 2)
        val odT = roiSrc.roiOdRectF.top.roundToInt().coerceIn(0, hFull - 2)
        val odR0 = roiSrc.roiOdRectF.right.roundToInt().coerceIn(odL + 1, wFull)
        val odB = roiSrc.roiOdRectF.bottom.roundToInt().coerceIn(odT + 1, hFull)

        val odR = max(odR0, midR).coerceIn(odL + 1, wFull)
        val roiOdGlobal0 = RectF(odL.toFloat(), odT.toFloat(), odR.toFloat(), odB.toFloat())

        val oiL0 = roiSrc.roiOiRectF.left.roundToInt().coerceIn(0, wFull - 2)
        val oiT = roiSrc.roiOiRectF.top.roundToInt().coerceIn(0, hFull - 2)
        val oiR = roiSrc.roiOiRectF.right.roundToInt().coerceIn(oiL0 + 1, wFull)
        val oiB = roiSrc.roiOiRectF.bottom.roundToInt().coerceIn(oiT + 1, hFull)

        val oiL = min(oiL0, midL).coerceIn(0, oiR - 1)
        val roiOiGlobal0 = RectF(oiL.toFloat(), oiT.toFloat(), oiR.toFloat(), oiB.toFloat())

        // ============================================================
        // Crop canónico (una sola verdad)
        // ============================================================
        val odEdges = cropRoiU8(edgemapfullu83250, wFull, hFull, roiOdGlobal0)
            ?: return FitPack3250(
                pxPerMmFaceD = roiSrc.pxPerMmGuessFace.toDouble(),
                placedOdUsed = null,
                placedOiUsed = null,
                rimOd = null,
                rimOi = null,
                dbgPath3250 = null
            )

        val roiOdGlobal = odEdges.roiGlobal
        val edgesOdU8 = odEdges.u8
        val wOd = odEdges.w
        val hOd = odEdges.h

        val dirOdU8 = cropRoiU8(dirfullu83250, wFull, hFull, roiOdGlobal)?.u8
            ?: return FitPack3250(
                pxPerMmFaceD = roiSrc.pxPerMmGuessFace.toDouble(),
                placedOdUsed = null,
                placedOiUsed = null,
                rimOd = null,
                rimOi = null,
                dbgPath3250 = null
            )

        val maskOdU8 = maskFullCanon?.let { cropRoiU8(it, wFull, hFull, roiOdGlobal)?.u8 }

        val oiEdges = cropRoiU8(edgemapfullu83250, wFull, hFull, roiOiGlobal0)
            ?: return FitPack3250(
                pxPerMmFaceD = roiSrc.pxPerMmGuessFace.toDouble(),
                placedOdUsed = null,
                placedOiUsed = null,
                rimOd = null,
                rimOi = null,
                dbgPath3250 = null
            )

        val roiOiGlobal = oiEdges.roiGlobal
        val edgesOiU8 = oiEdges.u8
        val wOi = oiEdges.w
        val hOi = oiEdges.h

        val dirOiU8 = cropRoiU8(dirfullu83250, wFull, hFull, roiOiGlobal)?.u8
            ?: return FitPack3250(
                pxPerMmFaceD = roiSrc.pxPerMmGuessFace.toDouble(),
                placedOdUsed = null,
                placedOiUsed = null,
                rimOd = null,
                rimOi = null,
                dbgPath3250 = null
            )

        val maskOiU8 = maskFullCanon?.let { cropRoiU8(it, wFull, hFull, roiOiGlobal)?.u8 }

        val roiOdCvCanon =
            Rect(roiOdGlobal.left.roundToInt(), roiOdGlobal.top.roundToInt(), wOd, hOd)
        val roiOiCvCanon =
            Rect(roiOiGlobal.left.roundToInt(), roiOiGlobal.top.roundToInt(), wOi, hOi)

        Log.d(
            TAG,
            "ROI3250 midLocal: OD=${midX - roiOdGlobal.left} / wOd=$wOd  OI=${midX - roiOiGlobal.left} / wOi=$wOi overlap=$overlapPx"
        )

        // ============================================================
        // Inner mm SOLO para seeds / fallback / lectura coherente
        // La verdad del detector la define detectRim() según profile3250
        // ============================================================
        val effectiveOverOd = effectiveFilOverPerSide3250(
            profile3250 = profileOd3250,
            filOverInnerMmPerSide = filOverInnerMmPerSide
        )
        val effectiveOverOi = effectiveFilOverPerSide3250(
            profile3250 = profileOi3250,
            filOverInnerMmPerSide = filOverInnerMmPerSide
        )

        val hboxInnerOdMm = (filHboxMm - 2.0 * effectiveOverOd).coerceAtLeast(10.0)
        val vboxInnerOdMm = (filVboxMm - 2.0 * effectiveOverOd).coerceAtLeast(10.0)
        val hboxInnerOiMm = (filHboxMm - 2.0 * effectiveOverOi).coerceAtLeast(10.0)
        val vboxInnerOiMm = (filVboxMm - 2.0 * effectiveOverOi).coerceAtLeast(10.0)

        stage?.onStage(R.string.dnp_stage_rim_detecting)

        // ============================================================
        // DEBUG VISUAL SIEMPRE: FULL edges + overlays + ROIs
        // ============================================================
        DebugDumpFace3250.dump(
            context = ctx,
            inp = DebugDumpFace3250.FaceOverlayInput3250(
                edgesFullU8 = edgemapfullu83250,
                wFull = wFull,
                hFull = hFull,
                maskfullu83250 = maskFullCanon,
                roiOdGlobal = roiOdGlobal,
                roiOiGlobal = roiOiGlobal,
                midlineXpx = roiSrc.midXBridgeGlobal3250,
                pupilOdGlobal = pm.pupilOdForRoi,
                pupilOiGlobal = pm.pupilOiForRoi,
                bridgeYpx = roiSrc.bridgeRowYGlobal3250,
                probeYpx = roiSrc.bridgeRowYGlobal3250,
                extraTag = "FACE"
            ),
            minIntervalMs = 250L
        )

        // ============================================================
        // RimDetector: edge map crudo + máscara + profile
        // ============================================================
        val detOd0: RimDetectPack3250? = RimDetectorBlock3250.detectRim(
            edgesU8 = edgesOdU8,
            dirU8 = dirOdU8,
            maskU8 = maskOdU8,
            w = wOd,
            h = hOd,
            roiGlobal = roiOdGlobal,
            midlineXpx = roiSrc.midXBridgeGlobal3250,
            browBottomYpx = roiSrc.browBottomOdY,
            filHboxMm = filHboxMm,
            filVboxMm = filVboxMm,
            filOverInnerMmPerSide3250 = filOverInnerMmPerSide,
            profile3250 = profileOd3250,
            bridgeRowYpxGlobal = roiSrc.bridgeRowYGlobal3250,
            pupilGlobal = pm.pupilOdForRoi,
            debugTag = "OD",
            pxPerMmGuessFace = roiSrc.pxPerMmGuessFace
        )

        val detOi0: RimDetectPack3250? = RimDetectorBlock3250.detectRim(
            edgesU8 = edgesOiU8,
            dirU8 = dirOiU8,
            maskU8 = maskOiU8,
            w = wOi,
            h = hOi,
            roiGlobal = roiOiGlobal,
            midlineXpx = roiSrc.midXBridgeGlobal3250,
            browBottomYpx = roiSrc.browBottomOiY,
            filHboxMm = filHboxMm,
            filVboxMm = filVboxMm,
            filOverInnerMmPerSide3250 = filOverInnerMmPerSide,
            profile3250 = profileOi3250,
            bridgeRowYpxGlobal = roiSrc.bridgeRowYGlobal3250,
            pupilGlobal = pm.pupilOiForRoi,
            debugTag = "OI",
            pxPerMmGuessFace = roiSrc.pxPerMmGuessFace
        )
        val pupilOdLocal3250 = toLocalPoint3250(pm.pupilOdForRoi, roiOdGlobal)
        val pupilOiLocal3250 = toLocalPoint3250(pm.pupilOiForRoi, roiOiGlobal)
        Log.d(TAG, "RETRY_CALL[OD] detOd0ok=${detOd0?.result?.ok == true}")

        val detOd1: RimDetectPack3250? =
            if (detOd0?.result?.ok == true) {
                detOd0
            } else {
                val r = retryRimWithAppearanceNorm3250(
                    stillBmp = stillBmp,
                    maskU8 = maskOdU8,
                    roiGlobal = roiOdGlobal,
                    midlineXpx = roiSrc.midXBridgeGlobal3250,
                    browBottomYpx = roiSrc.browBottomOdY,
                    filHboxMm = filHboxMm,
                    filVboxMm = filVboxMm,
                    filOverInnerMmPerSide3250 = filOverInnerMmPerSide,
                    profile3250 = profileOd3250,
                    bridgeRowYpxGlobal = roiSrc.bridgeRowYGlobal3250,
                    pupilGlobal = pm.pupilOdForRoi,
                    pupilLocal = pupilOdLocal3250,
                    debugTag = "OD",
                    pxPerMmGuessFace = roiSrc.pxPerMmGuessFace
                )
                Log.d(TAG, "RETRY_DONE[OD] hit=${r != null} ok=${r?.result?.ok == true}")
                r ?: detOd0
            }

        Log.d(TAG, "RETRY_CALL[OI] detOi0ok=${detOi0?.result?.ok == true}")

        val detOi1: RimDetectPack3250? =
            if (detOi0?.result?.ok == true) {
                detOi0
            } else {
                val r = retryRimWithAppearanceNorm3250(
                    stillBmp = stillBmp,
                    maskU8 = maskOiU8,
                    roiGlobal = roiOiGlobal,
                    midlineXpx = roiSrc.midXBridgeGlobal3250,
                    browBottomYpx = roiSrc.browBottomOiY,
                    filHboxMm = filHboxMm,
                    filVboxMm = filVboxMm,
                    filOverInnerMmPerSide3250 = filOverInnerMmPerSide,
                    profile3250 = profileOi3250,
                    bridgeRowYpxGlobal = roiSrc.bridgeRowYGlobal3250,
                    pupilGlobal = pm.pupilOiForRoi,
                    pupilLocal = pupilOiLocal3250,
                    debugTag = "OI",
                    pxPerMmGuessFace = roiSrc.pxPerMmGuessFace
                )
                Log.d(TAG, "RETRY_DONE[OI] hit=${r != null} ok=${r?.result?.ok == true}")
                r ?: detOi0
            }
        val odOk = detOd1?.result?.ok == true
        val oiOk = detOi1?.result?.ok == true

        val detOdFinal: RimDetectPack3250? = when {
            odOk -> detOd1
            oiOk -> RimDetectorBlock3250.mirrorFromOtherEye3250(
                primary = detOi1.result,
                targetEdgesU8 = edgesOdU8,
                w = wOd,
                h = hOd,
                targetRoiGlobal = roiOdGlobal,
                midlineXpx = roiSrc.midXBridgeGlobal3250,
                debugTag = "OD",
                srcTag = "OI",
                confScale = 0.60f
            )

            else -> null
        }

        val detOiFinal: RimDetectPack3250? = when {
            oiOk -> detOi1
            odOk -> RimDetectorBlock3250.mirrorFromOtherEye3250(
                primary = detOd1.result,
                targetEdgesU8 = edgesOiU8,
                w = wOi,
                h = hOi,
                targetRoiGlobal = roiOiGlobal,
                midlineXpx = roiSrc.midXBridgeGlobal3250,
                debugTag = "OI",
                srcTag = "OD",
                confScale = 0.60f
            )

            else -> null
        }

        val rimOd: RimDetectionResult? = detOdFinal?.result
        val rimOi: RimDetectionResult? = detOiFinal?.result

        DebugDump3250.dumpRimDetectorUsed3250(
            context = ctx,
            usedPack = detOdFinal,
            debugTag = "OD",
            filPtsDetectorGlobal800 = null
        )

        DebugDump3250.dumpRimDetectorUsed3250(
            context = ctx,
            usedPack = detOiFinal,
            debugTag = "OI",
            filPtsDetectorGlobal800 = null
        )
        fun pxFrom(
            r: RimDetectionResult?,
            profile3250: RimProfile3250
        ): Double? {
            if (r?.ok != true) return null
            val wPx = r.innerWidthPx
            if (!wPx.isFinite() || wPx <= 1f) return null

            val effectiveOver = effectiveFilOverPerSide3250(
                profile3250 = profile3250,
                filOverInnerMmPerSide = filOverInnerMmPerSide
            )
            val hboxInnerMm = (filHboxMm - 2.0 * effectiveOver).coerceAtLeast(10.0)

            val pxmm = wPx.toDouble() / hboxInnerMm
            return pxmm.takeIf { it.isFinite() && it > 1e-6 }
        }

        val odPx = pxFrom(rimOd, profileOd3250)
        val oiPx = pxFrom(rimOi, profileOi3250)

        val pxPerMmOfficialRaw: Double = when {
            odPx != null && oiPx != null -> {
                val avg = (odPx + oiPx) * 0.5
                val relDiff = abs(odPx - oiPx) / avg.coerceAtLeast(1e-9)
                if (relDiff <= 0.18) {
                    avg
                } else {
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

        val pxPerMmOfficialClamped = pxPerMmOfficialRaw.coerceIn(0.5, 30.0)
        Log.d(
            TAG,
            "PX/MM used[FACE]=$pxPerMmOfficialClamped (od=$odPx oi=$oiPx guess=${roiSrc.pxPerMmGuessFace})"
        )
// ============================================================
// Seeds + ArcFit + Sampler + Debug + Return
// ============================================================
        val seedOd = if (rimOd?.ok == true) {
            RimArcSeed3250.seedFromRimResult3250(
                roiCv = roiOdCvCanon,
                rim = rimOd,
                filHboxMm = hboxInnerOdMm,
                filVboxMm = vboxInnerOdMm,
                pxPerMmGuess = pxPerMmOfficialClamped.toFloat(),
                midlineXpxGlobal = roiSrc.midXBridgeGlobal3250,
                bridgeRowYpxGlobal = roiSrc.bridgeRowYGlobal3250,
                eyeSideSign3250 = -1
            )
        } else {
            RimArcSeed3250.seedFromPupilOrRoi3250(
                roiCv = roiOdCvCanon,
                filHboxMm = hboxInnerOdMm,
                filVboxMm = vboxInnerOdMm,
                pxPerMmGuess = pxPerMmOfficialClamped.toFloat(),
                midlineXpxGlobal = roiSrc.midXBridgeGlobal3250,
                pupilGlobal = pm.pupilOdForRoi,
                usePupil3250 = false,
                bridgeRowYpxGlobal = roiSrc.bridgeRowYGlobal3250
            )
        }

        val seedOi = if (rimOi?.ok == true) {
            RimArcSeed3250.seedFromRimResult3250(
                roiCv = roiOiCvCanon,
                rim = rimOi,
                filHboxMm = hboxInnerOiMm,
                filVboxMm = vboxInnerOiMm,
                pxPerMmGuess = pxPerMmOfficialClamped.toFloat(),
                midlineXpxGlobal = roiSrc.midXBridgeGlobal3250,
                bridgeRowYpxGlobal = roiSrc.bridgeRowYGlobal3250,
                eyeSideSign3250 = +1
            )
        } else {
            RimArcSeed3250.seedFromPupilOrRoi3250(
                roiCv = roiOiCvCanon,
                filHboxMm = hboxInnerOiMm,
                filVboxMm = vboxInnerOiMm,
                pxPerMmGuess = pxPerMmOfficialClamped.toFloat(),
                midlineXpxGlobal = roiSrc.midXBridgeGlobal3250,
                pupilGlobal = pm.pupilOiForRoi,
                usePupil3250 = false,
                bridgeRowYpxGlobal = roiSrc.bridgeRowYGlobal3250
            )
        }

        stage?.onStage(R.string.dnp_stage_arc_fit)

        val bottomAnchorOdYpxRoi = bottomAnchorFromRim3250(
            rim = rimOd,
            pack = detOdFinal,
            seed = seedOd,
            vboxInnerMm = vboxInnerOdMm,
            pxPerMmOfficial = pxPerMmOfficialClamped
        )

        val bottomAnchorOiYpxRoi = bottomAnchorFromRim3250(
            rim = rimOi,
            pack = detOiFinal,
            seed = seedOi,
            vboxInnerMm = vboxInnerOiMm,
            pxPerMmOfficial = pxPerMmOfficialClamped
        )

        val fitArcOd = detOdFinal?.let { pack ->
            ArcFitAdapter3250.arcFitFromRimPack3250(
                filPtsMm = filOdMm.ptsMm,
                filHboxMm = filHboxMm,
                filVboxMm = filVboxMm,
                filOverInnerMmPerSide3250 = filOverInnerMmPerSide,
                maskU8Roi = maskOdU8,
                pack = pack,
                seed = seedOd,
                eyeSideSign3250 = -1,
                bridgeRowYpxGlobal = roiSrc.bridgeRowYGlobal3250,
                profile3250 = profileOd3250,
                pxPerMmObservedOverride = null,
                bottomAnchorYpxRoiOverride = bottomAnchorOdYpxRoi,
                iters = 1
            )
        }

        val fitArcOi = detOiFinal?.let { pack ->
            ArcFitAdapter3250.arcFitFromRimPack3250(
                filPtsMm = filOiMm.ptsMm,
                filHboxMm = filHboxMm,
                filVboxMm = filVboxMm,
                filOverInnerMmPerSide3250 = filOverInnerMmPerSide,
                maskU8Roi = maskOiU8,
                pack = pack,
                seed = seedOi,
                eyeSideSign3250 = +1,
                bridgeRowYpxGlobal = roiSrc.bridgeRowYGlobal3250,
                profile3250 = profileOi3250,
                pxPerMmObservedOverride = null,
                bottomAnchorYpxRoiOverride = bottomAnchorOiYpxRoi,
                iters = 1
            )
        }

// ============================================================
// ArcFit px/mm: SOLO DEBUG por ahora
// NO usamos esto para decidir la escala final.
// ============================================================
        val pxArcOd = fitArcOd?.let {
            ArcFitAdapter3250.pxMmFromFit3250(
                fit = it,
                filHboxMm = filHboxMm,
                filVboxMm = filVboxMm,
                filOverInnerMmPerSide3250 = filOverInnerMmPerSide,
                profile3250 = profileOd3250,
                tolRelXY = 0.10
            )
        }

        val pxArcOi = fitArcOi?.let {
            ArcFitAdapter3250.pxMmFromFit3250(
                fit = it,
                filHboxMm = filHboxMm,
                filVboxMm = filVboxMm,
                filOverInnerMmPerSide3250 = filOverInnerMmPerSide,
                profile3250 = profileOi3250,
                tolRelXY = 0.10
            )
        }

        val placedOdGlobal0 = fitArcOd
            ?.takeIf { it.placedPxRoi.isNotEmpty() }
            ?.let { fit ->
                rimOd?.roiPx?.let { roiPx ->
                    roiLocalToGlobal3250(fit.placedPxRoi, roiPx)
                }
            }

        val placedOiGlobal0 = fitArcOi
            ?.takeIf { it.placedPxRoi.isNotEmpty() }
            ?.let { fit ->
                rimOi?.roiPx?.let { roiPx ->
                    roiLocalToGlobal3250(fit.placedPxRoi, roiPx)
                }
            }

        val pupilYGlobalUsed = roiSrc.bridgeRowYGlobal3250

        stage?.onStage(R.string.dnp_stage_sampler)

// Por ahora lo dejamos corriendo solo como diagnóstico.
// No estamos usando el resultado refinado todavía.
        val odRef = placedOdGlobal0?.let { placed ->
            rimOd?.roiPx?.let { roiPx ->
                RimRenderSampler3250.refinePlacedBySampling3250(
                    stillBmp = stillBmp,
                    roiGlobal = roiPx,
                    placedPtsGlobal = placed,
                    pupilYGlobal = pupilYGlobalUsed,
                    useBottomOnly = true
                )
            }
        }

        val oiRef = placedOiGlobal0?.let { placed ->
            rimOi?.roiPx?.let { roiPx ->
                RimRenderSampler3250.refinePlacedBySampling3250(
                    stillBmp = stillBmp,
                    roiGlobal = roiPx,
                    placedPtsGlobal = placed,
                    pupilYGlobal = pupilYGlobalUsed,
                    useBottomOnly = true
                )
            }
        }

// ArcFit usado visualmente = salida directa del arc-fit.
// El sampler todavía NO manda.
        val placedOdUsed = placedOdGlobal0

        DebugDump3250.dumpRimArcFitUsed3250(
            context = ctx,
            usedPack = detOdFinal,
            debugTag = "OD",
            filPtsArcFitGlobal800 = placedOdUsed
        )

        DebugDump3250.dumpRimArcFitUsed3250(
            context = ctx,
            usedPack = detOiFinal,
            debugTag = "OI",
            filPtsArcFitGlobal800 = placedOiGlobal0
        )

// ============================================================
// Escala final: por ahora queda LOCKED al detector / rim base.
// ArcFit solo informa/debuggea.
// ============================================================
        val arcOdOfficial = pxArcOd?.pxPerMmOfficial?.takeIf { it.isFinite() && it > 1e-6 }
        val arcOiOfficial = pxArcOi?.pxPerMmOfficial?.takeIf { it.isFinite() && it > 1e-6 }

        val pxPerMmFinalSrc3250 = "RIM_BASE_LOCKED"

        Log.d(
            TAG,
            "PXMM3250 FINAL rimBase=$pxPerMmOfficialClamped " +
                    "arcOd=$arcOdOfficial arcOi=$arcOiOfficial " +
                    "used=$pxPerMmOfficialClamped src=$pxPerMmFinalSrc3250"
        )

        stage?.onStage(R.string.dnp_stage_debug)

        val dbgBmp = buildFinalDebugBitmap3250(
            stillBmp = stillBmp,
            roiSrc = roiSrc,
            pm = pm,
            placedOdUsed = placedOdUsed,
            placedOiUsed = placedOiGlobal0
        )

        val dbgUri = saveDebugBitmapToCache3250(
            ctx = ctx,
            fileName = "DBG_FACE_${System.currentTimeMillis()}.png",
            bmp = dbgBmp,
            alsoCopyToGallery3250 = saveEdgeArcfitDebugToGallery3250
        )

        try {
            dbgBmp.recycle()
        } catch (_: Throwable) {
        }

        val dbgUriStr = dbgUri?.toString()
        Log.d(TAG, "DBG_URI_3250=$dbgUriStr")

        return FitPack3250(
            pxPerMmFaceD = pxPerMmOfficialClamped,
            placedOdUsed = placedOdUsed,
            placedOiUsed = placedOiGlobal0,
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
        alsoCopyToGallery3250: Boolean = false
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
                displayName = fileName
            )
            Log.d(TAG, "DBG_FACE copied to gallery: $g")
        }

        return dbgUri
    }

    private fun copyUriToGallery3250(
        ctx: Context,
        srcUri: Uri,
        displayName: String
    ): Uri? {
        return try {
            val resolver = ctx.contentResolver
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, displayName)
                put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/PrecalDNP")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }

            val dstUri =
                resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return null

            resolver.openInputStream(srcUri).use { ins ->
                resolver.openOutputStream(dstUri).use { outs ->
                    if (ins == null || outs == null) return null
                    ins.copyTo(outs)
                }
            }

            val done = ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }
            resolver.update(dstUri, done, null, null)

            dstUri
        } catch (t: Throwable) {
            Log.e(TAG, "copyUriToGallery3250 error", t)
            null
        }
    }

    fun metricsToMeasurementResult3250(
        m: DnpMetrics3250,
        originalUriStr: String,
        dbgUriStr: String? = null,

        // imagen FINAL (ROI + overlays). Si no la pasás, cae a dbg.
        finalAnnotatedUriStr: String? = null,

        // medidas legacy / fallback
        anchoMm: Double = Double.NaN,
        altoMm: Double = Double.NaN,
        diagMayorMm: Double = Double.NaN,

        // métricas del FIL
        filHboxMm: Double = Double.NaN,
        filVboxMm: Double = Double.NaN,
        filDblMm: Double = Double.NaN,
        filEyeSizeMm: Double = Double.NaN, // LEGACY: usar FED acá
        filCircMm: Double = Double.NaN,

        // stats arcfit (opcional)
        arcFitRotDeg: Double = Double.NaN,
        arcFitRmsPx: Double = Double.NaN,
        arcFitUsedSamples: Int = 0
    ): MeasurementResult {

        fun dd(x: Double): Double = if (x.isFinite()) x else Double.NaN

        fun prefer(primary: Double, fallback: Double): Double =
            if (primary.isFinite()) primary else fallback

        val annotated = finalAnnotatedUriStr?.takeIf { it.isNotBlank() }
            ?: dbgUriStr?.takeIf { it.isNotBlank() }
            ?: ""

        // VERDAD semántica: en este flujo DIAG = FED
        val filFedMm = dd(filEyeSizeMm)

        // VERDAD para Results: primero FIL, luego fallback legacy
        val outHboxMm = dd(prefer(filHboxMm, anchoMm))
        val outVboxMm = dd(prefer(filVboxMm, altoMm))
        val outFedMm = dd(prefer(filFedMm, diagMayorMm))

        return MeasurementResult(
            annotatedPath = annotated,
            originalPath = originalUriStr,

            // A / B / Diag que verá ResultsActivity
            anchoMm = outHboxMm,
            altoMm = outVboxMm,
            diagMayorMm = outFedMm,

            // Ø útil
            diamUtilOdMm = dd(m.diamUtilOdMm),
            diamUtilOiMm = dd(m.diamUtilOiMm),

            // DNP
            dnpOdMm = dd(m.dnpOdMm),
            dnpOiMm = dd(m.dnpOiMm),

            // alturas
            altOdMm = dd(m.heightOdMm),
            altOiMm = dd(m.heightOiMm),

            // puente: SOLO desde foto
            puenteMm = dd(m.bridgeMm),

            // 3250 extras
            pxPerMmFace3250 = dd(m.pxPerMmFaceD),

            filHboxMm3250 = dd(filHboxMm),
            filVboxMm3250 = dd(filVboxMm),
            filDblMm3250 = dd(filDblMm),

            // sigue con nombre legacy, pero contenido = FED
            filEyeSizeMm3250 = filFedMm,

            filCircMm3250 = dd(filCircMm),

            arcFitRotDeg3250 = dd(arcFitRotDeg),
            arcFitRmsPx3250 = dd(arcFitRmsPx),
            arcFitUsedSamples3250 = arcFitUsedSamples
        )
    }

    private fun roiLocalToGlobal3250(ptsRoi: List<PointF>, roiPx: RectF): List<PointF> {
        val l = roiPx.left
        val t = roiPx.top
        return ptsRoi.map { p -> PointF(p.x + l, p.y + t) }
    }

    private fun bottomAnchorFromRim3250(
        rim: RimDetectionResult?,
        pack: RimDetectPack3250?,
        seed: RimArcSeed3250.ArcSeed3250,
        vboxInnerMm: Double,
        pxPerMmOfficial: Double
    ): Float {
        val h = (pack?.h ?: 0).coerceAtLeast(2)
        val fallback = seed.originPxRoi.y + 0.40f * (vboxInnerMm * pxPerMmOfficial).toFloat()

        val yLocal = if (rim?.ok == true && rim.bottomYpx.isFinite()) {
            // ✅ GLOBAL -> ROI-local usando el roiPx del RESULT (no odT/oiT)
            (rim.bottomYpx - rim.roiPx.top)
        } else fallback

        return yLocal.coerceIn(0f, (h - 1).toFloat())
    }

    private class RoiU8(
        val u8: ByteArray,
        val w: Int,
        val h: Int,
        val roiGlobal: RectF
    )

    private fun cropRoiU8(full: ByteArray, wFull: Int, hFull: Int, roiG: RectF): RoiU8? {
        val l = roiG.left.roundToInt().coerceIn(0, wFull - 1)
        val t = roiG.top.roundToInt().coerceIn(0, hFull - 1)
        val r = roiG.right.roundToInt().coerceIn(0, wFull)
        val b = roiG.bottom.roundToInt().coerceIn(0, hFull)

        if (r <= l || b <= t) return null

        val w = (r - l).coerceAtLeast(1)
        val h = (b - t).coerceAtLeast(1)
        val out = ByteArray(w * h)

        var dst = 0
        for (y in 0 until h) {
            val srcRow = (t + y) * wFull + l
            full.copyInto(out, dst, srcRow, srcRow + w)
            dst += w
        }

        return RoiU8(
            u8 = out,
            w = w,
            h = h,
            roiGlobal = RectF(l.toFloat(), t.toFloat(), r.toFloat(), b.toFloat())
        )
    }

    private fun toLocalPoint3250(
        p: PointF?,
        roi: RectF
    ): PointF? {
        if (p == null) return null
        return PointF(
            p.x - roi.left,
            p.y - roi.top
        )
    }

    private fun cropBitmapFromRectF3250(
        src: Bitmap,
        roi: RectF
    ): Bitmap {
        val l = roi.left.toInt().coerceIn(0, src.width - 1)
        val t = roi.top.toInt().coerceIn(0, src.height - 1)
        val r = roi.right.toInt().coerceIn(l + 1, src.width)
        val b = roi.bottom.toInt().coerceIn(t + 1, src.height)
        return Bitmap.createBitmap(src, l, t, r - l, b - t)
    }

    private fun retryRimWithAppearanceNorm3250(
        stillBmp: Bitmap,
        maskU8: ByteArray?,
        roiGlobal: RectF,
        midlineXpx: Float,
        browBottomYpx: Float?,
        filHboxMm: Double,
        filVboxMm: Double,
        filOverInnerMmPerSide3250: Double,
        profile3250: RimProfile3250,
        bridgeRowYpxGlobal: Float,
        pupilGlobal: PointF?,
        pupilLocal: PointF?,
        debugTag: String,
        pxPerMmGuessFace: Float
    ): RimDetectPack3250? {

        val roiBmp = cropBitmapFromRectF3250(stillBmp, roiGlobal)
        val wRoi = roiBmp.width
        val hRoi = roiBmp.height
        if (wRoi <= 8 || hRoi <= 8) return null

        val filHboxInnerMm =
            (filHboxMm - 2.0 * filOverInnerMmPerSide3250).coerceAtLeast(1.0)
        val filVboxInnerMm =
            (filVboxMm - 2.0 * filOverInnerMmPerSide3250).coerceAtLeast(1.0)

        val pxGuess = pxPerMmGuessFace.toDouble().coerceAtLeast(0.5)
        val expWInnerPx = (filHboxInnerMm * pxGuess).toFloat().coerceAtLeast(40f)
        val expHInnerPx = (filVboxInnerMm * pxGuess).toFloat().coerceAtLeast(32f)

        val seamLocalX = (midlineXpx - roiGlobal.left).coerceIn(0f, (wRoi - 1).toFloat())

        val sideSign = when {
            pupilGlobal != null && pupilGlobal.x < midlineXpx -> -1f
            pupilGlobal != null && pupilGlobal.x > midlineXpx -> +1f
            roiGlobal.centerX() < midlineXpx -> -1f
            else -> +1f
        }

        val gapPx = (0.10f * expWInnerPx).coerceAtLeast(8f)

        val expectedCxLocal = (
                seamLocalX + sideSign * (gapPx + 0.5f * expWInnerPx)
                ).coerceIn(0f, (wRoi - 1).toFloat())

        val yRefLocal = (
                pupilLocal?.y ?: (bridgeRowYpxGlobal - roiGlobal.top)
                ).coerceIn(0f, (hRoi - 1).toFloat())

        val expectedCyLocal = (
                yRefLocal + 0.25f * expHInnerPx
                ).coerceIn(0f, (hRoi - 1).toFloat())

        val prep = RoiAppearanceNormalizer3250.normalizeForRim3250(
            roiBmp = roiBmp,
            filHboxInnerMm = filHboxInnerMm,
            filVboxInnerMm = filVboxInnerMm,
            pxPerMmGuess = pxGuess,
            expectedCxLocal = expectedCxLocal,
            expectedCyLocal = expectedCyLocal,
            eyeMaskU8 = maskU8,
            profileId3250 = RoiAppearanceNormalizer3250.FrameProfileId3250.CLEAR
        )

        Log.d(
            TAG,
            "ROI_NORM[$debugTag] " +
                    "profile=${prep.stats.profileId3250} " +
                    "sampleN=${prep.stats.sampleN} " +
                    "rgb=(${prep.stats.meanR},${prep.stats.meanG},${prep.stats.meanB}) " +
                    "gray=${prep.stats.meanGray} " +
                    "p10=${prep.stats.p10Lower} p90=${prep.stats.p90Lower} p98=${prep.stats.p98Lower} " +
                    "hiFrac=${"%.3f".format(prep.stats.hiFracLower)} " +
                    "cov=${"%.3f".format(prep.stats.normCoverage)} " +
                    "annCov=${"%.3f".format(prep.stats.annulusCoverage3250)} " +
                    "pres=${"%.3f".format(prep.stats.meanPreserve3250)} " +
                    "coh=${"%.3f".format(prep.stats.meanCoherence3250)} " +
                    "grad=${"%.3f".format(prep.stats.meanGrad3250)} " +
                    "tex=${"%.3f".format(prep.stats.textureSuppression3250)} " +
                    "target=${prep.stats.targetLuma} blurR=${prep.stats.blurRadius}"
        )

        val borderKillPxLocal = (
                min(prep.bitmap.width, prep.bitmap.height) * 0.01f
                ).toInt().coerceIn(0, 4)

        val edgePack = EdgeMapBuilder3250.buildFullFrameEdgePackFromBitmap3250(
            stillBmp = prep.bitmap,
            borderKillPx = borderKillPxLocal,
            topKillY = 0,
            maskFull = maskU8,
            debugTag = "${debugTag}_NORM"
        )

        return RimDetectorBlock3250.detectRim(
            edgesU8 = edgePack.edgesU8,
            dirU8 = edgePack.dirU8,
            maskU8 = maskU8,
            w = edgePack.w,
            h = edgePack.h,
            roiGlobal = roiGlobal,
            midlineXpx = midlineXpx,
            browBottomYpx = browBottomYpx,
            filHboxMm = filHboxMm,
            filVboxMm = filVboxMm,
            filOverInnerMmPerSide3250 = filOverInnerMmPerSide3250,
            profile3250 = profile3250,
            bridgeRowYpxGlobal = bridgeRowYpxGlobal,
            pupilGlobal = pupilGlobal,
            debugTag = "${debugTag}_NORM",
            pxPerMmGuessFace = pxPerMmGuessFace
        )
    }
}
