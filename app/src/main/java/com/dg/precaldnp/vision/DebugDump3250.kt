package com.dg.precaldnp.vision

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.net.Uri
import android.os.SystemClock
import android.provider.MediaStore
import android.util.Log
import androidx.core.graphics.createBitmap
import java.io.OutputStream
import kotlin.math.max
import kotlin.math.min

object DebugPaths3250 {
    const val REL_PATH = "Pictures/PrecalDNP/DEBUG"
}

object DebugDump3250 {

    private const val TAG = "DebugDump3250"
    private const val REL_PATH = DebugPaths3250.REL_PATH

    private const val FIL_LABEL_DETECTOR = "FIL DETECTOR"
    private const val FIL_LABEL_ARCFIT = "FIL ARC FIT"

    private const val FIL_COLOR_DETECTOR = 0xFFFF4DFF.toInt() // magenta-violeta
    private const val FIL_COLOR_ARCFIT = 0xFFFF5252.toInt()   // rojo suave

    private val lastByKey = HashMap<String, Long>()

    private fun allow(key: String, minIntervalMs: Long): Boolean {
        val now = SystemClock.elapsedRealtime()
        val last = lastByKey[key] ?: 0L
        if (now - last < minIntervalMs) return false
        lastByKey[key] = now
        return true
    }

    /**
     * Dump actual/legacy: detector usado.
     * Mantengo compatibilidad.
     */
    fun dumpRimUsed3250(
        context: Context,
        usedPack: RimDetectPack3250?,
        debugTag: String,
        filPtsGlobal800: List<PointF>? = null,
        minIntervalMs: Long = 1200L
    ) {
        if (usedPack == null) return

        val key = "RIM_USED_$debugTag"
        if (!allow(key, minIntervalMs)) return

        val overlayBmp = edgeU8ToBitmap(usedPack.edges, usedPack.w, usedPack.h)
            .copy(Bitmap.Config.ARGB_8888, true)

        drawRimOverlayOnEdgeBitmap(
            edgeBmp = overlayBmp,
            res = usedPack.result,
            filPtsGlobal800 = filPtsGlobal800,
            filLabel = if (filPtsGlobal800 != null) "FIL" else null,
            filColor = if (filPtsGlobal800 != null) FIL_COLOR_DETECTOR else null
        )

        saveBitmapToPictures(
            context = context,
            bmp = overlayBmp,
            baseName = "RIM_${debugTag}_USED_OVERLAY"
        )
    }

    /**
     * NUEVO:
     * Detector usado + FIL construido con la verdad del detector/rimBase.
     */
    fun dumpRimDetectorUsed3250(
        context: Context,
        usedPack: RimDetectPack3250?,
        debugTag: String,
        filPtsDetectorGlobal800: List<PointF>?,
        minIntervalMs: Long = 1200L
    ) {
        if (usedPack == null || filPtsDetectorGlobal800.isNullOrEmpty()) return

        val key = "RIM_FILDET_$debugTag"
        if (!allow(key, minIntervalMs)) return

        val overlayBmp = edgeU8ToBitmap(usedPack.edges, usedPack.w, usedPack.h)
            .copy(Bitmap.Config.ARGB_8888, true)

        drawRimOverlayOnEdgeBitmap(
            edgeBmp = overlayBmp,
            res = usedPack.result,
            filPtsGlobal800 = filPtsDetectorGlobal800,
            filLabel = FIL_LABEL_DETECTOR,
            filColor = FIL_COLOR_DETECTOR
        )

        saveBitmapToPictures(
            context = context,
            bmp = overlayBmp,
            baseName = "RIM_${debugTag}_FILDET_OVERLAY"
        )
    }

    /**
     * NUEVO:
     * Detector usado + FIL final que vino de ArcFit.
     * Esta es la que vamos a llamar después de ArcFit.
     */
    fun dumpRimArcFitUsed3250(
        context: Context,
        usedPack: RimDetectPack3250?,
        debugTag: String,
        filPtsArcFitGlobal800: List<PointF>?,
        minIntervalMs: Long = 1200L
    ) {
        if (usedPack == null || filPtsArcFitGlobal800.isNullOrEmpty()) return

        val key = "RIM_ARCFIT_$debugTag"
        if (!allow(key, minIntervalMs)) return

        val overlayBmp = edgeU8ToBitmap(usedPack.edges, usedPack.w, usedPack.h)
            .copy(Bitmap.Config.ARGB_8888, true)

        drawRimOverlayOnEdgeBitmap(
            edgeBmp = overlayBmp,
            res = usedPack.result,
            filPtsGlobal800 = filPtsArcFitGlobal800,
            filLabel = FIL_LABEL_ARCFIT,
            filColor = FIL_COLOR_ARCFIT
        )

        saveBitmapToPictures(
            context = context,
            bmp = overlayBmp,
            baseName = "RIM_${debugTag}_ARCFIT_OVERLAY"
        )
    }

    /**
     * NUEVO:
     * Si en algún punto querés emitir ambas en una sola llamada.
     */
    fun dumpRimDetectorAndArcFit3250(
        context: Context,
        usedPack: RimDetectPack3250?,
        debugTag: String,
        filPtsDetectorGlobal800: List<PointF>?,
        filPtsArcFitGlobal800: List<PointF>?,
        minIntervalMs: Long = 1200L
    ) {
        dumpRimDetectorUsed3250(
            context = context,
            usedPack = usedPack,
            debugTag = debugTag,
            filPtsDetectorGlobal800 = filPtsDetectorGlobal800,
            minIntervalMs = minIntervalMs
        )

        dumpRimArcFitUsed3250(
            context = context,
            usedPack = usedPack,
            debugTag = debugTag,
            filPtsArcFitGlobal800 = filPtsArcFitGlobal800,
            minIntervalMs = minIntervalMs
        )
    }

    /**
     * Compatibilidad con tu llamada actual del pipeline.
     * Internamente sigue dibujando SOLO el USED legacy.
     */
    @Suppress("UNUSED_PARAMETER")
    fun dumpRimRawNormAndUsed3250(
        context: Context,
        rawPack: RimDetectPack3250?,
        normPack: RimDetectPack3250?,
        usedPack: RimDetectPack3250?,
        debugTag: String,
        filPtsGlobal800: List<PointF>? = null,
        minIntervalMs: Long = 1200L
    ) {
        dumpRimUsed3250(
            context = context,
            usedPack = usedPack,
            debugTag = debugTag,
            filPtsGlobal800 = filPtsGlobal800,
            minIntervalMs = minIntervalMs
        )
    }

    /**
     * Base = edge map ROI del pack usado.
     * En este modo de depuración dibuja SOLO:
     * - bottom final
     * - marcas verticales inner L/R
     * - arcos inner/outer si existen
     *
     * No dibuja FIL, probe, walls ni seed.
     */
    private fun drawRimOverlayOnEdgeBitmap(
        edgeBmp: Bitmap,
        res: RimDetectionResult,
        filPtsGlobal800: List<PointF>? = null,
        filLabel: String? = null,
        filColor: Int? = null
    ) {
        val c = Canvas(edgeBmp)

        val x0 = res.roiPx.left
        val y0 = res.roiPx.top

        fun toLocalX(xGlobal: Float): Float = xGlobal - x0

        // Mantengo params por compatibilidad de firma.
        // En este modo de depuración no se dibuja FIL.
        val _unusedFilPts = filPtsGlobal800
        val _unusedFilLabel = filLabel
        val _unusedFilColor = filColor

        val pBottom = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = max(2.0f, edgeBmp.width * 0.0030f)
            color = Color.argb(145, 0, 255, 80)   // verde suave
        }

        val pInnerV = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = max(1.3f, edgeBmp.width * 0.0021f)
            color = Color.argb(85, 0, 229, 255)   // cian muy suave
        }

        val pInnerArc = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = max(1.8f, edgeBmp.width * 0.0027f)
            color = Color.argb(150, 0, 229, 255)  // cian suave
        }

        val pOuterArc = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = max(1.5f, edgeBmp.width * 0.0023f)
            color = Color.argb(105, 255, 152, 0)  // naranja suave
        }

        val pWhite = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = max(13f, edgeBmp.width * 0.020f)
            style = Paint.Style.FILL
            setShadowLayer(5f, 0f, 0f, Color.BLACK)
        }

        // =========================
        // Detector puro
        // =========================

        drawPolylineGlobal(
            canvas = c,
            ptsGlobal = res.bottomPolylinePx,
            x0 = x0,
            y0 = y0,
            paint = pBottom
        )

        drawPolylineGlobal(
            canvas = c,
            ptsGlobal = res.nasalInnerPolylinePx,
            x0 = x0,
            y0 = y0,
            paint = pInnerArc
        )

        drawPolylineGlobal(
            canvas = c,
            ptsGlobal = res.templeInnerPolylinePx,
            x0 = x0,
            y0 = y0,
            paint = pInnerArc
        )

        drawPolylineGlobal(
            canvas = c,
            ptsGlobal = res.nasalOuterPolylinePx,
            x0 = x0,
            y0 = y0,
            paint = pOuterArc
        )

        drawPolylineGlobal(
            canvas = c,
            ptsGlobal = res.templeOuterPolylinePx,
            x0 = x0,
            y0 = y0,
            paint = pOuterArc
        )

        val innerL = toLocalX(res.innerLeftXpx)
        if (innerL.isFinite()) {
            c.drawLine(innerL, 0f, innerL, edgeBmp.height.toFloat(), pInnerV)
        }

        val innerR = toLocalX(res.innerRightXpx)
        if (innerR.isFinite()) {
            c.drawLine(innerR, 0f, innerR, edgeBmp.height.toFloat(), pInnerV)
        }

        val botCount = res.bottomPolylinePx?.size ?: 0
        val nIn = res.nasalInnerPolylinePx?.size ?: 0
        val tIn = res.templeInnerPolylinePx?.size ?: 0
        val nOut = res.nasalOuterPolylinePx?.size ?: 0
        val tOut = res.templeOuterPolylinePx?.size ?: 0

        val dbgTxt =
            "bot=$botCount nIn=$nIn tIn=$tIn nOut=$nOut tOut=$tOut conf=${"%.3f".format(res.confidence)}"

        c.drawText(
            dbgTxt,
            12f,
            edgeBmp.height - pWhite.textSize - 12f,
            pWhite
        )
    }

    private fun drawPolylineGlobal(
        canvas: Canvas,
        ptsGlobal: List<PointF>?,
        x0: Float,
        y0: Float,
        paint: Paint
    ) {
        if (ptsGlobal.isNullOrEmpty()) return

        val path = Path()
        val p0 = ptsGlobal.first()
        path.moveTo(p0.x - x0, p0.y - y0)

        for (i in 1 until ptsGlobal.size) {
            val p = ptsGlobal[i]
            path.lineTo(p.x - x0, p.y - y0)
        }

        canvas.drawPath(path, paint)
    }

    private fun edgeU8ToBitmap(edges: ByteArray, w: Int, h: Int): Bitmap {
        val bmp = createBitmap(w, h)
        val px = IntArray(w * h)
        val n = min(px.size, edges.size)

        for (i in 0 until n) {
            val v = edges[i].toInt() and 0xFF
            px[i] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
        }

        bmp.setPixels(px, 0, w, 0, 0, w, h)
        return bmp
    }

    private fun saveBitmapToPictures(
        context: Context,
        bmp: Bitmap,
        baseName: String
    ): Uri? {
        val name = "${baseName}_${System.currentTimeMillis()}.png"

        val resolver = context.contentResolver
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, name)
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            put(MediaStore.Images.Media.RELATIVE_PATH, REL_PATH)
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }

        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
        if (uri == null) {
            Log.w(TAG, "saveBitmapToPictures insert returned null")
            return null
        }

        var os: OutputStream? = null
        try {
            os = resolver.openOutputStream(uri)
            if (os == null) {
                Log.w(TAG, "openOutputStream null")
                return null
            }
            bmp.compress(Bitmap.CompressFormat.PNG, 100, os)
        } catch (t: Throwable) {
            Log.e(TAG, "saveBitmapToPictures error", t)
        } finally {
            try {
                os?.close()
            } catch (_: Throwable) {
            }
        }

        val done = ContentValues().apply {
            put(MediaStore.Images.Media.IS_PENDING, 0)
        }
        resolver.update(uri, done, null, null)

        Log.d(TAG, "Saved debug png: $name")
        return uri
    }
}