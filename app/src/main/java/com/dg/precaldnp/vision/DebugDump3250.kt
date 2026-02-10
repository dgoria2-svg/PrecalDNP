package com.dg.precaldnp.vision

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.graphics.RectF
import android.net.Uri
import android.os.Build
import android.os.SystemClock
import android.provider.MediaStore
import android.util.Log
import java.io.OutputStream
import kotlin.math.max
import kotlin.math.min
object DebugPaths3250 {
    const val REL_PATH = "Pictures/PrecalDNP/DEBUG" // UNA sola carpeta
}
object DebugDump3250 {

    private const val TAG = "DebugDump3250"
    private const val REL_PATH = DebugPaths3250.REL_PATH

    // throttle por "canal" (OD_EDGES, OI_EDGES, OD_ROI, etc.)
    private val lastByKey = HashMap<String, Long>()

    private fun allow(key: String, minIntervalMs: Long): Boolean {
        val now = SystemClock.elapsedRealtime()
        val last = lastByKey[key] ?: 0L
        if (now - last < minIntervalMs) return false
        lastByKey[key] = now
        return true
    }

    /**
     * Guarda:
     *  - RIM_{tag}_ROI      : recorte del bitmap original según roiPx
     *  - RIM_{tag}_EDGES    : bitmap del edgeMap (U8 0..255)
     *  - RIM_{tag}_OVERLAY  : ROI con líneas/polylines del RimDetectionResult
     *
     * Opcional:
     *  - filPtsGlobal800: si lo pasás (size>=800), marca R1/R401/R601 sobre el ROI.
     */
    fun dumpRim(
        context: Context,
        srcBmp: Bitmap,
        pack: RimDetectPack3250,
        debugTag: String,
        filPtsGlobal800: List<PointF>? = null,
        minIntervalMs: Long = 1200L
    ) {
        val key = "RIM_${debugTag}"
        if (!allow(key, minIntervalMs)) return

        val res = pack.result
        val roi = res.roiPx

        val roiBmp = cropBitmapSafe(srcBmp, roi) ?: run {
            Log.w(TAG, "dumpRim[$debugTag] cropBitmapSafe=null")
            return
        }

        val edgesBmp = edgeU8ToBitmap(pack.edges, pack.w, pack.h)

        val overlayBmp = roiBmp.copy(Bitmap.Config.ARGB_8888, true)
        drawOverlayOnRoi(
            roiBmp = overlayBmp,
            res = res,
            filPtsGlobal800 = filPtsGlobal800
        )

        saveBitmapToPictures(context, roiBmp, "RIM_${debugTag}_ROI")
        saveBitmapToPictures(context, edgesBmp, "RIM_${debugTag}_EDGES")
        saveBitmapToPictures(context, overlayBmp, "RIM_${debugTag}_OVERLAY")
    }

    /**
     * Solo edge map (U8 0..255).
     * Útil para llamar aun si el resto falla.
     */
    fun dumpEdgesOnly(
        context: Context,
        pack: RimDetectPack3250,
        debugTag: String,
        minIntervalMs: Long = 1200L
    ) {
        val key = "EDGE_${debugTag}"
        if (!allow(key, minIntervalMs)) return

        val edgesBmp = edgeU8ToBitmap(pack.edges, pack.w, pack.h)
        saveBitmapToPictures(context, edgesBmp, "EDGE_${debugTag}")
    }

    // ---------------- drawing ----------------

    private fun drawOverlayOnRoi(
        roiBmp: Bitmap,
        res: RimDetectionResult,
        filPtsGlobal800: List<PointF>? = null
    ) {
        val c = Canvas(roiBmp)

        val x0 = res.roiPx.left
        val y0 = res.roiPx.top

        val pLine = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = max(2f, roiBmp.width * 0.004f)
        }

        fun drawH(yGlobal: Float, color: Int) {
            pLine.color = color
            val y = yGlobal - y0
            c.drawLine(0f, y, roiBmp.width.toFloat(), y, pLine)
        }

        fun drawV(xGlobal: Float, color: Int) {
            pLine.color = color
            val x = xGlobal - x0
            c.drawLine(x, 0f, x, roiBmp.height.toFloat(), pLine)
        }

        // probe/top/bottom
        drawH(res.probeYpx, Color.CYAN)
        drawH(res.topYpx, Color.YELLOW)
        drawH(res.bottomYpx, Color.GREEN)

        // inner L/R
        drawV(res.innerLeftXpx, Color.MAGENTA)
        drawV(res.innerRightXpx, Color.MAGENTA)

        // polylines (si existen)
        drawPolylineGlobal(c, res.bottomPolylinePx, x0, y0, Color.GREEN, pLine)
        drawPolylineGlobal(c, res.topPolylinePx, x0, y0, Color.YELLOW, pLine)

        // ✅ FIL markers (R1 / R401 / R601)
        filPtsGlobal800?.let { pts ->
            drawFilMarkersOnRoi(
                canvas = c,
                ptsGlobal800 = pts,
                x0 = x0,
                y0 = y0,
                roiW = roiBmp.width,
                roiH = roiBmp.height
            )
        }

        // texto mínimo
        val pTxt = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = max(18f, roiBmp.width * 0.035f)
            style = Paint.Style.FILL
            setShadowLayer(6f, 0f, 0f, Color.BLACK)
        }
        c.drawText(
            "ok=${res.ok} conf=${"%.2f".format(res.confidence)}",
            12f,
            pTxt.textSize + 8f,
            pTxt
        )
    }

    private fun drawPolylineGlobal(
        canvas: Canvas,
        ptsGlobal: List<PointF>?,
        x0: Float,
        y0: Float,
        color: Int,
        paint: Paint
    ) {
        if (ptsGlobal.isNullOrEmpty()) return
        paint.color = color

        val path = Path()
        val p0 = ptsGlobal.first()
        path.moveTo(p0.x - x0, p0.y - y0)
        for (i in 1 until ptsGlobal.size) {
            val p = ptsGlobal[i]
            path.lineTo(p.x - x0, p.y - y0)
        }
        canvas.drawPath(path, paint)
    }

    private fun drawFilMarkersOnRoi(
        canvas: Canvas,
        ptsGlobal800: List<PointF>,
        x0: Float,
        y0: Float,
        roiW: Int,
        roiH: Int
    ) {
        if (ptsGlobal800.size < 800) return

        val idxs = intArrayOf(0, 400, 600) // R1, R401, R601 (0-based)
        val labels = arrayOf("R1", "R401", "R601")
        val colors = intArrayOf(Color.CYAN, Color.YELLOW, Color.GREEN)

        val pDot = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
        val pRing = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = max(2f, roiW * 0.004f)
        }
        val pTxt = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = max(16f, roiW * 0.030f)
            style = Paint.Style.FILL
            setShadowLayer(6f, 0f, 0f, Color.BLACK)
        }

        val rDot = max(6f, roiW * 0.012f)

        for (k in idxs.indices) {
            val i = idxs[k]
            val pG = ptsGlobal800[i]
            if (!pG.x.isFinite() || !pG.y.isFinite()) continue

            val x = pG.x - x0
            val y = pG.y - y0

            // solo si cae dentro del ROI (con margen)
            if (x < -20f || y < -20f || x > roiW + 20f || y > roiH + 20f) continue

            pDot.color = colors[k]
            pRing.color = colors[k]

            canvas.drawCircle(x, y, rDot, pDot)
            canvas.drawCircle(x, y, rDot * 1.6f, pRing)
            canvas.drawText(labels[k], x + rDot * 1.9f, y - rDot * 1.2f, pTxt)
        }
    }

    // ---------------- bitmap helpers ----------------

    private fun cropBitmapSafe(src: Bitmap, roi: RectF): Bitmap? {
        val l = min(roi.left, roi.right).toInt()
        val t = min(roi.top, roi.bottom).toInt()
        val r = max(roi.left, roi.right).toInt()
        val b = max(roi.top, roi.bottom).toInt()

        val x0 = l.coerceIn(0, src.width - 1)
        val y0 = t.coerceIn(0, src.height - 1)
        val x1 = r.coerceIn(x0 + 1, src.width)
        val y1 = b.coerceIn(y0 + 1, src.height)

        val w = (x1 - x0).coerceAtLeast(1)
        val h = (y1 - y0).coerceAtLeast(1)

        return try {
            Bitmap.createBitmap(src, x0, y0, w, h)
        } catch (e: Throwable) {
            Log.e(TAG, "cropBitmapSafe error", e)
            null
        }
    }

    /** edges = U8 (0..255). Si te llega 0/1 también funciona (se ve negro/blanco). */
    private fun edgeU8ToBitmap(edges: ByteArray, w: Int, h: Int): Bitmap {
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val px = IntArray(w * h)
        val n = min(px.size, edges.size)
        for (i in 0 until n) {
            val v = edges[i].toInt() and 0xFF
            px[i] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
        }
        bmp.setPixels(px, 0, w, 0, 0, w, h)
        return bmp
    }

    // ---------------- saving (MediaStore) ----------------

    private fun saveBitmapToPictures(context: Context, bmp: Bitmap, baseName: String): Uri? {
        val name = "${baseName}_${System.currentTimeMillis()}.png"

        val resolver = context.contentResolver
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, name)
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.RELATIVE_PATH, REL_PATH)
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
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
            try { os?.close() } catch (_: Throwable) {}
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val done = ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }
            resolver.update(uri, done, null, null)
        }

        Log.d(TAG, "Saved debug png: $name")
        return uri
    }
}
