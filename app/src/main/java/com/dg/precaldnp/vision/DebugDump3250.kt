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
import kotlin.math.roundToInt

object DebugPaths3250 {
    const val REL_PATH = "Pictures/PrecalDNP/DEBUG" // UNA sola carpeta
}

/**
 * DebugDump3250 (núcleo único)
 *
 * - Guarda edgesU8 (U8 0..255) como PNG.
 * - Dibuja overlays de RimDetector (RIM_*).
 * - Dibuja overlays de FACE (FULL + ROIs OD/OI) aunque el detector falle.
 *
 * ✅ Un solo saver + un solo throttle + helpers únicos.
 */
object DebugDump3250 {

    private const val TAG = "DebugDump3250"
    private const val REL_PATH = DebugPaths3250.REL_PATH

    // throttle por "canal"
    private val lastByKey = HashMap<String, Long>()

    private fun allow(key: String, minIntervalMs: Long): Boolean {
        val now = SystemClock.elapsedRealtime()
        val last = lastByKey[key] ?: 0L
        if (now - last < minIntervalMs) return false
        lastByKey[key] = now
        return true
    }

    // ============================================================
    // RIM DUMPS (como ya tenías)
    // ============================================================

    /**
     * Guarda SOLO:
     *  - RIM_{tag}_EDGES
     *  - RIM_{tag}_EDGES_OVERLAY
     *
     * Opcional:
     *  - filPtsGlobal800: marca R1/R401/R601 sobre edgeMap.
     */
    fun dumpRim(
        context: Context,
        pack: RimDetectPack3250,
        debugTag: String,
        filPtsGlobal800: List<PointF>? = null,
        minIntervalMs: Long = 1200L
    ) {
        val key = "RIM_${debugTag}"
        if (!allow(key, minIntervalMs)) return

        val res = pack.result

        val edgesBmp = edgeU8ToBitmap(pack.edges, pack.w, pack.h)

        val edgesOverlayBmp = edgesBmp.copy(Bitmap.Config.ARGB_8888, true)
        drawRimOverlayOnEdgeBitmap(
            edgeBmp = edgesOverlayBmp,
            res = res,
            filPtsGlobal800 = filPtsGlobal800
        )

        saveBitmapToPictures(context, edgesBmp, "RIM_${debugTag}_EDGES")
        saveBitmapToPictures(context, edgesOverlayBmp, "RIM_${debugTag}_EDGES_OVERLAY")
    }

    /**
     * Solo edge map (U8 0..255).
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

    /**
     * “Attempt” overlay: para cuando todavía no hay RimDetectionResult bueno.
     * OJO: todos los parámetros que pasás acá deben estar en coords ROI-local si el bitmap es ROI.
     */
    fun dumpEdgesAttempt3250(
        context: Context,
        edgesU8: ByteArray,
        w: Int,
        h: Int,
        debugTag: String,
        roiGlobal: RectF,   // solo texto
        midlineXpx: Float? = null,
        pupilG: PointF? = null,
        probeYpx: Float? = null,
        topMinAllowedYpx: Float? = null,
        bottomYpx: Float? = null,
        seedXpx: Float? = null,
        expLeftXpx: Float? = null,
        expRightXpx: Float? = null,
        nasalInnerXpx: Float? = null,
        templeInnerXpx: Float? = null,
        note: String = "RIM_FAIL",
        minIntervalMs: Long = 1200L
    ) {
        val key = "ATTEMPT_${debugTag}"
        if (!allow(key, minIntervalMs)) return

        val edgesBmp = edgeU8ToBitmap(edgesU8, w, h)

        val overlayBmp = edgesBmp.copy(Bitmap.Config.ARGB_8888, true)
        drawAttemptOverlayOnEdgeBitmap(
            edgeBmp = overlayBmp,
            roiGlobal = roiGlobal,
            midlineXpx = midlineXpx,
            pupilG = pupilG,
            probeYpx = probeYpx,
            topMinAllowedYpx = topMinAllowedYpx,
            bottomYpx = bottomYpx,
            seedXpx = seedXpx,
            expLeftXpx = expLeftXpx,
            expRightXpx = expRightXpx,
            nasalInnerXpx = nasalInnerXpx,
            templeInnerXpx = templeInnerXpx,
            note = note
        )

        saveBitmapToPictures(context, edgesBmp, "ATT_${debugTag}_EDGES")
        saveBitmapToPictures(context, overlayBmp, "ATT_${debugTag}_EDGES_OVERLAY")
    }

    // ============================================================
    // FACE DUMP (núcleo único) — reemplaza DebugDumpFace3250 real
    // ============================================================

    class FaceOverlayInput3250(
        val edgesFullU8: ByteArray,   // size = wFull*hFull, 0/255
        val wFull: Int,
        val hFull: Int,
        val maskFullU8: ByteArray? = null,
        val roiOdGlobal: RectF,
        val roiOiGlobal: RectF,
        val midlineXpx: Float,
        val pupilOdGlobal: PointF?,
        val pupilOiGlobal: PointF?,
        val bridgeYpx: Float? = null,
        val probeYpx: Float? = null,
        val extraTag: String = "FACE"
    )

    /**
     * Guarda SIEMPRE registro visual aunque RimDetector falle:
     *  - EDGE_FULL + overlay (midline, pupilas, ROIs)
     *  - EDGE_ROI_OD + overlay
     *  - EDGE_ROI_OI + overlay
     */
    fun dumpFace(
        context: Context,
        inp: FaceOverlayInput3250,
        minIntervalMs: Long = 800L
    ) {
        val key = "FACE_${inp.extraTag}"
        if (!allow(key, minIntervalMs)) return

        if (inp.edgesFullU8.size != inp.wFull * inp.hFull) {
            Log.e(TAG, "dumpFace edgesFullU8 size mismatch ${inp.edgesFullU8.size} != ${inp.wFull * inp.hFull}")
            return
        }

        // 1) FULL edges + overlay general
        val fullBmp = edgeU8ToBitmap(inp.edgesFullU8, inp.wFull, inp.hFull)
        val fullOver = fullBmp.copy(Bitmap.Config.ARGB_8888, true)
        drawFullOverlay(fullOver, inp)
        saveBitmapToPictures(context, fullOver, "DBG_${inp.extraTag}_EDGE_FULL_OVER")

        // 2) OD ROI crop + overlay local (+ MASK)
        cropRoiU8(inp.edgesFullU8, inp.wFull, inp.hFull, inp.roiOdGlobal)?.let { roi ->
            val bmp = edgeU8ToBitmap(roi.u8, roi.w, roi.h)
            val over = bmp.copy(Bitmap.Config.ARGB_8888, true)

            // ✅ MASK: crop FULL -> ROI y pintá ROJO
            val maskRoi = inp.maskFullU8?.let { mf ->
                if (mf.size == inp.wFull * inp.hFull) cropRoiU8(mf, inp.wFull, inp.hFull, roi.roiGlobal) else null
            }
            maskRoi?.let { drawMaskOverlayRed(over, it.u8, it.w, it.h) }

            drawRoiOverlay(over, roi.roiGlobal, inp, sideTag = "OD")
            saveBitmapToPictures(context, over, "DBG_${inp.extraTag}_EDGE_ROI_OD_MASK_OVER")
        }
        // 3) OI ROI crop + overlay local (+ MASK)
        cropRoiU8(inp.edgesFullU8, inp.wFull, inp.hFull, inp.roiOiGlobal)?.let { roi ->
            val bmp = edgeU8ToBitmap(roi.u8, roi.w, roi.h)
            val over = bmp.copy(Bitmap.Config.ARGB_8888, true)

            val maskRoi = inp.maskFullU8?.let { mf ->
                if (mf.size == inp.wFull * inp.hFull) cropRoiU8(mf, inp.wFull, inp.hFull, roi.roiGlobal) else null
            }
            maskRoi?.let { drawMaskOverlayRed(over, it.u8, it.w, it.h) }

            drawRoiOverlay(over, roi.roiGlobal, inp, sideTag = "OI")
            saveBitmapToPictures(context, over, "DBG_${inp.extraTag}_EDGE_ROI_OI_MASK_OVER")
        }
    }

    // ============================================================
    // DRAWING — RIM
    // ============================================================

    private fun drawRimOverlayOnEdgeBitmap(
        edgeBmp: Bitmap,
        res: RimDetectionResult,
        filPtsGlobal800: List<PointF>? = null
    ) {
        val c = Canvas(edgeBmp)

        val x0 = res.roiPx.left
        val y0 = res.roiPx.top

        val pLine = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = max(2f, edgeBmp.width * 0.004f)
        }

        fun drawH(yGlobal: Float, color: Int) {
            val y = yGlobal - y0
            if (y < -40f || y > edgeBmp.height + 40f) return
            pLine.color = color
            c.drawLine(0f, y, edgeBmp.width.toFloat(), y, pLine)
        }

        fun drawV(xGlobal: Float, color: Int) {
            val x = xGlobal - x0
            if (x < -40f || x > edgeBmp.width + 40f) return
            pLine.color = color
            c.drawLine(x, 0f, x, edgeBmp.height.toFloat(), pLine)
        }

        // top (evito mostrar si parece eyelid pegado al probe)
        if (kotlin.math.abs(res.topYpx - res.probeYpx) >= 12f) {
            drawH(res.topYpx, Color.YELLOW)
        }

        // nasal/temple (INNER)
        drawV(res.nasalInnerXpx, Color.RED)
        drawV(res.templeInnerXpx, Color.BLUE)

        // inner segment @ probe
        run {
            val y = res.probeYpx - y0
            val xL = res.innerLeftXpx - x0
            val xR = res.innerRightXpx - x0
            if (y > -40f && y < edgeBmp.height + 40f) {
                pLine.color = Color.MAGENTA
                pLine.strokeWidth = max(3f, edgeBmp.width * 0.006f)
                c.drawLine(xL, y, xR, y, pLine)
            }
        }

        drawH(res.probeYpx, Color.CYAN)
        drawH(res.bottomYpx, Color.GREEN)

        drawV(res.innerLeftXpx, Color.MAGENTA)
        drawV(res.innerRightXpx, Color.MAGENTA)

        // OUTER
        res.outerLeftXpx?.let { drawV(it, 0xFFFF9800.toInt()) }
        res.outerRightXpx?.let { drawV(it, 0xFFFF9800.toInt()) }

        if (res.outerLeftXpx != null && res.outerRightXpx != null) {
            val y = res.probeYpx - y0
            val xL = res.outerLeftXpx - x0
            val xR = res.outerRightXpx - x0
            if (y > -40f && y < edgeBmp.height + 40f) {
                pLine.color = 0xFFFF9800.toInt()
                pLine.strokeWidth = max(3f, edgeBmp.width * 0.006f)
                c.drawLine(xL, y, xR, y, pLine)
            }
        }

        // polylines
        drawPolylineGlobal(c, res.bottomPolylinePx, x0, y0, Color.GREEN, pLine)
        drawPolylineGlobal(c, res.topPolylinePx, x0, y0, Color.YELLOW, pLine)

        // FIL markers
        filPtsGlobal800?.let { pts ->
            drawFilMarkersOnRoi(
                canvas = c,
                ptsGlobal800 = pts,
                x0 = x0,
                y0 = y0,
                roiW = edgeBmp.width,
                roiH = edgeBmp.height
            )
        }

        // texto mínimo
        val pTxt = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = max(18f, edgeBmp.width * 0.035f)
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

            if (x < -20f || y < -20f || x > roiW + 20f || y > roiH + 20f) continue

            pDot.color = colors[k]
            pRing.color = colors[k]

            canvas.drawCircle(x, y, rDot, pDot)
            canvas.drawCircle(x, y, rDot * 1.6f, pRing)
            canvas.drawText(labels[k], x + rDot * 1.9f, y - rDot * 1.2f, pTxt)
        }
    }

    // ============================================================
    // DRAWING — ATTEMPT
    // ============================================================

    private fun drawAttemptOverlayOnEdgeBitmap(
        edgeBmp: Bitmap,
        roiGlobal: RectF,
        midlineXpx: Float?,
        pupilG: PointF?,
        probeYpx: Float?,
        topMinAllowedYpx: Float?,
        bottomYpx: Float?,
        seedXpx: Float?,
        expLeftXpx: Float?,
        expRightXpx: Float?,
        nasalInnerXpx: Float?,
        templeInnerXpx: Float?,
        note: String
    ) {
        val c = Canvas(edgeBmp)

        val p = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = max(2f, edgeBmp.width * 0.006f)
        }

        fun drawH(y: Float?, color: Int) {
            if (y == null || !y.isFinite()) return
            p.color = color
            c.drawLine(0f, y, edgeBmp.width.toFloat(), y, p)
        }

        fun drawV(x: Float?, color: Int) {
            if (x == null || !x.isFinite()) return
            p.color = color
            c.drawLine(x, 0f, x, edgeBmp.height.toFloat(), p)
        }

        drawH(topMinAllowedYpx, Color.YELLOW)
        drawH(probeYpx, Color.CYAN)
        drawH(bottomYpx, Color.GREEN)

        drawV(midlineXpx, Color.WHITE)
        drawV(seedXpx, 0xFFFF9800.toInt())
        drawV(expLeftXpx, Color.MAGENTA)
        drawV(expRightXpx, Color.MAGENTA)
        drawV(nasalInnerXpx, Color.RED)
        drawV(templeInnerXpx, Color.BLUE)

        pupilG?.let {
            if (it.x.isFinite() && it.y.isFinite()) {
                val pDot = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    style = Paint.Style.FILL
                    color = 0xFF00BCD4.toInt()
                }
                c.drawCircle(it.x, it.y, max(6f, edgeBmp.width * 0.014f), pDot)
            }
        }

        val pTxt = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = max(18f, edgeBmp.width * 0.040f)
            style = Paint.Style.FILL
            setShadowLayer(6f, 0f, 0f, Color.BLACK)
        }
        c.drawText(note, 12f, pTxt.textSize + 10f, pTxt)

        val pTxt2 = Paint(pTxt).apply { textSize = max(16f, edgeBmp.width * 0.030f) }
        c.drawText(
            "ROI[g]=L${roiGlobal.left.toInt()} T${roiGlobal.top.toInt()} W${roiGlobal.width().toInt()} H${roiGlobal.height().toInt()}",
            12f,
            pTxt.textSize + pTxt2.textSize + 18f,
            pTxt2
        )
    }

    // ============================================================
    // DRAWING — FACE
    // ============================================================

    private fun drawFullOverlay(bmp: Bitmap, inp: FaceOverlayInput3250) {
        val c = Canvas(bmp)

        val p = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = max(3f, bmp.width * 0.004f)
        }

        // midline
        p.color = Color.CYAN
        c.drawLine(inp.midlineXpx, 0f, inp.midlineXpx, bmp.height.toFloat(), p)

        // ROIs
        p.color = 0xFFFF9800.toInt()
        c.drawRect(inp.roiOdGlobal, p)
        p.color = 0xFF00BCD4.toInt()
        c.drawRect(inp.roiOiGlobal, p)

        // bridge/probe
        inp.bridgeYpx?.let {
            p.color = Color.YELLOW
            c.drawLine(0f, it, bmp.width.toFloat(), it, p)
        }
        inp.probeYpx?.let {
            p.color = Color.MAGENTA
            c.drawLine(0f, it, bmp.width.toFloat(), it, p)
        }

        // pupilas
        drawDot(c, inp.pupilOdGlobal, Color.WHITE, bmp.width)
        drawDot(c, inp.pupilOiGlobal, Color.WHITE, bmp.width)

        val t = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = max(18f, bmp.width * 0.030f)
            style = Paint.Style.FILL
            setShadowLayer(6f, 0f, 0f, Color.BLACK)
        }
        c.drawText("FULL w=${inp.wFull} h=${inp.hFull}", 12f, t.textSize + 8f, t)
    }

    private fun drawRoiOverlay(bmp: Bitmap, roiGlobal: RectF, inp: FaceOverlayInput3250, sideTag: String) {
        val c = Canvas(bmp)
        val x0 = roiGlobal.left
        val y0 = roiGlobal.top

        val p = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = max(3f, bmp.width * 0.006f)
        }

        // midline local
        val midLocal = inp.midlineXpx - x0
        p.color = Color.CYAN
        c.drawLine(midLocal, 0f, midLocal, bmp.height.toFloat(), p)

        // bridge/probe local
        inp.bridgeYpx?.let {
            val y = it - y0
            p.color = Color.YELLOW
            c.drawLine(0f, y, bmp.width.toFloat(), y, p)
        }
        inp.probeYpx?.let {
            val y = it - y0
            p.color = Color.MAGENTA
            c.drawLine(0f, y, bmp.width.toFloat(), y, p)
        }

        // pupila local
        val pupilG = if (sideTag == "OD") inp.pupilOdGlobal else inp.pupilOiGlobal
        val pupilLocal = pupilG?.let { PointF(it.x - x0, it.y - y0) }
        drawDot(c, pupilLocal, Color.WHITE, bmp.width)

        // marco
        p.color = if (sideTag == "OD") 0xFFFF9800.toInt() else 0xFF00BCD4.toInt()
        c.drawRect(0f, 0f, bmp.width.toFloat(), bmp.height.toFloat(), p)

        val t = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = max(18f, bmp.width * 0.050f)
            style = Paint.Style.FILL
            setShadowLayer(6f, 0f, 0f, Color.BLACK)
        }
        c.drawText("$sideTag ROI ${bmp.width}x${bmp.height}", 12f, t.textSize + 8f, t)
    }

    private fun drawDot(canvas: Canvas, p: PointF?, color: Int, refW: Int) {
        if (p == null) return
        val r = max(6f, refW * 0.018f)
        val pd = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL; this.color = color }
        val pr = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = max(2f, refW * 0.004f); this.color = color }
        canvas.drawCircle(p.x, p.y, r, pd)
        canvas.drawCircle(p.x, p.y, r * 1.6f, pr)
    }

    // ============================================================
    // ROI crop U8 (FULL -> ROI)
    // ============================================================

    private class RoiU8(val u8: ByteArray, val w: Int, val h: Int, val roiGlobal: RectF)

    private fun cropRoiU8(full: ByteArray, wFull: Int, hFull: Int, roiG: RectF): RoiU8? {
        var l = roiG.left.roundToInt().coerceIn(0, wFull - 1)
        var t = roiG.top.roundToInt().coerceIn(0, hFull - 1)
        var r = roiG.right.roundToInt().coerceIn(0, wFull)
        var b = roiG.bottom.roundToInt().coerceIn(0, hFull)

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
        return RoiU8(out, w, h, RectF(l.toFloat(), t.toFloat(), r.toFloat(), b.toFloat()))
    }

    // ============================================================
    // Bitmap helpers
    // ============================================================

    /** edges = U8 (0..255). Si llega 0/1 también funciona (se ve negro/blanco). */
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

    // ============================================================
    // Saving (MediaStore)
    // ============================================================

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
    private fun drawMaskOverlayRed(bmp: Bitmap, mask: ByteArray, w: Int, h: Int) {
        if (bmp.width != w || bmp.height != h) return
        val overlay = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val px = IntArray(w * h)
        val n = min(px.size, mask.size)

        val redA = 0x66FF0000.toInt()

        for (i in 0 until n) {
            val v = mask[i].toInt() and 0xFF
            px[i] = if (v != 0) redA else 0x00000000
        }
        overlay.setPixels(px, 0, w, 0, 0, w, h)
        Canvas(bmp).drawBitmap(overlay, 0f, 0f, null)
    }

}
