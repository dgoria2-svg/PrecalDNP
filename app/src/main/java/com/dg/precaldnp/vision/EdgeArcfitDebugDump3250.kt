package com.dg.precaldnp.vision

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.net.Uri
import android.provider.MediaStore
import android.util.Log
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.roundToInt

object EdgeArcfitDebugDump3250 {

    private const val TAG = "EDGE_DUMP3250"

    /**
     * Guarda PNG en Galería (Pictures/PrecalDNP/DEBUG3250) con:
     * - edgeMap (ROI) de fondo
     * - marcas (inner/outer/top/bottom/probe)
     * - contorno ARC-FIT (placed)
     * - topEst/botEst (minY/maxY del placed)
     * - R1/R401/R601 (sobre placed)
     *
     * NOTA índices:
     * - R1 = NASAL
     * - R401 = TEMPORAL
     * - R601 = BOTTOM
     */
    fun saveEdgeWithArcfitMarksToGallery3250(
        ctx: Context,
        eyeTag: String,                 // "OD"/"OI" (o el tag que uses)
        edgeGray: ByteArray,
        w: Int,
        h: Int,

        innerLeftPx: Int?,
        innerRightPx: Int?,
        outerNasalPx: Int?,
        outerTemplePx: Int?,
        topPx: Int?,
        bottomPx: Int?,
        probeYPx: Int?,

        placedPxRoi: List<PointF>?
    ): Uri? {

        if (w <= 0 || h <= 0) return null
        val total = w * h
        if (edgeGray.size != total) {
            Log.w(TAG, "edge size mismatch ${edgeGray.size} != $total")
            return null
        }

        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val pix = IntArray(total)
        for (i in 0 until total) {
            val v = edgeGray[i].toInt() and 0xFF
            pix[i] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
        }
        bmp.setPixels(pix, 0, w, 0, 0, w, h)

        val c = Canvas(bmp)

        val pInner = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 2.5f; color = 0xFF00BCD4.toInt() }
        val pOuter = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 2.5f; color = 0xFFFF4081.toInt() }
        val pTB    = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 2.5f; color = 0xFFFFC107.toInt() }
        val pProbe = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 1.5f; color = 0xFF9E9E9E.toInt() }

        val pPoly  = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 2.0f; color = 0xFFFF3D00.toInt() }
        val pEst   = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 2.0f; color = 0xFF4CAF50.toInt() }

        val pDot   = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL; color = 0xFF2196F3.toInt() }
        val pText  = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL; color = 0xFFFFFFFF.toInt(); textSize = 18f }

        fun vLine(x: Int, paint: Paint) {
            if (x !in 0 until w) return
            c.drawLine(x.toFloat(), 0f, x.toFloat(), (h - 1).toFloat(), paint)
        }
        fun hLine(y: Int, paint: Paint) {
            if (y !in 0 until h) return
            c.drawLine(0f, y.toFloat(), (w - 1).toFloat(), y.toFloat(), paint)
        }

        // RimDetector marks
        innerLeftPx?.let { vLine(it, pInner) }
        innerRightPx?.let { vLine(it, pInner) }
        outerNasalPx?.let { vLine(it, pOuter) }
        outerTemplePx?.let { vLine(it, pOuter) }
        topPx?.let { hLine(it, pTB) }
        bottomPx?.let { hLine(it, pTB) }
        probeYPx?.let { hLine(it, pProbe) }

        // ARC-FIT placed poly + topEst/botEst + R markers
        var topEst: Int? = null
        var botEst: Int? = null

        placedPxRoi?.takeIf { it.size >= 3 }?.let { pts ->
            var minY = Float.POSITIVE_INFINITY
            var maxY = Float.NEGATIVE_INFINITY
            for (p in pts) {
                if (p.y < minY) minY = p.y
                if (p.y > maxY) maxY = p.y
            }
            if (minY.isFinite() && maxY.isFinite()) {
                topEst = minY.roundToInt()
                botEst = maxY.roundToInt()
            }

            val path = Path()
            path.moveTo(pts[0].x, pts[0].y)
            for (i in 1 until pts.size) path.lineTo(pts[i].x, pts[i].y)
            path.close()
            c.drawPath(path, pPoly)

            topEst?.let { hLine(it, pEst) }
            botEst?.let { hLine(it, pEst) }

            fun mark(idx1based: Int, label: String) {
                val i = (idx1based - 1).coerceIn(0, pts.lastIndex)
                val p = pts[i]
                c.drawCircle(p.x, p.y, 6.5f, pDot)
                c.drawText(label, p.x + 8f, p.y - 8f, pText)
            }

            if (pts.size >= 601) {
                mark(1,   "R1 NASAL")
                mark(401, "R401 TEMP")
                mark(601, "R601 BOT")
            } else {
                mark(1, "R1 NASAL")
            }
        }

        val info = buildString {
            append("EDGE+$eyeTag  ${w}x$h\n")
            append("inner=[${innerLeftPx ?: "-"},${innerRightPx ?: "-"}]  outer=[${outerNasalPx ?: "-"},${outerTemplePx ?: "-"}]\n")
            append("top/bot=[${topPx ?: "-"},${bottomPx ?: "-"}]  probeY=${probeYPx ?: "-"}  topEst/botEst=[${topEst ?: "-"},${botEst ?: "-"}]")
        }
        var ty = 22f
        for (ln in info.split('\n')) {
            c.drawText(ln, 10f, ty, pText)
            ty += 20f
        }

        val ts = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())
        val safeTag = eyeTag.replace(Regex("[^A-Za-z0-9_\\-]"), "_")
        val displayName = "edge_arcfit_${safeTag}_$ts.png"

        val uri = saveBitmapToPictures3250(
            ctx = ctx,
            bmp = bmp,
            displayName = displayName,
            relativePath = DebugPaths3250.REL_PATH

        )

        try { bmp.recycle() } catch (_: Throwable) {}
        return uri
    }

    private fun saveBitmapToPictures3250(
        ctx: Context,
        bmp: Bitmap,
        displayName: String,
        relativePath: String
    ): Uri? {
        val resolver = ctx.contentResolver

        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, displayName)
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            put(MediaStore.Images.Media.RELATIVE_PATH, relativePath)
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }

        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
        if (uri == null) {
            Log.w(TAG, "MediaStore insert failed")
            return null
        }

        var os: OutputStream? = null
        return try {
            os = resolver.openOutputStream(uri)
            if (os == null) throw IllegalStateException("openOutputStream null")
            if (!bmp.compress(Bitmap.CompressFormat.PNG, 100, os)) {
                throw IllegalStateException("compress returned false")
            }
            os.flush()

            val done = ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }
            resolver.update(uri, done, null, null)

            Log.d(TAG, "Saved debug PNG to gallery: $uri")
            uri
        } catch (t: Throwable) {
            Log.w(TAG, "saveBitmapToPictures failed: ${t.message}", t)
            try { resolver.delete(uri, null, null) } catch (_: Throwable) {}
            null
        } finally {
            try { os?.close() } catch (_: Throwable) {}
        }
    }
}
