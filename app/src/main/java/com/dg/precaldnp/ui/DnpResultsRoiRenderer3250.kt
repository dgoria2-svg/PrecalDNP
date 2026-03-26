package com.dg.precaldnp.ui

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.graphics.Rect
import android.graphics.RectF
import android.provider.MediaStore
import android.util.Log
import androidx.core.graphics.createBitmap
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.roundToInt

object DnpResultsRoiRenderer3250 {

    private const val TAG = "DnpResultsRoiRenderer3250"
    private const val REL_PATH = "Pictures/PrecalDNP/RESULTS"

    // Colores acordados
    private const val ORANGE = 0xFFFF9800.toInt()   // armazón / métricas de aro
    private const val CYAN = 0xFF00BCD4.toInt()     // pupilas / referencias

    private data class Tile3250(
        val label: String,
        val roi: Rect,
        val bmp: Bitmap,
        val pupil: PointF?,
        val placed: List<PointF>?,
        val rimBottomY: Float?,
        val eyeCenterX: Float,
        var dstX: Float = 0f,
        var dstY: Float = 0f
    )

    /**
     * Devuelve un content:// (string) de la imagen FINAL:
     * - 2 ROIs (OD/OI) en un solo PNG
     * - ambos PEGADOS en midline
     * - ambos ALINEADOS por irisline (yRef)
     * - círculo Ø útil + radio usado (sin texto)
     * - línea pupilar + marcadores pupila
     * - altura pupila->bottom rim (si existe)
     * - puente (línea + ticks en nasales) en la línea pupilar
     *
     * NO dibuja el FIL.
     */
    fun renderAndSaveFinalRois3250(
        ctx: Context,
        stillBmp: Bitmap,
        pm: DnpFacePipeline3250.PupilsMidPack3250,
        fit: DnpFacePipeline3250.FitPack3250,
        metrics: DnpMetrics3250,
        orderId3250: String
    ): String? {
        val wFull = stillBmp.width
        val hFull = stillBmp.height
        if (wFull <= 8 || hFull <= 8) return null

        val pxPerMm = metrics.pxPerMmFaceD.takeIf { it.isFinite() && it > 1e-6 } ?: fit.pxPerMmFaceD
        val padPx = ((10.0 * pxPerMm).coerceIn(30.0, 160.0)).toFloat()

        val odPupil = if (pm.odOkReal) pm.pupilOdDet else null
        val oiPupil = if (pm.oiOkReal) pm.pupilOiDet else null

        // yRef = irisline / pupil line oficial
        val yRef = run {
            val y = when {
                odPupil != null && oiPupil != null -> 0.5f * (odPupil.y + oiPupil.y)
                odPupil != null -> odPupil.y
                oiPupil != null -> oiPupil.y
                else -> (0.55f * hFull)
            }
            y.coerceIn(0f, (hFull - 1).toFloat())
        }

        // midline oficial en esa irisline
        val midRef = pm.midline3250.xAt(yRef, wFull).coerceIn(0f, (wFull - 1).toFloat())

        fun boundsOf(pts: List<PointF>?): RectF? {
            if (pts.isNullOrEmpty()) return null
            var minX = Float.POSITIVE_INFINITY
            var minY = Float.POSITIVE_INFINITY
            var maxX = Float.NEGATIVE_INFINITY
            var maxY = Float.NEGATIVE_INFINITY
            for (p in pts) {
                if (!p.x.isFinite() || !p.y.isFinite()) continue
                if (p.x < minX) minX = p.x
                if (p.y < minY) minY = p.y
                if (p.x > maxX) maxX = p.x
                if (p.y > maxY) maxY = p.y
            }
            if (!minX.isFinite() || !minY.isFinite() || !maxX.isFinite() || !maxY.isFinite()) return null
            if (maxX - minX < 6f || maxY - minY < 6f) return null
            return RectF(minX, minY, maxX, maxY)
        }

        fun fallbackBaseFromPupil(p: PointF): RectF {
            return RectF(
                p.x - 260f,
                p.y - 220f,
                p.x + 260f,
                p.y + 340f
            )
        }

        fun buildMidlineAnchoredRoi3250(
            base: RectF,
            eyeCenterX: Float,
            midX: Float,
            yRefGlobal: Float,
            pad: Float,
            w: Int,
            h: Int
        ): Rect {
            val eyeIsLeftOfMid = eyeCenterX < midX
            val midI = midX.roundToInt().coerceIn(1, w - 1)

            var l = (base.left - pad).roundToInt()
            var t = (minOf(base.top - pad, yRefGlobal - pad * 0.35f)).roundToInt()
            var r = (base.right + pad).roundToInt()
            var b = (maxOf(base.bottom + pad, yRefGlobal + pad * 0.35f)).roundToInt()

            if (eyeIsLeftOfMid) {
                // el ROI izquierdo termina EXACTO en midline
                r = midI
                if (l >= r - 2) {
                    l = (r - max(120, (pad * 2f).roundToInt())).coerceAtLeast(0)
                }
            } else {
                // el ROI derecho empieza EXACTO en midline
                l = midI
                if (r <= l + 2) {
                    r = (l + max(120, (pad * 2f).roundToInt())).coerceAtMost(w)
                }
            }

            l = l.coerceIn(0, w - 2)
            t = t.coerceIn(0, h - 2)
            r = r.coerceIn(l + 2, w)
            b = b.coerceIn(t + 2, h)

            return Rect(l, t, r, b)
        }

        val bOd = boundsOf(fit.placedOdUsed)
        val bOi = boundsOf(fit.placedOiUsed)

        val baseOd = bOd ?: odPupil?.let(::fallbackBaseFromPupil)
        val baseOi = bOi ?: oiPupil?.let(::fallbackBaseFromPupil)

        val roiOd0 = baseOd?.let {
            buildMidlineAnchoredRoi3250(
                base = it,
                eyeCenterX = odPupil?.x ?: it.centerX(),
                midX = midRef,
                yRefGlobal = yRef,
                pad = padPx,
                w = wFull,
                h = hFull
            )
        }

        val roiOi0 = baseOi?.let {
            buildMidlineAnchoredRoi3250(
                base = it,
                eyeCenterX = oiPupil?.x ?: it.centerX(),
                midX = midRef,
                yRefGlobal = yRef,
                pad = padPx,
                w = wFull,
                h = hFull
            )
        }

        if (roiOd0 == null && roiOi0 == null) return null

        // Fallback: si falta uno, duplicamos el otro para no romper UI
        val rOd = roiOd0 ?: roiOi0!!
        val rOi = roiOi0 ?: roiOd0!!

        val bmpOd = Bitmap.createBitmap(stillBmp, rOd.left, rOd.top, rOd.width(), rOd.height())
        val bmpOi = Bitmap.createBitmap(stillBmp, rOi.left, rOi.top, rOi.width(), rOi.height())

        val tileOd = Tile3250(
            label = "OD",
            roi = rOd,
            bmp = bmpOd,
            pupil = odPupil,
            placed = fit.placedOdUsed,
            rimBottomY = fit.rimOd?.bottomYpx,
            eyeCenterX = odPupil?.x ?: rOd.exactCenterX()
        )

        val tileOi = Tile3250(
            label = "OI",
            roi = rOi,
            bmp = bmpOi,
            pupil = oiPupil,
            placed = fit.placedOiUsed,
            rimBottomY = fit.rimOi?.bottomYpx,
            eyeCenterX = oiPupil?.x ?: rOi.exactCenterX()
        )

        // Orden VISUAL izquierda->derecha según la foto
        val tilesVisual = listOf(tileOd, tileOi).sortedBy { it.eyeCenterX }

        // Pegados en midline: sin gap
        tilesVisual[0].dstX = 0f
        tilesVisual[1].dstX = tilesVisual[0].bmp.width.toFloat()

        // Alineación vertical por irisline
        fun localIrisY(tile: Tile3250): Float = (yRef - tile.roi.top)

        val irisLocal0 = localIrisY(tilesVisual[0])
        val irisLocal1 = localIrisY(tilesVisual[1])

        val topNeed = max(irisLocal0, irisLocal1)
        val bottomNeed = max(
            tilesVisual[0].bmp.height - irisLocal0,
            tilesVisual[1].bmp.height - irisLocal1
        )

        tilesVisual[0].dstY = topNeed - irisLocal0
        tilesVisual[1].dstY = topNeed - irisLocal1

        val outW = tilesVisual[0].bmp.width + tilesVisual[1].bmp.width
        val outH = (topNeed + bottomNeed).roundToInt().coerceAtLeast(max(bmpOd.height, bmpOi.height))

        val outBmp = createBitmap(outW, outH)
        val c = Canvas(outBmp)
        c.drawColor(Color.BLACK)

        // Pegamos los crops ya alineados por irisline
        for (tile in tilesVisual) {
            c.drawBitmap(tile.bmp, tile.dstX, tile.dstY, null)
        }

        fun mapX(globalX: Float, tile: Tile3250): Float = tile.dstX + (globalX - tile.roi.left)
        fun mapY(globalY: Float, tile: Tile3250): Float = tile.dstY + (globalY - tile.roi.top)

        // strokes adaptativos
        val baseStroke = (max(outW, outH) / 260f).coerceIn(3f, 7f)
        val paintCyan = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            color = CYAN
            strokeWidth = baseStroke
        }
        val paintOrange = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            color = ORANGE
            strokeWidth = baseStroke * 1.10f
        }
        val paintDot = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            color = CYAN
        }

        // Línea pupilar única, ya horizontal porque alineamos por irisline
        c.drawLine(0f, topNeed, (outW - 1).toFloat(), topNeed, paintCyan)

        // Marcadores pupila
        fun drawPupil(p: PointF?, tile: Tile3250) {
            if (p == null) return
            val x = mapX(p.x, tile)
            val y = mapY(p.y, tile)
            c.drawCircle(x, y, baseStroke * 1.8f, paintDot)
            c.drawCircle(x, y, baseStroke * 3.0f, paintCyan)
        }
        drawPupil(tileOd.pupil, tileOd)
        drawPupil(tileOi.pupil, tileOi)

        // Altura (pupila -> bottom rim)
        fun drawHeight(p: PointF?, bottomY: Float?, tile: Tile3250) {
            if (p == null || bottomY == null || !bottomY.isFinite()) return
            if (bottomY <= p.y - 1f) return
            val x = mapX(p.x, tile)
            val y0 = mapY(p.y, tile)
            val y1 = mapY(bottomY, tile)
            c.drawLine(x, y0, x, y1, paintOrange)
            c.drawLine(x - 16f, y1, x + 16f, y1, paintOrange)
        }
        drawHeight(tileOd.pupil, tileOd.rimBottomY, tileOd)
        drawHeight(tileOi.pupil, tileOi.rimBottomY, tileOi)

        // Ø útil + radio usado
        fun drawDiamUtil(
            pupil: PointF?,
            placed: List<PointF>?,
            tile: Tile3250
        ) {
            if (pupil == null) return
            if (placed.isNullOrEmpty()) return
            if (!pxPerMm.isFinite() || pxPerMm <= 1e-6) return

            val cx = mapX(pupil.x, tile)
            val cy = mapY(pupil.y, tile)

            val far = farthestPointFromPupil3250(pupil, placed) ?: return

            val fx = mapX(far.x, tile)
            val fy = mapY(far.y, tile)

            val rBasePx = hypot((far.x - pupil.x).toDouble(), (far.y - pupil.y).toDouble()).toFloat()
            val extraPx = pxPerMm.toFloat() // +1 mm de radio = +2 mm de diámetro
            val rPx = (rBasePx + extraPx).coerceIn(6f, 2000f)

            c.drawCircle(cx, cy, rPx, paintOrange)
            c.drawLine(cx, cy, fx, fy, paintOrange)
            c.drawCircle(fx, fy, baseStroke * 1.8f, paintOrange)
        }

        drawDiamUtil(
            pupil = tileOd.pupil,
            placed = tileOd.placed,
            tile = tileOd
        )

        drawDiamUtil(
            pupil = tileOi.pupil,
            placed = tileOi.placed,
            tile = tileOi
        )

        val odSideSign = when {
            odPupil != null && odPupil.x < midRef -> -1f
            odPupil != null && odPupil.x >= midRef -> +1f
            else -> -1f
        }

        val oiSideSign = when {
            oiPupil != null && oiPupil.x < midRef -> -1f
            oiPupil != null && oiPupil.x >= midRef -> +1f
            else -> +1f
        }

        val bandHalfPx = (3.0 * pxPerMm).toFloat()

        fun nasalFromPlaced(
            placed: List<PointF>?,
            sideSign: Float,
            midX: Float,
            yRefGlobal: Float,
            bandHalf: Float,
            pupilX: Float?
        ): Float? {
            val pts = placed ?: return null
            if (pts.size < 6) return null

            val pupilDistToMid = pupilX?.takeIf { it.isFinite() }?.let { abs(it - midX) }
            val maxFromMid = (pupilDistToMid?.let { (it * 1.20f).coerceIn(30f, 260f) } ?: 220f)

            val xs = ArrayList<Float>(64)
            for (p in pts) {
                if (!p.x.isFinite() || !p.y.isFinite()) continue
                if (abs(p.y - yRefGlobal) > bandHalf) continue
                val dFromMidSigned = (p.x - midX) * sideSign
                if (dFromMidSigned < -2f) continue
                if (dFromMidSigned > maxFromMid) continue
                xs.add(p.x)
            }
            if (xs.isEmpty()) return null

            xs.sort()
            val q = if (sideSign > 0f) 0.12f else 0.88f
            val idx = (q * (xs.size - 1)).toInt().coerceIn(0, xs.size - 1)
            val i0 = (idx - 1).coerceAtLeast(0)
            val i2 = (idx + 1).coerceAtMost(xs.size - 1)
            return (xs[i0] + xs[idx] + xs[i2]) / 3f
        }

        val nasalOdX = nasalFromPlaced(
            placed = fit.placedOdUsed,
            sideSign = odSideSign,
            midX = midRef,
            yRefGlobal = yRef,
            bandHalf = bandHalfPx,
            pupilX = odPupil?.x
        )

        val nasalOiX = nasalFromPlaced(
            placed = fit.placedOiUsed,
            sideSign = oiSideSign,
            midX = midRef,
            yRefGlobal = yRef,
            bandHalf = bandHalfPx,
            pupilX = oiPupil?.x
        )

        if (nasalOdX != null && nasalOiX != null) {
            val xOd = mapX(nasalOdX, tileOd)
            val xOi = mapX(nasalOiX, tileOi)

            // puente ya horizontal porque ambos tiles quedaron alineados por yRef
            c.drawLine(xOd, topNeed, xOi, topNeed, paintOrange)

            val tick = 22f.coerceIn(14f, 34f)
            c.drawLine(xOd, topNeed - tick, xOd, topNeed + tick, paintOrange)
            c.drawLine(xOi, topNeed - tick, xOi, topNeed + tick, paintOrange)
        } else {
            Log.w(TAG, "bridge draw skipped: nasalOdX=$nasalOdX nasalOiX=$nasalOiX")
        }

        Log.d(
            TAG,
            "render final rois: midRef=$midRef yRef=$yRef " +
                    "roiOd=(${rOd.left},${rOd.top},${rOd.right},${rOd.bottom}) " +
                    "roiOi=(${rOi.left},${rOi.top},${rOi.right},${rOi.bottom}) " +
                    "dstOd=(${tileOd.dstX},${tileOd.dstY}) dstOi=(${tileOi.dstX},${tileOi.dstY})"
        )

        // Guardar
        val uriStr = savePngToMediaStore3250(ctx, outBmp, orderId3250)

        try {
            bmpOd.recycle()
        } catch (_: Throwable) {
        }
        try {
            bmpOi.recycle()
        } catch (_: Throwable) {
        }

        return uriStr
    }

    private fun savePngToMediaStore3250(ctx: Context, bmp: Bitmap, orderId: String): String? {
        return try {
            val name = "DNP_${orderId}_${System.currentTimeMillis()}.png"
            val cv = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, name)
                put(MediaStore.MediaColumns.MIME_TYPE, "image/png")
                put(MediaStore.MediaColumns.RELATIVE_PATH, REL_PATH)
            }
            val uri = ctx.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, cv) ?: return null
            ctx.contentResolver.openOutputStream(uri)?.use { os ->
                bmp.compress(Bitmap.CompressFormat.PNG, 100, os)
            }
            uri.toString()
        } catch (t: Throwable) {
            Log.e(TAG, "savePngToMediaStore error", t)
            null
        }
    }

    private fun farthestPointFromPupil3250(
        pupilPx: PointF,
        placedFilPx: List<PointF>?
    ): PointF? {
        if (placedFilPx.isNullOrEmpty()) return null

        var bestPt: PointF? = null
        var bestD2 = -1.0

        for (p in placedFilPx) {
            if (!p.x.isFinite() || !p.y.isFinite()) continue
            val dx = (p.x - pupilPx.x).toDouble()
            val dy = (p.y - pupilPx.y).toDouble()
            val d2 = dx * dx + dy * dy
            if (d2 > bestD2) {
                bestD2 = d2
                bestPt = p
            }
        }
        return bestPt
    }

}