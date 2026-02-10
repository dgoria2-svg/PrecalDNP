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
import kotlin.math.max
import kotlin.math.roundToInt

object DnpResultsRoiRenderer3250 {

    private const val TAG = "DnpResultsRoiRenderer3250"
    private const val REL_PATH = "Pictures/PrecalDNP/RESULTS"

    // Colores acordados
    private const val ORANGE = 0xFFFF9800.toInt()   // armazón / métricas de aro
    private const val CYAN = 0xFF00BCD4.toInt()     // pupilas / referencias

    /**
     * Devuelve un content:// (string) de la imagen FINAL:
     * - 2 ROIs (OD/OI) en un solo PNG
     * - círculo Ø útil + radio usado (sin texto)
     * - línea pupilar (segmentos) + marcadores pupila
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

        fun expandClamp(r: RectF, pad: Float): Rect {
            val l = (r.left - pad).roundToInt().coerceIn(0, wFull - 2)
            val t = (r.top - pad).roundToInt().coerceIn(0, hFull - 2)
            val rr = (r.right + pad).roundToInt().coerceIn(l + 2, wFull)
            val bb = (r.bottom + pad).roundToInt().coerceIn(t + 2, hFull)
            return Rect(l, t, rr, bb)
        }

        val bOd = boundsOf(fit.placedOdUsed)
        val bOi = boundsOf(fit.placedOiUsed)

        val roiOd = when {
            bOd != null -> expandClamp(bOd, padPx)
            odPupil != null -> {
                val r = RectF(odPupil.x - 260f, odPupil.y - 220f, odPupil.x + 260f, odPupil.y + 340f)
                expandClamp(r, 0f)
            }
            else -> null
        }

        val roiOi = when {
            bOi != null -> expandClamp(bOi, padPx)
            oiPupil != null -> {
                val r = RectF(oiPupil.x - 260f, oiPupil.y - 220f, oiPupil.x + 260f, oiPupil.y + 340f)
                expandClamp(r, 0f)
            }
            else -> null
        }

        if (roiOd == null && roiOi == null) return null

        // Si falta uno, duplicamos el que hay (para no romper UI). Igual te deja ver algo.
        val rOd = roiOd ?: roiOi!!
        val rOi = roiOi ?: roiOd!!

        val bmpOd = Bitmap.createBitmap(stillBmp, rOd.left, rOd.top, rOd.width(), rOd.height())
        val bmpOi = Bitmap.createBitmap(stillBmp, rOi.left, rOi.top, rOi.width(), rOi.height())

        val gap = max(10, (0.02f * max(bmpOd.width, bmpOi.width)).roundToInt())
        val outW = bmpOd.width + gap + bmpOi.width
        val outH = max(bmpOd.height, bmpOi.height)

        val outBmp = createBitmap(outW, outH)
        val c = Canvas(outBmp)
        c.drawColor(Color.BLACK)

        // pegamos los crops
        c.drawBitmap(bmpOd, 0f, 0f, null)
        c.drawBitmap(bmpOi, (bmpOd.width + gap).toFloat(), 0f, null)

        val offOdX = 0f
        val offOiX = (bmpOd.width + gap).toFloat()

        fun mapX(globalX: Float, roi: Rect, offX: Float) = offX + (globalX - roi.left)
        fun mapY(globalY: Float, roi: Rect) = (globalY - roi.top)

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

        // yRef = línea pupilar (igual que en métricas)
        val yRef = run {
            val y = when {
                odPupil != null && oiPupil != null -> 0.5f * (odPupil.y + oiPupil.y)
                odPupil != null -> odPupil.y
                oiPupil != null -> oiPupil.y
                else -> (0.55f * hFull)
            }
            y.coerceIn(0f, (hFull - 1).toFloat())
        }
        val midRef = pm.midline3250.xAt(yRef, wFull).coerceIn(0f, (wFull - 1).toFloat())

        // Pupilas (puntos) + línea pupilar (segmentos, y “puente” visual entre tiles)
        val yOd = mapY(yRef, rOd)
        val yOi = mapY(yRef, rOi)
        c.drawLine(offOdX, yOd, (bmpOd.width - 1).toFloat(), yOd, paintCyan)
        c.drawLine(offOiX, yOi, offOiX + (bmpOi.width - 1).toFloat(), yOi, paintCyan)
        // conector en el gap (puede quedar levemente inclinado si los crops tienen distinto top)
        c.drawLine((bmpOd.width - 1).toFloat(), yOd, offOiX, yOi, paintCyan)

        // Marcadores pupila
        fun drawPupil(p: PointF?, roi: Rect, offX: Float) {
            if (p == null) return
            val x = mapX(p.x, roi, offX)
            val y = mapY(p.y, roi)
            c.drawCircle(x, y, baseStroke * 1.8f, paintDot)
            c.drawCircle(x, y, baseStroke * 3.0f, paintCyan)
        }
        drawPupil(odPupil, rOd, offOdX)
        drawPupil(oiPupil, rOi, offOiX)

        // Altura (pupila -> bottom rim)
        fun drawHeight(p: PointF?, bottomY: Float?, roi: Rect, offX: Float) {
            if (p == null || bottomY == null || !bottomY.isFinite()) return
            if (bottomY <= p.y + 1f) return
            val x = mapX(p.x, roi, offX)
            val y0 = mapY(p.y, roi)
            val y1 = mapY(bottomY, roi)
            c.drawLine(x, y0, x, y1, paintOrange)
            // tick abajo
            c.drawLine(x - 16f, y1, x + 16f, y1, paintOrange)
        }
        drawHeight(odPupil, fit.rimOd?.bottomYpx, rOd, offOdX)
        drawHeight(oiPupil, fit.rimOi?.bottomYpx, rOi, offOiX)

        // Ø útil (círculo) + “R usado” (radio dibujado)
        fun drawDiamUtil(p: PointF?, diamMm: Double, roi: Rect, offX: Float) {
            if (p == null) return
            if (!diamMm.isFinite() || diamMm <= 1e-6) return
            if (!pxPerMm.isFinite() || pxPerMm <= 1e-6) return

            val rPx = (0.5 * diamMm * pxPerMm).toFloat().coerceIn(6f, 2000f)
            val cx = mapX(p.x, roi, offX)
            val cy = mapY(p.y, roi)

            c.drawCircle(cx, cy, rPx, paintOrange)

            // radio “R usado” hacia la derecha (sin texto)
            c.drawLine(cx, cy, cx + rPx, cy, paintOrange)
            c.drawCircle(cx + rPx, cy, baseStroke * 1.8f, paintOrange)
        }
        drawDiamUtil(odPupil, metrics.diamUtilOdMm, rOd, offOdX)
        drawDiamUtil(oiPupil, metrics.diamUtilOiMm, rOi, offOiX)

        // Puente (nasal-nasal) con ticks verticales en donde lo toma
        // Recalculamos nasal sobre placed (misma idea que métricas)
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
            sideSign = +1f,
            midX = midRef,
            yRefGlobal = yRef,
            bandHalf = bandHalfPx,
            pupilX = odPupil?.x
        )
        val nasalOiX = nasalFromPlaced(
            placed = fit.placedOiUsed,
            sideSign = -1f,
            midX = midRef,
            yRefGlobal = yRef,
            bandHalf = bandHalfPx,
            pupilX = oiPupil?.x
        )

        if (nasalOdX != null && nasalOiX != null) {
            val xOd = mapX(nasalOdX, rOd, offOdX)
            val xOi = mapX(nasalOiX, rOi, offOiX)

            // línea puente (en yRef)
            c.drawLine(xOd, yOd, xOi, yOi, paintOrange)

            // ticks verticales
            val tick = (22f).coerceIn(14f, 34f)
            c.drawLine(xOd, yOd - tick, xOd, yOd + tick, paintOrange)
            c.drawLine(xOi, yOi - tick, xOi, yOi + tick, paintOrange)
        } else {
            Log.w(TAG, "bridge draw skipped: nasalOdX=$nasalOdX nasalOiX=$nasalOiX")
        }

        // Guardar
        val uriStr = savePngToMediaStore3250(ctx, outBmp, orderId3250)
        try { bmpOd.recycle() } catch (_: Throwable) {}
        try { bmpOi.recycle() } catch (_: Throwable) {}

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
}
