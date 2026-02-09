package com.dg.precaldnp.overlay

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import kotlin.math.*
import com.dg.precaldnp.model.DnpShotHint3250

/**
 * Overlay FIL para DNP (3250):
 * - Dibuja contorno OD (izq) + OI (der) a partir de 800 radios en mm.
 * - NO dibuja marcas R200/R400/R600 (se sacaron).
 * - Une R1 (index 0) OD y OI con una línea roja (puente).
 * - Escala para dejar margen lateral (~15%) => entran orejas.
 * - Expone lens boxes en px para ROIs: getLensBoxRectPx / getLensBoxRectPxOi.
 */
class DnpFilOverlayFrame3250 @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    // ====== Ajustes ======
    private val fitMarginFrac3250 = 0.38f          // 15% por lado
    private val gapFracOfBoxW3250 = 0.3f          // gap entre lentes (relativo a boxW)
    private val boxPadFrac3250 = 0.08f             // padding ROI (8%)
    private var centerYFrac3250 = 0.45f            // ojo un poco arriba del centro

    // Si querés usar midline real (Iris), setearlo desde afuera (si no, usa w/2)
    private var midlineXOverridePx3250: Float? = null

    // ====== Datos FIL ======
    private var radiiMm3250: DoubleArray? = null
    private var odContourMm3250: ContourMm3250? = null
    private var oiContourMm3250: ContourMm3250? = null

    // ====== Layout calculado ======
    private var layoutDirty3250 = true
    private var scalePxPerMm3250 = 1f
    private val odBoxPx3250 = RectF()
    private val oiBoxPx3250 = RectF()
    private val odCenterPx3250 = PointF()
    private val oiCenterPx3250 = PointF()
    private var odR1Mm3250 = PointF()
    private var oiR1Mm3250 = PointF()

    // ====== Paints ======
    private val contourPaint3250 = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(3f)
        color = Color.rgb(0, 255, 0) // verde
        strokeJoin = Paint.Join.ROUND
        strokeCap = Paint.Cap.ROUND
    }

    private val midlinePaint3250 = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(2f)
        color = Color.argb(170, 255, 255, 255) // blanco tenue
    }

    private val bridgePaint3250 = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(2f)
        color = Color.RED
        strokeCap = Paint.Cap.ROUND
    }

    // ====== API pública ======

    /** Opcional: si querés que el midline sea el del iris engine (px en el overlay). */
    fun setMidlineXpx3250(xPx: Float?) {
        midlineXOverridePx3250 = xPx
        layoutDirty3250 = true
        postInvalidateOnAnimation()
    }

    /** Opcional: si querés ajustar la altura del overlay. */
    fun setCenterYFrac3250(frac: Float) {
        centerYFrac3250 = frac.coerceIn(0.20f, 0.70f)
        layoutDirty3250 = true
        postInvalidateOnAnimation()
    }

    /**
     * Set FIL radii (800) en mm.
     * shiftSteps3250: corrimiento circular de índices (mismo sentido que tu rotateR de Precal).
     * - Para “AS-IS” (lo que está escrito en el FIL): shiftSteps3250 = 0.
     */
    fun setFilRadiiMm3250(radiiMm: DoubleArray, shiftSteps3250: Int = 0) {
        radiiMm3250 = if (shiftSteps3250 == 0) {
            radiiMm.copyOf()
        } else {
            rotateRStepsLikePrecal3250(radiiMm, shiftSteps3250)
        }

        val r = radiiMm3250 ?: return
        odContourMm3250 = buildContourMm3250(r, mirrorX = false)
        oiContourMm3250 = buildContourMm3250(r, mirrorX = true)

        // R1 (index 0) de ambos (para unir con línea roja)
        odR1Mm3250 = odContourMm3250?.ptsMm?.firstOrNull() ?: PointF()
        oiR1Mm3250 = oiContourMm3250?.ptsMm?.firstOrNull() ?: PointF()

        layoutDirty3250 = true
        postInvalidateOnAnimation()
    }

    fun getLensBoxRectPx(): Rect? {
        if (odBoxPx3250.isEmpty) return null
        return paddedRectInt3250(odBoxPx3250)
    }

    fun getLensBoxRectPxOi(): Rect? {
        if (oiBoxPx3250.isEmpty) return null
        return paddedRectInt3250(oiBoxPx3250)
    }

    // ====== View ======

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        layoutDirty3250 = true
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val od = odContourMm3250 ?: return
        val oi = oiContourMm3250 ?: return
        if (width <= 0 || height <= 0) return

        ensureLayout3250(od, oi)

        // Midline
        val midX = midlineXOverridePx3250 ?: (width * 0.5f)
        canvas.drawLine(midX, 0f, midX, height.toFloat(), midlinePaint3250)

        // Contornos
        canvas.drawPath(buildPathFromContour3250(od.ptsMm, odCenterPx3250), contourPaint3250)
        canvas.drawPath(buildPathFromContour3250(oi.ptsMm, oiCenterPx3250), contourPaint3250)

        // Línea roja uniendo R1 OD y R1 OI (puente)
        val pOd = mapMmToPx3250(odR1Mm3250, odCenterPx3250)
        val pOi = mapMmToPx3250(oiR1Mm3250, oiCenterPx3250)
        canvas.drawLine(pOd.x, pOd.y, pOi.x, pOi.y, bridgePaint3250)
    }

    // ====== Internals ======

    private data class ContourMm3250(
        val ptsMm: List<PointF>,
        val hboxMm: Float,
        val vboxMm: Float
    )

    private fun buildContourMm3250(radiiMm: DoubleArray, mirrorX: Boolean): ContourMm3250 {
        val n = radiiMm.size.coerceAtLeast(1)
        val twoPi = (2.0 * Math.PI)
        val dTh = twoPi / n

        val pts = ArrayList<PointF>(n + 1)
        var minX = 1e9
        var maxX = -1e9
        var minY = 1e9
        var maxY = -1e9

        for (i in 0 until n) {
            val r = radiiMm[i].toFloat()
            val a = (dTh * i)
            var x = (r * cos(a)).toFloat()   // R1 (i=0) => +X
            val y = (r * sin(a)).toFloat()   // R200 => +Y (arriba en mm)
            if (mirrorX) x = -x

            pts.add(PointF(x, y))
            minX = min(minX, x.toDouble()); maxX = max(maxX, x.toDouble())
            minY = min(minY, y.toDouble()); maxY = max(maxY, y.toDouble())
        }

        // Centrar por bounding box (evita deriva por ruido)
        val cx = ((minX + maxX) * 0.5).toFloat()
        val cy = ((minY + maxY) * 0.5).toFloat()
        for (i in pts.indices) {
            val p = pts[i]
            pts[i] = PointF(p.x - cx, p.y - cy)
        }

        // Recalcular bounds centrados
        minX = 1e9; maxX = -1e9; minY = 1e9; maxY = -1e9
        for (p in pts) {
            minX = min(minX, p.x.toDouble()); maxX = max(maxX, p.x.toDouble())
            minY = min(minY, p.y.toDouble()); maxY = max(maxY, p.y.toDouble())
        }

        val hboxMm = (maxX - minX).toFloat()
        val vboxMm = (maxY - minY).toFloat()

        // cerrar
        if (pts.isNotEmpty()) pts.add(PointF(pts[0].x, pts[0].y))

        return ContourMm3250(pts, hboxMm, vboxMm)
    }

    private fun ensureLayout3250(od: ContourMm3250, oi: ContourMm3250) {
        if (!layoutDirty3250) return

        val w = width.toFloat()
        val h = height.toFloat()

        val marginX = w * fitMarginFrac3250
        val marginY = h * fitMarginFrac3250
        val availW = (w - 2f * marginX).coerceAtLeast(1f)
        val availH = (h - 2f * marginY).coerceAtLeast(1f)

        val hboxMm = od.hboxMm.coerceAtLeast(1e-3f)
        val vboxMm = od.vboxMm.coerceAtLeast(1e-3f)

        // Queremos que entren DOS cajas + gap, dentro del ancho disponible
        val totalWFactor = 2f + gapFracOfBoxW3250
        val scaleFromW = availW / (hboxMm * totalWFactor)
        val scaleFromH = availH / vboxMm
        scalePxPerMm3250 = min(scaleFromW, scaleFromH).coerceAtLeast(0.1f)

        val boxW = hboxMm * scalePxPerMm3250
        val boxH = vboxMm * scalePxPerMm3250
        val gapPx = boxW * gapFracOfBoxW3250

        val midX = midlineXOverridePx3250 ?: (w * 0.5f)
        var cy = h * centerYFrac3250

        // Clamp vertical para que no se vaya de pantalla
        val minCy = marginY + boxH * 0.5f
        val maxCy = h - marginY - boxH * 0.5f
        cy = cy.coerceIn(minCy, maxCy)

        val cxLeft = midX - (gapPx * 0.5f + boxW * 0.5f)
        val cxRight = midX + (gapPx * 0.5f + boxW * 0.5f)

        odCenterPx3250.set(cxLeft, cy)
        oiCenterPx3250.set(cxRight, cy)

        odBoxPx3250.set(
            cxLeft - boxW * 0.5f,
            cy - boxH * 0.5f,
            cxLeft + boxW * 0.5f,
            cy + boxH * 0.5f
        )
        oiBoxPx3250.set(
            cxRight - boxW * 0.5f,
            cy - boxH * 0.5f,
            cxRight + boxW * 0.5f,
            cy + boxH * 0.5f
        )

        layoutDirty3250 = false
    }

    private fun buildPathFromContour3250(ptsMm: List<PointF>, centerPx: PointF): Path {
        val path = Path()
        if (ptsMm.isEmpty()) return path

        val first = mapMmToPx3250(ptsMm[0], centerPx)
        path.moveTo(first.x, first.y)

        for (i in 1 until ptsMm.size) {
            val p = mapMmToPx3250(ptsMm[i], centerPx)
            path.lineTo(p.x, p.y)
        }
        return path
    }

    private fun mapMmToPx3250(pMm: PointF, centerPx: PointF): PointF {
        // mm tiene Y hacia arriba; Canvas tiene Y hacia abajo => invertimos Y
        val x = centerPx.x + pMm.x * scalePxPerMm3250
        val y = centerPx.y - pMm.y * scalePxPerMm3250
        return PointF(x, y)
    }

    private fun paddedRectInt3250(src: RectF): Rect {
        val padX = src.width() * boxPadFrac3250
        val padY = src.height() * boxPadFrac3250
        val l = (src.left - padX).roundToInt()
        val t = (src.top - padY).roundToInt()
        val r = (src.right + padX).roundToInt()
        val b = (src.bottom + padY).roundToInt()
        return Rect(l, t, r, b)
    }

    private fun rotateRStepsLikePrecal3250(src: DoubleArray, shiftSteps: Int): DoubleArray {
        val n = src.size
        if (n == 0) return src
        val shift = ((shiftSteps % n) + n) % n
        if (shift == 0) return src.copyOf()

        // Mismo sentido que tu rotateR() en Precal: out[i] = src[(i - shift + n) % n]
        val out = DoubleArray(n)
        for (i in 0 until n) out[i] = src[(i - shift + n) % n]
        return out
    }

    private fun dp(v: Float): Float = v * resources.displayMetrics.density


}