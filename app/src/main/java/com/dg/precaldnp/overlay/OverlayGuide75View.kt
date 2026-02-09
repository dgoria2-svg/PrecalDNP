package com.dg.precaldnp.overlay

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

class OverlayGuide75View @JvmOverloads constructor(
    ctx: Context,
    attrs: AttributeSet? = null
) : View(ctx, attrs) {

    // ——— Estilo ———
    private val p = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        // color se setea en onDraw para asegurar contraste
    }

    private val pShadow = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = 0x88000000.toInt()
        strokeWidth = 2f
    }

    // ——— Almacenamiento en px (coordenadas de DIBUJO en la vista) ———
    private var cxPx: Float = 0f
    private var cyPx: Float = 0f
    private var rPx: Float = 0f
    private var hasCircle = false

    // ——— Buffer de la cámara (para mapear buffer→vista con fitCenter) ———
    private var bufW = 0
    private var bufH = 0

    // ——— Pendientes si llaman antes de que la vista mida ———
    private var pendN: Boolean = false
    private var pendCxN = 0.5f
    private var pendCyN = 0.5f
    private var pendRN = 0.3f

    init {
        // Nos aseguramos de que la vista se marque como "puedo dibujar"
        setWillNotDraw(false)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (!hasCircle) return

        // Aro (borde doble para contraste sobre la imagen de cámara)
        canvas.drawCircle(cxPx, cyPx, rPx + 3f, pShadow)
        p.color = 0xFFFFFFFF.toInt()
        p.strokeWidth = 6f
        canvas.drawCircle(cxPx, cyPx, rPx, p)
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        if (pendN) {
            applyNormalizedToPx(pendCxN, pendCyN, pendRN)
            pendN = false
            postInvalidateOnAnimation()
        }
    }

    /**
     * Establece el círculo en coordenadas NORMALIZADAS DE LA VISTA (0..1).
     * cxN, cyN relativos al ancho/alto actual de la vista.
     * rN relativo al lado menor (diámetro = rN*2*min(w,h)).
     */
    fun setCircleNormalized(cxN: Float, cyN: Float, rN: Float) {
        if (width == 0 || height == 0) {
            // Guardar y aplicar cuando la vista tenga tamaño
            pendN = true
            pendCxN = cxN.coerceIn(0f, 1f)
            pendCyN = cyN.coerceIn(0f, 1f)
            pendRN = rN.coerceAtLeast(0f)
        } else {
            applyNormalizedToPx(cxN, cyN, rN)
            postInvalidateOnAnimation()
        }
        hasCircle = true
    }

    private fun applyNormalizedToPx(cxN: Float, cyN: Float, rN: Float) {
        val w = width - paddingStart - paddingEnd
        val h = height - paddingTop - paddingBottom
        val left = paddingStart.toFloat()
        val top = paddingTop.toFloat()

        val k = min(w, h).toFloat()
        cxPx = left + (cxN.coerceIn(0f, 1f) * w)
        cyPx = top + (cyN.coerceIn(0f, 1f) * h)
        rPx  = (rN.coerceAtLeast(0f) * k)
        hasCircle = true
    }

    /**
     * Configura el tamaño del buffer de cámara (por ej., 1280×720) para mapear buffer→vista.
     * Usá esto si tus coordenadas vienen en PX de BUFFER (ImageAnalysis).
     */
    fun setBufferSize(bufferW: Int, bufferH: Int) {
        bufW = bufferW
        bufH = bufferH
    }

    /**
     * Establece el círculo a partir de coordenadas en PX DE BUFFER (cx, cy, r) usando mapeo FIT_CENTER.
     * Útil si tu analyzer detecta el círculo en el frame y querés dibujarlo alineado al PreviewView.
     */
    fun setCircleFromBuffer(cxBuf: Float, cyBuf: Float, rBuf: Float) {
        if (bufW <= 0 || bufH <= 0 || width == 0 || height == 0) return

        // Mapeo FIT_CENTER: escala uniforme + letterbox centrado
        val vw = width.toFloat()
        val vh = height.toFloat()
        val s = min(vw / bufW, vh / bufH)
        val dw = bufW * s
        val dh = bufH * s
        val offX = (vw - dw) / 2f + paddingStart
        val offY = (vh - dh) / 2f + paddingTop

        cxPx = offX + cxBuf * s
        cyPx = offY + cyBuf * s
        rPx  = rBuf * s
        hasCircle = true
        postInvalidateOnAnimation()
    }

    /**
     * Opcional: limpiar el círculo (por si querés ocultar la guía en algún momento).
     */
    fun clearCircle() {
        hasCircle = false
        postInvalidateOnAnimation()
    }
}
