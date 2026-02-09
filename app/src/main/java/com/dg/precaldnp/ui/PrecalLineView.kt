package com.dg.precaldnp.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.util.AttributeSet
import android.util.TypedValue
import android.view.View
import androidx.core.content.ContextCompat
import androidx.core.graphics.withTranslation
import kotlin.math.min

/**
 * Dibuja el trazo EXACTO (lista de PointF en PX del buffer).
 * Encaje tipo fitCenter sobre la vista.
 *
 * Optimizado para no alocar en onDraw / layout:
 * - Path único que se "rewind" y se reconstruye solo cuando cambian los puntos.
 * - dp→px precalculado en init.
 * - Nada de objetos temporales dentro de onDraw.
 */
class PrecalLineView @JvmOverloads constructor(
    ctx: Context, attrs: AttributeSet? = null
) : View(ctx, attrs) {

    private fun dp(dp: Float) =
        TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, dp, resources.displayMetrics)

    // === Paints reusables
    private val paintLine = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(2f)
        // Colores del framework (android.R) para no depender de recursos propios.
        color = ContextCompat.getColor(ctx, android.R.color.holo_red_light)
        strokeJoin = Paint.Join.ROUND
        strokeCap  = Paint.Cap.ROUND
    }
    private val paintCenter = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = ContextCompat.getColor(ctx, android.R.color.holo_green_dark)
    }

    // === Datos del buffer
    private var bufW = 0
    private var bufH = 0

    // === Datos del trazo + centro opcional
    private var pts: List<PointF> = emptyList()
    private var center: PointF? = null
    private var closePath = true

    // === Path cacheado
    private val path: Path = Path()
    private var hasPath = false

    // === Tamaños precalculados
    private val centerRadiusPx: Float = dp(3f)

    /** Cambia el grosor sin DP allocations en draw */
    fun setStrokeDp(dpWidth: Float) {
        paintLine.strokeWidth = dp(dpWidth)
        postInvalidateOnAnimation()
    }

    fun setLineColor(colorInt: Int) {
        paintLine.color = colorInt
        postInvalidateOnAnimation()
    }

    fun setBufferSize(bufferWidth: Int, bufferHeight: Int) {
        val newW = if (bufferWidth  > 0) bufferWidth  else 1
        val newH = if (bufferHeight > 0) bufferHeight else 1
        if (newW != bufW || newH != bufH) {
            bufW = newW
            bufH = newH
            // La transformación se hace con translate/scale en draw.
            postInvalidateOnAnimation()
        }
    }

    /**
     * Versión NUEVA (sin centro): puntos + si cerrar el polígono.
     * Útil para llamadas como setTracePx(points, true)
     */
    fun setTracePx(points: List<PointF>, close: Boolean) {
        setTraceInternal(points, centerPx = null, close = close)
    }

    /**
     * Versión LEGACY (compat): puntos + centro opcional + cerrar polígono.
     */
    fun setTracePx(points: List<PointF>, centerPx: PointF?, close: Boolean = true) {
        setTraceInternal(points, centerPx, close)
    }

    private fun setTraceInternal(points: List<PointF>, centerPx: PointF?, close: Boolean) {
        pts = points
        center = centerPx
        closePath = close

        path.rewind()
        if (pts.isNotEmpty()) {
            val p0 = pts[0]
            path.moveTo(p0.x, p0.y)
            for (i in 1 until pts.size) {
                val p = pts[i]
                path.lineTo(p.x, p.y)
            }
            if (closePath && pts.size >= 3) path.close()
            hasPath = true
        } else {
            hasPath = false
        }
        postInvalidateOnAnimation()
    }

    fun clear() {
        pts = emptyList()
        center = null
        closePath = true
        path.rewind()
        hasPath = false
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (!hasPath || bufW <= 0 || bufH <= 0) return

        // FitCenter del buffer (bufW x bufH) dentro del canvas
        val vw = width.toFloat()
        val vh = height.toFloat()
        val scale = min(vw / bufW, vh / bufH)
        val rw = bufW * scale
        val rh = bufH * scale
        val ox = (vw - rw) * 0.5f
        val oy = (vh - rh) * 0.5f

        canvas.withTranslation(ox, oy) {
            scale(scale, scale)
            // Trazo
            drawPath(path, paintLine)
            // Centro (si hay)
            center?.let { c ->
                drawCircle(c.x, c.y, centerRadiusPx, paintCenter)
            }
        }
    }
}
