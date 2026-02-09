package com.dg.precaldnp.ui

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.util.AttributeSet
import android.view.View
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

class CompareLineView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    // Curva izquierda (outline de la cámara, en mm)
    private var leftShape: List<PointF> = emptyList()

    // Curva derecha (forma reconstruida desde los radios del FIL, en mm)
    private var rightShape: List<PointF> = emptyList()

    // Rotación SOLO para la curva derecha (en grados)
    private var rotationDeg: Float = 0f

    private val paintLeft = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 2f * resources.displayMetrics.density
    }

    private val paintRight = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.BLUE
        style = Paint.Style.STROKE
        strokeWidth = 2f * resources.displayMetrics.density
    }
    private val paintAxis = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 1.5f * resources.displayMetrics.density
    }

    /**
     * Setea las dos curvas a comparar.
     * Ambas listas están en "mm" (o unidades de trabajo), NO en píxeles de pantalla.
     */
    fun setTraces(leftPx: List<PointF>, rightPx: List<PointF>) {
        leftShape = leftPx
        rightShape = rightPx
        invalidate()
    }

    /**
     * Rotación (en grados) de la curva derecha (FIL).
     * Esto es lo que te pedía el compilador: setRotationDegrees(...)
     */
    fun setRotationDegrees(degrees: Float) {
        rotationDeg = degrees
        invalidate()
    }

    @SuppressLint("DrawAllocation")
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (leftShape.size < 2 || rightShape.size < 2) return

        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0f || h <= 0f) return

        val halfW = w / 2f
        val margin = 16f * resources.displayMetrics.density

        // ===== 1) Bounding boxes de cada forma (en su sistema propio, mm) =====
        val (lMinX, lMaxX, lMinY, lMaxY) = boundsOf(leftShape)
        val (rMinX, rMaxX, rMinY, rMaxY) = boundsOf(rightShape)

        val lW = lMaxX - lMinX
        val lH = lMaxY - lMinY
        val rW = rMaxX - rMinX
        val rH = rMaxY - rMinY

        if (lW <= 0f || lH <= 0f || rW <= 0f || rH <= 0f) return

        // ===== 2) Escalas para que entren en su mitad y respeten aspecto =====
        val leftScale = 0.9f * min((halfW - 2 * margin) / lW, (h - 2 * margin) / lH)
        val rightScale = 0.9f * min((halfW - 2 * margin) / rW, (h - 2 * margin) / rH)

        // Centros de cada mitad en pantalla
        val leftCenterX = halfW / 2f
        val rightCenterX = halfW + halfW / 2f
        val centerY = h / 2f

        // Centros geométricos (en mm)
        val lCx = (lMinX + lMaxX) / 2f
        val lCy = (lMinY + lMaxY) / 2f
        val rCx = (rMinX + rMaxX) / 2f
        val rCy = (rMinY + rMaxY) / 2f

        // ===== 3) Path izquierda (sin rotación, rojo) =====
        val pathLeft = Path()
        var first = true
        for (p in leftShape) {
            val nx = (p.x - lCx) * leftScale
            val ny = (p.y - lCy) * leftScale

            val sx = leftCenterX + nx
            val sy = centerY - ny // invertimos Y para coord. de pantalla

            if (first) {
                pathLeft.moveTo(sx, sy)
                first = false
            } else {
                pathLeft.lineTo(sx, sy)
            }
        }
        // Cerramos por las dudas
        if (!leftShape.isEmpty()) {
            val p0 = leftShape[0]
            val nx0 = (p0.x - lCx) * leftScale
            val ny0 = (p0.y - lCy) * leftScale
            pathLeft.lineTo(leftCenterX + nx0, centerY - ny0)
        }

        canvas.drawPath(pathLeft, paintLeft)

        // ===== 4) Path derecha (con rotación, azul) =====
        val rad = Math.toRadians(rotationDeg.toDouble()).toFloat()
        val cosA = cos(rad)
        val sinA = sin(rad)

        val pathRight = Path()
        first = true
        for (p in rightShape) {
            // centramos
            val dx = p.x - rCx
            val dy = p.y - rCy

            // rotación 2D
            val rx = dx * cosA - dy * sinA
            val ry = dx * sinA + dy * cosA

            val nx = rx * rightScale
            val ny = ry * rightScale

            val sx = rightCenterX + nx
            val sy = centerY - ny

            if (first) {
                pathRight.moveTo(sx, sy)
                first = false
            } else {
                pathRight.lineTo(sx, sy)
            }
        }
        // cerrar
        if (!rightShape.isEmpty()) {
            val p0 = rightShape[0]
            val dx0 = p0.x - rCx
            val dy0 = p0.y - rCy
            val rx0 = dx0 * cosA - dy0 * sinA
            val ry0 = dx0 * sinA + dy0 * cosA
            val nx0 = rx0 * rightScale
            val ny0 = ry0 * rightScale
            pathRight.lineTo(rightCenterX + nx0, centerY - ny0)
        }
// ===== 5) Raya horizontal guía (solo lado derecho, “eje de grados”) =====

// Tomamos un ancho basado en el tamaño de la lente en pantalla
        val lensHalfWidthPx = 0.5f * min(rW, rH) * rightScale
        val axisHalfWidth = lensHalfWidthPx * 0.9f

        val axisStartX = rightCenterX - axisHalfWidth
        val axisEndX   = rightCenterX + axisHalfWidth

        canvas.drawLine(axisStartX, centerY, axisEndX, centerY, paintAxis)
        canvas.drawPath(pathRight, paintRight)
    }

    private data class Bounds(
        val minX: Float,
        val maxX: Float,
        val minY: Float,
        val maxY: Float
    )

    private fun boundsOf(pts: List<PointF>): Bounds {
        var minX = Float.POSITIVE_INFINITY
        var maxX = Float.NEGATIVE_INFINITY
        var minY = Float.POSITIVE_INFINITY
        var maxY = Float.NEGATIVE_INFINITY
        for (p in pts) {
            if (p.x < minX) minX = p.x
            if (p.x > maxX) maxX = p.x
            if (p.y < minY) minY = p.y
            if (p.y > maxY) maxY = p.y
        }
        return Bounds(minX, maxX, minY, maxY)
    }
}
