// app/src/main/java/com/dg/precaldnp/vision/FilFitToRing.kt
package com.dg.precaldnp.vision

import android.graphics.Matrix
import android.graphics.PointF
import kotlin.math.min

object FilFitToRing {

    data class RingInner(
        val centerPx: PointF,
        val rxPx: Float,   // semieje X (px) del borde interior
        val ryPx: Float    // semieje Y (px) del borde interior
    )

    /**
     * Devuelve una Matrix que mapea (xMm,yMm cartesiano del FIL) -> (xPx,yPx pantalla).
     * - Escala uniforme para entrar en la elipse interior del aro
     * - Invierte Y (Y-up -> Y-down)
     * - Opcional: espejo en X (mirrorX)
     * - Traslada al centro del aro
     */
    fun fit(
        contourMaxAbsXmm: Double,
        contourMaxAbsYmm: Double,
        ring: RingInner,
        mirrorX: Boolean = false
    ): Matrix {
        val safeX = if (contourMaxAbsXmm > 1e-9) contourMaxAbsXmm else 1e-9
        val safeY = if (contourMaxAbsYmm > 1e-9) contourMaxAbsYmm else 1e-9

        val scaleX = ring.rxPx.toDouble() / safeX
        val scaleY = ring.ryPx.toDouble() / safeY
        val s = min(scaleX, scaleY).toFloat()

        val sx = if (mirrorX) - s else s

        return Matrix().apply {
            postScale(sx, s) // mirrorX opcional + Y invertida
            postTranslate(ring.centerPx.x, ring.centerPx.y)
        }
    }
}
