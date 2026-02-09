package com.dg.precaldnp.vision

import android.graphics.PointF

/**
 * Construye polígonos GLOBAL para enmascarar zonas "enemigas":
 *  - ojo izquierdo (rect)
 *  - ojo derecho (rect)
 *  - ceja izquierda (rect banda)
 *  - ceja derecha (rect banda)
 *
 * La idea: no hace falta precisión milimétrica, solo bloquear borde de ceja/pestaña/iris.
 */
object MaskPolysFromIrisPack3250 {

    data class Params(
        val eyeHalfW: Float = 80f,
        val eyeHalfH: Float = 45f,
        val browBandH: Float = 55f,
        val browPadDown: Float = 8f
    )

    fun buildMaskPolysGlobal3250(
        irisLeftPx: PointF?,
        irisRightPx: PointF?,
        leftBrowBottomYpx: Float?,
        rightBrowBottomYpx: Float?,
        imgW: Int,
        imgH: Int,
        p: Params = Params()
    ): List<List<PointF>> {

        val polys = ArrayList<List<PointF>>(4)

        fun clampX(x: Float) = x.coerceIn(0f, (imgW - 1).toFloat())
        fun clampY(y: Float) = y.coerceIn(0f, (imgH - 1).toFloat())

        fun rectPoly(cx: Float, cy: Float, halfW: Float, halfH: Float): List<PointF> {
            val x0 = clampX(cx - halfW)
            val x1 = clampX(cx + halfW)
            val y0 = clampY(cy - halfH)
            val y1 = clampY(cy + halfH)
            return listOf(
                PointF(x0, y0),
                PointF(x1, y0),
                PointF(x1, y1),
                PointF(x0, y1)
            )
        }

        // OJOS (bloquea iris+párpados+pestañas)
        if (irisLeftPx != null) {
            polys += rectPoly(irisLeftPx.x, irisLeftPx.y, p.eyeHalfW, p.eyeHalfH)
        }
        if (irisRightPx != null) {
            polys += rectPoly(irisRightPx.x, irisRightPx.y, p.eyeHalfW, p.eyeHalfH)
        }

        // CEJAS (banda por arriba del browBottom)
        // La ceja real suele estar arriba del "bottom"; acá bloqueamos una franja encima.
        fun browBand(iris: PointF?, browBottomY: Float?): List<PointF>? {
            if (iris == null || browBottomY == null) return null
            val yBottom = clampY(browBottomY + p.browPadDown)
            val yTop = clampY(yBottom - p.browBandH)
            val yMid = (yTop + yBottom) * 0.5f
            // misma X que iris, ancho un poco mayor que ojo
            return rectPoly(iris.x, yMid, p.eyeHalfW * 1.15f, (yBottom - yTop) * 0.5f)
        }

        browBand(irisLeftPx, leftBrowBottomYpx)?.let { polys += it }
        browBand(irisRightPx, rightBrowBottomYpx)?.let { polys += it }

        return polys
    }
}
