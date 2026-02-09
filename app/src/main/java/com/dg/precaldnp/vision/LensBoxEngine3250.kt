// app/src/main/java/com/dg/precaldnp/vision/LensBoxEngine3250.kt
package com.dg.precaldnp.vision

import android.graphics.PointF
import android.util.Log
import org.opencv.core.Point
import org.opencv.core.Rect
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max

/**
 * LensBoxEngine3250:
 * - Recibe pupilas (px) + boxes (px) + escala px/mm.
 * - Usa HBOX/VBOX DEL FIL (mm) SOLO para ancho/alto/diag (no sale de la caja).
 * - Calcula: DNP OD/OI respecto a midline (recomendado: centro del bitmap),
 *           alturas (pupila->borde inferior),
 *           puente (gap entre boxes),
 *           diámetro útil aprox (por corners de la box) mientras no haya inner-rim real.
 *
 * Convención IMPORTANTE:
 * - En cámara trasera apuntando a una persona, OD (ojo derecho del paciente) aparece a la IZQUIERDA en la imagen.
 * - Por eso, acá “OD” lo tomamos como el lado de menor X en la imagen.
 */
class LensBoxEngine3250 {

    companion object {
        private const val TAG = "LensBoxEngine3250"
    }

    data class Pupils3250(
        val left: PointF,   // pupila más a la izquierda en la imagen (OD típico)
        val right: PointF   // pupila más a la derecha en la imagen (OI típico)
    )

    data class LensBoxes3250(
        val leftBox: Rect,   // box más a la izquierda (OD típico)
        val rightBox: Rect   // box más a la derecha (OI típico)
    )

    data class DnpMetrics3250(
        val anchoMm: Double,       // HBOX del FIL
        val altoMm: Double,        // VBOX del FIL
        val diagMayorMm: Double,   // hypot(HBOX/2, VBOX/2) (tu “Diag” actual)

        val puenteMm: Double,      // gap entre boxes (>=0)

        val dnpOdMm: Double,       // OD desde midline
        val dnpOiMm: Double,       // OI desde midline
        val dnpTotalMm: Double,    // distancia pupila-pupila (horizontal)

        val altOdMm: Double,       // pupila->borde inferior OD
        val altOiMm: Double,       // pupila->borde inferior OI

        val diamUtilOdMm: Double,  // aprox (2 * radio útil por corners)
        val diamUtilOiMm: Double,  // aprox

        val pxPerMmUsed3250: Double
    )

    /**
     * API alineada a tu llamado actual (named args).
     */
    fun computeDnpAndUsefulRadii(
        pupils: Pupils3250,
        boxes: LensBoxes3250,
        pxPerMmUsed: Double,
        filHboxMm: Double,
        filVboxMm: Double,
        midlineXOverridePx: Double? = null
    ): DnpMetrics3250? {
        if (pxPerMmUsed <= 0.0) return null
        if (filHboxMm <= 0.0 || filVboxMm <= 0.0) return null

        // Enforzamos orden por X (izq->der) para pupilas y boxes
        val (pL, pR) =
            if (pupils.left.x <= pupils.right.x) pupils.left to pupils.right
            else pupils.right to pupils.left

        val (bL, bR) =
            if (boxes.leftBox.x <= boxes.rightBox.x) boxes.leftBox to boxes.rightBox
            else boxes.rightBox to boxes.leftBox

        val midX = midlineXOverridePx ?: ((pL.x + pR.x) / 2.0)

        // DNP total horizontal
        val dnpTotalMm = abs(pR.x - pL.x) / pxPerMmUsed

        // Monoculares desde midline
        val dnpOdMm = abs(midX - pL.x) / pxPerMmUsed
        val dnpOiMm = abs(pR.x - midX) / pxPerMmUsed

        // Alturas: pupila -> borde inferior de su box
        val altOdMm = ((bL.y + bL.height).toDouble() - pL.y) / pxPerMmUsed
        val altOiMm = ((bR.y + bR.height).toDouble() - pR.y) / pxPerMmUsed

        // Puente: gap entre boxes (interno) en px -> mm
        val gapPx = (bR.x - (bL.x + bL.width)).toDouble()
        val puenteMm = max(0.0, gapPx / pxPerMmUsed)

        fun usefulRadiusMm(p: PointF, box: Rect): Double {
            val corners = listOf(
                Point(box.x.toDouble(), box.y.toDouble()),
                Point((box.x + box.width).toDouble(), box.y.toDouble()),
                Point(box.x.toDouble(), (box.y + box.height).toDouble()),
                Point((box.x + box.width).toDouble(), (box.y + box.height).toDouble())
            )
            val maxDistPx = corners.maxOf { c -> hypot(c.x - p.x, c.y - p.y) }
            return maxDistPx / pxPerMmUsed
        }

        val radOd = usefulRadiusMm(pL, bL)
        val radOi = usefulRadiusMm(pR, bR)
        val diamOd = 2.0 * radOd
        val diamOi = 2.0 * radOi

        // Tu "Diag" actual: hypot(a,b) con a=HBOX/2, b=VBOX/2
        val a = filHboxMm * 0.5
        val b = filVboxMm * 0.5
        val diag = hypot(a, b)

        Log.d(TAG, "FIL h=$filHboxMm v=$filVboxMm diag=$diag | puente=$puenteMm | OD=$dnpOdMm OI=$dnpOiMm tot=$dnpTotalMm")

        return DnpMetrics3250(
            anchoMm = filHboxMm,
            altoMm = filVboxMm,
            diagMayorMm = diag,
            puenteMm = puenteMm,
            dnpOdMm = dnpOdMm,
            dnpOiMm = dnpOiMm,
            dnpTotalMm = dnpTotalMm,
            altOdMm = altOdMm,
            altOiMm = altOiMm,
            diamUtilOdMm = diamOd,
            diamUtilOiMm = diamOi,
            pxPerMmUsed3250 = pxPerMmUsed
        )
    }
}
