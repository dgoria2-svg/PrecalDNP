// app/src/main/java/com/dg/precaldnp/model/FilFace3250.kt
package com.dg.precaldnp.model

import android.graphics.PointF
import android.graphics.RectF
import kotlin.math.abs

/**
 * FIL-base: lo que sale del trazador / Precal, siempre en mm.
 *
 * Ojo: esto NO toca el .FIL mecánico real ni los R=.
 * Es sólo un resumen geométrico que ya tenés de antes
 * (HBOX/VBOX, ojo derecho/izquierdo, etc.).
 */
data class FilBase3250(
    val hboxMm: Double,
    val vboxMm: Double,
    val isOd: Boolean,     // true = ojo derecho, false = ojo izquierdo
)

/**
 * Versión "face" del FIL:
 *
 * - pxPerMmFace: escala oficial de la CARA para esta foto.
 * - hboxMm/vboxMm: iguales al FIL (no se tocan).
 * - boxOdPx/boxOiPx: cajas OD/OI en píxeles en la foto de cara.
 * - pupilOdPx/pupilOiPx: pupilas OD/OI en píxeles (ya validadas).
 * - midlineFaceXpx: línea media oficial (nariz+boca) en píxeles.
 *
 * Además guardamos una QA opcional:
 * - hboxMmFromFace / vboxMmFromFace: lo que "mediríamos" de HBOX/VBOX
 *   si rearmamos mm desde las cajas + pxPerMmFace.
 * - diffHboxPct / diffVboxPct: diferencia % entre lo del FIL y lo que
 *   se ve en la cara (sólo para avisos, nunca para bloquear medición).
 */
data class FilFace3250(
    // Escala oficial de la cara
    val pxPerMmFace: Double,

    // Geometría del FIL en mm (verdad mecánica)
    val hboxMm: Double,
    val vboxMm: Double,

    // Cajas en la foto (OD = izquierda imagen, OI = derecha imagen)
    val boxOdPx: RectF,
    val boxOiPx: RectF,

    // Pupilas en la foto
    val pupilOdPx: PointF?,
    val pupilOiPx: PointF?,

    // Midline oficial de la cara (nariz + boca)
    val midlineFaceXpx: Float,

    // QA opcional: cómo se ve HBOX/VBOX en la cara con esta escala
    val hboxMmFromFace: Double,
    val vboxMmFromFace: Double,
    val diffHboxPct: Double,
    val diffVboxPct: Double,
)

/**
 * Utilidades para construir el FIL "face" a partir de:
 * - FIL mecánico (HBOX/VBOX en mm).
 * - Escala de cara pxPerMmFace.
 * - Cajas OD/OI en la foto.
 * - Pupilas y midline en la foto.
 *
 * NO modifica el .FIL original, ni los R=, ni nada mecánico.
 * Sólo crea un snapshot consistente para esta foto de cara.
 */
object FilFaceUtils3250 {

    /**
     * Construye un FilFace3250 coherente:
     *
     * @param fil    FIL-base (mm) de este ojo (HBOX/VBOX, OD/OI).
     * @param pxPerMmFace escala oficial de la cara (px/mm).
     * @param boxOd caja OD en la foto (izquierda imagen, en px).
     * @param boxOi caja OI en la foto (derecha imagen, en px).
     * @param pupilOd pupila OD en la foto (puede ser null si no hay).
     * @param pupilOi pupila OI en la foto (puede ser null si no hay).
     * @param midlineFaceXpx midline de la cara (nariz+boca) en px.
     */
    fun buildFilFace3250(
        fil: FilBase3250,
        pxPerMmFace: Double,
        boxOd: RectF,
        boxOi: RectF,
        pupilOd: PointF?,
        pupilOi: PointF?,
        midlineFaceXpx: Float
    ): FilFace3250 {
        require(pxPerMmFace > 0.0) {
            "pxPerMmFace debe ser > 0 (escala de cara inválida)"
        }
        require(fil.hboxMm > 0.0 && fil.vboxMm > 0.0) {
            "HBOX/VBOX del FIL deben ser > 0"
        }

        // HBOX/VBOX "vistos" desde la cara con esta escala:
        // usamos el promedio de las dos cajas para suavizar.
        val avgBoxWidthPx = ((boxOd.width() + boxOi.width()) * 0.5).toDouble()
        val avgBoxHeightPx = ((boxOd.height() + boxOi.height()) * 0.5).toDouble()

        val hboxMmFromFace = avgBoxWidthPx / pxPerMmFace
        val vboxMmFromFace = avgBoxHeightPx / pxPerMmFace

        // Diferencias porcentuales vs FIL mecánico (solo QA, no bloquean nada)
        val diffHboxPct = percentDiff(fil.hboxMm, hboxMmFromFace)
        val diffVboxPct = percentDiff(fil.vboxMm, vboxMmFromFace)

        return FilFace3250(
            pxPerMmFace = pxPerMmFace,
            hboxMm = fil.hboxMm,
            vboxMm = fil.vboxMm,
            boxOdPx = RectF(boxOd),
            boxOiPx = RectF(boxOi),
            pupilOdPx = pupilOd?.let { PointF(it.x, it.y) },
            pupilOiPx = pupilOi?.let { PointF(it.x, it.y) },
            midlineFaceXpx = midlineFaceXpx,
            hboxMmFromFace = hboxMmFromFace,
            vboxMmFromFace = vboxMmFromFace,
            diffHboxPct = diffHboxPct,
            diffVboxPct = diffVboxPct
        )
    }

    private fun percentDiff(refMm: Double, fromFaceMm: Double): Double {
        if (refMm <= 0.0) return 0.0
        return abs(fromFaceMm / refMm - 1.0) * 100.0
    }
}
