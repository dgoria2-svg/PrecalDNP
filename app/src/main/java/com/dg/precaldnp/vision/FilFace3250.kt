package com.dg.precaldnp.vision

/**
 * Información del FIL en mm y relación con la cara.
 *
 * hboxMm / vboxMm / eyesizMm vienen del trazador mecánico (.FIL).
 * pxPerMmFace es la escala de cara ya resuelta por tu pipeline 3250.
 */
data class FilFace3250(
    val hboxMm: Double,
    val vboxMm: Double,
    val eyesizMm: Double,
    val pxPerMmFace: Double
) {
    init {
        require(hboxMm > 0.0) { "hboxMm debe ser > 0" }
        require(vboxMm > 0.0) { "vboxMm debe ser > 0" }
        require(pxPerMmFace > 0.0) { "pxPerMmFace debe ser > 0" }
    }

    /** Relación de aspecto HBOX/VBOX (útil para ajustar cajas). */
    val aspect: Double get() = hboxMm / vboxMm

    /** Ancho del FIL sobre la cara, en píxeles (usando pxPerMmFace). */
    val hboxPx: Double get() = hboxMm * pxPerMmFace

    /** Alto del FIL sobre la cara, en píxeles (usando pxPerMmFace). */
    val vboxPx: Double get() = vboxMm * pxPerMmFace
}
