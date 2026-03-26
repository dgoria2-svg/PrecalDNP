package com.dg.precaldnp.model

import android.os.Parcelable
import kotlinx.parcelize.IgnoredOnParcel
import kotlinx.parcelize.Parcelize

@Parcelize
data class MeasurementResult(
    val annotatedPath: String,
    val originalPath: String,

    // Métricas que muestra ResultsActivity
    val anchoMm: Double,
    val altoMm: Double,
    val diagMayorMm: Double, // en este flujo: DIAG = FED

    // Diámetros útiles por ojo
    val diamUtilOdMm: Double,
    val diamUtilOiMm: Double,

    // DNP por ojo
    val dnpOdMm: Double,
    val dnpOiMm: Double,

    // Alturas pupila→borde inferior
    val altOdMm: Double,
    val altOiMm: Double,

    // Puente
    val puenteMm: Double,

    // ---------------- 3250 extras ----------------
    // Escala FINAL usada para el render (mm -> px)
    val pxPerMmFace3250: Double = Double.NaN,

    // Métricas del FIL
    val filHboxMm3250: Double = Double.NaN,
    val filVboxMm3250: Double = Double.NaN,
    val filDblMm3250: Double = Double.NaN,

    // LEGACY: este campo hoy transporta FED del FIL
    val filEyeSizeMm3250: Double = Double.NaN,

    val filCircMm3250: Double = Double.NaN,

    // Stats útiles para depurar ARC-FIT
    val arcFitRotDeg3250: Double = Double.NaN,
    val arcFitRmsPx3250: Double = Double.NaN,
    val arcFitUsedSamples3250: Int = 0
) : Parcelable {

    // Alias semántico correcto
    @IgnoredOnParcel
    val filFedMm3250: Double
        get() = filEyeSizeMm3250
}
