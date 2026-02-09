package com.dg.precaldnp.model

import android.os.Parcelable
import kotlinx.parcelize.Parcelize

@Parcelize
data class MeasurementResult(
    val annotatedPath: String,
    val originalPath: String,

    // Métricas del aro derecho
    val anchoMm: Double,
    val altoMm: Double,
    val diagMayorMm: Double,

    // Diámetros útiles por ojo
    val diamUtilOdMm: Double,
    val diamUtilOiMm: Double,

    // DNP por ojo (nasopupilar)
    val dnpOdMm: Double,
    val dnpOiMm: Double,

    // Alturas pupila→borde inferior
    val altOdMm: Double,
    val altOiMm: Double,

    // Puente
    val puenteMm: Double,

    // ---------------- 3250 extras (para dibujar + FIL) ----------------
    // Escala FINAL usada para el render (mm -> px)
    val pxPerMmFace3250: Double = Double.NaN,

    // Métricas del FIL (lo que querés mostrar)
    val filHboxMm3250: Double = Double.NaN,
    val filVboxMm3250: Double = Double.NaN,
    val filDblMm3250: Double = Double.NaN,
    val filEyeSizeMm3250: Double = Double.NaN,
    val filCircMm3250: Double = Double.NaN,

    // Stats útiles para depurar ARC-FIT (opcional, pero vale oro)
    val arcFitRotDeg3250: Double = Double.NaN,
    val arcFitRmsPx3250: Double = Double.NaN,
    val arcFitUsedSamples3250: Int = 0
) : Parcelable

