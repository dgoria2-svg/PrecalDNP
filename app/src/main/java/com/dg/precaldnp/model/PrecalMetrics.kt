package com.dg.precaldnp.model

import android.os.Parcelable
import kotlinx.parcelize.Parcelize

@Parcelize
data class PrecalMetrics(
    // Radiales (N) desde el centro en PX de BUFFER y en MM
    val radiiPx: FloatArray,
    val radiiMm: FloatArray,

    // Centro en PX de BUFFER
    val cxPx: Float,
    val cyPx: Float,

    // Tamaño del buffer donde se midió
    val bufW: Int,
    val bufH: Int,

    // Calibración usada
    val pxPerMm: Float,
    val cameraId: String,
    val zoom: Float,

    // Calidad
    val coverage: Float,      // 0..1
    val sharpness: Float,     // métrica de nitidez
    val confidence: Float,    // 0..1

    // Gatekeeper
    val valid: Boolean,
    val reasonIfInvalid: String? = null,

    // (Opcional) contorno para anotar PNG (en coords de BUFFER)
    // Ahora lo estamos guardando como lista de pares (x, y) en px
    val outlineBuf: List<Pair<Float, Float>> = emptyList()
) : Parcelable {

    // texto que mostramos en el bloque gris abajo de la foto
    fun toDisplayString(): String {
        val sb = StringBuilder()

        sb.append("px/mm: ")
        sb.append("%.3f".format(pxPerMm))

        sb.append("\nPuntos contorno: ")
        // Antes: radiiPx.size → daba 0 siempre
        sb.append(outlineBuf.size)

        // info debug opcional:
        // sb.append("\nCobertura: %.2f".format(coverage))
        // sb.append("\nConfianza: %.2f".format(confidence))

        return sb.toString()
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as PrecalMetrics

        if (cxPx != other.cxPx) return false
        if (cyPx != other.cyPx) return false
        if (bufW != other.bufW) return false
        if (bufH != other.bufH) return false
        if (pxPerMm != other.pxPerMm) return false
        if (zoom != other.zoom) return false
        if (coverage != other.coverage) return false
        if (sharpness != other.sharpness) return false
        if (confidence != other.confidence) return false
        if (valid != other.valid) return false
        if (!radiiPx.contentEquals(other.radiiPx)) return false
        if (!radiiMm.contentEquals(other.radiiMm)) return false
        if (cameraId != other.cameraId) return false
        if (reasonIfInvalid != other.reasonIfInvalid) return false
        if (outlineBuf != other.outlineBuf) return false

        return true
    }

    override fun hashCode(): Int {
        var result = cxPx.hashCode()
        result = 31 * result + cyPx.hashCode()
        result = 31 * result + bufW
        result = 31 * result + bufH
        result = 31 * result + pxPerMm.hashCode()
        result = 31 * result + zoom.hashCode()
        result = 31 * result + coverage.hashCode()
        result = 31 * result + sharpness.hashCode()
        result = 31 * result + confidence.hashCode()
        result = 31 * result + valid.hashCode()
        result = 31 * result + radiiPx.contentHashCode()
        result = 31 * result + radiiMm.contentHashCode()
        result = 31 * result + cameraId.hashCode()
        result = 31 * result + (reasonIfInvalid?.hashCode() ?: 0)
        result = 31 * result + outlineBuf.hashCode()
        return result
    }
}
