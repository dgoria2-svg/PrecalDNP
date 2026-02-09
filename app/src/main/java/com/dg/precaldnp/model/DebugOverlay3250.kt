package com.dg.precaldnp.model

import android.graphics.PointF
import android.graphics.RectF
import android.os.Parcelable
import kotlinx.parcelize.Parcelize

@Parcelize
data class DebugOverlay3250(
    val stillW: Int,
    val stillH: Int,

    val midXpx: Float,

    val roiOd: RectF?,
    val roiOi: RectF?,

    val pupilOd: PointF?,
    val pupilOi: PointF?,

    // Línea donde medís “puente/inner/outer” (ideal: pupilY o bridgeRowY)
    val bridgeRowYpx: Float?,

    // FIL colocado (global px) empaquetado como x0,y0,x1,y1...
    val placedFilOdXY: FloatArray?,
    val placedFilOiXY: FloatArray?,

    // Marcas RIM (global px) — pares x,y. Ej: nasalInner, nasalOuter, temporalInner, temporalOuter
    val rimMarksOdXY: FloatArray?,
    val rimMarksOiXY: FloatArray?
) : Parcelable {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as DebugOverlay3250

        if (stillW != other.stillW) return false
        if (stillH != other.stillH) return false
        if (midXpx != other.midXpx) return false
        if (bridgeRowYpx != other.bridgeRowYpx) return false
        if (roiOd != other.roiOd) return false
        if (roiOi != other.roiOi) return false
        if (pupilOd != other.pupilOd) return false
        if (pupilOi != other.pupilOi) return false
        if (!placedFilOdXY.contentEquals(other.placedFilOdXY)) return false
        if (!placedFilOiXY.contentEquals(other.placedFilOiXY)) return false
        if (!rimMarksOdXY.contentEquals(other.rimMarksOdXY)) return false
        if (!rimMarksOiXY.contentEquals(other.rimMarksOiXY)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = stillW
        result = 31 * result + stillH
        result = 31 * result + midXpx.hashCode()
        result = 31 * result + (bridgeRowYpx?.hashCode() ?: 0)
        result = 31 * result + (roiOd?.hashCode() ?: 0)
        result = 31 * result + (roiOi?.hashCode() ?: 0)
        result = 31 * result + (pupilOd?.hashCode() ?: 0)
        result = 31 * result + (pupilOi?.hashCode() ?: 0)
        result = 31 * result + (placedFilOdXY?.contentHashCode() ?: 0)
        result = 31 * result + (placedFilOiXY?.contentHashCode() ?: 0)
        result = 31 * result + (rimMarksOdXY?.contentHashCode() ?: 0)
        result = 31 * result + (rimMarksOiXY?.contentHashCode() ?: 0)
        return result
    }
}
