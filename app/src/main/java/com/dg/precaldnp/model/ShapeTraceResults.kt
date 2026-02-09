package com.dg.precaldnp.model

import android.graphics.PointF
import android.net.Uri
import android.os.Parcel
import android.os.Parcelable

/**
 * Resultado que vamos a mandar a ResultsPrecalActivity.
 *
 * - outlinePx: lista de puntos del borde en píxeles
 * - bboxWidthMm / bboxHeightMm: caja que encierra la lente en mm aprox
 * - pxPerMm: escala usada para convertir px->mm
 * - axisRotationDeg: giro aplicado al FIL (en grados, CCW) desde Precal
 * - sharpness: nitidez estimada de la foto
 * - srcWidth/srcHeight: tamaño original de la imagen capturada
 * - imageUri: la foto para previsualizar en ResultsPrecal
 */
@Suppress("DEPRECATION")
// app/src/main/java/com/dg/precaldnp/model/ShapeTraceResult.kt
data class ShapeTraceResult(
    val pxPerMm: Float,
    val srcWidth: Int,
    val srcHeight: Int,
    val outlinePx: List<PointF>,
    val bboxWidthMm: Float?,
    val bboxHeightMm: Float?,
    val axisRotationDeg: Float = 0f,   // <-- giro guardado
    val sharpness: Float? = null,
    val imageUri: Uri? = null
) : Parcelable {

    val bufW: Int get() = srcWidth
    val bufH: Int get() = srcHeight

    constructor(parcel: Parcel) : this(
        pxPerMm = parcel.readFloat(),
        srcWidth = parcel.readInt(),
        srcHeight = parcel.readInt(),
        outlinePx = mutableListOf<PointF>().apply {
            val n = parcel.readInt()
            repeat(n) {
                val x = parcel.readFloat()
                val y = parcel.readFloat()
                add(PointF(x, y))
            }
        },
        bboxWidthMm = parcel.readValue(Float::class.java.classLoader) as? Float,
        bboxHeightMm = parcel.readValue(Float::class.java.classLoader) as? Float,
        axisRotationDeg = parcel.readFloat(),  // <-- AHORA SÍ
        sharpness = parcel.readValue(Float::class.java.classLoader) as? Float,
        imageUri = parcel.readParcelable(Uri::class.java.classLoader)
    )

    override fun writeToParcel(parcel: Parcel, flags: Int) {
        parcel.writeFloat(pxPerMm)
        parcel.writeInt(srcWidth)
        parcel.writeInt(srcHeight)

        parcel.writeInt(outlinePx.size)
        outlinePx.forEach { p ->
            parcel.writeFloat(p.x)
            parcel.writeFloat(p.y)
        }

        parcel.writeValue(bboxWidthMm)
        parcel.writeValue(bboxHeightMm)
        parcel.writeFloat(axisRotationDeg)          // <-- y acá
        parcel.writeValue(sharpness)
        parcel.writeParcelable(imageUri, flags)
    }

    override fun describeContents() = 0

    companion object CREATOR : Parcelable.Creator<ShapeTraceResult> {
        override fun createFromParcel(parcel: Parcel) = ShapeTraceResult(parcel)
        override fun newArray(size: Int): Array<ShapeTraceResult?> = arrayOfNulls(size)
    }
}
