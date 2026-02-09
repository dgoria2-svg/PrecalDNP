package com.dg.precaldnp.model

import android.graphics.PointF
import android.graphics.RectF
import kotlin.math.max
import kotlin.math.min

enum class WorkEye3250 { RIGHT_OD, LEFT_OI }

/**
 * Snapshot al momento del disparo (coordenadas en espacio PREVIEWVIEW).
 *
 * IMPORTANTES:
 * - leftBoxPreview / rightBoxPreview: cajas reales si fueron dibujadas (si no, null).
 * - leftPupilRoiPreview / rightPupilRoiPreview: ROIs guía (pueden ser estimadas).
 * - midlineXPreview3250: midline real (ideal: IrisLandmarker o tu detector de cara).
 */
data class DnpShotHint3250(
    val previewWidth: Int,
    val previewHeight: Int,

    // Cajas dibujadas (REALES). Si no dibujaste una, queda null.
    val leftBoxPreview: RectF?,
    val rightBoxPreview: RectF?,

    // Flags de autenticidad de cajas
    val leftBoxReal3250: Boolean = true,
    val rightBoxReal3250: Boolean = true,

    // ROIs para buscar pupilas (pueden existir aunque falte la box real)
    val leftPupilRoiPreview: RectF? = null,
    val rightPupilRoiPreview: RectF? = null,
    val leftPupilRoiEstimated3250: Boolean = false,
    val rightPupilRoiEstimated3250: Boolean = false,

    // Pupilas opcionales (solo para debug/preview)
    val pupilLeftPreview: PointF? = null,
    val pupilRightPreview: PointF? = null,

    // Midline en espacio preview (clave para OD/OI y para ROIs guiadas)
    val midlineXPreview3250: Float? = null,
    val midlineEstimated3250: Boolean = false,

    // Escala de cara (si la tenés en preview; si no, 0f y listo)
    val pxPerMmFace3250: Float,

    // Debug/robustez
    val rotationDegPreview3250: Int = 0,   // 0/90/180/270 si querés guardarlo
    val isMirrored3250: Boolean = false,   // false en back camera

    // Ojo de trabajo: por defecto OD
    val workEye3250: WorkEye3250 = WorkEye3250.RIGHT_OD
) {
    fun hasAnyRealBox(): Boolean =
        (leftBoxPreview != null && leftBoxReal3250) || (rightBoxPreview != null && rightBoxReal3250)

    fun midlineOrCenter3250(): Float =
        midlineXPreview3250 ?: (previewWidth * 0.5f)
}

/**
 * Reproyecta el snapshot al espacio de una foto fija stillWidth×stillHeight (regla de tres).
 * Esto asume que PREVIEW y STILL comparten el mismo crop (ideal: UseCaseGroup + ViewPort).
 */
fun DnpShotHint3250.toStillSpace(stillWidth: Int, stillHeight: Int): DnpShotHint3250 {
    if (previewWidth <= 0 || previewHeight <= 0) {
        return copy(previewWidth = stillWidth, previewHeight = stillHeight)
    }

    val sx = stillWidth.toFloat() / previewWidth.toFloat()
    val sy = stillHeight.toFloat() / previewHeight.toFloat()

    fun RectF.scale(): RectF = RectF(left * sx, top * sy, right * sx, bottom * sy)
    fun PointF.scale(): PointF = PointF(x * sx, y * sy)

    return copy(
        previewWidth = stillWidth,
        previewHeight = stillHeight,
        leftBoxPreview = leftBoxPreview?.scale(),
        rightBoxPreview = rightBoxPreview?.scale(),
        leftPupilRoiPreview = leftPupilRoiPreview?.scale(),
        rightPupilRoiPreview = rightPupilRoiPreview?.scale(),
        pupilLeftPreview = pupilLeftPreview?.scale(),
        pupilRightPreview = pupilRightPreview?.scale(),
        midlineXPreview3250 = midlineXPreview3250?.let { it * sx }
    )
}

/**
 * Si solo hay box REAL del ojo de trabajo (p.ej. OD), genera ROI de pupila del ojo contrario.
 * NO crea leftBoxPreview; crea leftPupilRoiPreview (ESTIMADA) como guía.
 *
 * dxMm suele ser ~65mm (2*32.50) o el valor que quieras.
 * roiScale reduce ROI para evitar “comerse” ceja/nariz.
 */
fun DnpShotHint3250.withOppositePupilRoiGuidance3250(
    dxMm: Float = 65f,
    roiScale: Float = 0.90f
): DnpShotHint3250 {
    val ppm = pxPerMmFace3250
    if (ppm <= 0f) return this
    if (previewWidth <= 0 || previewHeight <= 0) return this

    val dxPx = dxMm * ppm
    val mid = midlineOrCenter3250()

    fun scaledRoiFromBox(box: RectF): RectF {
        val cx = box.centerX()
        val cy = box.centerY()
        val hw = (box.width() * 0.5f * roiScale)
        val hh = (box.height() * 0.5f * roiScale)
        return RectF(cx - hw, cy - hh, cx + hw, cy + hh)
    }

    return when (workEye3250) {
        WorkEye3250.RIGHT_OD -> {
            if (leftPupilRoiPreview != null) return this

            val rightReal = rightBoxPreview?.takeIf { rightBoxReal3250 } ?: return this
            val roiBase = scaledRoiFromBox(rightReal)

            // Si la box OD está a la izquierda del midline, el OI está a la derecha (y viceversa)
            val sign = if (rightReal.centerX() < mid) +1f else -1f
            val leftRoi = RectF(roiBase).apply { offset(sign * dxPx, 0f) }

            copy(
                leftPupilRoiPreview = leftRoi.clampInside3250(previewWidth, previewHeight),
                leftPupilRoiEstimated3250 = true
            )
        }

        WorkEye3250.LEFT_OI -> {
            if (rightPupilRoiPreview != null) return this

            val leftReal = leftBoxPreview?.takeIf { leftBoxReal3250 } ?: return this
            val roiBase = scaledRoiFromBox(leftReal)

            val sign = if (leftReal.centerX() < mid) +1f else -1f
            val rightRoi = RectF(roiBase).apply { offset(sign * dxPx, 0f) }

            copy(
                rightPupilRoiPreview = rightRoi.clampInside3250(previewWidth, previewHeight),
                rightPupilRoiEstimated3250 = true
            )
        }
    }
}

private fun RectF.clampInside3250(w: Int, h: Int): RectF {
    val l = left.coerceIn(0f, (w - 1).toFloat())
    val t = top.coerceIn(0f, (h - 1).toFloat())
    val r = right.coerceIn(0f, (w - 1).toFloat())
    val b = bottom.coerceIn(0f, (h - 1).toFloat())
    return RectF(min(l, r), min(t, b), max(l, r), max(t, b))
}
