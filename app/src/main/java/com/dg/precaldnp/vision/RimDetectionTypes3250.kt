package com.dg.precaldnp.vision

import android.graphics.PointF
import android.graphics.RectF

/**
 * Resultado “humano” del rim, en coordenadas GLOBAL (pixeles de STILL).
 * - innerLeft/innerRight: bordes verticales del inner rim en probeY (aprox).
 * - nasalInner/templeInner: el borde “hacia midline” y el opuesto.
 * - topY/bottomY: líneas horizontales (aprox) del rim dentro del ROI.
 */
data class RimDetectionResult(
    val ok: Boolean,
    val confidence: Float,

    val roiPx: RectF,

    val probeYpx: Float,

    val topYpx: Float,
    val bottomYpx: Float,

    val innerLeftXpx: Float,
    val innerRightXpx: Float,

    val nasalInnerXpx: Float,
    val templeInnerXpx: Float,

    val heightPx: Float,
    val innerWidthPx: Float,

    // Para debug overlay (opcionales)
    val topPolylinePx: List<PointF>? = null,
    val bottomPolylinePx: List<PointF>? = null
)

/**
 * Paquete de salida del detector:
 * - result: lo que usa la Activity para anchors/bridge/debug
 * - edgeMap: ByteArray binario (0/255) en coords ROI (w*h) para ARC-FIT
 *
 * NOTA: NO es data class para evitar warning de equals/hashCode por ByteArray.
 */
/**
 * Paquete de salida del detector:
 * - result: lo que usa la Activity
 * - edges: ByteArray binario (0/255) en coords ROI (w*h) para ARC-FIT
 *
 * NOTA: NO data class (evita warning equals/hashCode por ByteArray).
 */
class RimDetectPack3250(
    val result: RimDetectionResult,
    val edges: ByteArray,
    val w: Int,
    val h: Int
) {
    // Alias por compatibilidad (si en algún lado seguís usando edgeMap)
    val edgeMap: ByteArray get() = edges
}

