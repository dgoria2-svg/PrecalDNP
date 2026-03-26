package com.dg.precaldnp.vision

import android.graphics.PointF
import android.graphics.RectF

/**
 * Resultado “humano” del rim, en coordenadas GLOBAL (pixeles de STILL).
 * - innerLeft/innerRight: bordes verticales del inner rim en probeY / wallsY.
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

    // fila donde se midieron L/R (yWalls)
    val wallsYpx: Float,

    // seed X usado para center→out
    val seedXpx: Float,

    // INNER (medidos en wallsYpx)
    val innerLeftXpx: Float,
    val innerRightXpx: Float,
    val nasalInnerXpx: Float,
    val templeInnerXpx: Float,

    val heightPx: Float,
    val innerWidthPx: Float,

    val topPolylinePx: List<PointF>?,
    val bottomPolylinePx: List<PointF>?,

    // diag opcional
    val pxPerMmSeedX: Float? = null,

    // OUTER (actual / futuro)
    val outerLeftXpx: Float? = null,
    val outerRightXpx: Float? = null,
    val nasalOuterXpx: Float? = null,
    val templeOuterXpx: Float? = null,
    val rimThicknessPx: Float? = null,
    val outerWidthPx: Float? = null,

    // ✅ NUEVO: arcos candidatos detector (GLOBAL)
    val nasalInnerPolylinePx: List<PointF>? = null,
    val templeInnerPolylinePx: List<PointF>? = null,
    val nasalOuterPolylinePx: List<PointF>? = null,
    val templeOuterPolylinePx: List<PointF>? = null,
    val innerArcPolylinePx: List<PointF>? = null
)

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