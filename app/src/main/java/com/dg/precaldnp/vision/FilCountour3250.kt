package com.dg.precaldnp.vision

import android.graphics.PointF

/**
 * Contorno del FIL en milímetros (cartesiano Y-up, como tu viewer PC).
 * - ptsMm: lista de puntos (x,y) en mm
 * - maxAbsX/maxAbsY: máximos absolutos (usado para fit uniforme al aro)
 */
data class FilContourMm3250(
    val ptsMm: List<PointF>,
    val maxAbsX: Double,
    val maxAbsY: Double ,

)
