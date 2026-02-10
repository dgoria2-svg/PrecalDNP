package com.dg.precaldnp.vision

import android.graphics.PointF

// Eye mask (TOP-LEVEL, shared)
data class EyeEllipseMask3250(
    val centerPx: PointF,      // GLOBAL px
    val rxPx: Float,           // semieje X (GLOBAL px)
    val ryPx: Float,           // semieje Y (GLOBAL px)
    val angleDeg: Float = 0f,  // opcional (por ahora 0)
    val score3250: Float = 1f  // opcional
)