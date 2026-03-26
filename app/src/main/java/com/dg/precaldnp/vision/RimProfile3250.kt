package com.dg.precaldnp.vision

enum class RimProfile3250 {
    FULL_RIM,
    RANURADO,
    PERFORADO
}

fun effectiveFilOverPerSide3250(
    profile3250: RimProfile3250,
    filOverInnerMmPerSide: Double
): Double {
    val over = filOverInnerMmPerSide.takeIf { it.isFinite() } ?: 0.5
    val overClamped = over.coerceIn(0.0, 2.0)

    return when (profile3250) {
        RimProfile3250.FULL_RIM -> overClamped
        RimProfile3250.RANURADO,
        RimProfile3250.PERFORADO -> 0.0
    }
}