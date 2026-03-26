package com.dg.precaldnp.vision

import android.content.Context
import android.graphics.PointF
import android.graphics.RectF

/**
 * Compatibilidad solamente.
 *
 * Antes este wrapper delegaba al debug de cara.
 * Ahora no hacemos nada porque el único debug que nos interesa
 * es el del detector de rim VALIDADO/USADO en DebugDump3250.
 */
object DebugDumpFace3250 {

    data class FaceOverlayInput3250(
        val edgesFullU8: ByteArray,
        val wFull: Int,
        val hFull: Int,
        val maskfullu83250: ByteArray? = null,
        val roiOdGlobal: RectF,
        val roiOiGlobal: RectF,
        val midlineXpx: Float,
        val pupilOdGlobal: PointF?,
        val pupilOiGlobal: PointF?,
        val bridgeYpx: Float? = null,
        val probeYpx: Float? = null,
        val extraTag: String = "FACE"
    )

    fun dump(
        context: Context,
        inp: FaceOverlayInput3250,
        minIntervalMs: Long = 800L
    ) {
        // no-op a propósito
    }
}

    