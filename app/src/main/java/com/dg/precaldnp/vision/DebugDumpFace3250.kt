package com.dg.precaldnp.vision

import android.content.Context

/**
 * Wrapper: mantiene tu API/nombres actuales,
 * pero delega TODO al único núcleo DebugDump3250.
 */
object DebugDumpFace3250 {

    data class FaceOverlayInput3250(
        val edgesFullU8: ByteArray,
        val wFull: Int,
        val hFull: Int,

        // ✅ NUEVO: máscara FULL (wFull*hFull), la que querés ver en ROI
        val maskFullU8_3250: ByteArray? = null,

        val roiOdGlobal: android.graphics.RectF,
        val roiOiGlobal: android.graphics.RectF,
        val midlineXpx: Float,
        val pupilOdGlobal: android.graphics.PointF?,
        val pupilOiGlobal: android.graphics.PointF?,
        val bridgeYpx: Float? = null,
        val probeYpx: Float? = null,
        val extraTag: String = "FACE"
    )

    fun dump(
        context: Context,
        inp: FaceOverlayInput3250,
        minIntervalMs: Long = 800L
    ) {
        DebugDump3250.dumpFace(
            context = context,
            inp = DebugDump3250.FaceOverlayInput3250(
                edgesFullU8 = inp.edgesFullU8,
                wFull = inp.wFull,
                hFull = inp.hFull,
                maskFullU8 = inp.maskFullU8_3250, // ✅ forward real

                roiOdGlobal = inp.roiOdGlobal,
                roiOiGlobal = inp.roiOiGlobal,
                midlineXpx = inp.midlineXpx,
                pupilOdGlobal = inp.pupilOdGlobal,
                pupilOiGlobal = inp.pupilOiGlobal,
                bridgeYpx = inp.bridgeYpx,
                probeYpx = inp.probeYpx,
                extraTag = inp.extraTag
            ),
            minIntervalMs = minIntervalMs
        )
    }
}
