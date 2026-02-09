package com.dg.precaldnp.ui

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.core.content.FileProvider
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import kotlin.math.max

object DnpBurstOrchestrator3250 {

    private const val TAG = "DnpBurstOrch3250"

    data class BurstConfig3250(
        val durationMs: Long = 1000L,
        val maxShots: Int = 10,
        val delayMs: Long = 80L,
        val topK: Int = 3,
        // si querés dataset: guardar además los topK aunque fallen en measure
        val keepTopKOnFail: Boolean = false
    )

    data class Result3250<T : Any>(
        val bestOut: T?,
        val bestUri: Uri?,
        // borrar estos tokens al salir de Results (winner + debug(s) si agregás)
        val cleanupOnExit: List<String>,
        val capturedCount: Int,
        val keptCount: Int
    )

    private data class Shot3250(
        val file: File,
        val uri: Uri,
        var sharpness: Double
    )

    fun <T : Any> run3250(
        ctx: Context,
        imageCapture: ImageCapture,
        authority: String,                  // "${packageName}.fileprovider"
        orderId: String,
        cfg: BurstConfig3250,
        bgExecutor: ExecutorService,
        measure: (Uri) -> T?,               // tu measureArcFitOnlyCore3250
        score: (out: T, sharpness: Double) -> Double,
        cleanupTokens: (T) -> List<String> = { emptyList() },
        onDoneMain: (Result3250<T>) -> Unit
    ) {
        Runner3250(
            ctx = ctx,
            imageCapture = imageCapture,
            authority = authority,
            orderId = orderId,
            cfg = cfg,
            bgExecutor = bgExecutor,
            measure = measure,
            score = score,
            cleanupTokens = cleanupTokens,
            onDoneMain = onDoneMain
        ).start()
    }

    private class Runner3250<T : Any>(
        private val ctx: Context,
        private val imageCapture: ImageCapture,
        private val authority: String,
        private val orderId: String,
        private val cfg: BurstConfig3250,
        private val bgExecutor: ExecutorService,
        private val measure: (Uri) -> T?,
        private val score: (T, Double) -> Double,
        private val cleanupTokens: (T) -> List<String>,
        private val onDoneMain: (Result3250<T>) -> Unit
    ) {
        private val main = Handler(Looper.getMainLooper())

        private val deadline = System.currentTimeMillis() + cfg.durationMs
        private var capturedCount = 0
        private var finalized = false

        // mantenemos solo TOP-K por sharpness, ya persistidos como PNG lossless
        private val bestK = ArrayList<Shot3250>(max(1, cfg.topK))

        fun start() {
            captureNext()
        }

        private fun buildTempFilePng(): File {
            val dir = File(ctx.cacheDir, "dnp_burst_3250").apply { mkdirs() }
            val name = "face_${orderId}_${SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())}.png"
            return File(dir, name)
        }

        private fun uriForFile(f: File): Uri =
            FileProvider.getUriForFile(ctx, authority, f)

        private fun captureNext() {
            if (finalized) return

            val now = System.currentTimeMillis()
            if (now >= deadline || capturedCount >= cfg.maxShots) {
                finalizeAndMeasure()
                return
            }

            imageCapture.takePicture(
                bgExecutor,
                object : ImageCapture.OnImageCapturedCallback() {

                    override fun onError(exc: ImageCaptureException) {
                        Log.w(TAG, "shot error: ${exc.message}", exc)
                        main.postDelayed({ captureNext() }, cfg.delayMs)
                    }

                    override fun onCaptureSuccess(image: ImageProxy) {
                        try {
                            capturedCount++

                            val sharp = computeSharpnessScoreFromImageProxy3250(image)
                            val k = max(1, cfg.topK)

                            // ¿entra al topK?
                            val minSharp = bestK.minOfOrNull { it.sharpness } ?: Double.NEGATIVE_INFINITY
                            val qualifies = (bestK.size < k) || (sharp > minSharp)

                            if (qualifies) {
                                // Persistimos como PNG lossless (ya rotado). Esto es lo que alimenta RimDetector.
                                val bmp = imageProxyToBitmapArgb88883250(image)
                                if (bmp == null) {
                                    Log.w(TAG, "skip frame: cannot decode to bitmap (fmt=${image.format} planes=${image.planes.size})")
                                    return
                                }

                                val rot = image.imageInfo.rotationDegrees
                                val bmpRot = rotateBitmapIfNeeded3250(bmp, rot)

                                val file = buildTempFilePng()
                                savePng3250(file, bmpRot)

                                // liberar memoria
                                try { if (bmpRot !== bmp) bmpRot.recycle() } catch (_: Throwable) {}
                                try { bmp.recycle() } catch (_: Throwable) {}

                                val uri = uriForFile(file)
                                bestK.add(Shot3250(file = file, uri = uri, sharpness = sharp))

                                // recortar a K (borrar el peor)
                                if (bestK.size > k) {
                                    bestK.sortByDescending { it.sharpness }
                                    val drop = bestK.removeAt(bestK.size - 1)
                                    safeDeleteToken(drop.file.absolutePath)
                                }
                            }
                        } catch (t: Throwable) {
                            Log.w(TAG, "captureSuccess processing failed", t)
                        } finally {
                            try { image.close() } catch (_: Throwable) {}
                            main.postDelayed({ captureNext() }, cfg.delayMs)
                        }
                    }
                }
            )
        }

        private fun finalizeAndMeasure() {
            if (finalized) return
            finalized = true

            bgExecutor.execute {
                try {
                    if (bestK.isEmpty()) {
                        main.post {
                            onDoneMain(
                                Result3250(
                                    bestOut = null,
                                    bestUri = null,
                                    cleanupOnExit = emptyList(),
                                    capturedCount = capturedCount,
                                    keptCount = 0
                                )
                            )
                        }
                        return@execute
                    }

                    // medir solo los TOP-K persistidos (PNG lossless)
                    bestK.sortByDescending { it.sharpness }

                    var bestOut: T? = null
                    var bestShot: Shot3250? = null
                    var bestScore = Double.NEGATIVE_INFINITY

                    for (s in bestK) {
                        val out = try { measure(s.uri) } catch (t: Throwable) {
                            Log.w(TAG, "measure failed: ${s.uri}", t)
                            null
                        }
                        if (out != null) {
                            val sc = score(out, s.sharpness)
                            if (sc > bestScore) {
                                bestScore = sc
                                bestOut = out
                                bestShot = s
                            }
                        }
                    }

                    // borrar losers YA (todo menos winner)
                    val winner = bestShot
                    if (winner != null) {
                        for (s in bestK) {
                            if (s.file.absolutePath != winner.file.absolutePath) {
                                safeDeleteToken(s.file.absolutePath)
                            }
                        }
                    } else if (!cfg.keepTopKOnFail) {
                        // si nadie midió bien y no querés dataset -> borrar todo
                        for (s in bestK) safeDeleteToken(s.file.absolutePath)
                    }

                    // cleanup para exit (winner + tokens del out)
                    val cleanup = ArrayList<String>(8)
                    if (winner != null) cleanup.add(winner.file.absolutePath)
                    if (bestOut != null) cleanup.addAll(cleanupTokens(bestOut))

                    main.post {
                        onDoneMain(
                            Result3250(
                                bestOut = bestOut,
                                bestUri = winner?.uri,
                                cleanupOnExit = cleanup,
                                capturedCount = capturedCount,
                                keptCount = bestK.size
                            )
                        )
                    }
                } catch (t: Throwable) {
                    Log.e(TAG, "finalize failed", t)
                    try { for (s in bestK) safeDeleteToken(s.file.absolutePath) } catch (_: Throwable) {}
                    main.post {
                        onDoneMain(
                            Result3250(
                                bestOut = null,
                                bestUri = null,
                                cleanupOnExit = emptyList(),
                                capturedCount = capturedCount,
                                keptCount = bestK.size
                            )
                        )
                    }
                }
            }
        }

        private fun safeDeleteToken(token: String) {
            try {
                when {
                    token.startsWith("content:") -> ctx.contentResolver.delete(Uri.parse(token), null, null)
                    token.startsWith("file:") -> Uri.parse(token).path?.let { File(it).delete() }
                    else -> File(token).delete()
                }
            } catch (_: Throwable) { }
        }

        // ---------------------------
        // Sharpness (YUV rápido o JPEG con decode chico)
        // ---------------------------
        private fun computeSharpnessScoreFromImageProxy3250(image: ImageProxy): Double {
            return try {
                when (image.format) {
                    ImageFormat.YUV_420_888 -> computeSharpnessFromYPlane3250(image)
                    ImageFormat.JPEG -> computeSharpnessFromJpegProxy3250(image)
                    else -> 0.0
                }
            } catch (_: Throwable) {
                0.0
            }
        }

        private fun computeSharpnessFromYPlane3250(image: ImageProxy): Double {
            if (image.planes.isEmpty()) return 0.0
            val yPlane = image.planes[0]
            val yBuf = yPlane.buffer.duplicate().apply { rewind() }

            val w = image.width
            val h = image.height
            if (w < 8 || h < 8) return 0.0

            val yRowStride = yPlane.rowStride
            val yPixStride = yPlane.pixelStride
            val yBytes = ByteArray(yBuf.remaining())
            yBuf.get(yBytes)

            var n = 0
            var sum = 0.0
            var sumSq = 0.0

            val step = 2 // subsample
            var y = 1
            while (y < h - 1) {
                val row = y * yRowStride
                val rowU = (y - 1) * yRowStride
                val rowD = (y + 1) * yRowStride

                var x = 1
                while (x < w - 1) {
                    val idx = row + x * yPixStride
                    if (idx !in yBytes.indices) return 0.0

                    val c = (yBytes[idx].toInt() and 0xFF)
                    val l = (yBytes[row + (x - 1) * yPixStride].toInt() and 0xFF)
                    val r = (yBytes[row + (x + 1) * yPixStride].toInt() and 0xFF)
                    val u = (yBytes[rowU + x * yPixStride].toInt() and 0xFF)
                    val d = (yBytes[rowD + x * yPixStride].toInt() and 0xFF)

                    val lap = (l + r + u + d) - 4 * c
                    val v = lap.toDouble()
                    sum += v
                    sumSq += v * v
                    n++

                    x += step
                }
                y += step
            }

            return if (n <= 10) 0.0 else {
                val mean = sum / n
                val varLap = (sumSq / n) - (mean * mean)
                varLap.coerceAtLeast(0.0)
            }
        }

        private fun computeSharpnessFromJpegProxy3250(image: ImageProxy): Double {
            if (image.planes.isEmpty()) return 0.0
            val buf = image.planes[0].buffer.duplicate().apply { rewind() }
            val bytes = ByteArray(buf.remaining())
            buf.get(bytes)

            val opt = BitmapFactory.Options().apply {
                inPreferredConfig = Bitmap.Config.ARGB_8888
                inSampleSize = 8 // 4 si querés más fino (más CPU)
            }
            val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opt) ?: return 0.0
            val sc = computeSharpnessFromSmallBitmap3250(bmp)
            try { bmp.recycle() } catch (_: Throwable) {}
            return sc
        }

        private fun computeSharpnessFromSmallBitmap3250(bmp: Bitmap): Double {
            val w = bmp.width
            val h = bmp.height
            if (w < 8 || h < 8) return 0.0

            val px = IntArray(w * h)
            bmp.getPixels(px, 0, w, 0, 0, w, h)

            fun luma(c: Int): Int {
                val r = (c shr 16) and 0xFF
                val g = (c shr 8) and 0xFF
                val b = c and 0xFF
                return (77 * r + 150 * g + 29 * b) shr 8
            }

            var n = 0
            var sum = 0.0
            var sumSq = 0.0
            val step = 2

            var y = 1
            while (y < h - 1) {
                var x = 1
                while (x < w - 1) {
                    val c = luma(px[y * w + x])
                    val l = luma(px[y * w + (x - 1)])
                    val r = luma(px[y * w + (x + 1)])
                    val u = luma(px[(y - 1) * w + x])
                    val d = luma(px[(y + 1) * w + x])

                    val lap = (l + r + u + d) - 4 * c
                    val v = lap.toDouble()
                    sum += v
                    sumSq += v * v
                    n++

                    x += step
                }
                y += step
            }

            if (n <= 10) return 0.0
            val mean = sum / n
            return ((sumSq / n) - mean * mean).coerceAtLeast(0.0)
        }

        // ---------------------------
        // ImageProxy -> Bitmap ARGB_8888 (robusto: U/V stride separado)
        // ---------------------------
        private fun imageProxyToBitmapArgb88883250(image: ImageProxy): Bitmap? {
            val w = image.width
            val h = image.height
            val fmt = image.format
            val planes = image.planes

            if (fmt == ImageFormat.JPEG && planes.isNotEmpty()) {
                val buffer = planes[0].buffer.duplicate().apply { rewind() }
                val bytes = ByteArray(buffer.remaining())
                buffer.get(bytes)
                val decoded = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: return null
                return if (decoded.config == Bitmap.Config.ARGB_8888) decoded
                else decoded.copy(Bitmap.Config.ARGB_8888, false)
            }

            if (fmt != ImageFormat.YUV_420_888 || planes.size < 3) {
                Log.w(TAG, "Unsupported ImageProxy for decode: fmt=$fmt planes=${planes.size}")
                return null
            }

            val yPlane = planes[0]
            val uPlane = planes[1]
            val vPlane = planes[2]

            val yBuf = yPlane.buffer.duplicate().apply { rewind() }
            val uBuf = uPlane.buffer.duplicate().apply { rewind() }
            val vBuf = vPlane.buffer.duplicate().apply { rewind() }

            val yBytes = ByteArray(yBuf.remaining()); yBuf.get(yBytes)
            val uBytes = ByteArray(uBuf.remaining()); uBuf.get(uBytes)
            val vBytes = ByteArray(vBuf.remaining()); vBuf.get(vBytes)

            val yRowStride = yPlane.rowStride
            val yPixStride = yPlane.pixelStride

            // ✅ clave: stride separado para U y V
            val uRowStride = uPlane.rowStride
            val vRowStride = vPlane.rowStride
            val uPixStride = uPlane.pixelStride
            val vPixStride = vPlane.pixelStride

            val out = IntArray(w * h)
            var yp = 0

            for (yy in 0 until h) {
                val yRow = yy * yRowStride
                val uRow = (yy / 2) * uRowStride
                val vRow = (yy / 2) * vRowStride

                for (xx in 0 until w) {
                    val yIdx = yRow + xx * yPixStride
                    val uIdx = uRow + (xx / 2) * uPixStride
                    val vIdx = vRow + (xx / 2) * vPixStride

                    if (yIdx !in yBytes.indices || uIdx !in uBytes.indices || vIdx !in vBytes.indices) {
                        Log.w(
                            TAG,
                            "YUV index OOB (y=$yIdx u=$uIdx v=$vIdx) sizes y=${yBytes.size} u=${uBytes.size} v=${vBytes.size} " +
                                    "strides y(r=$yRowStride p=$yPixStride) u(r=$uRowStride p=$uPixStride) v(r=$vRowStride p=$vPixStride)"
                        )
                        return null
                    }

                    val Y = (yBytes[yIdx].toInt() and 0xFF)
                    val U = (uBytes[uIdx].toInt() and 0xFF)
                    val V = (vBytes[vIdx].toInt() and 0xFF)

                    // YUV -> RGB (BT.601)
                    val c = Y - 16
                    val d = U - 128
                    val e = V - 128

                    var r = (298 * c + 409 * e + 128) shr 8
                    var g = (298 * c - 100 * d - 208 * e + 128) shr 8
                    var b = (298 * c + 516 * d + 128) shr 8

                    if (r < 0) r = 0 else if (r > 255) r = 255
                    if (g < 0) g = 0 else if (g > 255) g = 255
                    if (b < 0) b = 0 else if (b > 255) b = 255

                    out[yp++] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }

            return Bitmap.createBitmap(out, w, h, Bitmap.Config.ARGB_8888)
        }

        private fun rotateBitmapIfNeeded3250(bmp: Bitmap, rotationDeg: Int): Bitmap {
            val deg = ((rotationDeg % 360) + 360) % 360
            if (deg == 0) return bmp
            val m = Matrix().apply { postRotate(deg.toFloat()) }
            return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
        }

        private fun savePng3250(file: File, bmp: Bitmap) {
            FileOutputStream(file).use { out ->
                bmp.compress(Bitmap.CompressFormat.PNG, 100, out) // PNG: lossless
            }
            Log.d(TAG, "saved PNG: ${file.absolutePath}  (${bmp.width}x${bmp.height})")
        }
    }
}
