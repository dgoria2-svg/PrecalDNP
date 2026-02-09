// app/src/main/java/com/dg/precaldnp/ui/StillCaptureActivity.kt
@file:Suppress("DEPRECATION", "SameParameterValue")

package com.dg.precaldnp.ui

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.graphics.ImageFormat
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.util.Size
import android.view.Gravity
import android.view.Surface
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.graphics.scale
import androidx.lifecycle.Observer
import com.dg.precaldnp.R
import com.dg.precaldnp.model.ShapeTraceResult
import com.dg.precaldnp.overlay.OverlayGuide75View
import com.dg.precaldnp.vision.LensTracer
import com.dg.precaldnp.vision.RefCalibrator75
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import org.opencv.core.Size as CvSize

@Suppress("DEPRECATION", "SameParameterValue")
class StillCaptureActivity : ComponentActivity() {

    companion object {
        private const val TAG = "Precal/StillBlend"
        // Zoom inicial moderado
        private const val START_ZOOM_RATIO = 1f
    }

    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayGuide75View
    private lateinit var tvHint: TextView
    private lateinit var btnCapturar: ImageButton

    // ====== Overlay de espera (solo “cartelito” mientras procesa) ======
    private lateinit var processingOverlay: FrameLayout
    private lateinit var processingText: TextView

    private var imageCapture: ImageCapture? = null
    private var camera: Camera? = null
    private var cameraId: String = "back"
    private var zoomRatio: Float = 1f

    private val exec: Executor by lazy { ContextCompat.getMainExecutor(this) }
    private val bgExec = Executors.newSingleThreadExecutor()

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera() else {
            Toast.makeText(this, getString(R.string.msg_permiso_cam), Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_still_capture)

        previewView = findViewById(R.id.previewView)
        overlay = findViewById(R.id.overlayGuide)
        tvHint = findViewById(R.id.tvHint)
        btnCapturar = findViewById(R.id.btnCapturar)

        tvHint.text = getString(R.string.capture_hint)
        btnCapturar.setOnClickListener { takePhoto() }

        // Guía inicial centrada (por si aún no detectamos el aro)
        overlay.post { overlay.setCircleNormalized(0.5f, 0.5f, 0.35f) }
        overlay.bringToFront()

        // ====== crea el “cartelito” de espera por arriba de TODO ======
        setupProcessingOverlay()

        checkPermissionAndStart()
    }

    override fun onDestroy() {
        super.onDestroy()
        bgExec.shutdownNow()
    }

    private fun setupProcessingOverlay() {
        val root = findViewById<ViewGroup>(android.R.id.content)

        processingOverlay = FrameLayout(this).apply {
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            setBackgroundColor(0xAA000000.toInt()) // negro semi-transparente
            visibility = View.GONE
            isClickable = true // bloquea taps mientras procesa
            isFocusable = true
        }

        val density = resources.displayMetrics.density
        val pad = (20f * density).toInt()

        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setPadding(pad, pad, pad, pad)
            // gris muy oscuro semi opaco
            setBackgroundColor(0xCC111111.toInt())
        }

        val pb = ProgressBar(this).apply {
            isIndeterminate = true
        }

        processingText = TextView(this).apply {
            text = getString(R.string.precal_processing)
            setTextColor(0xFFFFFFFF.toInt())
            textSize = 16f
            gravity = Gravity.CENTER
            setPadding(0, (10f * density).toInt(), 0, 0)
        }

        card.addView(pb)
        card.addView(processingText)

        val lp = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT,
            FrameLayout.LayoutParams.WRAP_CONTENT
        ).apply { gravity = Gravity.CENTER }

        processingOverlay.addView(card, lp)
        root.addView(processingOverlay)
        processingOverlay.bringToFront()
    }

    private fun showProcessing(show: Boolean, message: String = getString(R.string.precal_processing)) {
        runOnUiThread {
            processingText.text = message
            processingOverlay.visibility = if (show) View.VISIBLE else View.GONE
            btnCapturar.isEnabled = !show
        }
    }

    private fun checkPermissionAndStart() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    @OptIn(ExperimentalCamera2Interop::class)
    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val cameraProvider = providerFuture.get()
            cameraProvider.unbindAll()

            // PREVIEW liviana 4:3 para rendimiento (NO afecta captura final)
            val preview = Preview.Builder()
                .setTargetRotation(previewView.display?.rotation ?: Surface.ROTATION_0)
                .setTargetResolution(Size(1280, 960))
                .build()
                .also { it.surfaceProvider = previewView.surfaceProvider }

            // CAPTURA en ALTA calidad (sin fijar resolución para que use la nativa alta)
            imageCapture = ImageCapture.Builder()
                .setTargetRotation(previewView.display?.rotation ?: Surface.ROTATION_0)
                // .setTargetResolution(Size(2048, 1536)) // <- si querés forzar 4:3 alta
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .setJpegQuality(100)
                .build()

            // ANALYSIS liviano para overlay del aro 75
            val analysis = ImageAnalysis.Builder()
                .setTargetRotation(previewView.display?.rotation ?: Surface.ROTATION_0)
                .setTargetResolution(Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .apply { setAnalyzer(bgExec, Guide75Analyzer(overlay)) }

            camera = cameraProvider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                imageCapture,
                analysis
            )

            camera?.let { cam ->
                cameraId = runCatching { Camera2CameraInfo.from(cam.cameraInfo).cameraId }
                    .getOrElse { "back" }

                zoomRatio = cam.cameraInfo.zoomState.value?.zoomRatio ?: 1f
                cam.cameraInfo.zoomState.observe(this, Observer { zs ->
                    zoomRatio = zs?.zoomRatio ?: 1f
                })

                // Zoom inicial moderado
                runCatching { cam.cameraControl.setZoomRatio(START_ZOOM_RATIO) }
                    .onFailure { Log.w(TAG, "Zoom no soportado: ${it.message}") }

                // IMPORTANTE: dejamos AF continuo por defecto (sin forzar LENS_FOCUS_DISTANCE)
            }
        }, exec)
    }

    private fun takePhoto() {
        val capture = imageCapture ?: return

        val name = "PRECAL_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
            .format(System.currentTimeMillis())

        val contentValues = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "$name.jpg")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/medirDNP")
        }

        val output = ImageCapture.OutputFileOptions.Builder(
            contentResolver,
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            contentValues
        ).build()

        capture.takePicture(output, exec, object : ImageCapture.OnImageSavedCallback {
            override fun onError(exc: ImageCaptureException) {
                Toast.makeText(
                    this@StillCaptureActivity,
                    getString(R.string.msg_error_captura),
                    Toast.LENGTH_LONG
                ).show()
            }

            override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                val uri = outputFileResults.savedUri ?: run {
                    Toast.makeText(
                        this@StillCaptureActivity,
                        getString(R.string.msg_error_captura),
                        Toast.LENGTH_LONG
                    ).show()
                    return
                }

                // ====== ACA SOLO: cartelito de espera mientras procesa ======
                showProcessing(true, getString(R.string.precal_processing))

                bgExec.execute { processPhoto(uri) }
            }
        })
    }

    // ===== Pipeline: guía 75 → px/mm (RefCalibrator75 o fallback Hough con guardas) → trazo → Results
    private fun processPhoto(uri: Uri) {
        // Como el viejo: mayor tamaño = borde más limpio
        val bmp = decodeBitmap(uri, maxDim = 2800)
            ?: return runOnUi {
                showProcessing(false)
                Toast.makeText(
                    this,
                    getString(R.string.msg_error_load_photo),
                    Toast.LENGTH_LONG
                ).show()
                finish()
            }

        // 1) Calibrador pro (preferido)
        val ref = RefCalibrator75.detect(bmp)
        val pxPerMm = if (ref != null) {
            Log.d(TAG, "PX/MM=%.4f (RefCalibrator75)".format(ref.pxPerMm))
            ref.pxPerMm
        } else {
            // 2) Fallback Hough con anti-brillos y guardas
            val g = detectGuide75(bmp)
            val accepted = g?.let { acceptFallback(it, bmp.width, bmp.height) } ?: false
            if (accepted) {
                Log.d(TAG, "PX/MM=%.4f (Fallback Hough) score=%.1f".format(g.pxPerMm, g.score))
                g.pxPerMm
            } else {
                Log.w(TAG, "Fallback Hough rechazado (score/centro/rango).")
                null
            }
        } ?: return runOnUi {
            showProcessing(false)
            Toast.makeText(this, getString(R.string.precal_toast_no_ref75), Toast.LENGTH_LONG)
                .show()
            finish()
        }

        val metrics = LensTracer.trace(
            bitmap = bmp,
            pxPerMm = pxPerMm,
            cameraId = cameraId,
            zoom = zoomRatio
        ) ?: return runOnUi {
            showProcessing(false)
            Toast.makeText(this, getString(R.string.precal_toast_trace_fail), Toast.LENGTH_LONG)
                .show()
            finish()
        }

        val outlinePx = metrics.outlineBuf.map { (x, y) -> android.graphics.PointF(x, y) }

        val shapeTrace = ShapeTraceResult(
            pxPerMm = metrics.pxPerMm,
            srcWidth = metrics.bufW,
            srcHeight = metrics.bufH,
            outlinePx = outlinePx,
            bboxWidthMm = null,
            bboxHeightMm = null,
            sharpness = metrics.sharpness,
            imageUri = uri
        )

        runOnUi {
            showProcessing(false)
            val intent = Intent(this, ResultsPrecalActivity::class.java)
                .putExtra(ResultsPrecalActivity.EXTRA_SHAPE_TRACE, shapeTrace)
                .putExtra(ResultsPrecalActivity.EXTRA_PRECAL_METRICS, metrics)
            startActivity(intent)
            finish()
        }
    }

    // ===== Fallback Hough: estima px/mm detectando el círculo Ø75 con corrección de brillos =====
    private data class Guide75Detection(
        val cxPx: Float,
        val cyPx: Float,
        val rPx: Float,
        val pxPerMm: Float,
        val score: Double
    )

    private fun detectGuide75(bmp: Bitmap): Guide75Detection? {
        val rgba = Mat()
        Utils.bitmapToMat(bmp, rgba)

        // Trabajamos en copia reducida (rápido, promedia ruido)
        val maxDimTarget = 1000.0
        val maxSide = max(rgba.cols(), rgba.rows()).toDouble()
        val scale = if (maxSide > maxDimTarget) maxDimTarget / maxSide else 1.0

        val work = Mat()
        if (scale < 1.0) {
            Imgproc.resize(rgba, work, CvSize(rgba.cols() * scale, rgba.rows() * scale))
        } else {
            rgba.copyTo(work)
        }

        val gray = Mat()
        Imgproc.cvtColor(work, gray, Imgproc.COLOR_RGBA2GRAY)

        // Máscara de brillos (zonas casi saturadas)
        val highlights = Mat()
        Imgproc.threshold(gray, highlights, 235.0, 255.0, Imgproc.THRESH_BINARY)
        Imgproc.morphologyEx(
            highlights,
            highlights,
            Imgproc.MORPH_CLOSE,
            Mat.ones(3, 3, CvType.CV_8U)
        )

        // Equalize + blur, y blanqueo de brillos para ignorar bordes internos
        Imgproc.equalizeHist(gray, gray)
        Imgproc.GaussianBlur(gray, gray, CvSize(7.0, 7.0), 0.0)
        gray.setTo(Scalar(230.0), highlights)

        val minDim = min(gray.rows(), gray.cols())
        val minR = (0.18 * minDim).toInt()
        val maxR = (0.48 * minDim).toInt()

        var bestR = 0.0
        var bestX = 0.0
        var bestY = 0.0
        var bestScore = -1.0

        val circles = Mat()
        val dps = doubleArrayOf(1.2, 1.4)
        val param2s = doubleArrayOf(38.0, 32.0, 26.0)

        for (dp in dps) {
            for (p2 in param2s) {
                Imgproc.HoughCircles(
                    gray, circles, Imgproc.HOUGH_GRADIENT,
                    dp, /*minDist*/ minDim * 0.6, /*param1*/ 160.0, /*param2*/ p2,
                    minR, maxR
                )
                if (circles.empty()) continue
                for (i in 0 until circles.cols()) {
                    val c = circles.get(0, i) ?: continue
                    val x = c[0]; val y = c[1]; val r = c[2]
                    val cx = gray.cols() / 2.0; val cy = gray.rows() / 2.0
                    val centerPenalty = hypot(x - cx, y - cy) / minDim
                    val score = r - 200.0 * centerPenalty
                    if (score > bestScore) {
                        bestScore = score; bestR = r; bestX = x; bestY = y
                    }
                }
            }
        }

        rgba.release(); work.release(); gray.release(); highlights.release(); circles.release()

        if (bestR <= 0.0) return null

        val invScale = if (scale < 1.0) (1.0 / scale) else 1.0
        val rFull = bestR * invScale
        val cxFull = bestX * invScale
        val cyFull = bestY * invScale
        val diameterPx = 2.0 * rFull
        val pxPerMm = (diameterPx / 75.0).toFloat()

        Log.d(TAG, "Hough75 r=$rFull @ ($cxFull,$cyFull) → diamPx=$diameterPx → px/mm=$pxPerMm (score=$bestScore)")

        return Guide75Detection(
            cxPx = cxFull.toFloat(),
            cyPx = cyFull.toFloat(),
            rPx = rFull.toFloat(),
            pxPerMm = pxPerMm,
            score = bestScore
        )
    }

    private fun acceptFallback(g: Guide75Detection, w: Int, h: Int): Boolean {
        val minDim = min(w, h).toFloat()
        val dx = g.cxPx - w / 2f
        val dy = g.cyPx - h / 2f
        val centerBias = hypot(dx.toDouble(), dy.toDouble()) / minDim
        val inRange = g.pxPerMm in 3f..20f
        // centrado razonable y score positivo
        return inRange && centerBias < 0.03 && g.score > 0.0
    }

    // ================= Helpers =================
    @SuppressLint("UseKtx")
    private fun decodeBitmap(uri: Uri, maxDim: Int = 2200): Bitmap? {
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                val source = ImageDecoder.createSource(contentResolver, uri)
                ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
                    decoder.isMutableRequired = false
                    decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                    val w = info.size.width
                    val h = info.size.height
                    val scale = if (w >= h) maxDim.toFloat() / w else maxDim.toFloat() / h
                    if (scale < 1f) decoder.setTargetSize((w * scale).toInt(), (h * scale).toInt())
                }
            } else {
                @Suppress("DEPRECATION")
                val legacy = MediaStore.Images.Media.getBitmap(contentResolver, uri)
                val w = legacy.width
                val h = legacy.height
                val scale = if (w >= h) maxDim.toFloat() / w else maxDim.toFloat() / h
                if (scale < 1f) legacy.scale((w * scale).toInt(), (h * scale).toInt()) else legacy
            }
        } catch (_: Throwable) { null }
    }

    private inline fun runOnUi(crossinline action: () -> Unit) {
        runOnUiThread { action() }
    }
}

/**
 * Analyzer en vivo para detectar el aro Ø75 en el papel y mover el overlay.
 */
private class Guide75Analyzer(
    private val overlay: OverlayGuide75View
) : ImageAnalysis.Analyzer {

    private var bufW: Int = 0
    private var bufH: Int = 0

    // Estado para estabilizar la detección
    private var lastCx = 0f
    private var lastCy = 0f
    private var lastR = 0f
    private var hasLast = false
    private var stableCount = 0

    override fun analyze(image: ImageProxy) {
        try {
            if (image.format != ImageFormat.YUV_420_888) {
                image.close(); return
            }

            val w = image.width
            val h = image.height
            if (bufW != w || bufH != h) {
                bufW = w; bufH = h
                overlay.post { overlay.setBufferSize(w, h) }
            }

            val matGray = image.toGrayMat()
            if (matGray.empty()) { image.close(); return }

            // Downscale rápido (~600 px lado mayor)
            val maxDimTarget = 600.0
            val maxSide = max(matGray.cols(), matGray.rows()).toDouble()
            val scale = if (maxSide > maxDimTarget) maxDimTarget / maxSide else 1.0

            val work = Mat()
            if (scale < 1.0) {
                Imgproc.resize(matGray, work, CvSize(matGray.cols() * scale, matGray.rows() * scale))
            } else {
                matGray.copyTo(work)
            }

            Imgproc.equalizeHist(work, work)
            Imgproc.GaussianBlur(work, work, CvSize(7.0, 7.0), 0.0)

            val highlights = Mat()
            Imgproc.threshold(work, highlights, 245.0, 255.0, Imgproc.THRESH_BINARY)
            Imgproc.morphologyEx(highlights, highlights, Imgproc.MORPH_CLOSE, Mat.ones(3, 3, CvType.CV_8U))
            work.setTo(Scalar(250.0), highlights)

            val minDim = min(work.rows(), work.cols())
            val minR = (0.18 * minDim).toInt()
            val maxR = (0.48 * minDim).toInt()

            val circles = Mat()
            Imgproc.HoughCircles(
                work, circles, Imgproc.HOUGH_GRADIENT,
                1.2, /*minDist*/ minDim * 0.6, /*param1*/ 160.0, /*param2*/ 32.0,
                minR, maxR
            )

            if (!circles.empty()) {
                val c = circles.get(0, 0)
                if (c != null) {
                    val xSmall = c[0].toFloat()
                    val ySmall = c[1].toFloat()
                    val rSmall = c[2].toFloat()

                    val invScale = if (scale < 1.0) (1.0 / scale).toFloat() else 1f
                    val cxBuf = xSmall * invScale
                    val cyBuf = ySmall * invScale
                    val rBuf = rSmall * invScale

                    if (hasLast) {
                        val dx = cxBuf - lastCx
                        val dy = cyBuf - lastCy
                        val dr = rBuf - lastR
                        val dist = hypot(dx.toDouble(), dy.toDouble())
                        val relPos = dist / min(bufW, bufH).toDouble()
                        val relR = kotlin.math.abs(dr) / rBuf.toDouble()
                        if (relPos < 0.02 && relR < 0.03) stableCount++ else stableCount = 0
                    } else {
                        stableCount = 0; hasLast = true
                    }

                    lastCx = cxBuf; lastCy = cyBuf; lastR = rBuf

                    if (stableCount >= 3) {
                        overlay.post { overlay.setCircleFromBuffer(cxBuf, cyBuf, rBuf) }
                    }
                }
            }

            work.release(); highlights.release(); circles.release(); matGray.release()
        } catch (_: Throwable) {
            // Frame perdido: ignorar
        } finally {
            image.close()
        }
    }
}

/**
 * Conversión de ImageProxy (YUV_420_888) a Mat en gris respetando rowStride.
 */
private fun ImageProxy.toGrayMat(): Mat {
    if (format != ImageFormat.YUV_420_888) return Mat()

    val yPlane = planes[0]
    val buffer = yPlane.buffer
    val rowStride = yPlane.rowStride
    val w = width
    val h = height

    val data = ByteArray(buffer.remaining())
    buffer.get(data)
    buffer.rewind()

    val mat = Mat(h, w, CvType.CV_8UC1)
    var srcPos = 0
    for (row in 0 until h) {
        mat.put(row, 0, data, srcPos, w)
        srcPos += rowStride
    }
    return mat
}
