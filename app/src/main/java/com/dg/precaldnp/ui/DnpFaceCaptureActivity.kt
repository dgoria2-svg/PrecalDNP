package com.dg.precaldnp.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.util.Rational
import android.view.Surface
import android.view.View
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.ViewPort
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.dg.precaldnp.R
import com.dg.precaldnp.model.DnpShotHint3250
import com.dg.precaldnp.model.ShapeTraceResult
import com.dg.precaldnp.model.WorkEye3250
import com.dg.precaldnp.model.withOppositePupilRoiGuidance3250
import com.dg.precaldnp.overlay.DnpFilOverlayFrame3250
import com.dg.precaldnp.vision.FilContourBuilder3250
import com.dg.precaldnp.vision.FilParser3250
import com.dg.precaldnp.vision.IrisDnpLandmarker3250
import com.dg.precaldnp.vision.PupilFrameEngine3250
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

@Suppress("unused")
class DnpFaceCaptureActivity : ComponentActivity() {

    companion object {
        private const val TAG = "DnpFaceCapture"

        const val EXTRA_FIL_URI = "extra_fil_uri"
        const val EXTRA_ORDER_ID = "extra_order_id"
        const val EXTRA_SHAPE_TRACE = "extra_shape_trace"

        private const val FIL_INNER_REDUCTION_TOTAL_MM_3250 = 2.0
        private const val FIL_OVER_INNER_MM_PER_SIDE_3250 =
            FIL_INNER_REDUCTION_TOTAL_MM_3250 / 2.0 // = 1.0

        const val EXTRA_CLEANUP_ON_EXIT_3250 = "extra_cleanup_on_exit_3250"
    }

    // --- Extras
    private lateinit var filUri3250: Uri
    private lateinit var orderId3250: String
    private lateinit var shapeTrace3250: ShapeTraceResult

    // --- FIL
    private lateinit var filOdMm3250: FilContourBuilder3250.ContourMm3250
    private lateinit var filOiMm3250: FilContourBuilder3250.ContourMm3250
    private var filHboxMm3250: Double = Double.NaN
    private var filVboxMm3250: Double = Double.NaN

    // --- CameraX
    private var cameraProvider: ProcessCameraProvider? = null
    private var camera: Camera? = null
    private var imageCapture: ImageCapture? = null

    // --- UI
    private lateinit var previewView: PreviewView
    private lateinit var overlay: DnpFilOverlayFrame3250
    private lateinit var tvInfo: TextView
    private lateinit var btnCapture: ImageButton
    private lateinit var layoutProcessing: View

    // --- Exec (separados)
    private lateinit var analysisExecutor: ExecutorService
    private lateinit var measureExecutor: ExecutorService

    // --- Engines
    private lateinit var pupilEngine3250: PupilFrameEngine3250
    private var irisEngine3250: IrisDnpLandmarker3250? = null

    // --- Runtime
    private var lastShotHint3250: DnpShotHint3250? = null
    private var frames3250 = 0

    @Volatile
    private var busyCapture3250 = false

    private val reqCam = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera() else {
            Toast.makeText(
                this,
                getString(R.string.dnp_camera_permission_required),
                Toast.LENGTH_LONG
            ).show()
            finish()
        }
    }

    private fun hasCameraPermission(): Boolean =
        ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_dnp_face_capture)

        previewView = findViewById(R.id.previewView)
        overlay = findViewById(R.id.overlayFrame)
        tvInfo = findViewById(R.id.tvInfoDnp)
        btnCapture = findViewById(R.id.btnCaptureFace)
        layoutProcessing = findViewById(R.id.layoutProcessing)

        previewView.implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        layoutProcessing.bringToFront()
        showProcessing(false)

        pupilEngine3250 = PupilFrameEngine3250(this)

        analysisExecutor = Executors.newSingleThreadExecutor()
        measureExecutor = Executors.newSingleThreadExecutor()

        // ===== extras
        val filUri = if (Build.VERSION.SDK_INT >= 33) {
            intent.getParcelableExtra(EXTRA_FIL_URI, Uri::class.java)
        } else {
            @Suppress("DEPRECATION") intent.getParcelableExtra(EXTRA_FIL_URI)
        }
        val orderId = intent.getStringExtra(EXTRA_ORDER_ID)
        val shapeTrace = if (Build.VERSION.SDK_INT >= 33) {
            intent.getParcelableExtra(EXTRA_SHAPE_TRACE, ShapeTraceResult::class.java)
        } else {
            @Suppress("DEPRECATION") intent.getParcelableExtra(EXTRA_SHAPE_TRACE)
        }

        if (filUri == null || orderId.isNullOrBlank() || shapeTrace == null) {
            Toast.makeText(this, getString(R.string.dnp_missing_inputs), Toast.LENGTH_LONG).show()
            finish()
            return
        }

        filUri3250 = filUri
        orderId3250 = orderId
        shapeTrace3250 = shapeTrace

        // ===== parse FIL
        val filText = DnpFacePipeline3250.readTextFromUri(this, filUri3250) ?: run {
            Toast.makeText(this, getString(R.string.dnp_cannot_read_fil), Toast.LENGTH_LONG).show()
            finish()
            return
        }

        val parsed = FilParser3250.parseFromText(filText)
        overlay.setFilRadiiMm3250(parsed.radiiMm, shiftSteps3250 = 0)

        val od = FilContourBuilder3250.buildOdAsIs3250(parsed.radiiMm)
        val oi = FilContourBuilder3250.flipOdToOi3250(od)
        filOdMm3250 = od
        filOiMm3250 = oi

        filHboxMm3250 = (parsed.hboxMm ?: (od.maxX - od.minX)).coerceAtLeast(0.0)
        filVboxMm3250 = (parsed.vboxMm ?: (od.maxY - od.minY)).coerceAtLeast(0.0)

        Log.d(TAG, "FIL mm: H=${"%.2f".format(filHboxMm3250)} V=${"%.2f".format(filVboxMm3250)}")

        irisEngine3250 = try {
            IrisDnpLandmarker3250(this)
        } catch (_: Throwable) {
            null
        }

        tvInfo.text = getString(R.string.dnp_face_order_label, orderId3250)

        btnCapture.setOnClickListener { startBurstAndMeasure3250() }
    }

    override fun onStart() {
        super.onStart()
        if (!hasCameraPermission()) reqCam.launch(Manifest.permission.CAMERA) else startCamera()
    }

    override fun onStop() {
        super.onStop()
        cameraProvider?.unbindAll()
    }

    override fun onDestroy() {
        super.onDestroy()
        try { analysisExecutor.shutdown() } catch (_: Throwable) {}
        try { measureExecutor.shutdown() } catch (_: Throwable) {}
        try { irisEngine3250?.close() } catch (_: Throwable) {}
    }

    // ============================================================
    // CameraX
    // ============================================================
    private fun startCamera() {
        previewView.post {
            val providerFuture = ProcessCameraProvider.getInstance(this)
            providerFuture.addListener({
                val provider = try {
                    providerFuture.get()
                } catch (t: Throwable) {
                    Log.e(TAG, "CameraProvider get() error", t)
                    Toast.makeText(
                        this,
                        getString(R.string.dnp_camera_start_failed),
                        Toast.LENGTH_LONG
                    ).show()
                    finish()
                    return@addListener
                }

                cameraProvider = provider

                val rotation = previewView.display?.rotation ?: Surface.ROTATION_0

                val resolutionSelector = ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                    .build()

                val preview = Preview.Builder()
                    .setResolutionSelector(resolutionSelector)
                    .setTargetRotation(rotation)
                    .build()
                    .also { it.surfaceProvider = previewView.surfaceProvider }

                /**
                 * ✅ CAPTURE “PNG real”:
                 * - Forzamos buffer YUV en memoria (sin JPEG intermedio).
                 * - El PNG final lo genera DnpBurstOrchestrator3250 (Bitmap.compress PNG).
                 * - MINIMIZE_LATENCY para poder meter 10 frames en ~1s.
                 */
                val ic = ImageCapture.Builder()
                    .setResolutionSelector(resolutionSelector)
                    .setTargetRotation(rotation)
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .setFlashMode(ImageCapture.FLASH_MODE_OFF)
                    .build()

                val ia = ImageAnalysis.Builder()
                    .setResolutionSelector(resolutionSelector)
                    .setTargetRotation(rotation)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()

                ia.setAnalyzer(analysisExecutor) { image ->
                    try {
                        frames3250++
                        val vw = previewView.width
                        val vh = previewView.height

                        if (vw > 0 && vh > 0) {
                            val hint = DnpFacePipeline3250.buildBaselineShotHint3250(
                                viewW = vw,
                                viewH = vh,
                                rotationDeg = image.imageInfo.rotationDegrees,
                                filHboxMm = filHboxMm3250,
                                filVboxMm = filVboxMm3250
                            )
                            runOnUiThread {
                                lastShotHint3250 = hint
                                overlay.invalidate()
                            }
                        }

                        if (frames3250 % 30 == 0) {
                            Log.d(
                                TAG,
                                "analyzer alive: frame=$frames3250 vw=${previewView.width} vh=${previewView.height} hint=${lastShotHint3250 != null}"
                            )
                        }
                    } catch (t: Throwable) {
                        Log.e(TAG, "analyzer error", t)
                    } finally {
                        image.close()
                    }
                }

                val vw = previewView.width
                val vh = previewView.height
                val aspect = if (vw > 0 && vh > 0) Rational(vw, vh) else Rational(4, 3)

                val viewPort = ViewPort.Builder(aspect, rotation)
                    .setScaleType(ViewPort.FIT)
                    .build()

                val group = UseCaseGroup.Builder()
                    .setViewPort(viewPort)
                    .addUseCase(preview)
                    .addUseCase(ic)
                    .addUseCase(ia)
                    .build()

                provider.unbindAll()
                camera = provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, group)
                imageCapture = ic

                try { camera?.cameraControl?.setZoomRatio(1.0f) } catch (_: Throwable) {}

            }, ContextCompat.getMainExecutor(this))
        }
    }

    // ============================================================
    // Burst + Measure
    // ============================================================

    private fun startBurstAndMeasure3250() {
        val ic = imageCapture ?: return
        if (busyCapture3250) return

        val snap = lastShotHint3250 ?: run {
            Toast.makeText(this, getString(R.string.dnp_overlay_not_ready), Toast.LENGTH_LONG).show()
            return
        }

        val okWorkBox = when (snap.workEye3250) {
            WorkEye3250.RIGHT_OD -> snap.rightBoxPreview != null && snap.rightBoxReal3250
            WorkEye3250.LEFT_OI -> snap.leftBoxPreview != null && snap.leftBoxReal3250
        }
        if (!okWorkBox) {
            Toast.makeText(this, getString(R.string.dnp_overlay_missing_box), Toast.LENGTH_LONG).show()
            return
        }

        val snapFixed = snap.withOppositePupilRoiGuidance3250(dxMm = 65f, roiScale = 0.90f)

        busyCapture3250 = true
        btnCapture.isEnabled = false
        showProcessing(true)

        Toast.makeText(this, getString(R.string.dnp_capturing_best_frames), Toast.LENGTH_SHORT).show()

        val cfg = DnpBurstOrchestrator3250.BurstConfig3250(
            durationMs = 1000L,
            maxShots = 10,
            delayMs = 80L,
            topK = 3
        )

        DnpBurstOrchestrator3250.run3250(
            ctx = this,
            imageCapture = ic,
            authority = "${packageName}.fileprovider",
            orderId = orderId3250,
            cfg = cfg,
            bgExecutor = measureExecutor,

            measure = { u ->
                DnpFacePipeline3250.measureArcFitOnlyCore3250(
                    ctx = this,
                    savedUri = u,
                    st = shapeTrace3250,
                    sp = snapFixed,
                    filHboxMm = filHboxMm3250,
                    filVboxMm = filVboxMm3250,
                    filOdMm = filOdMm3250,
                    filOiMm = filOiMm3250,
                    irisEngine = irisEngine3250,
                    pupilEngine = pupilEngine3250,
                    filOverInnerMmPerSide = FIL_OVER_INNER_MM_PER_SIDE_3250,
                    saveRingRoiDebug = false,
                    stage = DnpFacePipeline3250.StageCb3250 { resId ->
                        runOnUiThread {
                            // si querés mantener el orderId arriba, concatená en vez de pisar
                            tvInfo.text = getString(resId)
                        }
                    }
                )
            },

            score = { _, sharpness -> sharpness },

            cleanupTokens = { out ->
                val toks = ArrayList<String>(2)
                val dbg = out.dbgPath3250
                if (!dbg.isNullOrBlank()) toks.add(dbg)
                toks
            }

        ) { res ->
            busyCapture3250 = false
            btnCapture.isEnabled = true
            showProcessing(false)

            val out = res.bestOut
            if (out == null) {
                Toast.makeText(this, getString(R.string.dnp_measure_failed), Toast.LENGTH_LONG).show()
                return@run3250
            }

            val result = DnpFacePipeline3250.metricsToMeasurementResult3250(
                m = out.metrics,
                originalUriStr = out.originalUriStr,
                dbgUriStr = out.dbgPath3250
            )

            val intent = Intent(this, ResultsActivity::class.java).apply {
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)

                putExtra(ResultsActivity.KEY_RESULT, result)
                putExtra(ResultsActivity.KEY_IMAGE_URI, out.originalUriStr)
                putExtra(ResultsActivity.EXTRA_DEBUG_URI_3250, out.dbgPath3250)
                putStringArrayListExtra(EXTRA_CLEANUP_ON_EXIT_3250, ArrayList(res.cleanupOnExit))
            }

            startActivity(intent)
            finish()
        }
    }

    private fun showProcessing(show: Boolean) {
        layoutProcessing.visibility = if (show) View.VISIBLE else View.GONE
        if (show) layoutProcessing.bringToFront()
    }
}
