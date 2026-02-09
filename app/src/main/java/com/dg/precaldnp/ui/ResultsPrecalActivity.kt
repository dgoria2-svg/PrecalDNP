package com.dg.precaldnp.ui

import android.content.Intent
import android.graphics.PointF
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.text.Editable
import android.text.TextWatcher
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.core.content.FileProvider
import androidx.core.net.toUri
import com.dg.precaldnp.R
import com.dg.precaldnp.io.OutlineToFil
import com.dg.precaldnp.model.PrecalMetrics
import com.dg.precaldnp.model.ShapeTraceResult
import java.io.File
import java.io.FileOutputStream
import java.util.Locale
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.roundToInt
import kotlin.math.sin

class ResultsPrecalActivity : ComponentActivity() {

    companion object {
        const val EXTRA_SHAPE_TRACE = "extra_shape_trace"
        const val EXTRA_PRECAL_METRICS = "extra_precal_metrics"

        private const val STATE_LAST_PNG = "state_last_png"
        private const val STATE_LAST_FIL = "state_last_fil"
    }

    // UI
    private lateinit var ivPhoto: ImageView
    private lateinit var traceView: PrecalLineView
    private lateinit var tvStatus: TextView
    private lateinit var etOrder: EditText
    private lateinit var btnSave: Button
    private lateinit var btnShare: Button
    private lateinit var btnBack: Button
    private lateinit var compareView: CompareLineView
    private lateinit var seekRotate: SeekBar
    private lateinit var btnDnp: Button

    // Estado
    private var lastPngUri: Uri? = null
    private var lastFilUri: Uri? = null
    private var shapeTrace: ShapeTraceResult? = null
    private var precalMetrics: PrecalMetrics? = null

    // Geometría para preview / FIL
    private var simpleGeo: SimpleGeo? = null

    // Ángulo actual del slider (°). Afecta preview y exportación.
    private var currentRotationDeg: Float = 0f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_results_precal)

        ivPhoto      = findViewById(R.id.ivPhoto)
        traceView    = findViewById(R.id.traceView)
        tvStatus     = findViewById(R.id.tvMetrics)
        etOrder      = findViewById(R.id.etOrderNumber)
        btnSave      = findViewById(R.id.btnSaveFil)
        btnShare     = findViewById(R.id.btnShareFil)
        btnBack      = findViewById(R.id.btnMenu)
        compareView  = findViewById(R.id.compareView)
        seekRotate   = findViewById(R.id.seekRotate)
        btnDnp = findViewById(R.id.btnDnp)

        if (savedInstanceState != null) {
            lastPngUri = savedInstanceState.getString(STATE_LAST_PNG)?.toUri()
            lastFilUri = savedInstanceState.getString(STATE_LAST_FIL)?.toUri()
        }

        // ShapeTrace para preview visual
        shapeTrace = if (Build.VERSION.SDK_INT >= 33) {
            intent.getParcelableExtra(EXTRA_SHAPE_TRACE, ShapeTraceResult::class.java)
        } else {
            @Suppress("DEPRECATION")
            intent.getParcelableExtra(EXTRA_SHAPE_TRACE)
        }

        // PrecalMetrics para cocinar el FIL real (si llegó)
        @Suppress("DEPRECATION")
        precalMetrics = intent.getParcelableExtra(EXTRA_PRECAL_METRICS)

        val st = shapeTrace
        if (st == null) {
            Toast.makeText(this, getString(R.string.precal_toast_missing_trace), Toast.LENGTH_LONG).show()
            tvStatus.setText(R.string.precal_status_missing)
            disableActions()
        } else {
            bindPreview(st)
            tvStatus.setText(R.string.precal_status_ok)
            disableActions()
            btnDnp.isEnabled = true

            // 1) Geometría base (si hay métricas, desde R; si no, fallback desde contorno)
            simpleGeo = precalMetrics?.let { metrics ->
                try { buildSimpleGeoFromPrecalMetrics(metrics) } catch (_: Throwable) { computeSimpleGeo(st) }
            } ?: computeSimpleGeo(st)

            // 2) Preview 2D: contorno (izq, rojo) vs FIL (der, azul)
            buildPreviewCompare()

            // 3) Métricas visibles iniciales (derivadas de R)
            updateMetricsUI()
        }

        etOrder.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: Editable?) {
                val orderNum = s?.toString()?.trim().orEmpty()
                val ok = orderNum.matches(Regex("^[0-9]+$"))
                btnSave.isEnabled = ok
                btnShare.isEnabled = ok
            }

        })


        // Rotación del FIL en el preview; HBOX/VBOX cambian con el eje; FED (2·Rmax) no.
        seekRotate.max = 360
        seekRotate.progress = 0
        seekRotate.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                currentRotationDeg = progress.toFloat() // sentido directo (tu CompareLineView gira la derecha)
                compareView.setRotationDegrees(currentRotationDeg)
                updateMetricsUI()
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {}
        })

        btnSave.setOnClickListener {
            val stLocal = shapeTrace ?: return@setOnClickListener
            val orderNum = etOrder.text.toString().trim()
            if (!orderNum.matches(Regex("^[0-9]+$"))) {
                Toast.makeText(this, getString(R.string.precal_toast_invalid_order), Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val uri = saveFilToDocuments(orderNum, stLocal)
            if (uri != null) {
                lastFilUri = uri
                Toast.makeText(this, getString(R.string.precal_toast_saved_ok), Toast.LENGTH_LONG).show()
            } else {
                Toast.makeText(this, getString(R.string.precal_toast_saved_error), Toast.LENGTH_LONG).show()
            }
        }

        btnShare.setOnClickListener {
            val stLocal = shapeTrace ?: return@setOnClickListener
            val orderNum = etOrder.text.toString().trim()
            if (!orderNum.matches(Regex("^[0-9]+$"))) {
                Toast.makeText(this, getString(R.string.precal_toast_invalid_order), Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            shareFil(orderNum, stLocal)
        }

        btnBack.setOnClickListener {
            val intent = Intent(this, WelcomeActivity::class.java)
            intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP)
            startActivity(intent)
            finish()
        }
        btnDnp.setOnClickListener {
            val stLocal = shapeTrace
            if (stLocal == null) {
                Toast.makeText(
                    this,
                    getString(R.string.precal_toast_missing_trace),
                    Toast.LENGTH_LONG
                ).show()
                return@setOnClickListener
            }

            val orderNum = etOrder.text.toString().trim()
            if (!orderNum.matches(Regex("^[0-9]+$"))) {
                Toast.makeText(
                    this,
                    "Ingresá un número de orden válido antes de medir DNP",
                    Toast.LENGTH_LONG
                ).show()
                return@setOnClickListener
            }

            val filUriLocal = lastFilUri
            if (filUriLocal == null) {
                Toast.makeText(
                    this,
                    "Guardá el FIL antes de medir DNP",
                    Toast.LENGTH_LONG
                ).show()
                return@setOnClickListener
            }

            val intent = Intent(this, DnpFaceCaptureActivity::class.java).apply {
                putExtra(DnpFaceCaptureActivity.EXTRA_SHAPE_TRACE, stLocal)
                putExtra(DnpFaceCaptureActivity.EXTRA_FIL_URI, filUriLocal)
                putExtra(DnpFaceCaptureActivity.EXTRA_ORDER_ID, orderNum)
            }

            android.util.Log.d(
                "ResultsPrecal",
                "Lanzando DnpFaceCapture: filUri=$filUriLocal orderId=$orderNum st.src=${stLocal.srcWidth}x${stLocal.srcHeight}"
            )

            startActivity(intent)
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putString(STATE_LAST_PNG, lastPngUri?.toString())
        outState.putString(STATE_LAST_FIL, lastFilUri?.toString())
    }

    private fun bindPreview(st: ShapeTraceResult) {
        st.imageUri?.let { uri ->
            ivPhoto.setImageURI(uri)
            lastPngUri = uri
        }
        traceView.setBufferSize(st.srcWidth, st.srcHeight)
        traceView.setTracePx(st.outlinePx, true)
    }

    private fun disableActions() {
        btnSave.isEnabled = false
        btnShare.isEnabled = false

    }

    // ============================ Geometría simple / mm ============================

    private data class SimpleGeo(
        val radiiHundredths: IntArray, // R en centésimas de mm (800)
        val hboxMm: Float,
        val vboxMm: Float,
        val circMm: Float, // perímetro
        val fedMm: Float   // EYESIZ = 2·Rmax
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is SimpleGeo) return false
            if (!radiiHundredths.contentEquals(other.radiiHundredths)) return false
            if (hboxMm != other.hboxMm) return false
            if (vboxMm != other.vboxMm) return false
            if (circMm != other.circMm) return false
            if (fedMm != other.fedMm) return false
            return true
        }
        override fun hashCode(): Int {
            var result = radiiHundredths.contentHashCode()
            result = 31 * result + hboxMm.hashCode()
            result = 31 * result + vboxMm.hashCode()
            result = 31 * result + circMm.hashCode()
            result = 31 * result + fedMm.hashCode()
            return result
        }
    }

    private data class GeoFromR(
        val hboxMm: Float,
        val vboxMm: Float,
        val fedMm: Float // 2·Rmax
    )

    private fun computeGeoFromRadiiHundredths(radiiHundredths: IntArray): GeoFromR {
        val n = radiiHundredths.size
        if (n == 0) return GeoFromR(0f, 0f, 0f)

        val twoPi = 2.0 * Math.PI
        val dTheta = twoPi / n

        var minX = 1e9
        var maxX = -1e9
        var minY = 1e9
        var maxY = -1e9
        var maxR = 0.0

        for (i in 0 until n) {
            val rMm = radiiHundredths[i] / 100.0
            val a = dTheta * i
            val x = -rMm * cos(a)
            val y =  rMm * sin(a)
            if (x < minX) minX = x
            if (x > maxX) maxX = x
            if (y < minY) minY = y
            if (y > maxY) maxY = y
            val rr = hypot(x, y)
            if (rr > maxR) maxR = rr
        }

        val hbox = (maxX - minX).toFloat()
        val vbox = (maxY - minY).toFloat()
        val fed  = (2.0 * maxR).toFloat()

        return GeoFromR(hbox, vbox, fed)
    }

    /** Desde métricas: usa radiiMm → R centésimas y deriva geometría. */
    private fun buildSimpleGeoFromPrecalMetrics(metrics: PrecalMetrics): SimpleGeo? {
        val rm = metrics.radiiMm
        if (rm.isEmpty()) return null

        val R = IntArray(rm.size) { i -> (rm[i] * 100f).roundToInt() }
        val geo = computeGeoFromRadiiHundredths(R)

        // perímetro aproximado en mm (polígono polar uniforme)
        val n = R.size
        val twoPi = 2.0 * Math.PI
        val dTheta = twoPi / n
        var circ = 0.0
        var prevX = -(R[0] / 100.0) * cos(0.0)
        var prevY =  (R[0] / 100.0) * sin(0.0)
        for (i in 1..n) {
            val a = dTheta * (i % n)
            val rMm = R[i % n] / 100.0
            val x = -rMm * cos(a)
            val y =  rMm * sin(a)
            circ += hypot(x - prevX, y - prevY)
            prevX = x; prevY = y
        }

        return SimpleGeo(
            radiiHundredths = R,
            hboxMm = geo.hboxMm,
            vboxMm = geo.vboxMm,
            circMm = circ.toFloat(),
            fedMm  = geo.fedMm
        )
    }

    /** Fallback simple si no hay métricas: aproxima R desde el contorno (discreto). */
    private fun computeSimpleGeo(st: ShapeTraceResult): SimpleGeo {
        // Tomamos outlinePx y lo convertimos a mm respecto del centroide (y-up)
        val leftMm = outlinePxToMmCentered(st)
        // “R” aproximado: distancia radial sobre N=800 ángulos
        val N = 800
        val dTh = (2.0 * Math.PI) / N
        val R = IntArray(N)
        for (i in 0 until N) {
            val a = dTh * i
            val dirX = -cos(a)
            val dirY =  sin(a)
            // proyección máx en la dirección (dirX,dirY)
            var rMax = 0.0
            for (p in leftMm) {
                val r = p.x * dirX + p.y * dirY
                if (r > rMax) rMax = r
            }
            R[i] = (rMax * 100.0).roundToInt()
        }
        val geo = computeGeoFromRadiiHundredths(R)

        // perímetro aproximado desde la polilínea “mm” original
        var circ = 0.0
        for (i in 0 until leftMm.size) {
            val a = leftMm[i]
            val b = leftMm[(i + 1) % leftMm.size]
            circ += hypot((b.x - a.x).toDouble(), (b.y - a.y).toDouble())
        }

        return SimpleGeo(
            radiiHundredths = R,
            hboxMm = geo.hboxMm,
            vboxMm = geo.vboxMm,
            circMm = circ.toFloat(),
            fedMm  = geo.fedMm
        )
    }

    // ============================ Preview rojo vs azul ============================

    private fun buildPreviewCompare() {
        val st = shapeTrace ?: return
        val geo = simpleGeo ?: return

        // IZQUIERDA (rojo): contorno de la foto → mm centrado (y-up)
        val leftMm = outlinePxToMmCentered(st)

        // DERECHA (azul): polilínea desde R (mm, centro 0,0)
        val rightMm = rToPolylineMm(geo.radiiHundredths)

        compareView.setTraces(leftMm, rightMm)
        compareView.setRotationDegrees(-currentRotationDeg)
    }

    private fun outlinePxToMmCentered(st: ShapeTraceResult): List<PointF> {
        val ptsPx = st.outlinePx
        if (ptsPx.isEmpty()) return emptyList()
        // Centroide en px
        var sx = 0f; var sy = 0f
        for (p in ptsPx) { sx += p.x; sy += p.y }
        val cx = sx / ptsPx.size
        val cy = sy / ptsPx.size

        val ppm = st.pxPerMm
        if (ppm <= 0f) return ptsPx // fallback: sin escala

        // A mm con origen en el centroide y eje Y hacia arriba
        return ptsPx.map { p ->
            val dx = (p.x - cx) / ppm
            val dy = (p.y - cy) / ppm
            PointF(dx, -dy)
        }
    }

    private fun rToPolylineMm(R: IntArray): List<PointF> {
        val n = R.size
        if (n == 0) return emptyList()
        val dTh = (2.0 * Math.PI) / n
        val out = ArrayList<PointF>(n + 1)
        for (i in 0 until n) {
            val mm = R[i] / 100.0
            val a  = dTh * i
            // convenio consistente con visor/FIL: x=-r cos, y=+r sin
            out.add(PointF((mm * cos(a)).toFloat(), (mm * sin(a)).toFloat()))
        }
        if (n >= 1) out.add(out.first())
        return out
    }

    // ============================ Métricas UI (derivadas de R) ============================

    private fun updateMetricsUI() {
        val geo = simpleGeo ?: return
        val R = rotateR(geo.radiiHundredths, currentRotationDeg)
        val g = computeGeoFromRadiiHundredths(R)
        tvStatus.text = "Trazado OK - HBOX %.2f VBOX %.2f FED %.2f".format(
            Locale.US, g.hboxMm, g.vboxMm, g.fedMm
        )
    }

    private fun rotateR(src: IntArray, angleDeg: Float): IntArray {
        if (src.isEmpty()) return src
        val n = src.size
        val shift = ((angleDeg / 360f) * n).toInt().mod(n)
        if (shift == 0) return src.clone()
        val out = IntArray(n)
        for (i in 0 until n) out[i] = src[(i - shift + n) % n]
        return out
    }

    // ============================ Guardar / Compartir FIL ============================

    private fun buildFilContent(orderNum: String, st: ShapeTraceResult): String {
        // Base desde métricas si hay; si no, generamos desde simpleGeo (R)
        val base: String = precalMetrics?.let { m ->
            runCatching { OutlineToFil.buildFilFromPrecal(orderNum, m) }.getOrElse {
                buildFilFromR(orderNum, simpleGeo?.radiiHundredths ?: IntArray(0))
            }
        } ?: buildFilFromR(orderNum, simpleGeo?.radiiHundredths ?: IntArray(0))

        // Aplicar giro actual a R= y recalcular métricas
        return applyAxisRotationToFilText(base, currentRotationDeg)
    }

    private fun buildFilFromR(orderNum: String, R: IntArray): String {
        val n = R.size
        val geo = computeGeoFromRadiiHundredths(R)
        val sb = StringBuilder()
        sb.appendLine("REQ=FIL")
        sb.appendLine("JOB=\"$orderNum\"")
        sb.appendLine("STATUS=0")
        sb.appendLine("TRCFMT=1;$n;E;R;F")
        val group = 8
        var i = 0
        while (i < n) {
            val end = minOf(i + group, n)
            sb.append("R=")
            for (k in i until end) {
                sb.append(R[k])
                if (k < end - 1) sb.append(';')
            }
            sb.appendLine()
            i = end
        }
        sb.appendLine()
        sb.appendLine("CIRC=%.2f;?".format(Locale.US, geoCircMm(R)))
        sb.appendLine("FED=%.2f;?".format(Locale.US, geo.fedMm))
        sb.appendLine("HBOX=%.2f;?".format(Locale.US, geo.hboxMm))
        sb.appendLine("VBOX=%.2f;?".format(Locale.US, geo.vboxMm))
        sb.appendLine()
        sb.appendLine("FMFR=PRECAL")
        sb.appendLine("FRAM=$orderNum")
        sb.appendLine("EYESIZ=%.2f".format(Locale.US, geo.fedMm))
        sb.appendLine()
        return sb.toString()
    }

    private fun geoCircMm(R: IntArray): Double {
        val pts = rToPolylineMm(R)
        var c = 0.0
        for (i in 0 until pts.size - 1) {
            val a = pts[i]; val b = pts[i + 1]
            c += hypot((b.x - a.x).toDouble(), (b.y - a.y).toDouble())
        }
        return c
    }

    /** Parsea R=..., los rota con el ángulo y reescribe R/HBOX/VBOX/FED/CIRC. */
    private fun applyAxisRotationToFilText(base: String, angleDeg: Float): String {
        val lines = base.split("\n")
        val R = ArrayList<Int>()
        val passthrough = ArrayList<String>()

        for (raw in lines) {
            val line = raw.trim()
            if (line.startsWith("R=")) {
                val body = line.substring(2)
                val toks = body.split(';')
                for (t in toks) {
                    val v = t.trim()
                    if (v.isNotEmpty()) v.toIntOrNull()?.let(R::add)
                }
            } else if (!line.startsWith("HBOX=") &&
                !line.startsWith("VBOX=") &&
                !line.startsWith("FED=")  &&
                !line.startsWith("CIRC=") &&
                line.isNotEmpty()) {
                passthrough.add(line)
            }
        }

        if (R.isEmpty()) return base // nada que rotar

        val rotated = rotateR(R.toIntArray(), angleDeg)
        val geo = computeGeoFromRadiiHundredths(rotated)
        val circ = geoCircMm(rotated)

        val out = StringBuilder()
        // 1) encabezado y todo lo que no sea R/HBOX/VBOX/FED/CIRC
        for (l in passthrough) out.appendLine(l)
        // 2) R= reescritos
        val group = 8
        var i = 0
        while (i < rotated.size) {
            val end = minOf(i + group, rotated.size)
            out.append("R=")
            for (k in i until end) {
                out.append(rotated[k])
                if (k < end - 1) out.append(';')
            }
            out.appendLine()
            i = end
        }
        out.appendLine()
        // 3) métricas recalculadas
        out.appendLine("CIRC=%.2f;?".format(Locale.US, circ))
        out.appendLine("FED=%.2f;?".format(Locale.US, geo.fedMm))
        out.appendLine("HBOX=%.2f;?".format(Locale.US, geo.hboxMm))
        out.appendLine("VBOX=%.2f;?".format(Locale.US, geo.vboxMm))
        out.appendLine()
        return out.toString()
    }

    private fun saveFilToDocuments(orderNum: String, st: ShapeTraceResult): Uri? {
        val text = buildFilContent(orderNum, st)
        // Carpeta app-specific (no requiere permisos ni API 29)
        val dir = getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)?.let { File(it, "Precal") }
            ?: File(filesDir, "PrecalDocs")
        if (!dir.exists()) dir.mkdirs()
        val file = File(dir, "$orderNum.FIL")
        return try {
            FileOutputStream(file).use { it.write(text.toByteArray(Charsets.UTF_8)) }
            file.toUri()
        } catch (_: Throwable) { null }
    }

    private fun shareFil(orderNum: String, st: ShapeTraceResult) {
        val text = buildFilContent(orderNum, st)
        // Archivo temporal en cache para compartir
        val cacheDir = File(cacheDir, "share"); if (!cacheDir.exists()) cacheDir.mkdirs()
        val file = File(cacheDir, "$orderNum.FIL")
        try {
            FileOutputStream(file).use { it.write(text.toByteArray(Charsets.UTF_8)) }
        } catch (_: Throwable) {
            Toast.makeText(this, getString(R.string.precal_toast_saved_error), Toast.LENGTH_LONG).show()
            return
        }
        val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", file)
        val send = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(Intent.createChooser(send, getString(R.string.trace_share_title)))
    }
}
