package com.dg.precaldnp.ui

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.core.content.FileProvider
import androidx.core.net.toUri
import com.dg.precaldnp.R
import com.dg.precaldnp.model.MeasurementResult
import java.io.File
import java.util.Locale

@Suppress("DEPRECATION")
class ResultsActivity : ComponentActivity() {

    companion object {
        const val KEY_RESULT = "key_measurement_result"
        private const val STATE_RESULT = "state_result"

        // RAW (still original que querés conservar)
        const val KEY_IMAGE_URI = "KEY_IMAGE_URI"

        // DEBUG (por si querés ver dumps)
        const val EXTRA_DEBUG_URI_3250 = "EXTRA_DEBUG_URI_3250"
    }

    private var result: MeasurementResult? = null
    private var shareUri: Uri? = null

    private lateinit var iv: ImageView
    private lateinit var tvMetrics: TextView
    private lateinit var btnShare: Button
    private lateinit var btnNext: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_results)

        iv = findViewById(R.id.ivResult)
        tvMetrics = findViewById(R.id.tvMetrics)
        btnShare = findViewById(R.id.btnShare)
        btnNext = findViewById(R.id.btnNext)

        // 1) Resultado (puede ser null, NO cerramos la pantalla)
        result = savedInstanceState?.getParcelable(STATE_RESULT) ?: run {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                intent.getParcelableExtra(KEY_RESULT, MeasurementResult::class.java)
            } else {
                @Suppress("DEPRECATION")
                intent.getParcelableExtra(KEY_RESULT)
            }
        }

        // 2) Extras de imagen (RAW/DEBUG)
        val rawExtra = intent.getStringExtra(KEY_IMAGE_URI)
        val dbgExtra = intent.getStringExtra(EXTRA_DEBUG_URI_3250)

        val rawUri = parseUriForView(rawExtra)
        val dbgUri = parseUriForView(dbgExtra)

        // 3) URIs desde MeasurementResult (ANNOTATED/ORIGINAL)
        val annotatedUri = parseUriForView(result?.annotatedPath)
        val originalUri = parseUriForView(result?.originalPath)

        // 4) Qué mostramos: ANNOTATED > RAW > ORIGINAL > DEBUG
        val imgUriForView: Uri? = annotatedUri ?: rawUri ?: originalUri ?: dbgUri

        // 5) Qué compartimos: ANNOTATED > RAW > ORIGINAL > DEBUG
        shareUri = annotatedUri ?: rawUri ?: originalUri ?: dbgUri

        // 6) UI
        bindUi(
            imgUri = imgUriForView,
            rawExtra = rawExtra,
            dbgExtra = dbgExtra
        )

        btnShare.setOnClickListener {
            val u = shareUri ?: return@setOnClickListener
            val shareable = toShareableUri(u) ?: return@setOnClickListener

            val text = buildMetricsTextOrDebug(
                r = result,
                rawExtra = rawExtra,
                dbgExtra = dbgExtra,
                viewUri = imgUriForView,
                shareUri = u
            )

            // fallback silencioso
            copyToClipboard(text)

            val shareIntent = Intent(Intent.ACTION_SEND).apply {
                type = contentResolver.getType(shareable) ?: "image/*"
                putExtra(Intent.EXTRA_STREAM, shareable)
                putExtra(Intent.EXTRA_TEXT, text)
                putExtra(Intent.EXTRA_SUBJECT, getString(R.string.share_annotated_title))
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                clipData = ClipData.newRawUri("measurement", shareable)
            }

            startActivity(
                Intent.createChooser(
                    shareIntent,
                    getString(R.string.share_annotated_title)
                )
            )
        }

        btnNext.setOnClickListener {
            startActivity(Intent(this, DnpFaceCaptureActivity::class.java))
            finish()
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        result?.let { outState.putParcelable(STATE_RESULT, it) }
    }

    private fun bindUi(imgUri: Uri?, rawExtra: String?, dbgExtra: String?) {
        iv.contentDescription = getString(R.string.results_image_cd)

        // reset para evitar “cache visual” raro en algunos devices
        iv.setImageDrawable(null)
        if (imgUri != null) iv.setImageURI(imgUri)

        tvMetrics.text = buildMetricsTextOrDebug(
            r = result,
            rawExtra = rawExtra,
            dbgExtra = dbgExtra,
            viewUri = imgUri,
            shareUri = shareUri
        )

        btnShare.isEnabled = (shareUri != null)
        btnNext.isEnabled = true
    }

    /**
     * Convierte file:// (o paths sin scheme) a content:// usando FileProvider.
     * Si ya es content:// lo devuelve tal cual.
     */
    private fun toShareableUri(u: Uri): Uri? {
        return when (u.scheme) {
            "content" -> u
            "file", null -> {
                val path = u.path ?: return null
                val f = File(path)
                if (!f.exists()) return null
                FileProvider.getUriForFile(this, "${packageName}.fileprovider", f)
            }
            else -> u
        }
    }

    private fun copyToClipboard(text: String) {
        try {
            val cm = getSystemService(CLIPBOARD_SERVICE) as ClipboardManager
            cm.setPrimaryClip(ClipData.newPlainText("results", text))
        } catch (_: Throwable) { }
    }

    private fun parseUriForView(s: String?): Uri? {
        val str = s?.trim().orEmpty()
        if (str.isEmpty()) return null

        return when {
            str.startsWith("content://") -> str.toUri()
            str.startsWith("file://") -> str.toUri()
            str.startsWith("/") -> Uri.fromFile(File(str))
            else -> str.toUri()
        }
    }

    private fun buildMetricsTextOrDebug(
        r: MeasurementResult?,
        rawExtra: String?,
        dbgExtra: String?,
        viewUri: Uri?,
        shareUri: Uri?
    ): String {
        if (r != null) return buildMetricsText(r)

        fun exists(u: Uri?): String {
            if (u == null) return "null"
            return when (u.scheme) {
                "file", null -> {
                    val p = u.path
                    if (p.isNullOrBlank()) "file:? (no path)" else "file exists=${File(p).exists()}"
                }
                "content" -> "content:// (ok)"
                else -> "${u.scheme}://"
            }
        }

        return buildString {
            appendLine("DEBUG RESULT MODE (result=null)")
            appendLine("rawExtra=${rawExtra ?: "null"}")
            appendLine("dbgExtra=${dbgExtra ?: "null"}")
            appendLine("viewUri=${viewUri ?: "null"} -> ${exists(viewUri)}")
            appendLine("shareUri=${shareUri ?: "null"} -> ${exists(shareUri)}")
            appendLine()
            appendLine("Tip: mandame EXTRA_DEBUG_URI_3250 como String content:// o file:// y se ve igual.")
        }
    }

    private fun buildMetricsText(r: MeasurementResult): String {
        fun f(x: Double) = if (x.isFinite()) String.format(Locale.US, "%.2f", x) else "-"

        val dnpTotalMm =
            if (r.dnpOdMm.isFinite() && r.dnpOiMm.isFinite()) (r.dnpOdMm + r.dnpOiMm) else Double.NaN

        return buildString {
            appendLine(getString(R.string.results_line_main_fmt, f(r.anchoMm), f(r.altoMm), f(r.diagMayorMm)))
            appendLine(getString(R.string.results_line_bridge_fmt, f(r.puenteMm)))
            appendLine(getString(R.string.results_line_dnp_fmt, f(r.dnpOdMm), f(r.dnpOiMm)))
            appendLine(getString(R.string.results_line_dnp_total_fmt, f(dnpTotalMm)))
            appendLine(getString(R.string.results_line_alt_fmt, f(r.altOdMm), f(r.altOiMm)))
            append(getString(R.string.results_line_diam_fmt, f(r.diamUtilOdMm), f(r.diamUtilOiMm)))
        }
    }
}
