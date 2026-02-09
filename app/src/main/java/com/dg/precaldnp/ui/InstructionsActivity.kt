@file:Suppress("OVERRIDE_DEPRECATION")

package com.dg.precaldnp.ui

import android.content.ActivityNotFoundException
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.core.content.FileProvider
import com.dg.precaldnp.R
import com.dg.precaldnp.util.PrefsTerms
import java.io.File
import java.io.FileOutputStream

class InstructionsActivity : ComponentActivity() {

    companion object {
        // Nombre con el que se maneja el PDF dentro de la app
        private const val PDF_FILE_NAME = "precal_demolens_a4_100mm.pdf"
        private const val MIME_PDF = "application/pdf"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_instructions)

        // IDs tal cual tu XML
        val btnDownloadPdf: Button = findViewById(R.id.btnDownloadPdf)
        val btnSharePdf: Button = findViewById(R.id.btnSharePdf)
        val btnBack: Button = findViewById(R.id.btnBack)
        val btnAccept: Button = findViewById(R.id.btnAccept)

        // Ver / "descargar" PDF (abre visor si existe)
        btnDownloadPdf.setOnClickListener {
            openLocalPdf()
        }

        // Compartir PDF (WhatsApp, mail, imprimir, etc.)
        btnSharePdf.setOnClickListener {
            shareLocalPdf()
        }

        // Volver → NO acepta términos, vuelve a Welcome
        btnBack.setOnClickListener {
            PrefsTerms.setAccepted(this, false)
            goToWelcome()
        }

        // Aceptar → marca aceptado y vuelve a Welcome
        btnAccept.setOnClickListener {
            PrefsTerms.setAccepted(this, true)
            goToWelcome()
        }
    }

    // Si tocan back físico, mismo comportamiento que "Volver"
    @Suppress("DEPRECATION")
    override fun onBackPressed() {
        super.onBackPressed()
        PrefsTerms.setAccepted(this, false)
        goToWelcome()
    }

    private fun goToWelcome() {
        val intent = Intent(this, WelcomeActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP)
        }
        startActivity(intent)
        finish()
    }

    /**
     * Copia el PDF de res/raw a cache y devuelve un Uri de FileProvider.
     */
    private fun prepareLocalPdfUri(): Uri? {
        return try {
            // Recurso real: res/raw/precaldemolensA4.pdf
            val inputStream = resources.openRawResource(R.raw.precal_demolens_a4_100mm)

            val outFile = File(cacheDir, PDF_FILE_NAME)
            FileOutputStream(outFile).use { out ->
                inputStream.use { inp ->
                    inp.copyTo(out)
                }
            }

            FileProvider.getUriForFile(
                this,
                "$packageName.fileprovider",
                outFile
            )
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(
                this,
                getString(R.string.instr_download_failed),
                Toast.LENGTH_SHORT
            ).show()
            null
        }
    }

    /**
     * Intenta abrir el PDF local. Si no hay visor de PDF, avisa.
     */
    private fun openLocalPdf() {
        val uri = prepareLocalPdfUri() ?: return

        val intent = Intent(Intent.ACTION_VIEW).apply {
            setDataAndType(uri, MIME_PDF)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }

        try {
            startActivity(intent)
        } catch (_: ActivityNotFoundException) {
            Toast.makeText(
                this,
                getString(R.string.instr_download_failed),
                Toast.LENGTH_LONG
            ).show()
        }
    }

    /**
     * Comparte el PDF local (WhatsApp, mail, imprimir, etc.).
     */
    private fun shareLocalPdf() {
        val uri = prepareLocalPdfUri() ?: return

        val shareIntent = Intent(Intent.ACTION_SEND).apply {
            type = MIME_PDF
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }

        startActivity(
            Intent.createChooser(
                shareIntent,
                getString(R.string.instr_share_pdf)
            )
        )
    }
}
