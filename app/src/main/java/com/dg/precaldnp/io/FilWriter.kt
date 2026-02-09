// app/src/main/java/com/dg/precaldnp/io/FilWriter.kt
package com.dg.precaldnp.io

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.File
import java.io.FileOutputStream
import java.nio.charset.StandardCharsets
import java.util.Locale
import kotlin.math.hypot
import kotlin.math.min
import kotlin.math.roundToInt

object FilWriter {

    /**
     * Guarda un "trazado" (texto plano) SIN extensión en Documents/medirDNP.
     * Formato compatible visualmente con las líneas R= del .FIL (centésimas de mm).
     *
     * @param baseName  Nombre de archivo SIN extensión (p.ej. "TRAZ_20251103_174620")
     * @param pxPerMm   Escala usada (informativa)
     * @param radiiMm   Serie polar (N = 800 típicamente) en mm, ya rotada al “page angle”
     * @param centerMm  Centro estimado (x,y) en mm (informativo)
     * @param hboxMm    Ancho caja en mm (opcional; si null se calcula del trazado)
     * @param notes     Notas libres (theta, coverage, sharpness, cam, zoom, etc.)
     */
    fun saveTrazadoText(
        context: Context,
        baseName: String,
        pxPerMm: Float,
        radiiMm: FloatArray,
        centerMm: Pair<Float, Float>,
        hboxMm: Float? = null,
        notes: String? = null
    ): Uri? {
        // Caja desde el trazado (independiente del centro)
        val (hboxFromRadii, vboxFromRadii, eyeFromRadii) = computeBoxFromRadii(radiiMm)
        val usedHbox = hboxMm ?: hboxFromRadii

        val content = buildString {
            appendLine("TRAZADO=1")
            appendLine("PXPERMM=${pxPerMm.formatUS(4)}")
            appendLine("CENTERMM=${centerMm.first.formatUS(2)};${centerMm.second.formatUS(2)}")

            if (usedHbox > 0f) appendLine("HBOX=${usedHbox.formatUS(2)}")
            if (vboxFromRadii > 0f) appendLine("VBOX=${vboxFromRadii.formatUS(2)}")
            if (eyeFromRadii > 0f) appendLine("EYESIZ=${eyeFromRadii.formatUS(2)}")

            if (!notes.isNullOrBlank()) appendLine("NOTES=$notes")

            // R en centésimas de mm (14 por línea)
            if (radiiMm.isNotEmpty()) {
                val hundredths = IntArray(radiiMm.size) { i -> (radiiMm[i] * 100f).roundToInt() }
                val group = 14
                var idx = 0
                while (idx < hundredths.size) {
                    val end = min(idx + group, hundredths.size)
                    append("R=")
                    for (j in idx until end) {
                        append(hundredths[j])
                        append(';')
                    }
                    appendLine()
                    idx = end
                }
            }
        }.toByteArray(StandardCharsets.UTF_8)

        // Guardar en Documents/medirDNP SIN extensión
        return if (Build.VERSION.SDK_INT >= 29) {
            val cv = ContentValues().apply {
                put(MediaStore.Files.FileColumns.DISPLAY_NAME, baseName) // sin extensión
                put(MediaStore.Files.FileColumns.MIME_TYPE, "text/plain")
                put(MediaStore.Files.FileColumns.RELATIVE_PATH, "Documents/medirDNP")
            }
            val uri = context.contentResolver.insert(
                MediaStore.Files.getContentUri("external"), cv
            ) ?: return null

            context.contentResolver.openOutputStream(uri)?.use { it.write(content) } ?: return null
            uri
        } else {
            @Suppress("DEPRECATION")
            val docs = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
            val dir = File(docs, "medirDNP").apply { if (!exists()) mkdirs() }
            val outFile = File(dir, baseName) // sin extensión
            FileOutputStream(outFile).use { it.write(content) }
            Uri.fromFile(outFile)
        }
    }

    /**
     * Guarda un archivo .FIL completo (texto ya armado) en Documents/medirDNP.
     */
    fun saveFil(
        context: Context,
        baseName: String,
        filText: String
    ): Uri? {
        val bytes = filText.toByteArray(StandardCharsets.ISO_8859_1)
        val name = "$baseName.FIL"

        return if (Build.VERSION.SDK_INT >= 29) {
            val cv = ContentValues().apply {
                put(MediaStore.Files.FileColumns.DISPLAY_NAME, name)
                put(MediaStore.Files.FileColumns.MIME_TYPE, "text/plain")
                put(MediaStore.Files.FileColumns.RELATIVE_PATH, "Documents/medirDNP")
            }
            val uri = context.contentResolver.insert(
                MediaStore.Files.getContentUri("external"), cv
            ) ?: return null

            context.contentResolver.openOutputStream(uri)?.use { it.write(bytes) } ?: return null
            uri
        } else {
            @Suppress("DEPRECATION")
            val docs = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
            val dir = File(docs, "medirDNP").apply { if (!exists()) mkdirs() }
            val outFile = File(dir, name)
            FileOutputStream(outFile).use { it.write(bytes) }
            Uri.fromFile(outFile)
        }
    }

    fun saveDummyFil(
        context: Context,
        baseName: String,
        pxPerMm: Float,
        note: String? = null
    ): Uri? {
        val name = "$baseName.fil"
        val cv = ContentValues().apply {
            put(MediaStore.Files.FileColumns.DISPLAY_NAME, name)
            put(MediaStore.Files.FileColumns.MIME_TYPE, "text/plain")
            put(MediaStore.Files.FileColumns.RELATIVE_PATH, "Documents/medirDNP")
        }
        val uri = context.contentResolver.insert(
            MediaStore.Files.getContentUri("external"), cv
        )

        val body = buildString {
            appendLine("FIL v1")
            appendLine("pxPerMm=${pxPerMm.formatUS(4)}")
            appendLine("outline=[]")
            if (!note.isNullOrBlank()) appendLine("note=$note")
        }.toByteArray(StandardCharsets.UTF_8)

        uri?.let { context.contentResolver.openOutputStream(it)?.use { os -> os.write(body) } }
        return uri
    }

    // ----- helpers -----

    private fun Float.formatUS(decimals: Int): String =
        String.format(Locale.US, "%.${decimals}f", this)

    /**
     * Caja (HBOX/VBOX/EYESIZ) a partir de los radios en mm (distribuidos 360°).
     * Usa el mismo barrido angular que LensTracer: theta = -2π i / N.
     * Es independiente del centro que uses (la traslación no afecta anchos/altos).
     */
    private fun computeBoxFromRadii(radiiMm: FloatArray): Triple<Float, Float, Float> {
        if (radiiMm.isEmpty()) return Triple(0f, 0f, 0f)

        val n = radiiMm.size
        var minX = Float.POSITIVE_INFINITY
        var maxX = Float.NEGATIVE_INFINITY
        var minY = Float.POSITIVE_INFINITY
        var maxY = Float.NEGATIVE_INFINITY

        for (i in 0 until n) {
            val r = radiiMm[i]
            val theta = (-2.0 * Math.PI * i / n) // horario (consistente con LensTracer)
            val x = (r * kotlin.math.cos(theta)).toFloat()
            val y = (r * kotlin.math.sin(theta)).toFloat()
            if (x < minX) minX = x
            if (x > maxX) maxX = x
            if (y < minY) minY = y
            if (y > maxY) maxY = y
        }

        val hbox = (maxX - minX).coerceAtLeast(0f)
        val vbox = (maxY - minY).coerceAtLeast(0f)
        val eye  = if (hbox > 0f && vbox > 0f) hypot(hbox.toDouble(), vbox.toDouble()).toFloat() else 0f

        return Triple(hbox, vbox, eye)
    }
}
