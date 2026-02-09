package com.dg.precaldnp.vision

import java.io.File
import kotlin.math.min

object FilParser3250 {

    data class FilRadii3250(
        val radiiMm: DoubleArray, // tamaño 800
        val n: Int,
        val hboxMm: Double?,      // puede venir null si no está en el FIL
        val vboxMm: Double?       // puede venir null si no está en el FIL
    )

    fun parseFromText(text: String, expectedN: Int = 800): FilRadii3250 {
        val values = ArrayList<Double>(expectedN)

        // -------- HBOX / VBOX (primer match que aparezca) --------
        fun parseMmField(fieldName: String): Double? {
            // admite: HBOX=55.10  |  HBOX = 5510  |  HBOX=55.10;
            val re = Regex("""(?i)\b${fieldName}\s*=\s*([-+]?\d+(?:\.\d+)?)""")
            val m = re.find(text) ?: return null
            val raw = m.groupValues[1].toDoubleOrNull() ?: return null
            // misma heurística que R: si es grande, probablemente centésimas
            return if (raw >= 200.0) raw / 100.0 else raw
        }

        val hboxMm = parseMmField("HBOX")
        val vboxMm = parseMmField("VBOX")

        // -------- R= (pueden venir separadas en varias líneas) --------
        // Captura líneas R= ... (pueden venir separadas en varias líneas)
        // Acepta separadores ; , espacios, tabs
        val regexR = Regex("""(?m)^\s*R\s*=\s*(.+)$""")
        val matches = regexR.findAll(text)

        for (m in matches) {
            val payload = m.groupValues[1]
            val tokens = payload
                .replace(',', ';')
                .split(';', ' ', '\t')
                .map { it.trim() }
                .filter { it.isNotEmpty() }

            for (t in tokens) {
                val v = t.toDoubleOrNull() ?: continue
                // En muchos FIL los radios vienen en centésimas (int). Si ya vienen en mm (float), también sirve.
                val mm = if (v >= 200.0) v / 100.0 else v
                values.add(mm)
                if (values.size >= expectedN) break
            }
            if (values.size >= expectedN) break
        }

        val out = DoubleArray(expectedN)
        val n = min(values.size, expectedN)
        for (i in 0 until n) out[i] = values[i]
        // Si faltan, repetimos el último para no crashear (mejor que null)
        for (i in n until expectedN) out[i] = if (n > 0) out[n - 1] else 0.0

        return FilRadii3250(
            radiiMm = out,
            n = expectedN,
            hboxMm = hboxMm,
            vboxMm = vboxMm
        )
    }

    fun parseFromFile(file: File, expectedN: Int = 800): FilRadii3250 =
        parseFromText(file.readText(), expectedN)
}
