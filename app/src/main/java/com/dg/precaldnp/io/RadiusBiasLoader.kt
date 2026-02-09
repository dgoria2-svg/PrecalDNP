package com.dg.precaldnp.io

import android.content.Context

object RadiusBiasLoader {

    fun loadFromAssets(
        context: Context,
        assetName: String = "RadiusBias.dat"
    ): FloatArray {
        context.assets.open(assetName).bufferedReader().use { br ->
            val values = br.lineSequence()
                .map { it.trim() }
                .filter { it.isNotEmpty() && !it.startsWith("#") }
                .map { it.toFloat() }
                .toList()

            require(values.size == 800) {
                "RadiusBias.dat debe tener 800 valores, tiene ${values.size}"
            }
            return values.toFloatArray()
        }
    }
}
