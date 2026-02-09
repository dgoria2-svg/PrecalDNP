// app/src/main/java/com/dg/medirdnp/App.kt
package com.dg.precaldnp

import android.app.Application
import android.util.Log
import com.dg.precaldnp.io.OutlineToFil
import com.dg.precaldnp.io.RadiusBiasLoader

class App : Application() {


    override fun onCreate() {
        super.onCreate()
        Log.i("App", "Precaldnp init")
        // Carga nativa de OpenCV (AAR 4.11+)
        try {
            org.opencv.android.OpenCVLoader.initLocal()
            Log.i("OpenCV", " Motor IA OK")
        } catch (t: Throwable) {
            Log.e("OpenCV", "Fallo del motor IA", t)
        }
        val bias800 = RadiusBiasLoader.loadFromAssets(this)
        OutlineToFil.setRadiusBias(bias800)
    }

}
