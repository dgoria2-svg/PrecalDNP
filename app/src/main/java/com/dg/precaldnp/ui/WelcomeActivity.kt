// app/src/main/java/com/dg/precaldnp/ui/WelcomeActivity.kt
package com.dg.precaldnp.ui

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import com.dg.precaldnp.R
import com.dg.precaldnp.util.PrefsTerms

class WelcomeActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        installSplashScreen()
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_welcome)

        val tvTitle = findViewById<TextView>(R.id.tvTitle)
        val tvSub   = findViewById<TextView>(R.id.tvSub)
        val btnLeft = findViewById<Button>(R.id.btnLeft)
        val btnRight= findViewById<Button>(R.id.btnRight)

        tvTitle.text = getString(R.string.welcome_title)
        tvSub.text   = getString(R.string.welcome_sub)

        // Etiquetas fijas: Ver instrucciones / Continuar
        btnLeft.text  = getString(R.string.btn_ver_instr)
        btnRight.text = getString(R.string.btn_continuar)

        // Ver instrucciones (el usuario acepta ahí)
        btnLeft.setOnClickListener {
            startActivity(Intent(this, InstructionsActivity::class.java))
        }

        // Continuar: sólo avanza si ya aceptó; si no, lo lleva a Instrucciones
        btnRight.setOnClickListener {
            if (PrefsTerms.isAccepted(this)) {
                startActivity(Intent(this, StillCaptureActivity::class.java))
                finish()
            } else {
                // Sin cerrar Welcome, abrimos Instrucciones para que acepte
                startActivity(Intent(this, InstructionsActivity::class.java))
            }
        }
    }
}
