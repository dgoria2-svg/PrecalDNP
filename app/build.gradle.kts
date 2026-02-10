@file:Suppress("DEPRECATION")

import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.parcelize)
}

android {
    namespace = "com.dg.precaldnp"
    compileSdk = rootProject.extra["compileSdkVersion"] as Int

    androidResources {
        noCompress += "onnx"
    }

    defaultConfig {
        applicationId = "com.dg.precaldnp"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "1.0.0"
        vectorDrawables { useSupportLibrary = true }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        debug { isDebuggable = true }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures { viewBinding = true }

    packaging {
        resources { excludes += "/META-INF/{AL2.0,LGPL2.1}" }
        jniLibs { useLegacyPackaging = false }
    }

}

// ✅ Toolchain + jvmTarget 17 (para Kotlin)
kotlin { jvmToolchain(17) }

tasks.withType<KotlinCompile>().configureEach {
    compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
}

// (Opcional) yo lo sacaría salvo que lo necesites sí o sí:
configurations.all {
    resolutionStrategy {
        force(
            "androidx.core:core:1.13.1",
            "androidx.core:core-ktx:1.13.1"
        )
    }
}

dependencies {
    // ⚠️ Importante: NO mezcles "libs.androidx.camera.*" con strings con versión distinta.
    // Elegí uno solo (catálogo o hardcode) y que TODAS las camera-* tengan la MISMA versión.

    implementation("androidx.camera:camera-core:1.5.3")
    implementation(libs.androidx.camera.camera2)
    implementation("androidx.camera:camera-lifecycle:1.5.3")
    implementation("androidx.camera:camera-view:1.5.3")
    implementation("androidx.camera:camera-video:1.5.3")
    implementation("androidx.camera:camera-extensions:1.5.3")

    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.24.2")
    implementation("org.opencv:opencv:4.13.0")

    // ...el resto de tus deps

    implementation(libs.androidx.core.ktx.v1131)
    implementation(libs.androidx.appcompat.v170)
    implementation("androidx.activity:activity-ktx:1.12.4")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.10.0")
    implementation("androidx.annotation:annotation:1.9.1")
    implementation("androidx.concurrent:concurrent-futures-ktx:1.3.0")
    implementation("androidx.exifinterface:exifinterface:1.4.2")

    implementation(libs.tasks.vision)
    implementation(libs.face.detection)

    implementation("com.google.code.gson:gson:2.13.2")

    implementation(libs.material.v1120)
    implementation(libs.androidx.core.splashscreen)
    implementation(libs.androidx.remote.creation.compose)
    implementation(libs.androidx.ui.geometry)
}
