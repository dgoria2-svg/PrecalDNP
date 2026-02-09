plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("kotlin-parcelize")
}

android {
    namespace = "com.dg.precaldnp"
    compileSdk = 35

    // reemplaza a aaptOptions { noCompress "onnx" }
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
        debug {
            isDebuggable = true
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    kotlin {
        jvmToolchain(17)
    }

    buildFeatures {
        viewBinding = true
    }

    packaging {
        resources {
            // excluye metadatos duplicados
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
        jniLibs {
            // empaquetado moderno para .so
            useLegacyPackaging = false
        }
    }
}

dependencies {
    // --- CameraX 1.4.1 ---
    implementation("androidx.camera:camera-core:1.4.1")
    implementation("androidx.camera:camera-camera2:1.4.1")
    implementation("androidx.camera:camera-lifecycle:1.4.1")
    implementation("androidx.camera:camera-view:1.4.1")
    implementation("androidx.camera:camera-video:1.4.1")
    implementation("androidx.camera:camera-extensions:1.4.1") // si lo usás

    // --- ONNX Runtime ---
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.18.0")

    // --- Jetpack base ---
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.2")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.6")
    implementation("androidx.annotation:annotation:1.8.2")
    implementation("androidx.concurrent:concurrent-futures-ktx:1.2.0")
    implementation("androidx.exifinterface:exifinterface:1.3.7")

    // --- OpenCV (Android) ---

    implementation("org.opencv:opencv:4.11.0")
    // MediaPipe Tasks Vision (FaceLandmarker)
    implementation("com.google.mediapipe:tasks-vision:0.20230731")
// ML Kit Face Detection (modelo embebido en la app)
    implementation("com.google.mlkit:face-detection:16.1.7")


    // --- JSON (Prefs) ---
    implementation("com.google.code.gson:gson:2.11.0")

    // --- Material ---
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.core:core-splashscreen:1.0.1")
    implementation(libs.androidx.remote.creation.compose)
    implementation(libs.androidx.ui.geometry)


    configurations.all {
        resolutionStrategy {
            force(
                "androidx.core:core:1.13.1",
                "androidx.core:core-ktx:1.13.1"
            )
        }
    }
}
