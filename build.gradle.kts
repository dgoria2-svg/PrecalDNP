// Top-level build file where you can add configuration options common to all sub-projects/modules.
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.parcelize) apply false
}
val compileSdkVersion by extra(36)
val kotlinJvmTarget by extra(JavaVersion.VERSION_17)
