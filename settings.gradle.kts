pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
    plugins {
        // Android Gradle Plugin
        id("com.android.application") version "8.5.2"
        // Kotlin Android
        id("org.jetbrains.kotlin.android") version "1.9.24"
        // (opcional, pero prolijo para parcelize)
        id("org.jetbrains.kotlin.plugin.parcelize") version "1.9.24"
    }
}

plugins {
    id("org.gradle.toolchains.foojay-resolver-convention") version "0.8.0"
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "PrecalDNP"
include(":app")
