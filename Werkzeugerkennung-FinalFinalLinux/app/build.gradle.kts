import com.chaquo.python.PythonExtension
import org.gradle.api.file.SourceDirectorySet
import org.gradle.api.plugins.ExtensionAware

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.serialization")
    id("com.chaquo.python")
}

android {
    namespace = "com.example.werkzeugerkennung"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.werkzeugerkennung"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        (this as ExtensionAware).extensions.configure<PythonExtension>("python") {
            version = "3.11"
            // Force Chaquopy to use a supported build-time Python (not 3.13).
            // On macOS with Homebrew, `python3.11` is available after `brew install python@3.11`.
            buildPython = listOf("python3.11")
            pip {
                install("numpy")
                install("Pillow")
            }
        }

        ndk {
            abiFilters += listOf("armeabi-v7a", "arm64-v8a", "x86", "x86_64")
        }
    }

    buildFeatures {
        compose = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.11"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }

    sourceSets {
        getByName("main") {
            val pythonSrc = (this as ExtensionAware)
                .extensions.getByName("python") as SourceDirectorySet
            pythonSrc.srcDir("src/main/python")
        }
    }
}

dependencies {
    val lifecycleVersion = "2.7.0"
    val retrofitVersion = "2.11.0"
    val okHttpVersion = "4.12.0"
    val serializationVersion = "1.6.3"
    val cameraXVersion = "1.3.3"

    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-compose:1.9.0")
    implementation("androidx.compose.ui:ui:1.6.7")
    implementation("androidx.compose.material3:material3:1.2.1")
    implementation("androidx.compose.ui:ui-tooling-preview:1.6.7")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:${lifecycleVersion}")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:${lifecycleVersion}")

    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:${serializationVersion}")
    implementation("com.squareup.retrofit2:retrofit:${retrofitVersion}")
    implementation("com.jakewharton.retrofit:retrofit2-kotlinx-serialization-converter:0.8.0")
    implementation("com.squareup.okhttp3:okhttp:${okHttpVersion}")
    implementation("com.squareup.okhttp3:logging-interceptor:${okHttpVersion}")

    implementation("androidx.camera:camera-core:${cameraXVersion}")
    implementation("androidx.camera:camera-camera2:${cameraXVersion}")
    implementation("androidx.camera:camera-lifecycle:${cameraXVersion}")
    implementation("androidx.camera:camera-view:${cameraXVersion}")

    implementation("io.coil-kt:coil-compose:2.6.0")

    debugImplementation("androidx.compose.ui:ui-tooling:1.6.7")
}
