package com.example.werkzeugerkennung

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import java.io.ByteArrayOutputStream
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
fun CameraCapture(
    modifier: Modifier = Modifier,
    onImageCaptured: (ByteArray) -> Unit,
    onError: (Throwable) -> Unit,
    captureTrigger: Boolean
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val previewView = remember {
        PreviewView(context).apply {
            // Use TextureView to avoid SurfaceView covering Compose text on some devices.
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }
    val imageCaptureState = remember { mutableStateOf<ImageCapture?>(null) }
    val scope = rememberCoroutineScope()
    val onImageCapturedState = rememberUpdatedState(onImageCaptured)
    val onErrorState = rememberUpdatedState(onError)

    LaunchedEffect(Unit) {
        val cameraProvider = ProcessCameraProvider.getInstance(context).get()
        val preview = Preview.Builder()
            .setTargetResolution(Size(1280, 720))
            .build()
            .also { it.setSurfaceProvider(previewView.surfaceProvider) }
        val imageCapture = ImageCapture.Builder()
            .setTargetResolution(Size(1280, 720))
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .build()
        val selector = CameraSelector.DEFAULT_BACK_CAMERA
        cameraProvider.unbindAll()
        cameraProvider.bindToLifecycle(lifecycleOwner, selector, preview, imageCapture)
        imageCaptureState.value = imageCapture
    }

    DisposableEffect(lifecycleOwner) {
        onDispose {
            val cameraProvider = ProcessCameraProvider.getInstance(context).get()
            cameraProvider.unbindAll()
        }
    }

    LaunchedEffect(captureTrigger) {
        if (captureTrigger) {
            val imageCapture = imageCaptureState.value
            if (imageCapture == null) {
                onErrorState.value(IllegalStateException("Camera not ready"))
                return@LaunchedEffect
            }
            val executor = ContextCompat.getMainExecutor(context)
            imageCapture.takePicture(
                executor,
                object : ImageCapture.OnImageCapturedCallback() {
                    override fun onCaptureSuccess(image: ImageProxy) {
                        scope.launch {
                            val bytes = image.toJpegByteArray()
                            image.close()
                            onImageCapturedState.value(bytes)
                        }
                    }

                    override fun onError(exception: ImageCaptureException) {
                        onErrorState.value(exception)
                    }
                }
            )
        }
    }

    Box(modifier = modifier) {
        AndroidView(
            factory = { previewView },
            modifier = Modifier
                .fillMaxWidth()
                .height(260.dp)
        )
    }
}

@SuppressLint("UnsafeOptInUsageError")
private suspend fun ImageProxy.toJpegByteArray(): ByteArray = withContext(Dispatchers.Default) {
    val buffer = planes.firstOrNull()?.buffer
    if (buffer != null && format == ImageFormat.JPEG) {
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return@withContext bytes
    }
    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)
    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuvImage = android.graphics.YuvImage(
        nv21,
        ImageFormat.NV21,
        width,
        height,
        null
    )
    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 90, out)
    val bitmap = BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())
    val rotated = bitmap.rotate(imageInfo.rotationDegrees)
    val result = ByteArrayOutputStream()
    rotated.compress(Bitmap.CompressFormat.JPEG, 90, result)
    result.toByteArray()
}

private fun Bitmap.rotate(rotationDegrees: Int): Bitmap {
    if (rotationDegrees == 0) return this
    val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
    return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
}
