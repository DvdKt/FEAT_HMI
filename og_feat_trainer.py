package com.example.werkzeugerkennung

import java.util.concurrent.TimeUnit
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import com.jakewharton.retrofit2.converter.kotlinx.serialization.asConverterFactory

class RetrofitBackendRepository(
    private val api: BackendApi
) : BackendRepository {
    override suspend fun getEnvSpec(): List<EnvSpecItem> = api.getEnvSpec()

    override suspend fun listObjects(): List<ObjectRecord> = api.listObjects()

    override suspend fun createObject(name: String): ObjectRecord =
        api.createObject(mapOf("object_name" to name))

    override suspend fun getNextShot(objectId: String): NextShotResponse = api.getNextShot(objectId)

    override suspend fun submitShot(
        objectId: String,
        envCode: String,
        accept: Boolean,
        imageBytes: ByteArray
    ): SubmitShotResponse {
        val envBody = envCode.toRequestBody("text/plain".toMediaType())
        val acceptBody = accept.toString().toRequestBody("text/plain".toMediaType())
        val imagePart = MultipartBody.Part.createFormData(
            name = "image_file",
            filename = "shot.jpg",
            body = imageBytes.toRequestBody("image/jpeg".toMediaType())
        )
        return api.submitShot(objectId, envBody, acceptBody, imagePart)
    }

    override suspend fun selectTrainingEnvironment(envCode: String): TrainingSelectionState {
        return api.confirmTrainingEnv(TrainingEnvConfirm(envCode, true))
    }

    override suspend fun getSelectedTrainingEnvs(): TrainingSelectionState {
        return api.getSelectedTrainingEnvs()
    }

    override suspend fun buildTrainingSets(): List<TrainingList> = api.buildTrainingSets()

    override suspend fun getTrainingLists(): List<TrainingList> = api.getTrainingLists()

    override suspend fun trainFeatModel(): FeatTrainingResult {
        return api.trainFeatModel()
    }

    override suspend fun getThresholds(): ThresholdConfigResponse {
        return api.getThresholds()
    }

    override suspend fun setThresholds(
        confThreshold: Double,
        marginThreshold: Double
    ): ThresholdConfigResponse {
        return api.setThresholds(ThresholdConfigRequest(confThreshold, marginThreshold))
    }

    override suspend fun getInferenceMode(): InferenceModeResponse {
        return api.getInferenceMode()
    }

    override suspend fun setInferenceMode(mode: String): InferenceModeResponse {
        return api.setInferenceMode(InferenceModeRequest(mode))
    }

    override suspend fun getSoftResetStatus(): SoftResetStatusResponse {
        return api.getSoftResetStatus()
    }

    override suspend fun softReset(): SoftResetResponse {
        return api.softReset()
    }

    override suspend fun resetAll(): Boolean {
        return api.resetAll().ok
    }

    override suspend fun runInference(imageBytes: ByteArray): InferenceResponse {
        val imagePart = MultipartBody.Part.createFormData(
            name = "image_file",
            filename = "inference.jpg",
            body = imageBytes.toRequestBody("image/jpeg".toMediaType())
        )
        return api.runInference(imagePart)
    }

    override suspend fun confirmPrediction(
        pendingId: String,
        userConfirms: Boolean
    ): ConfirmPredictionResponse {
        return api.confirmPrediction(ConfirmPredictionRequest(pendingId, userConfirms))
    }

    override suspend fun submitCorrection(
        pendingId: String,
        objectId: String,
        userConfirms: Boolean
    ): ConfirmPredictionResponse {
        return api.submitCorrection(SubmitCorrectionRequest(pendingId, objectId, userConfirms))
    }

    override suspend fun unknownDecision(
        pendingId: String,
        isNew: Boolean
    ): UnknownDecisionResponse {
        return api.unknownDecision(UnknownDecisionRequest(pendingId, isNew))
    }

    override suspend fun createObjectFromPending(
        pendingId: String,
        objectName: String
    ): CreateObjectFromPendingResponse {
        return api.createObjectFromPending(CreateObjectFromPendingRequest(pendingId, objectName))
    }

    override suspend fun cancelPending(pendingId: String): Boolean {
        return api.cancelPending(CancelPendingRequest(pendingId)).canceled
    }

    companion object {
        fun create(baseUrl: String, apiKey: String): RetrofitBackendRepository {
            val json = Json { ignoreUnknownKeys = true }
            val logging = HttpLoggingInterceptor().apply {
                level = HttpLoggingInterceptor.Level.BASIC
            }
            val authKey = apiKey.trim()
            val okHttpClient = OkHttpClient.Builder()
                .addInterceptor { chain ->
                    val request = if (authKey.isNotEmpty()) {
                        chain.request().newBuilder()
                            .addHeader("X-API-Key", authKey)
                            .build()
                    } else {
                        chain.request()
                    }
                    chain.proceed(request)
                }
                .addInterceptor(logging)
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(120, TimeUnit.SECONDS)
                .writeTimeout(120, TimeUnit.SECONDS)
                .callTimeout(120, TimeUnit.SECONDS)
                .build()
            val retrofit = Retrofit.Builder()
                .baseUrl(baseUrl)
                .addConverterFactory(json.asConverterFactory("application/json".toMediaType()))
                .client(okHttpClient)
                .build()
            return RetrofitBackendRepository(retrofit.create(BackendApi::class.java))
        }
    }
}
