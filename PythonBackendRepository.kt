package com.example.werkzeugerkennung

import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import retrofit2.http.Path

interface BackendApi {
    @GET("/env-spec")
    suspend fun getEnvSpec(): List<EnvSpecItem>

    @GET("/objects")
    suspend fun listObjects(): List<ObjectRecord>

    @POST("/objects")
    suspend fun createObject(@Body body: Map<String, String>): ObjectRecord

    @GET("/objects/{object_id}/next-shot")
    suspend fun getNextShot(@Path("object_id") objectId: String): NextShotResponse

    @Multipart
    @POST("/objects/{object_id}/shots")
    suspend fun submitShot(
        @Path("object_id") objectId: String,
        @Part("env_code") envCode: RequestBody,
        @Part("accept") accept: RequestBody,
        @Part image_file: MultipartBody.Part
    ): SubmitShotResponse

    @GET("/training/selected-envs")
    suspend fun getSelectedTrainingEnvs(): TrainingSelectionState

    @POST("/training/selected-envs/confirm")
    suspend fun confirmTrainingEnv(@Body selection: TrainingEnvConfirm): TrainingSelectionState

    @POST("/training/selected-envs")
    suspend fun setSelectedTrainingEnvs(@Body selection: TrainingSelection)

    @POST("/training/build")
    suspend fun buildTrainingSets(): List<TrainingList>

    @GET("/training/lists")
    suspend fun getTrainingLists(): List<TrainingList>

    @POST("/training/feat")
    suspend fun trainFeatModel(): FeatTrainingResult

    @GET("/thresholds")
    suspend fun getThresholds(): ThresholdConfigResponse

    @POST("/thresholds")
    suspend fun setThresholds(@Body request: ThresholdConfigRequest): ThresholdConfigResponse

    @GET("/inference-mode")
    suspend fun getInferenceMode(): InferenceModeResponse

    @POST("/inference-mode")
    suspend fun setInferenceMode(@Body request: InferenceModeRequest): InferenceModeResponse

    @GET("/soft-reset/status")
    suspend fun getSoftResetStatus(): SoftResetStatusResponse

    @POST("/soft-reset")
    suspend fun softReset(): SoftResetResponse

    @POST("/reset")
    suspend fun resetAll(): ResetResponse

    @Multipart
    @POST("/inference")
    suspend fun runInference(
        @Part image_file: MultipartBody.Part
    ): InferenceResponse

    @POST("/confirm_prediction")
    suspend fun confirmPrediction(@Body request: ConfirmPredictionRequest): ConfirmPredictionResponse

    @POST("/submit_correction")
    suspend fun submitCorrection(@Body request: SubmitCorrectionRequest): ConfirmPredictionResponse

    @POST("/unknown_decision")
    suspend fun unknownDecision(@Body request: UnknownDecisionRequest): UnknownDecisionResponse

    @POST("/create_object_from_pending")
    suspend fun createObjectFromPending(
        @Body request: CreateObjectFromPendingRequest
    ): CreateObjectFromPendingResponse

    @POST("/cancel_pending")
    suspend fun cancelPending(@Body request: CancelPendingRequest): CancelPendingResponse
}
