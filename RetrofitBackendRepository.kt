package com.example.werkzeugerkennung

/**
 * Backend contract used by the UI.
 *
 * The concrete implementation can be:
 * - Python (embedded via Chaquopy, on-device), or
 * - Retrofit (HTTP, for remote/local server testing).
 */
interface BackendRepository {
    suspend fun getEnvSpec(): List<EnvSpecItem>
    suspend fun listObjects(): List<ObjectRecord>
    suspend fun createObject(name: String): ObjectRecord
    suspend fun getNextShot(objectId: String): NextShotResponse
    suspend fun submitShot(
        objectId: String,
        envCode: String,
        accept: Boolean,
        imageBytes: ByteArray
    ): SubmitShotResponse
    // Training selection is confirmed one-by-one by the user.
    suspend fun selectTrainingEnvironment(envCode: String): TrainingSelectionState
    // Read current selection progress (selected codes + remaining count).
    suspend fun getSelectedTrainingEnvs(): TrainingSelectionState
    suspend fun buildTrainingSets(): List<TrainingList>
    suspend fun getTrainingLists(): List<TrainingList>
    // Train FEAT from the 5-shot training lists.
    suspend fun trainFeatModel(): FeatTrainingResult
    // Phase 2 thresholds (set once, then lock).
    suspend fun getThresholds(): ThresholdConfigResponse
    suspend fun setThresholds(confThreshold: Double, marginThreshold: Double): ThresholdConfigResponse
    // Phase 2 inference mode selection.
    suspend fun getInferenceMode(): InferenceModeResponse
    suspend fun setInferenceMode(mode: String): InferenceModeResponse
    // Soft reset back to baseline.
    suspend fun getSoftResetStatus(): SoftResetStatusResponse
    suspend fun softReset(): SoftResetResponse
    // Reset backend state and delete all objects.
    suspend fun resetAll(): Boolean

    // Phase 2: inference + confirmation/correction flow.
    suspend fun runInference(imageBytes: ByteArray): InferenceResponse
    suspend fun confirmPrediction(pendingId: String, userConfirms: Boolean): ConfirmPredictionResponse
    suspend fun submitCorrection(
        pendingId: String,
        objectId: String,
        userConfirms: Boolean
    ): ConfirmPredictionResponse
    suspend fun unknownDecision(pendingId: String, isNew: Boolean): UnknownDecisionResponse
    suspend fun createObjectFromPending(
        pendingId: String,
        objectName: String
    ): CreateObjectFromPendingResponse
    suspend fun cancelPending(pendingId: String): Boolean
}
