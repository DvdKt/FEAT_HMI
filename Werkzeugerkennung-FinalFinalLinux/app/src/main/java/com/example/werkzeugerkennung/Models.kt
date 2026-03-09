package com.example.werkzeugerkennung

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class EnvSpecItem(
    @SerialName("env_code") val envCode: String,
    @SerialName("env_name") val envName: String = "",
    @SerialName("friendly_name") val friendlyName: String = "",
    @SerialName("is_clean") val isClean: Boolean = false
)

@Serializable
data class ShotRecord(
    @SerialName("env_code") val envCode: String,
    @SerialName("sequence") val sequence: Int? = null
)

@Serializable
data class ObjectRecord(
    @SerialName("object_id") val objectId: String,
    @SerialName("object_name") val objectName: String = "",
    @SerialName("instance_id") val instanceId: String = "",
    @SerialName("completed") val completed: Boolean = false
)

@Serializable
data class NextShotResponse(
    @SerialName("env_code") val envCode: String,
    @SerialName("remaining_needed_for_env") val remainingNeededForEnv: Int,
    @SerialName("object_completed") val objectCompleted: Boolean
)

@Serializable
data class SubmitShotResponse(
    @SerialName("status") val status: String,
    @SerialName("next_required_shot") val nextRequiredShot: NextShotResponse? = null,
    @SerialName("object_completed") val objectCompleted: Boolean = false
)

@Serializable
data class TrainingSelection(
    @SerialName("selected_env_codes") val selectedEnvCodes: List<String>
)

@Serializable
data class TrainingEnvConfirm(
    @SerialName("env_code") val envCode: String,
    @SerialName("confirm") val confirm: Boolean = true
)

@Serializable
data class TrainingSelectionState(
    @SerialName("selected_env_codes") val selectedEnvCodes: List<String>,
    @SerialName("remaining_to_select") val remainingToSelect: Int
)

@Serializable
data class TrainingList(
    @SerialName("object_id") val objectId: String? = null,
    @SerialName("training_file") val trainingFile: String? = null,
    @SerialName("shots") val shots: List<ShotRecord> = emptyList(),
    @SerialName("post_training_shots") val postTrainingShots: List<ShotRecord> = emptyList()
)

@Serializable
data class FeatTrainingResult(
    @SerialName("trained") val trained: Boolean,
    @SerialName("model_path") val modelPath: String = ""
)

@Serializable
data class InferenceProbability(
    @SerialName("object_id") val objectId: String? = null,
    @SerialName("object_name") val objectName: String? = null,
    @SerialName("prob") val prob: Double
)

@Serializable
data class InferencePrediction(
    @SerialName("object_id") val objectId: String? = null,
    @SerialName("object_name") val objectName: String? = null,
    @SerialName("probs_topk") val probsTopK: List<InferenceProbability> = emptyList(),
    @SerialName("max_prob") val maxProb: Double,
    @SerialName("second_prob") val secondProb: Double,
    @SerialName("margin") val margin: Double,
    @SerialName("passed_conf") val passedConf: Boolean,
    @SerialName("passed_margin") val passedMargin: Boolean,
    @SerialName("accepted_by_threshold") val acceptedByThreshold: Boolean,
    @SerialName("is_unknown") val isUnknown: Boolean,
    @SerialName("feat_confidence") val featConfidence: Double? = null
)

@Serializable
data class InferenceResponse(
    @SerialName("pending_id") val pendingId: String,
    @SerialName("predicted") val predicted: InferencePrediction,
    @SerialName("next_action") val nextAction: String
)

@Serializable
data class ConfirmPredictionRequest(
    @SerialName("pending_id") val pendingId: String,
    @SerialName("user_confirms") val userConfirms: Boolean
)

@Serializable
data class ConfirmPredictionResponse(
    @SerialName("needs_correction") val needsCorrection: Boolean = false,
    @SerialName("allow_new_object") val allowNewObject: Boolean = false,
    @SerialName("committed") val committed: Boolean = false,
    @SerialName("object_id") val objectId: String? = null,
    @SerialName("sequence") val sequence: String? = null
)

@Serializable
data class SubmitCorrectionRequest(
    @SerialName("pending_id") val pendingId: String,
    @SerialName("object_id") val objectId: String,
    @SerialName("user_confirms") val userConfirms: Boolean
)

@Serializable
data class UnknownDecisionRequest(
    @SerialName("pending_id") val pendingId: String,
    @SerialName("is_new") val isNew: Boolean
)

@Serializable
data class UnknownDecisionResponse(
    @SerialName("needs_correction") val needsCorrection: Boolean = false,
    @SerialName("needs_object_name") val needsObjectName: Boolean = false
)

@Serializable
data class CreateObjectFromPendingRequest(
    @SerialName("pending_id") val pendingId: String,
    @SerialName("object_name") val objectName: String
)

@Serializable
data class CreateObjectFromPendingResponse(
    @SerialName("created") val created: Boolean,
    @SerialName("object_id") val objectId: String? = null,
    @SerialName("object_name") val objectName: String? = null,
    @SerialName("committed") val committed: Boolean = false,
    @SerialName("ask_sequence") val askSequence: Boolean = false
)

@Serializable
data class CancelPendingRequest(
    @SerialName("pending_id") val pendingId: String
)

@Serializable
data class CancelPendingResponse(
    @SerialName("canceled") val canceled: Boolean
)

@Serializable
data class ThresholdConfigResponse(
    @SerialName("conf_threshold") val confThreshold: Double,
    @SerialName("margin_threshold") val marginThreshold: Double,
    @SerialName("locked") val locked: Boolean
)

@Serializable
data class ThresholdConfigRequest(
    @SerialName("conf_threshold") val confThreshold: Double,
    @SerialName("margin_threshold") val marginThreshold: Double
)

@Serializable
data class InferenceModeResponse(
    @SerialName("mode") val mode: String? = null
)

@Serializable
data class InferenceModeRequest(
    @SerialName("mode") val mode: String
)

@Serializable
data class SoftResetStatusResponse(
    @SerialName("can_soft_reset") val canSoftReset: Boolean,
    @SerialName("envs_selected") val envsSelected: Boolean,
    @SerialName("objects_created") val objectsCreated: Boolean,
    @SerialName("training_lists_complete") val trainingListsComplete: Boolean,
    @SerialName("thresholds_locked") val thresholdsLocked: Boolean,
    @SerialName("inference_started") val inferenceStarted: Boolean,
    @SerialName("baseline_objects") val baselineObjects: Int,
    @SerialName("missing") val missing: List<String> = emptyList()
)

@Serializable
data class SoftResetResponse(
    @SerialName("soft_reset") val softReset: Boolean,
    @SerialName("baseline_objects") val baselineObjects: Int,
    @SerialName("removed_objects") val removedObjects: List<String> = emptyList(),
    @SerialName("removed_post_training") val removedPostTraining: Map<String, Int> = emptyMap()
)

@Serializable
data class ResetResponse(
    @SerialName("ok") val ok: Boolean
)

@Serializable
data class ApiError(
    @SerialName("detail") val detail: String
)
