package com.example.werkzeugerkennung

import android.content.Context
import android.util.Base64
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class PythonBackendRepository(
    context: Context
) : BackendRepository {
    private val backendModule: PyObject

    init {
        // Start embedded Python and load the backend module.
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(context))
        }
        val python = Python.getInstance()
        backendModule = python.getModule("backend")

        // All backend state is stored under the app's private files directory.
        val baseDir = File(context.filesDir, "feat_backend").absolutePath
        val initResult = backendModule.callAttr("initialize_backend", baseDir).toMap()
        throwIfError(initResult)
    }

    override suspend fun getEnvSpec(): List<EnvSpecItem> = withContext(Dispatchers.IO) {
        val result = backendModule.callAttr("get_env_spec").toList()
        result.mapNotNull { item ->
            val map = item as? Map<*, *> ?: return@mapNotNull null
            EnvSpecItem(
                envCode = map["env_code"] as? String ?: "",
                envName = map["env_name"] as? String ?: "",
                friendlyName = map["friendly_name"] as? String ?: "",
                isClean = map["is_clean"] as? Boolean ?: false
            )
        }
    }

    override suspend fun listObjects(): List<ObjectRecord> = withContext(Dispatchers.IO) {
        val raw = backendModule.callAttr("list_objects").toAny()
        if (raw is Map<*, *>) {
            throwIfError(raw)
            return@withContext emptyList()
        }
        val list = raw as? List<*> ?: return@withContext emptyList()
        list.mapNotNull { item ->
            val map = item as? Map<*, *> ?: return@mapNotNull null
            ObjectRecord(
                objectId = map["object_id"] as? String ?: "",
                objectName = map["name"] as? String ?: map["object_name"] as? String ?: "",
                instanceId = map["instance_id"] as? String ?: "",
                completed = map["completed"] as? Boolean ?: false
            )
        }
    }

    override suspend fun createObject(name: String): ObjectRecord = withContext(Dispatchers.IO) {
        val result = backendModule.callAttr("create_object", name).toMap()
        throwIfError(result)
        ObjectRecord(
            objectId = result["object_id"] as? String ?: "",
            objectName = result["name"] as? String ?: "",
            instanceId = result["instance_id"] as? String ?: "",
            completed = result["completed"] as? Boolean ?: false
        )
    }

    override suspend fun getNextShot(objectId: String): NextShotResponse = withContext(Dispatchers.IO) {
        val result = backendModule.callAttr("get_next_required_shot", objectId).toMap()
        throwIfError(result)
        mapToNextShot(result)
    }

    override suspend fun submitShot(
        objectId: String,
        envCode: String,
        accept: Boolean,
        imageBytes: ByteArray
    ): SubmitShotResponse = withContext(Dispatchers.IO) {
        // Python expects base64 for image transport to keep the API simple and deterministic.
        val imageBase64 = Base64.encodeToString(imageBytes, Base64.NO_WRAP)
        val result = backendModule.callAttr(
            "submit_shot",
            objectId,
            envCode,
            imageBase64,
            accept,
            "jpg"
        ).toMap()
        throwIfError(result)
        val nextRequired = (result["next_required"] as? Map<*, *>)?.let { mapToNextShot(it) }
        val status = if (result["accepted"] as? Boolean == true) "accepted" else "rejected"
        val completed = result["object_completed"] as? Boolean ?: false
        SubmitShotResponse(
            status = status,
            nextRequiredShot = nextRequired,
            objectCompleted = completed
        )
    }

    override suspend fun selectTrainingEnvironment(envCode: String): TrainingSelectionState =
        withContext(Dispatchers.IO) {
            val result = backendModule.callAttr("select_training_environment", envCode, true).toMap()
            throwIfError(result)
            val selected = (result["selected_env_codes"] as? List<*>)?.mapNotNull { it as? String } ?: emptyList()
            TrainingSelectionState(
                selectedEnvCodes = selected,
                remainingToSelect = (result["remaining_to_select"] as? Number)?.toInt() ?: 0
            )
        }

    override suspend fun getSelectedTrainingEnvs(): TrainingSelectionState = withContext(Dispatchers.IO) {
        val result = backendModule.callAttr("get_selected_training_environments").toMap()
        throwIfError(result)
        val selected = (result["selected_env_codes"] as? List<*>)?.mapNotNull { it as? String } ?: emptyList()
        TrainingSelectionState(
            selectedEnvCodes = selected,
            remainingToSelect = (result["remaining_to_select"] as? Number)?.toInt() ?: 0
        )
    }

    override suspend fun buildTrainingSets(): List<TrainingList> = withContext(Dispatchers.IO) {
        val raw = backendModule.callAttr("build_training_sets_for_ui").toAny()
        if (raw is Map<*, *>) {
            throwIfError(raw)
            return@withContext emptyList()
        }
        val list = raw as? List<*> ?: return@withContext emptyList()
        list.mapNotNull { item ->
            val map = item as? Map<*, *> ?: return@mapNotNull null
            val shots = (map["shots"] as? List<*>)?.mapNotNull { shotItem ->
                val shotMap = shotItem as? Map<*, *> ?: return@mapNotNull null
                ShotRecord(
                    envCode = shotMap["env_code"] as? String ?: "",
                    sequence = (shotMap["sequence"] as? Number)?.toInt()
                )
            } ?: emptyList()
            val postTrainingShots = (map["post_training_shots"] as? List<*>)?.mapNotNull { shotItem ->
                val shotMap = shotItem as? Map<*, *> ?: return@mapNotNull null
                ShotRecord(
                    envCode = shotMap["env_code"] as? String ?: "PostTraining",
                    sequence = (shotMap["sequence"] as? Number)?.toInt()
                )
            } ?: emptyList()
            TrainingList(
                objectId = map["object_id"] as? String,
                trainingFile = map["training_file"] as? String,
                shots = shots,
                postTrainingShots = postTrainingShots
            )
        }
    }

    override suspend fun getTrainingLists(): List<TrainingList> = withContext(Dispatchers.IO) {
        val raw = backendModule.callAttr("list_training_sets").toAny()
        if (raw is Map<*, *>) {
            throwIfError(raw)
            return@withContext emptyList()
        }
        val list = raw as? List<*> ?: return@withContext emptyList()
        list.mapNotNull { item ->
            val map = item as? Map<*, *> ?: return@mapNotNull null
            val shots = (map["shots"] as? List<*>)?.mapNotNull { shotItem ->
                val shotMap = shotItem as? Map<*, *> ?: return@mapNotNull null
                ShotRecord(
                    envCode = shotMap["env_code"] as? String ?: "",
                    sequence = (shotMap["sequence"] as? Number)?.toInt()
                )
            } ?: emptyList()
            val postTrainingShots = (map["post_training_shots"] as? List<*>)?.mapNotNull { shotItem ->
                val shotMap = shotItem as? Map<*, *> ?: return@mapNotNull null
                ShotRecord(
                    envCode = shotMap["env_code"] as? String ?: "PostTraining",
                    sequence = (shotMap["sequence"] as? Number)?.toInt()
                )
            } ?: emptyList()
            TrainingList(
                objectId = map["object_id"] as? String,
                trainingFile = map["training_file"] as? String,
                shots = shots,
                postTrainingShots = postTrainingShots
            )
        }
    }

    override suspend fun trainFeatModel(): FeatTrainingResult = withContext(Dispatchers.IO) {
        val result = backendModule.callAttr("train_feat_model").toMap()
        throwIfError(result)
        FeatTrainingResult(
            trained = result["trained"] as? Boolean ?: false,
            modelPath = result["model_path"] as? String ?: ""
        )
    }

    override suspend fun getThresholds(): ThresholdConfigResponse {
        throw IllegalStateException("Thresholds are only available on the remote backend.")
    }

    override suspend fun setThresholds(
        confThreshold: Double,
        marginThreshold: Double
    ): ThresholdConfigResponse {
        throw IllegalStateException("Thresholds are only available on the remote backend.")
    }

    override suspend fun getInferenceMode(): InferenceModeResponse {
        throw IllegalStateException("Inference mode is only available on the remote backend.")
    }

    override suspend fun setInferenceMode(mode: String): InferenceModeResponse {
        throw IllegalStateException("Inference mode is only available on the remote backend.")
    }

    override suspend fun getSoftResetStatus(): SoftResetStatusResponse {
        throw IllegalStateException("Soft reset is only available on the remote backend.")
    }

    override suspend fun softReset(): SoftResetResponse {
        throw IllegalStateException("Soft reset is only available on the remote backend.")
    }

    override suspend fun resetAll(): Boolean = withContext(Dispatchers.IO) {
        val result = backendModule.callAttr("reset_all").toMap()
        throwIfError(result)
        result["ok"] as? Boolean ?: false
    }

    override suspend fun runInference(imageBytes: ByteArray): InferenceResponse {
        throw IllegalStateException("Inference is only available on the remote backend.")
    }

    override suspend fun confirmPrediction(
        pendingId: String,
        userConfirms: Boolean
    ): ConfirmPredictionResponse {
        throw IllegalStateException("Inference is only available on the remote backend.")
    }

    override suspend fun submitCorrection(
        pendingId: String,
        objectId: String,
        userConfirms: Boolean
    ): ConfirmPredictionResponse {
        throw IllegalStateException("Inference is only available on the remote backend.")
    }

    override suspend fun unknownDecision(
        pendingId: String,
        isNew: Boolean
    ): UnknownDecisionResponse {
        throw IllegalStateException("Inference is only available on the remote backend.")
    }

    override suspend fun createObjectFromPending(
        pendingId: String,
        objectName: String
    ): CreateObjectFromPendingResponse {
        throw IllegalStateException("Inference is only available on the remote backend.")
    }

    override suspend fun cancelPending(pendingId: String): Boolean {
        throw IllegalStateException("Inference is only available on the remote backend.")
    }

    private fun mapToNextShot(map: Map<*, *>): NextShotResponse {
        val envCode = map["env_code"] as? String ?: ""
        val remaining = (map["remaining_needed_for_env"] as? Number)?.toInt() ?: 0
        val completed = map["object_completed"] as? Boolean ?: false
        return NextShotResponse(envCode = envCode, remainingNeededForEnv = remaining, objectCompleted = completed)
    }

    private fun throwIfError(result: Map<*, *>) {
        val error = result["error"] as? Map<*, *> ?: return
        val code = error["code"] as? String ?: "ERROR"
        val message = error["message"] as? String ?: "Unknown error"
        throw IllegalStateException("$code: $message")
    }

    private fun PyObject.toAny(): Any? {
        return pyToAny(this, maxDepth = 8, seen = mutableSetOf())
    }

    private fun PyObject.toMap(): Map<String, Any?> {
        val converted = pyToAny(this, maxDepth = 8, seen = mutableSetOf())
        if (converted is Map<*, *>) {
            @Suppress("UNCHECKED_CAST")
            return converted as Map<String, Any?>
        }
        throw IllegalStateException("Expected map from Python, got ${converted?.javaClass}")
    }

    private fun PyObject.toList(): List<Any?> {
        val converted = pyToAny(this, maxDepth = 8, seen = mutableSetOf())
        if (converted is List<*>) {
            @Suppress("UNCHECKED_CAST")
            return converted as List<Any?>
        }
        throw IllegalStateException("Expected list from Python, got ${converted?.javaClass}")
    }

    private fun pyToAny(
        obj: PyObject,
        maxDepth: Int,
        seen: MutableSet<Long>
    ): Any? {
        if (maxDepth <= 0) {
            return obj.toString()
        }
        val objId = try {
            obj.id()
        } catch (_: Exception) {
            -1L
        }
        if (objId != -1L && !seen.add(objId)) {
            return obj.toString()
        }

        when (obj.safeTypeName()) {
            "dict" -> {
                val map = obj.asMap()
                return map.entries.associate { (key, value) ->
                    key.toSafeString() to pyToAny(value, maxDepth - 1, seen)
                }
            }
            "list", "tuple", "set" -> {
                return obj.asList().map { item ->
                    pyToAny(item, maxDepth - 1, seen)
                }
            }
            "str" -> return obj.toJava(String::class.java)
            "bool" -> return obj.toBoolean()
            "int" -> return obj.toLong()
            "float" -> return obj.toDouble()
            "NoneType" -> return null
            null -> {
                runCatching { obj.asMap() }.getOrNull()?.let { map ->
                    return map.entries.associate { (key, value) ->
                        key.toSafeString() to pyToAny(value, maxDepth - 1, seen)
                    }
                }
                runCatching { obj.asList() }.getOrNull()?.let { list ->
                    return list.map { item ->
                        pyToAny(item, maxDepth - 1, seen)
                    }
                }
            }
        }

        return obj.toString()
    }

    private fun PyObject.safeTypeName(): String? {
        val typeObj = runCatching { type() }.getOrNull()
        val fromAttr = runCatching {
            typeObj?.get("__name__")?.toJava(String::class.java)
        }.getOrNull()
        if (!fromAttr.isNullOrBlank()) {
            return fromAttr
        }
        val typeString = runCatching { typeObj?.toString() }.getOrNull() ?: return null
        val start = typeString.indexOf('\'')
        val end = typeString.lastIndexOf('\'')
        if (start != -1 && end > start) {
            return typeString.substring(start + 1, end)
        }
        return null
    }

    private fun PyObject.toSafeString(): String {
        return runCatching {
            toJava(String::class.java)
        }.getOrElse {
            toString()
        }
    }
}
