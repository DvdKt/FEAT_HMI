package com.example.werkzeugerkennung

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import retrofit2.HttpException

/**
 * Phase 2 inference state machine.
 *
 * The UI follows this strict flow:
 * 1) runInference -> either CONFIRM_PREDICTION or ASK_NEW_OR_EXISTING
 * 2) if confirmed -> commit PostTraining shot
 * 3) if rejected/unknown -> correction loop or new object creation
 */
sealed class InferenceUiState {
    data object Idle : InferenceUiState()
    data object RunningInference : InferenceUiState()
    data class ConfirmPrediction(
        val pendingId: String,
        val prediction: InferencePrediction
    ) : InferenceUiState()
    data class AutoAccepted(
        val prediction: InferencePrediction
    ) : InferenceUiState()
    data class RejectPredictionChoice(
        val pendingId: String,
        val prediction: InferencePrediction
    ) : InferenceUiState()
    data class AskNewOrExisting(
        val pendingId: String,
        val prediction: InferencePrediction
    ) : InferenceUiState()
    data class Correction(val pendingId: String) : InferenceUiState()
    data class NewObjectName(val pendingId: String) : InferenceUiState()
    data class AskStartSequence(
        val objectId: String,
        val objectName: String
    ) : InferenceUiState()
}

class InferenceViewModel(
    private val repository: BackendRepository
) : ViewModel() {
    private val modeFullAutomatic = "full-automatic"

    private val _uiState = MutableStateFlow<InferenceUiState>(InferenceUiState.Idle)
    val uiState: StateFlow<InferenceUiState> = _uiState.asStateFlow()

    private val _objects = MutableStateFlow<List<ObjectRecord>>(emptyList())
    val objects: StateFlow<List<ObjectRecord>> = _objects.asStateFlow()

    private val _thresholds = MutableStateFlow<ThresholdConfigResponse?>(null)
    val thresholds: StateFlow<ThresholdConfigResponse?> = _thresholds.asStateFlow()

    private val _inferenceMode = MutableStateFlow<InferenceModeResponse?>(null)
    val inferenceMode: StateFlow<InferenceModeResponse?> = _inferenceMode.asStateFlow()

    private val _lastPrediction = MutableStateFlow<InferencePrediction?>(null)
    val lastPrediction: StateFlow<InferencePrediction?> = _lastPrediction.asStateFlow()

    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage.asStateFlow()

    private val _isSubmitting = MutableStateFlow(false)
    val isSubmitting: StateFlow<Boolean> = _isSubmitting.asStateFlow()

    fun loadObjects() {
        viewModelScope.launch {
            try {
                _objects.value = repository.listObjects()
            } catch (error: HttpException) {
                _errorMessage.value = BackendErrorParser.parseHttpException(error)
            } catch (error: Exception) {
                _errorMessage.value = error.message ?: "Unknown error"
            }
        }
    }

    fun loadInferenceMode() {
        viewModelScope.launch {
            try {
                _inferenceMode.value = repository.getInferenceMode()
            } catch (error: HttpException) {
                _errorMessage.value = BackendErrorParser.parseHttpException(error)
            } catch (error: Exception) {
                _errorMessage.value = error.message ?: "Unknown error"
            }
        }
    }

    fun setInferenceMode(mode: String) {
        viewModelScope.launch {
            _isSubmitting.value = true
            _errorMessage.value = null
            try {
                _inferenceMode.value = repository.setInferenceMode(mode)
            } catch (error: HttpException) {
                _errorMessage.value = BackendErrorParser.parseHttpException(error)
            } catch (error: Exception) {
                _errorMessage.value = error.message ?: "Unknown error"
            } finally {
                _isSubmitting.value = false
            }
        }
    }

    fun runInference(imageBytes: ByteArray) {
        viewModelScope.launch {
            _isSubmitting.value = true
            _errorMessage.value = null
            val mode = _inferenceMode.value?.mode?.trim()?.lowercase()
            if (mode.isNullOrBlank()) {
                _uiState.value = InferenceUiState.Idle
                _errorMessage.value = "Select Semi-Automatic or Full-Automatic before inference."
                _isSubmitting.value = false
                return@launch
            }
            _uiState.value = InferenceUiState.RunningInference
            _lastPrediction.value = null
            try {
                val response = repository.runInference(imageBytes)
                _lastPrediction.value = response.predicted
                refreshInferenceConfig()
                when (response.nextAction) {
                    "CONFIRM_PREDICTION" -> {
                        if (mode == modeFullAutomatic) {
                            val prediction = response.predicted
                            val autoResult = runCatching {
                                repository.confirmPrediction(response.pendingId, true)
                            }.getOrElse { error ->
                                _errorMessage.value = if (error is HttpException) {
                                    BackendErrorParser.parseHttpException(error)
                                } else {
                                    error.message ?: "Failed to auto-accept prediction."
                                }
                                _uiState.value = InferenceUiState.ConfirmPrediction(
                                    pendingId = response.pendingId,
                                    prediction = prediction
                                )
                                return@launch
                            }
                            if (autoResult.committed) {
                                _uiState.value = InferenceUiState.AutoAccepted(prediction)
                            } else if (autoResult.needsCorrection) {
                                _uiState.value = InferenceUiState.Correction(response.pendingId)
                                loadObjects()
                            } else {
                                _uiState.value = InferenceUiState.ConfirmPrediction(
                                    pendingId = response.pendingId,
                                    prediction = prediction
                                )
                            }
                        } else {
                            _uiState.value = InferenceUiState.ConfirmPrediction(
                                pendingId = response.pendingId,
                                prediction = response.predicted
                            )
                        }
                    }
                    "ASK_NEW_OR_EXISTING" -> {
                        _uiState.value = InferenceUiState.AskNewOrExisting(
                            pendingId = response.pendingId,
                            prediction = response.predicted
                        )
                    }
                    else -> {
                        _uiState.value = InferenceUiState.Idle
                        _errorMessage.value = "Unknown next action: ${response.nextAction}"
                    }
                }
            } catch (error: HttpException) {
                _errorMessage.value = BackendErrorParser.parseHttpException(error)
                _uiState.value = InferenceUiState.Idle
            } catch (error: Exception) {
                _errorMessage.value = error.message ?: "Unknown error"
                _uiState.value = InferenceUiState.Idle
            } finally {
                _isSubmitting.value = false
            }
        }
    }

    private suspend fun refreshInferenceConfig() {
        runCatching { repository.getThresholds() }
            .onSuccess { _thresholds.value = it }
    }

    fun confirmPrediction(userConfirms: Boolean) {
        val state = _uiState.value
        if (state !is InferenceUiState.ConfirmPrediction) return
        viewModelScope.launch {
            _isSubmitting.value = true
            _errorMessage.value = null
            try {
                val result = repository.confirmPrediction(state.pendingId, userConfirms)
                if (!userConfirms) {
                    if (result.allowNewObject) {
                        _uiState.value = InferenceUiState.RejectPredictionChoice(
                            pendingId = state.pendingId,
                            prediction = state.prediction
                        )
                    } else {
                        _uiState.value = InferenceUiState.Correction(state.pendingId)
                        loadObjects()
                    }
                } else if (result.needsCorrection) {
                    _uiState.value = InferenceUiState.Correction(state.pendingId)
                    loadObjects()
                } else {
                    _uiState.value = InferenceUiState.Idle
                    _lastPrediction.value = null
                }
            } catch (error: HttpException) {
                _errorMessage.value = BackendErrorParser.parseHttpException(error)
            } catch (error: Exception) {
                _errorMessage.value = error.message ?: "Unknown error"
            } finally {
                _isSubmitting.value = false
            }
        }
    }

    fun chooseRejectedPrediction(useNewObject: Boolean) {
        val state = _uiState.value
        if (state !is InferenceUiState.RejectPredictionChoice) return
        if (useNewObject) {
            _uiState.value = InferenceUiState.NewObjectName(state.pendingId)
        } else {
            _uiState.value = InferenceUiState.Correction(state.pendingId)
            loadObjects()
        }
    }

    fun decideUnknown(isNew: Boolean) {
        val state = _uiState.value
        if (state !is InferenceUiState.AskNewOrExisting) return
        viewModelScope.launch {
            _isSubmitting.value = true
            _errorMessage.value = null
            try {
                val result = repository.unknownDecision(state.pendingId, isNew)
                if (result.needsCorrection) {
                    _uiState.value = InferenceUiState.Correction(state.pendingId)
                    loadObjects()
                } else if (result.needsObjectName) {
                    _uiState.value = InferenceUiState.NewObjectName(state.pendingId)
                } else {
                    _uiState.value = InferenceUiState.Idle
                    _lastPrediction.value = null
                }
            } catch (error: HttpException) {
                _errorMessage.value = BackendErrorParser.parseHttpException(error)
            } catch (error: Exception) {
                _errorMessage.value = error.message ?: "Unknown error"
            } finally {
                _isSubmitting.value = false
            }
        }
    }

    fun submitCorrection(objectId: String, userConfirms: Boolean) {
        val state = _uiState.value
        if (state !is InferenceUiState.Correction) return
        if (!userConfirms) return
        viewModelScope.launch {
            _isSubmitting.value = true
            _errorMessage.value = null
            try {
                val result = repository.submitCorrection(state.pendingId, objectId, true)
                if (result.needsCorrection) {
                    _uiState.value = InferenceUiState.Correction(state.pendingId)
                } else {
                    _uiState.value = InferenceUiState.Idle
                    _lastPrediction.value = null
                }
            } catch (error: HttpException) {
                _errorMessage.value = BackendErrorParser.parseHttpException(error)
            } catch (error: Exception) {
                _errorMessage.value = error.message ?: "Unknown error"
            } finally {
                _isSubmitting.value = false
            }
        }
    }

    fun createObjectFromPending(objectName: String) {
        val state = _uiState.value
        if (state !is InferenceUiState.NewObjectName) return
        viewModelScope.launch {
            _isSubmitting.value = true
            _errorMessage.value = null
            try {
                val result = repository.createObjectFromPending(state.pendingId, objectName)
                if (result.askSequence && result.objectId != null && result.objectName != null) {
                    _uiState.value = InferenceUiState.AskStartSequence(
                        objectId = result.objectId,
                        objectName = result.objectName
                    )
                } else {
                    _uiState.value = InferenceUiState.Idle
                    _lastPrediction.value = null
                }
            } catch (error: HttpException) {
                _errorMessage.value = BackendErrorParser.parseHttpException(error)
            } catch (error: Exception) {
                _errorMessage.value = error.message ?: "Unknown error"
            } finally {
                _isSubmitting.value = false
            }
        }
    }

    fun cancelPending() {
        val pendingId = when (val state = _uiState.value) {
            is InferenceUiState.ConfirmPrediction -> state.pendingId
            is InferenceUiState.RejectPredictionChoice -> state.pendingId
            is InferenceUiState.AskNewOrExisting -> state.pendingId
            is InferenceUiState.Correction -> state.pendingId
            is InferenceUiState.NewObjectName -> state.pendingId
            else -> null
        } ?: return

        viewModelScope.launch {
            _isSubmitting.value = true
            _errorMessage.value = null
            try {
                repository.cancelPending(pendingId)
                _uiState.value = InferenceUiState.Idle
                _lastPrediction.value = null
            } catch (error: HttpException) {
                _errorMessage.value = BackendErrorParser.parseHttpException(error)
            } catch (error: Exception) {
                _errorMessage.value = error.message ?: "Unknown error"
            } finally {
                _isSubmitting.value = false
            }
        }
    }

    fun resetToIdle() {
        _uiState.value = InferenceUiState.Idle
        _lastPrediction.value = null
    }

    fun clearError() {
        _errorMessage.value = null
    }
}
