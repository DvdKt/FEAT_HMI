package com.example.werkzeugerkennung

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import retrofit2.HttpException

class TrainingViewModel(
    private val repository: BackendRepository
) : ViewModel() {
    // Tracks which environments have been confirmed so far and how many remain.
    private val _selectionState = MutableStateFlow(TrainingSelectionState(emptyList(), 3))
    val selectionState: StateFlow<TrainingSelectionState> = _selectionState.asStateFlow()

    private val _trainingLists = MutableStateFlow<List<TrainingList>>(emptyList())
    val trainingLists: StateFlow<List<TrainingList>> = _trainingLists.asStateFlow()

    private val _featResult = MutableStateFlow<FeatTrainingResult?>(null)
    val featResult: StateFlow<FeatTrainingResult?> = _featResult.asStateFlow()

    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage.asStateFlow()

    private val _isSubmitting = MutableStateFlow(false)
    val isSubmitting: StateFlow<Boolean> = _isSubmitting.asStateFlow()

    fun loadSelection() {
        viewModelScope.launch {
            _isSubmitting.value = true
            _errorMessage.value = null
            try {
                // Pull current selection progress from the backend.
                _selectionState.value = repository.getSelectedTrainingEnvs()
            } catch (error: HttpException) {
                _errorMessage.value = BackendErrorParser.parseHttpException(error)
            } catch (error: Exception) {
                _errorMessage.value = error.message ?: "Unknown error"
            } finally {
                _isSubmitting.value = false
            }
        }
    }

    fun confirmSelection(envCode: String) {
        viewModelScope.launch {
            _isSubmitting.value = true
            _errorMessage.value = null
            try {
                // Confirms a single environment choice (one-by-one flow).
                _selectionState.value = repository.selectTrainingEnvironment(envCode)
            } catch (error: HttpException) {
                _errorMessage.value = BackendErrorParser.parseHttpException(error)
            } catch (error: Exception) {
                _errorMessage.value = error.message ?: "Unknown error"
            } finally {
                _isSubmitting.value = false
            }
        }
    }

    fun buildTrainingSets() {
        viewModelScope.launch {
            _isSubmitting.value = true
            _errorMessage.value = null
            _featResult.value = null
            try {
                // Build the 5-shot training lists, then trigger FEAT training.
                _trainingLists.value = repository.buildTrainingSets()
                _featResult.value = repository.trainFeatModel()
            } catch (error: HttpException) {
                _errorMessage.value = BackendErrorParser.parseHttpException(error)
            } catch (error: Exception) {
                _errorMessage.value = error.message ?: "Unknown error"
            } finally {
                _isSubmitting.value = false
            }
        }
    }

    fun loadTrainingLists() {
        viewModelScope.launch {
            _isSubmitting.value = true
            _errorMessage.value = null
            try {
                _trainingLists.value = repository.getTrainingLists()
            } catch (error: HttpException) {
                _errorMessage.value = BackendErrorParser.parseHttpException(error)
            } catch (error: Exception) {
                _errorMessage.value = error.message ?: "Unknown error"
            } finally {
                _isSubmitting.value = false
            }
        }
    }

    fun clearError() {
        _errorMessage.value = null
    }
}
