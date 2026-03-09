package com.example.werkzeugerkennung

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import retrofit2.HttpException

class ShotViewModel(
    private val repository: BackendRepository,
    private val objectId: String
) : ViewModel() {
    private val _nextShot = MutableStateFlow<NextShotResponse?>(null)
    val nextShot: StateFlow<NextShotResponse?> = _nextShot.asStateFlow()

    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage.asStateFlow()

    private val _isSubmitting = MutableStateFlow(false)
    val isSubmitting: StateFlow<Boolean> = _isSubmitting.asStateFlow()

    fun loadNextShot() {
        viewModelScope.launch {
            _errorMessage.value = null
            try {
                _nextShot.value = repository.getNextShot(objectId)
            } catch (error: HttpException) {
                _errorMessage.value = BackendErrorParser.parseHttpException(error)
            } catch (error: Exception) {
                _errorMessage.value = error.message ?: "Unknown error"
            }
        }
    }

    fun submitShot(envCode: String, accept: Boolean, imageBytes: ByteArray) {
        viewModelScope.launch {
            _isSubmitting.value = true
            _errorMessage.value = null
            try {
                val response = repository.submitShot(objectId, envCode, accept, imageBytes)
                _nextShot.value = if (response.objectCompleted) {
                    NextShotResponse(
                        envCode = "",
                        remainingNeededForEnv = 0,
                        objectCompleted = true
                    )
                } else {
                    response.nextRequiredShot
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

    fun clearError() {
        _errorMessage.value = null
    }
}
