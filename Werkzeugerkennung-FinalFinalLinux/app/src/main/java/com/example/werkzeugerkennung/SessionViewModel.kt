package com.example.werkzeugerkennung

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import retrofit2.HttpException

class SessionViewModel(
    private val repository: BackendRepository
) : ViewModel() {
    private val _envSpec = MutableStateFlow<List<EnvSpecItem>>(emptyList())
    val envSpec: StateFlow<List<EnvSpecItem>> = _envSpec.asStateFlow()

    private val _objects = MutableStateFlow<List<ObjectRecord>>(emptyList())
    val objects: StateFlow<List<ObjectRecord>> = _objects.asStateFlow()

    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage.asStateFlow()

    private val _thresholds = MutableStateFlow<ThresholdConfigResponse?>(null)
    val thresholds: StateFlow<ThresholdConfigResponse?> = _thresholds.asStateFlow()

    private val _softResetStatus = MutableStateFlow<SoftResetStatusResponse?>(null)
    val softResetStatus: StateFlow<SoftResetStatusResponse?> = _softResetStatus.asStateFlow()

    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()

    fun loadSession() {
        viewModelScope.launch {
            _isLoading.value = true
            _errorMessage.value = null
            try {
                _envSpec.value = repository.getEnvSpec()
                _objects.value = repository.listObjects()
            } catch (error: HttpException) {
                _errorMessage.value = BackendErrorParser.parseHttpException(error)
            } catch (error: Exception) {
                _errorMessage.value = error.message ?: "Unknown error"
            } finally {
                _isLoading.value = false
            }
        }
    }

    fun loadThresholds() {
        viewModelScope.launch {
            _errorMessage.value = null
            try {
                _thresholds.value = repository.getThresholds()
            } catch (error: HttpException) {
                _errorMessage.value = BackendErrorParser.parseHttpException(error)
            } catch (error: Exception) {
                _errorMessage.value = error.message ?: "Unknown error"
            }
        }
    }

    fun loadSoftResetStatus() {
        viewModelScope.launch {
            _errorMessage.value = null
            try {
                _softResetStatus.value = repository.getSoftResetStatus()
            } catch (error: HttpException) {
                _errorMessage.value = BackendErrorParser.parseHttpException(error)
            } catch (error: Exception) {
                _errorMessage.value = error.message ?: "Unknown error"
            }
        }
    }

    fun refreshObjects() {
        viewModelScope.launch {
            _errorMessage.value = null
            try {
                _objects.value = repository.listObjects()
            } catch (error: HttpException) {
                _errorMessage.value = BackendErrorParser.parseHttpException(error)
            } catch (error: Exception) {
                _errorMessage.value = error.message ?: "Unknown error"
            }
        }
    }

    fun createObject(name: String) {
        viewModelScope.launch {
            _isLoading.value = true
            _errorMessage.value = null
            try {
                repository.createObject(name)
                _objects.value = repository.listObjects()
            } catch (error: HttpException) {
                _errorMessage.value = BackendErrorParser.parseHttpException(error)
            } catch (error: Exception) {
                _errorMessage.value = error.message ?: "Unknown error"
            } finally {
                _isLoading.value = false
            }
        }
    }

    fun setThresholds(confThreshold: Double, marginThreshold: Double) {
        viewModelScope.launch {
            _isLoading.value = true
            _errorMessage.value = null
            try {
                _thresholds.value = repository.setThresholds(confThreshold, marginThreshold)
            } catch (error: HttpException) {
                _errorMessage.value = BackendErrorParser.parseHttpException(error)
            } catch (error: Exception) {
                _errorMessage.value = error.message ?: "Unknown error"
            } finally {
                _isLoading.value = false
            }
        }
    }

    fun softReset() {
        viewModelScope.launch {
            _isLoading.value = true
            _errorMessage.value = null
            try {
                repository.softReset()
                _objects.value = repository.listObjects()
                runCatching { repository.getSoftResetStatus() }
                    .onSuccess { _softResetStatus.value = it }
            } catch (error: HttpException) {
                _errorMessage.value = BackendErrorParser.parseHttpException(error)
            } catch (error: Exception) {
                _errorMessage.value = error.message ?: "Unknown error"
            } finally {
                _isLoading.value = false
            }
        }
    }

    fun resetAll() {
        viewModelScope.launch {
            _isLoading.value = true
            _errorMessage.value = null
            try {
                repository.resetAll()
                _objects.value = repository.listObjects()
                runCatching { repository.getThresholds() }
                    .onSuccess { _thresholds.value = it }
                runCatching { repository.getSoftResetStatus() }
                    .onSuccess { _softResetStatus.value = it }
            } catch (error: HttpException) {
                _errorMessage.value = BackendErrorParser.parseHttpException(error)
            } catch (error: Exception) {
                _errorMessage.value = error.message ?: "Unknown error"
            } finally {
                _isLoading.value = false
            }
        }
    }

    fun clearError() {
        _errorMessage.value = null
    }
}
