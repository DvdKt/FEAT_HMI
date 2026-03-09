package com.example.werkzeugerkennung

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonPrimitive
import retrofit2.HttpException

object BackendErrorParser {
    private val json = Json { ignoreUnknownKeys = true }

    fun parseHttpException(error: HttpException): String {
        val body = error.response()?.errorBody()?.string()
        if (body.isNullOrBlank()) {
            return "${error.code()} ${error.message()}"
        }
        val jsonElement = runCatching { json.parseToJsonElement(body) }.getOrNull()
        if (jsonElement is JsonObject) {
            val errorObj = jsonElement["error"]
            if (errorObj is JsonObject) {
                val message = errorObj["message"]?.jsonPrimitive?.content
                val code = errorObj["code"]?.jsonPrimitive?.content
                if (!message.isNullOrBlank()) {
                    return if (code.isNullOrBlank()) message else "$code: $message"
                }
            }
        }
        return runCatching {
            json.decodeFromString(ApiError.serializer(), body).detail
        }.getOrElse { body }
    }
}
