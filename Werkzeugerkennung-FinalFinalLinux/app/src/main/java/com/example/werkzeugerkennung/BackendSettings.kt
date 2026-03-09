package com.example.werkzeugerkennung

import android.content.Context

object BackendSettings {
    private const val PREFS_NAME = "backend_settings"
    private const val KEY_BASE_URL = "backend_base_url"
    private const val KEY_API_KEY = "backend_api_key"
    const val DEFAULT_BASE_URL = "http://10.0.2.2:8000"

    fun loadBaseUrl(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val stored = prefs.getString(KEY_BASE_URL, DEFAULT_BASE_URL) ?: DEFAULT_BASE_URL
        return normalizeBaseUrl(stored)
    }

    fun saveBaseUrl(context: Context, rawUrl: String): String {
        val normalized = normalizeBaseUrl(rawUrl)
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs.edit().putString(KEY_BASE_URL, normalized).apply()
        return normalized
    }

    fun loadApiKey(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getString(KEY_API_KEY, "") ?: ""
    }

    fun saveApiKey(context: Context, apiKey: String): String {
        val trimmed = apiKey.trim()
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs.edit().putString(KEY_API_KEY, trimmed).apply()
        return trimmed
    }

    fun normalizeBaseUrl(rawUrl: String): String {
        val trimmed = rawUrl.trim()
        val sanitized = if (trimmed.isBlank()) DEFAULT_BASE_URL else trimmed
        val withScheme = if (sanitized.startsWith("http://") || sanitized.startsWith("https://")) {
            sanitized
        } else {
            "http://$sanitized"
        }
        return if (withScheme.endsWith("/")) withScheme else "$withScheme/"
    }
}
