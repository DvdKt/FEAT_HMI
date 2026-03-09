package com.example.werkzeugerkennung

import android.content.Context

/**
 * Repository factory for remote backend access.
 */
@Suppress("UNUSED_PARAMETER")
fun createRepository(context: Context, url: String, apiKey: String): BackendRepository {
    return RetrofitBackendRepository.create(url, apiKey)
}
