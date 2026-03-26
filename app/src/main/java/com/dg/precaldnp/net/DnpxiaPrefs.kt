package com.dg.precaldnp.net

import android.content.Context

class DnpxiaPrefs(context: Context) {
    private val sp = context.getSharedPreferences("dnpxia_auth", Context.MODE_PRIVATE)

    fun saveToken(token: String) {
        sp.edit().putString("token", token).apply()
    }

    fun getToken(): String? = sp.getString("token", null)

    fun clear() {
        sp.edit().clear().apply()
    }
}