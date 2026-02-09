package com.dg.precaldnp.util

import android.content.Context

object PrefsTerms {
    private const val PREFS = "ui_terms"
    private const val KEY = "accepted"

    fun isAccepted(ctx: Context) =
        ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getBoolean(KEY, false)

    fun setAccepted(ctx: Context, v: Boolean) {
        ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putBoolean(KEY, v).apply()
    }
}
