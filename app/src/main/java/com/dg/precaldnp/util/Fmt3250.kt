package com.dg.precaldnp.util

import java.util.Locale

object Fmt3250 {

    fun fmt2(v: Double): String =
        if (!v.isFinite()) "NaN" else String.format(Locale.US, "%.2f", v)

    fun fmt2(v: Double?): String =
        if (v == null || !v.isFinite()) "NaN" else String.format(Locale.US, "%.2f", v)

    fun fmt1(v: Double): String =
        if (!v.isFinite()) "NaN" else String.format(Locale.US, "%.1f", v)

    fun fmt1(v: Double?): String =
        if (v == null || !v.isFinite()) "NaN" else String.format(Locale.US, "%.1f", v)

    fun fmt1(v: Float): String =
        if (!v.isFinite()) "NaN" else String.format(Locale.US, "%.1f", v)

    fun fmt1(v: Float?): String =
        if (v == null || !v.isFinite()) "NaN" else String.format(Locale.US, "%.1f", v)
}
