package com.dg.precaldnp.model

import androidx.annotation.StringRes

interface ProgressSink3250 {
    fun setBusy3250(busy: Boolean)
    fun stage3250(@StringRes msgRes: Int, vararg args: Any)
}
