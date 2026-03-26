package com.dg.precaldnp.net

data class LoginRequest(
    val email: String
)

data class LoginResponse(
    val ok: Boolean,
    val token: String?,
    val expires_at: String?,
    val error: String?
)

data class AuthMeResponse(
    val ok: Boolean,
    val user: AuthUser?,
    val error: String?
)

data class AuthUser(
    val id: String,
    val email: String,
    val tenant_id: String,
    val tenant_slug: String,
    val tenant_name: String,
    val subscription_status: String,
    val plan_code: String,
    val max_devices: Int
)

data class RegisterDeviceRequest(
    val deviceLabel: String,
    val platform: String = "android",
    val deviceFingerprint: String
)

data class RegisterDeviceResponse(
    val ok: Boolean,
    val reused: Boolean?,
    val device: RegisteredDevice?,
    val limits: DeviceLimits?,
    val error: String?
)

data class RegisteredDevice(
    val id: String,
    val device_label: String,
    val platform: String,
    val device_fingerprint: String,
    val status: String,
    val created_at: String
)

data class DeviceLimits(
    val plan_code: String,
    val max_devices: Int,
    val active_devices_after_insert: Int? = null,
    val active_devices: Int? = null
)