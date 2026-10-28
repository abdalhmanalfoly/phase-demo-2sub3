"""SAGA 3D wire protocol, version 1.0 (single source of truth for the host).

The browser mirrors these rules in ``sensor-client/src/protocol.js``; both are checked against the
shared examples in ``docs/examples/`` and against each other (see tests).

Conventions
-----------
* Every message carries ``protocol_version`` and ``message_type``.
* All fields are REQUIRED; "absent" is an explicit ``null`` (never a silently substituted value).
* Unknown fields are rejected (``extra="forbid"``): this also guarantees no raw video/image payload
  can ride along inside a message.
* Image coordinates are PIXELS in the frame of ``image_width`` x ``image_height`` (normative).
  Optional ``x_norm``/``y_norm`` are explicitly labelled normalized copies and must agree with pixels.
* ``captured_at_ms`` is the SENSOR clock (epoch ms). ``received_at_ms`` is assigned by the HOST and is
  never accepted from a client.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, model_validator

PROTOCOL_VERSION = "1.0"
OUT_OF_FRAME_MARGIN = 0.10          # landmarks may lie up to 10 % outside the frame (detector jitter)
MIN_EPOCH_MS = 1_577_836_800_000     # 2020-01-01
MAX_EPOCH_MS = 4_102_444_800_000     # 2100-01-01
MAX_SEQUENCE = 2**53 - 1             # largest integer a JS client can represent exactly
MAX_IMAGE_DIM = 8192
MAX_LANDMARKS = 32
NORM_TOLERANCE_PX = 1.0

LandmarkId = Literal["wrist", "thumb_tip", "index_tip", "index_mcp", "pinky_mcp"]
SessionId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{4,64}$")]
SensorId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,32}$")]
ClientId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{8,64}$")]
EpochMs = Annotated[int, Field(ge=MIN_EPOCH_MS, le=MAX_EPOCH_MS)]
Seq = Annotated[int, Field(ge=0, le=MAX_SEQUENCE)]
Dim = Annotated[int, Field(ge=1, le=MAX_IMAGE_DIM)]
Unit = Annotated[float, Field(ge=0.0, le=1.0)]

Permission = Literal["unknown", "prompt", "granted", "denied"]
CameraState = Literal["idle", "starting", "active", "unavailable", "stopped"]
InferenceState = Literal["idle", "loading", "ready", "detecting", "error"]
SensorStateName = Literal["online", "degraded", "offline", "disconnected"]

ERROR_CODES = (
    "MALFORMED_JSON", "MESSAGE_TOO_LARGE", "BINARY_NOT_SUPPORTED", "SCHEMA_VALIDATION_FAILED",
    "UNSUPPORTED_PROTOCOL_VERSION", "UNSUPPORTED_MESSAGE_TYPE", "UNKNOWN_SESSION", "SESSION_STOPPED",
    "SESSION_FULL", "NOT_REGISTERED", "ALREADY_REGISTERED", "SESSION_MISMATCH", "SENSOR_ID_MISMATCH",
    "DUPLICATE_SENSOR_ID", "HELLO_TIMEOUT", "STALE_SEQUENCE", "STALE_OBSERVATION",
    "TIMESTAMP_IN_FUTURE", "RATE_LIMITED",
)
ErrorCode = Literal[
    "MALFORMED_JSON", "MESSAGE_TOO_LARGE", "BINARY_NOT_SUPPORTED", "SCHEMA_VALIDATION_FAILED",
    "UNSUPPORTED_PROTOCOL_VERSION", "UNSUPPORTED_MESSAGE_TYPE", "UNKNOWN_SESSION", "SESSION_STOPPED",
    "SESSION_FULL", "NOT_REGISTERED", "ALREADY_REGISTERED", "SESSION_MISMATCH", "SENSOR_ID_MISMATCH",
    "DUPLICATE_SENSOR_ID", "HELLO_TIMEOUT", "STALE_SEQUENCE", "STALE_OBSERVATION",
    "TIMESTAMP_IN_FUTURE", "RATE_LIMITED",
]


class ProtocolError(Exception):
    """A rejected message. ``fatal`` means the host closes the socket after replying."""

    def __init__(self, code: str, message: str, *, details: list[dict] | None = None,
                 fatal: bool = False, related: str | None = None):
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.details, self.fatal, self.related = code, message, details, fatal, related


class _Msg(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


# --------------------------------------------------------------------------- shared pieces
class CameraInfo(_Msg):
    facing: Literal["user", "environment", "unknown"]
    width: Dim
    height: Dim


class Capabilities(_Msg):
    landmarks: Annotated[list[LandmarkId], Field(min_length=1, max_length=MAX_LANDMARKS)]
    max_update_hz: Annotated[float, Field(gt=0, le=120)]
    inference: Literal["on_device", "none"]
    camera: CameraInfo | None
    user_agent: Annotated[str, Field(max_length=256)] | None

    @model_validator(mode="after")
    def _unique(self):
        if len(set(self.landmarks)) != len(self.landmarks):
            raise ValueError("capabilities.landmarks must be unique")
        return self


class Health(_Msg):
    camera: CameraState
    inference: InferenceState
    observed_hz: Annotated[float, Field(ge=0, le=1000)] | None
    frames_skipped: Annotated[int, Field(ge=0, le=MAX_SEQUENCE)]   # intentionally skipped by the rate cap
    send_drops: Annotated[int, Field(ge=0, le=MAX_SEQUENCE)]       # observations dropped (back-pressure)


class Landmark(_Msg):
    id: LandmarkId
    x_px: float
    y_px: float
    confidence: Unit | None                                        # null = detector gave none
    x_norm: float | None                                           # normalized copy of x_px (0..1)
    y_norm: float | None
    z_model: float | None                                          # model-relative depth, NOT metric


class Quality(_Msg):
    inference_ms: Annotated[float, Field(ge=0, le=10_000)] | None
    handedness_score: Unit | None                                  # classifier score, not tracking confidence
    flags: Annotated[list[Literal["frames_dropped", "overloaded", "out_of_frame"]], Field(max_length=8)]


# --------------------------------------------------------------------------- client -> host
class Hello(_Msg):
    protocol_version: Literal["1.0"]
    message_type: Literal["HELLO"]
    session_id: SessionId
    client_id: ClientId                                            # stable per browser tab, enables resume
    requested_sensor_id: SensorId | None
    client_version: Annotated[str, Field(min_length=1, max_length=64)]
    model_version: Annotated[str, Field(max_length=128)] | None
    client_time_ms: EpochMs
    capabilities: Capabilities


class Heartbeat(_Msg):
    protocol_version: Literal["1.0"]
    message_type: Literal["HEARTBEAT"]
    session_id: SessionId
    sensor_id: SensorId
    sequence_number: Seq
    client_time_ms: EpochMs
    health: Health


class SensorStatus(_Msg):
    protocol_version: Literal["1.0"]
    message_type: Literal["SENSOR_STATUS"]
    session_id: SessionId
    sensor_id: SensorId
    client_time_ms: EpochMs
    permission: Permission
    camera: CameraState
    inference: InferenceState
    connection: Literal["connected", "reconnected"]
    reconnect_count: Annotated[int, Field(ge=0, le=1_000_000)]
    detail: Annotated[str, Field(max_length=200)] | None


class LandmarkObservation(_Msg):
    protocol_version: Literal["1.0"]
    message_type: Literal["LANDMARK_OBSERVATION"]
    session_id: SessionId
    sensor_id: SensorId
    sequence_number: Seq
    frame_id: Seq
    captured_at_ms: EpochMs
    image_width: Dim
    image_height: Dim
    status: Literal["tracked", "no_hand"]
    hand: Literal["left", "right", "unknown"]
    landmarks: Annotated[list[Landmark], Field(max_length=MAX_LANDMARKS)]
    quality: Quality

    @model_validator(mode="after")
    def _semantics(self):
        w, h, m = self.image_width, self.image_height, OUT_OF_FRAME_MARGIN
        if self.status == "no_hand":
            if self.landmarks:
                raise ValueError("status 'no_hand' requires an empty landmarks list")
            return self
        if not self.landmarks:
            raise ValueError("status 'tracked' requires at least one landmark")
        seen: set[str] = set()
        for i, lm in enumerate(self.landmarks):
            if lm.id in seen:
                raise ValueError(f"landmarks[{i}]: duplicate landmark id '{lm.id}'")
            seen.add(lm.id)
            if not (-m * w <= lm.x_px <= (1 + m) * w):
                raise ValueError(f"landmarks[{i}].x_px outside frame bounds (width {w}, margin {m})")
            if not (-m * h <= lm.y_px <= (1 + m) * h):
                raise ValueError(f"landmarks[{i}].y_px outside frame bounds (height {h}, margin {m})")
            if (lm.x_norm is None) != (lm.y_norm is None):
                raise ValueError(f"landmarks[{i}]: x_norm and y_norm must both be set or both null")
            if lm.x_norm is not None:
                if not (-m <= lm.x_norm <= 1 + m and -m <= lm.y_norm <= 1 + m):
                    raise ValueError(f"landmarks[{i}]: normalized coordinates out of range")
                if abs(lm.x_norm * w - lm.x_px) > NORM_TOLERANCE_PX or abs(lm.y_norm * h - lm.y_px) > NORM_TOLERANCE_PX:
                    raise ValueError(f"landmarks[{i}]: x_norm/y_norm disagree with pixel coordinates")
        return self


class Disconnect(_Msg):
    protocol_version: Literal["1.0"]
    message_type: Literal["DISCONNECT"]
    session_id: SessionId
    sensor_id: SensorId
    client_time_ms: EpochMs
    reason: Literal["user_stop", "page_unload", "error"]


# --------------------------------------------------------------------------- host -> client
class Registered(_Msg):
    protocol_version: Literal["1.0"]
    message_type: Literal["REGISTERED"]
    session_id: SessionId
    sensor_id: SensorId
    resumed: bool
    heartbeat_interval_ms: Annotated[int, Field(ge=100, le=60_000)]
    target_update_hz: Annotated[float, Field(gt=0, le=120)]
    max_update_hz: Annotated[float, Field(gt=0, le=120)]
    required_landmarks: list[LandmarkId]
    session_state: Literal["active"]
    host_time_ms: EpochMs
    host_version: Annotated[str, Field(max_length=32)]


class HeartbeatAck(_Msg):
    protocol_version: Literal["1.0"]
    message_type: Literal["HEARTBEAT_ACK"]
    session_id: SessionId
    sensor_id: SensorId
    sequence_number: Seq               # echo of the heartbeat
    client_time_ms: EpochMs            # echo (lets the sensor compute round-trip time)
    host_time_ms: EpochMs
    sensor_state: SensorStateName


class ErrorDetail(_Msg):
    field: Annotated[str, Field(max_length=120)]
    problem: Annotated[str, Field(max_length=200)]


class ErrorMessage(_Msg):
    protocol_version: Literal["1.0"]
    message_type: Literal["ERROR"]
    session_id: SessionId | None
    error_code: ErrorCode
    message: Annotated[str, Field(max_length=300)]
    related_message_type: Annotated[str, Field(max_length=40)] | None
    details: Annotated[list[ErrorDetail], Field(max_length=10)] | None
    fatal: bool
    host_time_ms: EpochMs


class SessionStop(_Msg):
    protocol_version: Literal["1.0"]
    message_type: Literal["SESSION_STOP"]
    session_id: SessionId
    reason: Literal["operator_request", "host_shutdown"]
    host_time_ms: EpochMs


CLIENT_MODELS: dict[str, type[_Msg]] = {
    "HELLO": Hello, "HEARTBEAT": Heartbeat, "SENSOR_STATUS": SensorStatus,
    "LANDMARK_OBSERVATION": LandmarkObservation, "DISCONNECT": Disconnect,
}
HOST_MODELS: dict[str, type[_Msg]] = {
    "REGISTERED": Registered, "HEARTBEAT_ACK": HeartbeatAck, "ERROR": ErrorMessage, "SESSION_STOP": SessionStop,
}


def _details(err: ValidationError, limit: int = 10) -> list[dict]:
    """Safe, value-free summary of validation failures (never echoes client-supplied values)."""
    out = []
    for e in err.errors()[:limit]:
        field = ".".join(str(p) for p in e["loc"]) or "$"
        msg = str(e["msg"]).removeprefix("Value error, ")
        out.append({"field": field[:120], "problem": msg[:200]})
    return out


def _parse(data: object, models: dict[str, type[_Msg]], unsupported_msg: str) -> _Msg:
    if not isinstance(data, dict):
        raise ProtocolError("MALFORMED_JSON", "top-level JSON value must be an object")
    if data.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError("UNSUPPORTED_PROTOCOL_VERSION",
                            f"this host speaks protocol_version '{PROTOCOL_VERSION}'")
    mtype = data.get("message_type")
    if not isinstance(mtype, str) or mtype not in models:
        raise ProtocolError("UNSUPPORTED_MESSAGE_TYPE", unsupported_msg)
    try:
        return models[mtype].model_validate(data)
    except ValidationError as exc:
        raise ProtocolError("SCHEMA_VALIDATION_FAILED", f"{mtype} failed validation",
                            details=_details(exc), related=mtype) from None


def parse_client_message(data: object) -> _Msg:
    """Validate an inbound (sensor -> host) message or raise :class:`ProtocolError`."""
    return _parse(data, CLIENT_MODELS, "unsupported or missing message_type for a sensor message")


def parse_host_message(data: object) -> _Msg:
    """Validate an outbound (host -> sensor) message (used by tests and tooling)."""
    return _parse(data, HOST_MODELS, "unsupported or missing message_type for a host message")
