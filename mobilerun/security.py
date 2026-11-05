"""Fail-closed local execution policy for this fork (not an OS sandbox)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from ipaddress import ip_address
from typing import Literal, Protocol
from urllib.parse import urlsplit, urlunsplit

Capability = Literal["CLOUD", "MCP", "REMOTE_APP_CARDS", "TRACING"]
LOCAL_MODEL_URL = "http://127.0.0.1:11434"
_LOCAL_MODEL_TOKEN = object()


class LocalModelClient(Protocol):
    """Structural interface for a loader-issued local inference client."""

    _mobilerun_local_only: object


def capability_enabled(capability: Capability) -> bool:
    """Require an exact, separate operator opt-in, not legacy credentials."""
    return os.environ.get(f"MOBILERUN_ALLOW_{capability}") == "1"


def require_capability(capability: Capability) -> None:
    """Reject disabled optional capabilities before side effects occur."""
    if not capability_enabled(capability):
        raise PermissionError(
            f"{capability} is disabled by local policy. Explicitly set "
            f"MOBILERUN_ALLOW_{capability}=1 only after reviewing its risks."
        )


def local_model_url(value: object) -> str:
    """Validate an HTTP(S) loopback origin without DNS or URL credentials."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("A local model URL is required.")
    if any(ord(char) < 33 for char in value) or "\\" in value:
        raise ValueError("Invalid local model URL.")
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port
        if host == "localhost":
            host = "127.0.0.1"
        if not host or not ip_address(host).is_loopback:
            raise ValueError("Model endpoints must use a loopback address.")
    except ValueError as error:
        raise ValueError(
            "Model endpoints must use a valid loopback address."
        ) from error
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or port == 0
    ):
        raise ValueError("Use a local HTTP(S) origin without credentials or paths.")
    authority = f"[{host}]" if ":" in host else host
    if port is not None:
        authority += f":{port}"
    return urlunsplit((parsed.scheme, authority, "", "", ""))


def validate_model_config(
    provider: str, model: object, options: Mapping[str, object]
) -> str:
    """Allow only explicit local Ollama models; never choose a substitute."""
    if provider != "Ollama":
        raise PermissionError("Only local Ollama models are permitted by this fork.")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Configure an already installed local Ollama model.")
    if "cloud" in model.lower() or "://" in model:
        raise PermissionError("Cloud-backed Ollama models are not permitted.")
    forbidden = {
        "api_base",
        "api_key",
        "client",
        "async_client",
        "http_client",
        "async_http_client",
        "client_kwargs",
        "async_client_kwargs",
        "callback_manager",
    }
    if forbidden.intersection(options):
        raise PermissionError(
            "Custom model transports, credentials and callbacks are disabled."
        )
    return local_model_url(options.get("base_url", LOCAL_MODEL_URL))


def mark_local_llm(llm: LocalModelClient) -> None:
    """Mark loader-created clients with an identity token, not a truthy flag."""
    llm._mobilerun_local_only = _LOCAL_MODEL_TOKEN


def validate_local_llm(llm: object) -> None:
    """Reject injected or mutated clients not built by the local-only loader."""
    if getattr(llm, "_mobilerun_local_only", None) is not _LOCAL_MODEL_TOKEN:
        raise PermissionError("Use load_llm('Ollama', model=...) for local inference.")
    local_model_url(getattr(llm, "base_url", None))
    model = getattr(llm, "model", None)
    validate_model_config("Ollama", model, {})


def reject_remote_model_login() -> None:
    """Reject model OAuth authentication independently of optional cloud access."""
    raise PermissionError("Remote model login is disabled by local-only policy.")
