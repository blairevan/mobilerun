"""Dependency-free regression tests for the local-only security policy."""

import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

POLICY_PATH = Path(__file__).resolve().parents[1] / "mobilerun" / "security.py"
SPEC = importlib.util.spec_from_file_location("local_security_policy", POLICY_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load the local security policy.")
policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy)


class LocalSecurityTests(unittest.TestCase):
    """Check local URL parsing, explicit opt-ins and model restrictions."""

    def test_optional_capabilities_require_exact_opt_in(self) -> None:
        """Old credentials and truthy strings must not enable capabilities."""
        for name in ("CLOUD", "MCP", "REMOTE_APP_CARDS", "TRACING"):
            for value in ("", "true", "yes", "0"):
                with (
                    self.subTest(name=name, value=value),
                    patch.dict(
                        os.environ, {f"MOBILERUN_ALLOW_{name}": value}, clear=True
                    ),
                ):
                    with self.assertRaises(PermissionError):
                        policy.require_capability(name)
            with patch.dict(os.environ, {f"MOBILERUN_ALLOW_{name}": "1"}, clear=True):
                policy.require_capability(name)

    def test_valid_loopback_origins(self) -> None:
        """Canonicalize localhost and preserve numeric loopback origins."""
        for url, expected in (
            ("http://localhost:11434/", "http://127.0.0.1:11434"),
            ("http://127.0.0.1:11434", "http://127.0.0.1:11434"),
            ("http://[::1]:11434", "http://[::1]:11434"),
        ):
            with self.subTest(url=url):
                self.assertEqual(policy.local_model_url(url), expected)

    def test_remote_and_ambiguous_urls_are_rejected(self) -> None:
        """Reject DNS names, credentials, redirects via paths and URL tricks."""
        for url in (
            "https://api.example.com",
            "http://192.168.1.2",
            "http://0.0.0.0",
            "http://localhost.example.com",
            "http://127.1",
            "http://2130706433",
            "http://127.0.0.1@evil.example",
            "http://x@127.0.0.1",
            "http://[::ffff:192.168.1.1]",
            "file:///tmp/model",
            "http://127.0.0.1/path",
            "http://127.0.0.1?host=evil",
            "http://127.0.0.1#evil",
            "http://127.0.0.1:0",
            "http://127.0.0.1:bad",
            "http://127.0.0.1\\@evil",
            " http://127.0.0.1",
            "http://local\nhost",
            "",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                policy.local_model_url(url)

    def test_models_fail_closed(self) -> None:
        """Reject public providers, cloud models and alternate clients."""
        for provider in ("OpenAI", "GoogleGenAI", "openai_oauth", "OpenAILike"):
            with self.subTest(provider=provider), self.assertRaises(PermissionError):
                policy.validate_model_config(provider, "test", {})
        for model in ("model-cloud", "model:cloud", "https://example.com/model"):
            with self.subTest(model=model), self.assertRaises(PermissionError):
                policy.validate_model_config("Ollama", model, {})
        for key in ("api_base", "client", "async_client", "client_kwargs", "api_key"):
            with self.subTest(key=key), self.assertRaises(PermissionError):
                policy.validate_model_config("Ollama", "test", {key: "override"})
        with self.assertRaises(ValueError):
            policy.validate_model_config("Ollama", None, {})
        self.assertEqual(
            policy.validate_model_config("Ollama", "test", {}), policy.LOCAL_MODEL_URL
        )

    def test_injected_remote_clients_and_oauth_are_rejected(self) -> None:
        """Reject unverified clients and revalidate endpoints at inference time."""
        for client in (
            SimpleNamespace(base_url=policy.LOCAL_MODEL_URL),
            SimpleNamespace(_mobilerun_local_only=True, base_url="https://example.com"),
        ):
            with (
                self.subTest(client=client),
                self.assertRaises((ValueError, PermissionError)),
            ):
                policy.validate_local_llm(client)
        with self.assertRaises(PermissionError):
            policy.reject_remote_model_login()

    def test_loader_identity_and_model_mutation_are_checked(self) -> None:
        """A truthy marker cannot substitute for a loader-issued identity token."""
        model = SimpleNamespace(
            _mobilerun_local_only=True, base_url=policy.LOCAL_MODEL_URL, model="test"
        )
        with self.assertRaises(PermissionError):
            policy.validate_local_llm(model)
        policy.mark_local_llm(model)
        policy.validate_local_llm(model)
        model.base_url = "https://example.com"
        with self.assertRaises(ValueError):
            policy.validate_local_llm(model)
        model.base_url = policy.LOCAL_MODEL_URL
        model.model = "test-cloud"
        with self.assertRaises(PermissionError):
            policy.validate_local_llm(model)


if __name__ == "__main__":
    unittest.main()
