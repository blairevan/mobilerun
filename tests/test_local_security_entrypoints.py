"""Exercise real entrypoint bodies without installing or importing dependencies.

AST isolation deliberately excludes package startup. These tests complement, not
replace, the dependency-backed integration suite and OS-level network isolation.
"""

import ast
import asyncio
import logging
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from test_local_security import policy

ROOT = Path(__file__).resolve().parents[1]


def entrypoint(path: str, name: str, namespace: dict, owner: str | None = None):
    """Compile the actual function body with deferred annotations and test doubles."""
    tree = ast.parse((ROOT / path).read_text())
    nodes = tree.body
    if owner:
        nodes = next(
            node
            for node in nodes
            if isinstance(node, ast.ClassDef) and node.name == owner
        ).body
    function = next(
        node
        for node in nodes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    function.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(ROOT / path), "exec"), namespace)
    return namespace[name]


class LocalEntrypointTests(unittest.TestCase):
    """Assert blocked operations stop before network or device side effects."""

    def setUp(self) -> None:
        """Make only the standalone policy importable, without package startup."""
        package = ModuleType("mobilerun")
        package.__path__ = []
        self.modules = patch.dict(
            sys.modules, {"mobilerun": package, "mobilerun.security": policy}
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.environment = patch.dict("os.environ", {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_setup_rejects_downloads_before_device_selection(self) -> None:
        """No path, latest and version options cannot invoke device discovery."""
        select = AsyncMock(side_effect=AssertionError("device discovery"))
        setup = entrypoint(
            "mobilerun/cli/main.py",
            "_setup_portal",
            {
                "Path": Path,
                "click": SimpleNamespace(ClickException=RuntimeError),
                "resolve_android_serial": select,
            },
        )
        for path, latest, version in (
            (None, False, None),
            ("test.apk", True, None),
            ("test.apk", False, "1"),
        ):
            with (
                self.subTest(path=path, latest=latest, version=version),
                self.assertRaises(RuntimeError),
            ):
                asyncio.run(setup(path, None, False, latest, version))
        select.assert_not_called()

    def test_setup_installs_only_local_apk_without_grants(self) -> None:
        """Explicit setup never uninstalls, grants permissions or invokes shell."""
        import os

        device = SimpleNamespace(install=AsyncMock(), shell=AsyncMock())
        adb = SimpleNamespace(device=AsyncMock(return_value=device))
        setup = entrypoint(
            "mobilerun/cli/main.py",
            "_setup_portal",
            {
                "Path": Path,
                "click": SimpleNamespace(ClickException=RuntimeError),
                "adb": adb,
                "console": Mock(),
                "nullcontext": nullcontext,
                "os": os,
            },
        )
        with tempfile.NamedTemporaryFile(suffix=".apk") as apk:
            asyncio.run(setup(apk.name, "usb-device", False))
            device.install.assert_awaited_once_with(
                str(Path(apk.name).resolve()),
                nolaunch=True,
                uninstall=False,
                flags=[],
                silent=True,
            )
        device.shell.assert_not_called()

    def test_model_loader_rejects_public_config_before_imports(self) -> None:
        """Old profiles and command-line endpoint overrides cannot load providers."""
        load = entrypoint(
            "mobilerun/agent/utils/llm_picker.py",
            "load_llm",
            {
                "normalize_provider_name": lambda value: value,
            },
        )
        for provider, options in (
            ("GoogleGenAI", {}),
            ("openai_oauth", {}),
            ("Ollama", {"base_url": "https://example.com"}),
            ("Ollama", {"client_kwargs": {"proxy": "https://example.com"}}),
        ):
            with (
                self.subTest(provider=provider, options=options),
                self.assertRaises((PermissionError, ValueError)),
            ):
                load(provider, model="test", **options)

    def test_model_loader_pins_transport_and_local_origin(self) -> None:
        """Local construction disables proxy environment and HTTP redirects."""
        import inspect

        class FakeOllama:
            """Record adapter construction without contacting any model server."""

            def __init__(
                self, *, client_kwargs: dict, async_client_kwargs: dict, **kwargs
            ):
                """Record model and transport settings supplied by the loader."""
                self.transport = client_kwargs
                self.async_transport = async_client_kwargs
                self.__dict__.update(kwargs)

        adapter = ModuleType("llama_index.llms.ollama")
        adapter.Ollama = FakeOllama
        load = entrypoint(
            "mobilerun/agent/utils/llm_picker.py",
            "load_llm",
            {
                "normalize_provider_name": lambda value: value,
                "_prepare_ollama_kwargs": lambda kwargs, cls: kwargs,
                "inspect": inspect,
                "logger": logging.getLogger(__name__),
            },
        )
        with patch.dict(sys.modules, {"llama_index.llms.ollama": adapter}):
            model = load("Ollama", "test", base_url="http://localhost:11434")
        self.assertEqual(model.base_url, policy.LOCAL_MODEL_URL)
        self.assertEqual(
            model.transport, {"trust_env": False, "follow_redirects": False}
        )
        self.assertEqual(model.async_transport, model.transport)
        policy.validate_local_llm(model)

    def test_inference_rejects_injected_remote_models(self) -> None:
        """All three inference APIs check the client before sending anything."""
        for name, args in (
            ("acall_with_retries", ([],)),
            ("acomplete_with_retries", ("secret",)),
            ("astructured_predict_with_retries", (object, "secret")),
        ):
            model = SimpleNamespace(
                achat=AsyncMock(),
                acomplete=AsyncMock(),
                astructured_predict=AsyncMock(),
            )
            call = entrypoint("mobilerun/agent/utils/inference.py", name, {})
            with self.subTest(name=name), self.assertRaises(PermissionError):
                asyncio.run(call(model, *args))
            model.achat.assert_not_called()
            model.acomplete.assert_not_called()
            model.astructured_predict.assert_not_called()

    def test_telemetry_discards_events_without_serialization(self) -> None:
        """Legacy enable flags cannot create a client, log content or track a user."""
        event = Mock()
        capture = entrypoint("mobilerun/telemetry/tracker.py", "capture", {})
        enabled = entrypoint(
            "mobilerun/telemetry/tracker.py", "is_telemetry_enabled", {}
        )
        user_id = entrypoint("mobilerun/telemetry/tracker.py", "get_user_id", {})
        flush = entrypoint("mobilerun/telemetry/tracker.py", "flush", {})
        with patch.dict("os.environ", {"MOBILERUN_TELEMETRY_ENABLED": "true"}):
            self.assertFalse(enabled(config_enabled=True))
            capture(event, "user", config_enabled=True)
            asyncio.run(flush(config_enabled=True))
        event.assert_not_called()
        event.model_dump.assert_not_called()
        self.assertEqual(user_id(), "local")

    def test_all_default_profiles_are_local_and_require_model_selection(self) -> None:
        """Helper and structured-output models must not retain public defaults."""
        profiles = entrypoint(
            "mobilerun/config_manager/config_manager.py",
            "_default_profiles",
            {"LLMProfile": SimpleNamespace},
            "MobileConfig",
        )()
        self.assertEqual(
            set(profiles),
            {"manager", "executor", "fast_agent", "app_opener", "structured_output"},
        )
        for name, profile in profiles.items():
            with self.subTest(name=name):
                self.assertEqual(profile.provider, "Ollama")
                self.assertEqual(profile.model, "")

    def test_device_listing_does_not_use_saved_cloud_credentials(self) -> None:
        """Default device discovery must not even read cloud credentials."""
        credentials = Mock(side_effect=AssertionError("cloud credential lookup"))
        cloud = AsyncMock(side_effect=AssertionError("cloud request"))
        devices = entrypoint(
            "mobilerun/cli/main.py",
            "devices",
            {
                "capability_enabled": policy.capability_enabled,
                "resolve_cloud_api_key": credentials,
                "_list_cloud_devices": cloud,
                "adb": SimpleNamespace(list=AsyncMock(return_value=[])),
                "console": Mock(),
            },
        )
        asyncio.run(devices(False, None))
        credentials.assert_not_called()
        cloud.assert_not_called()

    def test_optional_entrypoints_reject_legacy_enable_flags(self) -> None:
        """YAML enabled=true alone cannot start extensions or tracing."""
        cases = (
            (
                "mobilerun/mcp/client.py",
                "discover_tools",
                "MCPClientManager",
                (SimpleNamespace(config=SimpleNamespace(enabled=True)),),
            ),
            (
                "mobilerun/mcp/client.py",
                "_connect_server",
                "MCPClientManager",
                (object(), "test"),
            ),
            (
                "mobilerun/mcp/client.py",
                "call_tool",
                "MCPClientManager",
                (object(), "test", {}),
            ),
            (
                "mobilerun/app_cards/providers/server_provider.py",
                "load_app_card",
                "ServerAppCardProvider",
                (object(), "test", "secret"),
            ),
            (
                "mobilerun/agent/utils/tracing_setup.py",
                "setup_tracing",
                None,
                (SimpleNamespace(enabled=True),),
            ),
            (
                "mobilerun/cli/deviceauth.py",
                "_endpoint",
                None,
                ("https://example.com", "/device/code"),
            ),
            ("mobilerun/cli/main.py", "_list_cloud_devices", None, (None,)),
        )
        for path, name, owner, args in cases:
            call = entrypoint(
                path, name, {"require_capability": policy.require_capability}, owner
            )
            with self.subTest(name=name), self.assertRaises(PermissionError):
                result = call(*args)
                if asyncio.iscoroutine(result):
                    asyncio.run(result)

    def test_doctor_only_reads_installed_versions(self) -> None:
        """Diagnostics do not fetch release metadata or compatibility mappings."""

        def record(*args):
            """Represent a diagnostic result without importing its dependencies."""
            return args

        status = SimpleNamespace(PASS="pass", WARN="warn")
        sdk = entrypoint(
            "mobilerun/cli/doctor.py",
            "check_sdk_version",
            {
                "CheckResult": record,
                "Status": status,
                "__version__": "test",
            },
        )
        self.assertIn("remote check disabled", asyncio.run(sdk(False))[2])
        latest = entrypoint("mobilerun/cli/doctor.py", "_get_latest_portal_version", {})
        self.assertIsNone(latest())
        fix = entrypoint(
            "mobilerun/cli/doctor.py", "fix_enable_accessibility", {"console": Mock()}
        )
        device = SimpleNamespace(shell=AsyncMock())
        self.assertFalse(asyncio.run(fix(device, False)))
        device.shell.assert_not_called()

    def test_model_oauth_entrypoints_are_blocked(self) -> None:
        """Model login is forbidden even when cloud device access is opted in."""
        for name, args in (
            ("run_openai_oauth_login", ("test.json", None)),
            ("run_gemini_oauth_login", ("test.json", None)),
            ("run_grok_oauth_login", ("test.json", None)),
            ("run_anthropic_setup_token_oauth", ()),
        ):
            namespace = {
                "reject_remote_model_login": policy.reject_remote_model_login,
                "DEFAULT_OPENAI_OAUTH_CALLBACK_HOST": "127.0.0.1",
                "DEFAULT_OPENAI_OAUTH_CALLBACK_PATH": "/callback",
                "DEFAULT_OPENAI_OAUTH_CALLBACK_PORT": 0,
            }
            call = entrypoint("mobilerun/cli/oauth_actions.py", name, namespace)
            with self.subTest(name=name), self.assertRaises(PermissionError):
                call(*args)


if __name__ == "__main__":
    unittest.main()
