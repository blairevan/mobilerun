"""Regression tests for permanently disabled telemetry and state event compatibility."""

import asyncio
from unittest.mock import Mock

import pytest

from mobilerun.telemetry import tracker
from mobilerun.telemetry.events import PackageVisitEvent

PRIMARY_TELEMETRY_ENV = "MOBILERUN_TELEMETRY_ENABLED"


@pytest.mark.parametrize("value", ["true", "1", "yes", "false", ""])
def test_telemetry_cannot_be_enabled(monkeypatch, value: str) -> None:
    """Reject legacy opt-ins without creating clients or user identifiers."""
    monkeypatch.setenv("MOBILERUN_TELEMETRY_ENABLED", value)
    monkeypatch.setenv("DROIDRUN_TELEMETRY_ENABLED", value)
    assert tracker.is_telemetry_enabled(config_enabled=True) is False
    assert tracker.is_telemetry_enabled(config_enabled=False) is False


def test_capture_never_serializes_or_logs_event_content(monkeypatch) -> None:
    """Discard even explicitly enabled captures before accessing event contents."""
    event = Mock(spec=PackageVisitEvent)
    tracker.capture(event, config_enabled=True)
    asyncio.run(tracker.flush(config_enabled=True))
    event.model_dump.assert_not_called()
    assert tracker.get_user_id() == "local"
    assert not hasattr(tracker, "_posthog")


def test_package_event_forwards_state_telemetry_setting(monkeypatch) -> None:
    from mobilerun.agent.droid import state as state_module

    captured = []

    def fake_capture(event, user_id=None, *, config_enabled=True) -> None:
        captured.append((event, user_id, config_enabled))

    monkeypatch.setattr(state_module, "capture", fake_capture)
    state = state_module.MobileAgentState(
        user_id="state-user",
        step_number=7,
        telemetry_config_enabled=False,
    )

    state.update_current_app("com.android.settings", ".Settings")

    assert len(captured) == 1
    event, user_id, config_enabled = captured[0]
    assert event == PackageVisitEvent(
        package_name="com.android.settings",
        activity_name=".Settings",
        step_number=7,
    )
    assert user_id == "state-user"
    assert config_enabled is False


def test_mobile_agent_copies_telemetry_config_to_shared_state(monkeypatch) -> None:
    from mobilerun.agent.droid import droid_agent as agent_module
    from mobilerun.config_manager.config_manager import (
        AgentConfig,
        MobileConfig,
        TelemetryConfig,
    )

    monkeypatch.setattr(agent_module, "setup_tracing", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        agent_module.MobileAgent,
        "_configure_default_logging",
        lambda *args, **kwargs: None,
    )
    config = MobileConfig(
        agent=AgentConfig(name="telemetry-test-external-agent"),
        telemetry=TelemetryConfig(enabled=False),
    )

    agent = agent_module.MobileAgent("Check telemetry", config=config)

    assert agent.shared_state.telemetry_config_enabled is False


def test_mobile_agent_lifecycle_forwards_disabled_telemetry_config(
    monkeypatch,
) -> None:
    from llama_index.core.llms.mock import MockLLM
    from llama_index.core.workflow import StartEvent

    from mobilerun.agent.droid import droid_agent as agent_module
    from mobilerun.agent.droid.events import FinalizeEvent
    from mobilerun.config_manager.config_manager import MobileConfig, TelemetryConfig
    from mobilerun.telemetry.events import (
        MobileAgentFinalizeEvent,
        MobileAgentInitEvent,
    )

    class FakeDriver:
        platform = "Android"
        supported = set()
        supported_buttons = set()

        async def get_date(self) -> str:
            return "2026-08-31"

    class FakeStateProvider:
        requires_coordinate_tools = False
        supported = set()

    class FakeRegistry:
        def disable_unsupported(self, capabilities) -> None:
            return None

        def disable(self, names) -> None:
            return None

        def register_from_dict(self, tools) -> None:
            return None

    class FakeContext:
        def __init__(self):
            self.events = []
            self.store = self

        def write_event_to_stream(self, event) -> None:
            self.events.append(event)

        async def set(self, key, value) -> None:
            setattr(self, key, value)

        async def get(self, key, default=None):
            return getattr(self, key, default)

    captures = []
    flushes = []

    def fake_capture(event, user_id=None, *, config_enabled=True) -> None:
        captures.append((event, user_id, config_enabled))

    async def fake_flush(*, config_enabled=True) -> None:
        flushes.append(config_enabled)

    async def fake_build_tool_registry(**kwargs):
        return FakeRegistry(), {"tap"}

    monkeypatch.setenv(PRIMARY_TELEMETRY_ENV, "true")
    monkeypatch.delenv("MOBILERUN_STREAM_SCREENSHOTS", raising=False)
    monkeypatch.delenv("DROIDRUN_STREAM_SCREENSHOTS", raising=False)
    monkeypatch.setattr(agent_module, "setup_tracing", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        agent_module.MobileAgent,
        "_configure_default_logging",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(agent_module, "build_tool_registry", fake_build_tool_registry)
    monkeypatch.setattr(agent_module, "capture", fake_capture)
    monkeypatch.setattr(agent_module, "flush", fake_flush)

    agent = agent_module.MobileAgent(
        "Check telemetry lifecycle",
        config=MobileConfig(telemetry=TelemetryConfig(enabled=False)),
        llms=MockLLM(),
        driver=FakeDriver(),
        state_provider=FakeStateProvider(),
        user_id="lifecycle-user",
    )
    context = FakeContext()

    async def run_lifecycle() -> None:
        await agent.start_handler(context, StartEvent())
        await agent.finalize(
            context,
            FinalizeEvent(success=True, reason="Lifecycle complete"),
        )

    asyncio.run(run_lifecycle())

    assert len(captures) == 2
    assert isinstance(captures[0][0], MobileAgentInitEvent)
    assert isinstance(captures[1][0], MobileAgentFinalizeEvent)
    assert [capture[1] for capture in captures] == [
        "lifecycle-user",
        "lifecycle-user",
    ]
    assert [capture[2] for capture in captures] == [False, False]
    assert flushes == [False]


def test_print_telemetry_message_respects_config_false(monkeypatch) -> None:
    messages = []
    monkeypatch.setenv(PRIMARY_TELEMETRY_ENV, "true")
    monkeypatch.setattr(tracker.logger, "debug", messages.append)

    tracker.print_telemetry_message(config_enabled=False)

    assert messages == ["Telemetry is permanently disabled by local-only policy."]


def test_cli_run_and_test_forward_loaded_telemetry_config(monkeypatch) -> None:
    from mobilerun.cli import main as cli_main
    from mobilerun.config_manager.config_manager import MobileConfig, TelemetryConfig

    class StopAfterTelemetryMessage(Exception):
        pass

    forwarded = []
    config = MobileConfig(telemetry=TelemetryConfig(enabled=False))

    def capture_config(*, config_enabled=True) -> None:
        forwarded.append(config_enabled)
        raise StopAfterTelemetryMessage

    async def skip_keyboard_cleanup(config, serial=None) -> None:
        return None

    monkeypatch.setattr(cli_main.ConfigLoader, "load", lambda _: config)
    monkeypatch.setattr(cli_main, "_setup_cli_logging", lambda _: None)
    monkeypatch.setattr(cli_main, "print_telemetry_message", capture_config)
    monkeypatch.setattr(cli_main, "_cleanup_android_keyboard", skip_keyboard_cleanup)
    monkeypatch.setattr(cli_main.console, "print", lambda *args, **kwargs: None)

    assert asyncio.run(cli_main.run_command("Check telemetry", debug=False)) is False
    assert asyncio.run(cli_main.test("Check telemetry", debug=False)) is None
    assert forwarded == [False, False]
