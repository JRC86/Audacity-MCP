import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.server.fastmcp import FastMCP

from audacity_mcp.tools.plugin_tools import (
    _extract_json_array,
    _parse_commands,
    _parse_menu_categories,
)
from audacity_mcp_shared.error_codes import AudacityMCPError, ErrorCode


COMMANDS = [
    {
        "id": "Echo",
        "name": "Eko",
        "params": [
            {"key": "Delay", "type": "double", "default": 1.0},
            {"key": "Wet Only", "type": "bool", "default": "False"},
        ],
        "url": "Echo",
        "tip": "Localized effect tip",
    },
    {
        "id": "Chirp",
        "name": "Pip",
        "params": [
            {
                "key": "Wave-form",
                "type": "enum",
                "default": "Sinus",
                "enum": ["Sinus", "Triangel"],
            },
            {"key": "Count", "type": "size_t", "default": 1},
        ],
        "url": "Chirp",
        "tip": "Localized generator tip",
    },
    {
        "id": "HittaKlippning",
        "name": "Hitta klippning",
        "params": [{"key": "Threshold", "type": "int", "default": 3}],
        "url": "",
        "tip": "Localized analyzer tip",
    },
    {
        "id": "VerktygÅ",
        "name": "Verktyg Å",
        "params": [{"key": "Text", "type": "string", "default": ""}],
        "url": "",
        "tip": "Trusted local tool",
    },
    {
        "id": "Export2",
        "name": "Export",
        "params": [{"key": "Filename", "type": "string", "default": ""}],
        "url": "",
        "tip": "Must never become a plugin candidate",
    },
]

MENUS = [
    {"depth": 0, "flags": 0, "accel": ""},
    {"depth": 1, "flags": 0, "accel": "", "id": "ManageEffects"},
    {"depth": 1, "flags": 0, "accel": "", "id": "Echo"},
    {"depth": 1, "flags": 0, "accel": "", "id": "MenuOnly"},
    {"depth": 0, "flags": 0, "accel": ""},
    {"depth": 1, "flags": 0, "accel": "", "id": "ManageGenerators"},
    {"depth": 2, "flags": 0, "accel": "", "id": "Chirp"},
    {"depth": 0, "flags": 0, "accel": ""},
    {"depth": 1, "flags": 0, "accel": "", "id": "ManageAnalyzers"},
    {"depth": 1, "flags": 0, "accel": "", "id": "HittaKlippning"},
    {"depth": 0, "flags": 0, "accel": ""},
    {"depth": 1, "flags": 0, "accel": "", "id": "ManageTools"},
    {"depth": 1, "flags": 0, "accel": "", "id": "VerktygÅ"},
]


def _result(payload, *, field="message", success=True):
    result = {"success": success, "message": "", "raw": "", "data": {}}
    result[field] = json.dumps(payload, ensure_ascii=False, indent=2)
    return result


def _client(commands=COMMANDS, menus=MENUS):
    client = MagicMock()

    async def execute(command, **kwargs):
        assert command == "GetInfo"
        assert kwargs["Format"] == "JSON"
        if kwargs["Type"] == "Commands":
            return _result(commands)
        if kwargs["Type"] == "Menus":
            return _result(menus)
        raise AssertionError(f"Unexpected GetInfo type: {kwargs['Type']}")

    client.execute = AsyncMock(side_effect=execute)
    client.execute_long = AsyncMock(
        return_value={"success": True, "raw": "", "message": "", "data": {}}
    )
    return client


@pytest.fixture
def registered_tools():
    client = _client()
    mcp = FastMCP("TestPlugins")
    with patch("audacity_mcp.main.client", client):
        from audacity_mcp.tools.plugin_tools import register

        register(mcp)
    return mcp._tool_manager._tools, client


class TestJsonExtraction:
    def test_extracts_multiline_message(self):
        assert _extract_json_array(_result([{"id": "Echo"}]), "test") == [{"id": "Echo"}]

    def test_falls_back_to_raw_and_ignores_batch_status(self):
        result = _result([], field="raw")
        result["raw"] += "\nBatchCommand finished: OK"
        assert _extract_json_array(result, "test") == []

    def test_rejects_failed_response_even_with_json(self):
        with pytest.raises(AudacityMCPError) as exc:
            _extract_json_array(_result([], success=False), "test")
        assert exc.value.code == ErrorCode.COMMAND_FAILED

    def test_rejects_malformed_json(self):
        with pytest.raises(AudacityMCPError) as exc:
            _extract_json_array(
                {"success": True, "message": "not json", "raw": "", "data": {}},
                "test",
            )
        assert exc.value.code == ErrorCode.COMMAND_FAILED


class TestMetadataParsing:
    def test_preserves_unicode_and_parameter_keys(self):
        commands = _parse_commands(_result(COMMANDS))
        assert commands["VerktygÅ"]["name"] == "Verktyg Å"
        assert commands["Echo"]["params"][1]["key"] == "Wet Only"
        assert commands["Chirp"]["params"][0]["key"] == "Wave-form"

    def test_rejects_duplicate_commands(self):
        with pytest.raises(AudacityMCPError) as exc:
            _parse_commands(_result([COMMANDS[0], COMMANDS[0]]))
        assert exc.value.code == ErrorCode.COMMAND_FAILED

    def test_classifies_localized_menu_sections_by_stable_sentinel(self):
        categories = _parse_menu_categories(_result(MENUS))
        assert categories["Echo"] == {"effect"}
        assert categories["Chirp"] == {"generate"}
        assert categories["HittaKlippning"] == {"analyze"}
        assert categories["VerktygÅ"] == {"tool"}
        assert "ManageEffects" not in categories

    def test_rejects_missing_category_sentinels(self):
        menus = [{"depth": 0, "label": "File"}, {"depth": 1, "id": "Export2"}]
        with pytest.raises(AudacityMCPError) as exc:
            _parse_menu_categories(_result(menus))
        assert exc.value.code == ErrorCode.COMMAND_FAILED

    def test_rejects_item_before_top_level_menu(self):
        with pytest.raises(AudacityMCPError) as exc:
            _parse_menu_categories(_result([{"depth": 1, "id": "Echo"}]))
        assert exc.value.code == ErrorCode.COMMAND_FAILED


class TestPluginTools:
    @pytest.mark.asyncio
    async def test_three_tools_registered(self, registered_tools):
        tools, _ = registered_tools
        assert {"plugin_list", "plugin_get", "plugin_apply"} <= set(tools)

    @pytest.mark.asyncio
    async def test_list_is_compact_sorted_filtered_and_paginated(self, registered_tools):
        tools, _ = registered_tools
        result = await tools["plugin_list"].fn(category="all", query="", limit=2, offset=0)
        assert result["total"] == 4
        assert result["next_offset"] == 2
        assert [entry["name"] for entry in result["plugins"]] == ["Eko", "Hitta klippning"]
        assert "params" not in result["plugins"][0]

        filtered = await tools["plugin_list"].fn(
            category="generate", query="PIP", limit=50, offset=0
        )
        assert [entry["id"] for entry in filtered["plugins"]] == ["Chirp"]
        assert filtered["next_offset"] is None

    @pytest.mark.asyncio
    async def test_list_intersects_commands_and_menus(self, registered_tools):
        tools, _ = registered_tools
        result = await tools["plugin_list"].fn()
        ids = {entry["id"] for entry in result["plugins"]}
        assert "Export2" not in ids
        assert "MenuOnly" not in ids

    @pytest.mark.asyncio
    async def test_get_returns_full_metadata(self, registered_tools):
        tools, _ = registered_tools
        result = await tools["plugin_get"].fn(plugin_id="Echo")
        assert result["params"][0]["key"] == "Delay"
        assert result["url"] == "Echo"
        assert result["categories"] == ["effect"]
        assert result["authorized"] is True

    @pytest.mark.asyncio
    async def test_get_unknown_id_raises_not_found(self, registered_tools):
        tools, _ = registered_tools
        with pytest.raises(AudacityMCPError) as exc:
            await tools["plugin_get"].fn(plugin_id="Export2")
        assert exc.value.code == ErrorCode.COMMAND_NOT_FOUND

    @pytest.mark.asyncio
    async def test_invalid_plugin_id_fails_before_discovery(self, registered_tools):
        tools, client = registered_tools
        with pytest.raises(AudacityMCPError) as exc:
            await tools["plugin_get"].fn(plugin_id="bad\nid")
        assert exc.value.code == ErrorCode.INJECTION_DETECTED
        client.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tool_is_listed_but_blocked_without_allowlist(
        self, registered_tools, monkeypatch
    ):
        monkeypatch.delenv("AUDACITY_MCP_ALLOWED_TOOL_PLUGINS", raising=False)
        tools, client = registered_tools
        result = await tools["plugin_get"].fn(plugin_id="VerktygÅ")
        assert result["authorized"] is False
        assert "blocked_reason" in result

        with pytest.raises(AudacityMCPError) as exc:
            await tools["plugin_apply"].fn(plugin_id="VerktygÅ", parameters={})
        assert exc.value.code == ErrorCode.COMMAND_REJECTED
        client.execute_long.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("allowlist", ["verktygå", "Verktyg", "Other,Verktyg"])
    async def test_tool_allowlist_requires_exact_case_sensitive_id(
        self, registered_tools, monkeypatch, allowlist
    ):
        monkeypatch.setenv("AUDACITY_MCP_ALLOWED_TOOL_PLUGINS", allowlist)
        tools, client = registered_tools
        with pytest.raises(AudacityMCPError) as exc:
            await tools["plugin_apply"].fn(plugin_id="VerktygÅ", parameters={})
        assert exc.value.code == ErrorCode.COMMAND_REJECTED
        client.execute_long.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_allowlisted_tool_executes(self, registered_tools, monkeypatch):
        monkeypatch.setenv(
            "AUDACITY_MCP_ALLOWED_TOOL_PLUGINS", "Other, VerktygÅ"
        )
        tools, client = registered_tools
        await tools["plugin_apply"].fn(
            plugin_id="VerktygÅ", parameters={"Text": "safe text"}
        )
        client.execute_long.assert_awaited_once_with(
            "VerktygÅ", extra_params={"Text": "safe text"}
        )

    @pytest.mark.asyncio
    async def test_safe_category_executes_with_exact_keys(self, registered_tools):
        tools, client = registered_tools
        await tools["plugin_apply"].fn(
            plugin_id="Echo", parameters={"Delay": 0.5, "Wet Only": True}
        )
        client.execute_long.assert_awaited_once_with(
            "Echo", extra_params={"Delay": 0.5, "Wet Only": True}
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("plugin_id", "parameters", "expected_code"),
        [
            ("Echo", {"Missing": 1}, ErrorCode.INVALID_PARAMETER),
            ("Echo", {"Delay": True}, ErrorCode.INVALID_PARAMETER),
            ("Echo", {"Delay": float("inf")}, ErrorCode.INVALID_PARAMETER),
            ("Echo", {"Wet Only": 1}, ErrorCode.INVALID_PARAMETER),
            ("Chirp", {"Wave-form": "Triangle"}, ErrorCode.INVALID_PARAMETER),
            ("Chirp", {"Count": -1}, ErrorCode.INVALID_PARAMETER),
            ("VerktygÅ", {"Text": "bad\nvalue"}, ErrorCode.INJECTION_DETECTED),
            ("VerktygÅ", {"Text": "x" * 4097}, ErrorCode.VALUE_OUT_OF_RANGE),
        ],
    )
    async def test_invalid_parameters_never_execute(
        self, registered_tools, monkeypatch, plugin_id, parameters, expected_code
    ):
        monkeypatch.setenv("AUDACITY_MCP_ALLOWED_TOOL_PLUGINS", "VerktygÅ")
        tools, client = registered_tools
        with pytest.raises(AudacityMCPError) as exc:
            await tools["plugin_apply"].fn(plugin_id=plugin_id, parameters=parameters)
        assert exc.value.code == expected_code
        client.execute_long.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_excessive_parameter_count_never_executes(
        self, registered_tools
    ):
        tools, client = registered_tools
        with pytest.raises(AudacityMCPError) as exc:
            await tools["plugin_apply"].fn(
                plugin_id="Echo",
                parameters={f"key-{index}": index for index in range(129)},
            )
        assert exc.value.code == ErrorCode.VALUE_OUT_OF_RANGE
        client.execute_long.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_invalid_list_arguments_fail_before_discovery(self, registered_tools):
        tools, client = registered_tools
        with pytest.raises(AudacityMCPError) as exc:
            await tools["plugin_list"].fn(category="files")
        assert exc.value.code == ErrorCode.INVALID_PARAMETER
        client.execute.assert_not_awaited()
