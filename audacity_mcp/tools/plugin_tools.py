import json
import math
import os
import re

from mcp.server.fastmcp import FastMCP

from audacity_mcp_shared.error_codes import AudacityMCPError, ErrorCode
from audacity_mcp_shared.pipe_protocol import has_dangerous_chars


_CATEGORY_SENTINELS = {
    "effect": "ManageEffects",
    "generate": "ManageGenerators",
    "analyze": "ManageAnalyzers",
    "tool": "ManageTools",
}
_SAFE_CATEGORIES = {"effect", "generate", "analyze"}
_JSON_STRING_TOKEN = re.compile(r'"(?:\\.|[^"\\])*"')
_WINDOWS_DRIVE_PATH = re.compile(r"[A-Za-z]:\\")
_TOOL_ALLOWLIST_ENV = "AUDACITY_MCP_ALLOWED_TOOL_PLUGINS"
_MAX_IDENTIFIER_LENGTH = 512
_MAX_QUERY_LENGTH = 256
_MAX_PARAMETERS = 128
_MAX_STRING_LENGTH = 4096
_MAX_PAGE_SIZE = 200


def _command_error(message: str) -> AudacityMCPError:
    return AudacityMCPError(ErrorCode.COMMAND_FAILED, message)


def _repair_windows_paths(text: str) -> str:
    """Escape drive-letter paths that Audacity 3.x emits as invalid JSON."""

    def repair_token(match: re.Match) -> str:
        content = match.group(0)[1:-1]
        path_match = _WINDOWS_DRIVE_PATH.search(content)
        if path_match is None:
            return match.group(0)

        repaired: list[str] = []
        path_start = path_match.start()
        index = 0
        while index < len(content):
            char = content[index]
            if char != "\\" or index < path_start:
                repaired.append(char)
                index += 1
                continue
            if index + 1 < len(content) and content[index + 1] == "\\":
                repaired.append("\\\\")
                index += 2
                continue
            repaired.append("\\\\")
            index += 1
        return f'"{"".join(repaired)}"'

    return _JSON_STRING_TOKEN.sub(repair_token, text)


def _decode_json_array(text: str) -> list | None:
    decoder = json.JSONDecoder()
    try:
        parsed, _ = decoder.raw_decode(text)
    except json.JSONDecodeError:
        repaired = _repair_windows_paths(text)
        if repaired == text:
            return None
        try:
            parsed, _ = decoder.raw_decode(repaired)
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, list) else None


def _extract_json_array(result: dict, response_name: str) -> list:
    """Extract a JSON array from current and older Audacity client responses."""
    if not isinstance(result, dict):
        raise _command_error(f"{response_name} returned an invalid response")
    if result.get("success") is False:
        detail = result.get("message") or "Audacity reported command failure"
        raise _command_error(f"{response_name} failed: {detail}")

    for field in ("message", "raw"):
        value = result.get(field)
        if not isinstance(value, str) or not value.strip():
            continue

        # Modern parse_response puts the JSON in message. Older server
        # versions may leave it in raw, optionally followed by batch status.
        text = value.strip()
        parsed = _decode_json_array(text)
        if parsed is not None:
            return parsed

    raise _command_error(f"{response_name} did not return a valid JSON array")


def _validate_text(value: str, field: str, max_length: int) -> None:
    if not isinstance(value, str) or not value:
        raise _command_error(f"{field} must be a non-empty string")
    if len(value) > max_length:
        raise _command_error(f"{field} exceeds the maximum length of {max_length}")
    if has_dangerous_chars(value):
        raise _command_error(f"{field} contains illegal control characters")


def _parse_commands(result: dict) -> dict[str, dict]:
    rows = _extract_json_array(result, "GetInfo Commands")
    commands: dict[str, dict] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise _command_error(f"Command metadata row {index} is not an object")

        command_id = row.get("id")
        _validate_text(command_id, f"Command metadata row {index} id", _MAX_IDENTIFIER_LENGTH)
        if command_id in commands:
            raise _command_error(f"Duplicate command metadata for id: {command_id}")

        name = row.get("name")
        if not isinstance(name, str):
            raise _command_error(f"Command metadata for {command_id} has an invalid name")

        params = row.get("params", [])
        if not isinstance(params, list):
            raise _command_error(f"Command metadata for {command_id} has invalid parameters")
        if len(params) > _MAX_PARAMETERS:
            raise _command_error(
                f"Command metadata for {command_id} exceeds {_MAX_PARAMETERS} parameters"
            )

        seen_keys: set[str] = set()
        for param_index, param in enumerate(params):
            if not isinstance(param, dict):
                raise _command_error(
                    f"Parameter {param_index} for {command_id} is not an object"
                )
            key = param.get("key")
            _validate_text(
                key,
                f"Parameter {param_index} key for {command_id}",
                _MAX_IDENTIFIER_LENGTH,
            )
            if key in seen_keys:
                raise _command_error(f"Duplicate parameter key for {command_id}: {key}")
            seen_keys.add(key)
            if not isinstance(param.get("type"), str):
                raise _command_error(f"Parameter {key} for {command_id} has no valid type")
            if param.get("type") == "enum":
                values = param.get("enum")
                if not isinstance(values, list) or not all(
                    isinstance(value, str) for value in values
                ):
                    raise _command_error(
                        f"Enum parameter {key} for {command_id} has invalid choices"
                    )

        commands[command_id] = {
            "id": command_id,
            "name": name,
            "params": params,
            "url": row.get("url", "") if isinstance(row.get("url", ""), str) else "",
            "tip": row.get("tip", "") if isinstance(row.get("tip", ""), str) else "",
        }
    return commands


def _parse_menu_categories(result: dict) -> dict[str, set[str]]:
    rows = _extract_json_array(result, "GetInfo Menus")
    sections: list[list[dict]] = []
    current: list[dict] | None = None

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise _command_error(f"Menu metadata row {index} is not an object")
        depth = row.get("depth")
        if type(depth) is not int or depth < 0:
            raise _command_error(f"Menu metadata row {index} has an invalid depth")
        menu_id = row.get("id")
        if menu_id is not None:
            _validate_text(menu_id, f"Menu metadata row {index} id", _MAX_IDENTIFIER_LENGTH)

        if depth == 0:
            current = []
            sections.append(current)
        elif current is None:
            raise _command_error("Menu metadata contains an item before a top-level menu")
        current.append(row)

    if not sections:
        raise _command_error("GetInfo Menus returned no top-level menus")

    sentinel_to_category = {
        sentinel: category for category, sentinel in _CATEGORY_SENTINELS.items()
    }
    category_sections: dict[str, list[dict]] = {}
    for section in sections:
        found = {
            sentinel_to_category[row["id"]]
            for row in section
            if row.get("id") in sentinel_to_category
        }
        if len(found) > 1:
            raise _command_error("Audacity menu categories are structurally ambiguous")
        if found:
            category = found.pop()
            if category in category_sections:
                raise _command_error(f"Duplicate Audacity menu section for {category}")
            category_sections[category] = section

    if not category_sections:
        raise _command_error("Could not identify any plugin-capable Audacity menus")

    categories: dict[str, set[str]] = {}
    sentinel_ids = set(_CATEGORY_SENTINELS.values())
    for category, section in category_sections.items():
        for row in section:
            menu_id = row.get("id")
            if menu_id and menu_id not in sentinel_ids:
                categories.setdefault(menu_id, set()).add(category)
    return categories


def _tool_allowlist() -> set[str]:
    return {
        item.strip()
        for item in os.environ.get(_TOOL_ALLOWLIST_ENV, "").split(",")
        if item.strip()
    }


def _authorization(command_id: str, categories: set[str]) -> tuple[bool, str | None]:
    if categories & _SAFE_CATEGORIES:
        return True, None
    if categories == {"tool"}:
        if command_id in _tool_allowlist():
            return True, None
        return (
            False,
            f"Tool plugins require exact authorization in {_TOOL_ALLOWLIST_ENV}",
        )
    return False, "Command is not in an allowed plugin category"


async def _discover_plugins(client) -> dict[str, dict]:
    command_result = await client.execute("GetInfo", Type="Commands", Format="JSON")
    menu_result = await client.execute("GetInfo", Type="Menus", Format="JSON")
    commands = _parse_commands(command_result)
    menu_categories = _parse_menu_categories(menu_result)

    plugins: dict[str, dict] = {}
    for command_id in commands.keys() & menu_categories.keys():
        metadata = commands[command_id]
        categories = menu_categories[command_id]
        authorized, blocked_reason = _authorization(command_id, categories)
        plugin = {
            **metadata,
            "categories": sorted(categories),
            "authorized": authorized,
        }
        if blocked_reason:
            plugin["blocked_reason"] = blocked_reason
        plugins[command_id] = plugin
    return plugins


def _validate_plugin_id(plugin_id: str) -> None:
    if not isinstance(plugin_id, str) or not plugin_id:
        raise AudacityMCPError(ErrorCode.INVALID_PARAMETER, "plugin_id must not be empty")
    if len(plugin_id) > _MAX_IDENTIFIER_LENGTH:
        raise AudacityMCPError(
            ErrorCode.VALUE_OUT_OF_RANGE,
            f"plugin_id must not exceed {_MAX_IDENTIFIER_LENGTH} characters",
        )
    if has_dangerous_chars(plugin_id):
        raise AudacityMCPError(
            ErrorCode.INJECTION_DETECTED,
            "plugin_id contains illegal control characters",
        )


def _find_plugin(plugins: dict[str, dict], plugin_id: str) -> dict:
    _validate_plugin_id(plugin_id)
    plugin = plugins.get(plugin_id)
    if plugin is None:
        raise AudacityMCPError(
            ErrorCode.COMMAND_NOT_FOUND,
            f"Plugin candidate not found or not in an allowed menu category: {plugin_id}",
        )
    return plugin


def _validate_parameters(plugin: dict, parameters: dict | None) -> dict:
    if parameters is None:
        return {}
    if not isinstance(parameters, dict):
        raise AudacityMCPError(
            ErrorCode.INVALID_PARAMETER, "parameters must be an object"
        )
    if len(parameters) > _MAX_PARAMETERS:
        raise AudacityMCPError(
            ErrorCode.VALUE_OUT_OF_RANGE,
            f"parameters must contain at most {_MAX_PARAMETERS} entries",
        )

    schema = {param["key"]: param for param in plugin["params"]}
    validated: dict[str, str | int | float | bool] = {}
    for key, value in parameters.items():
        if not isinstance(key, str) or key not in schema:
            raise AudacityMCPError(
                ErrorCode.INVALID_PARAMETER,
                f"Unknown parameter for {plugin['id']}: {key!r}",
            )

        param_type = schema[key]["type"]
        if param_type == "bool":
            valid = type(value) is bool
        elif param_type == "int":
            valid = type(value) is int
        elif param_type == "size_t":
            valid = type(value) is int and value >= 0
        elif param_type in {"float", "double"}:
            valid = type(value) in {int, float}
            if valid:
                try:
                    valid = math.isfinite(value)
                except OverflowError:
                    valid = False
        elif param_type == "string":
            valid = isinstance(value, str)
            if valid and len(value) > _MAX_STRING_LENGTH:
                raise AudacityMCPError(
                    ErrorCode.VALUE_OUT_OF_RANGE,
                    f"String parameter {key} exceeds {_MAX_STRING_LENGTH} characters",
                )
            if valid and has_dangerous_chars(value):
                raise AudacityMCPError(
                    ErrorCode.INJECTION_DETECTED,
                    f"String parameter {key} contains illegal control characters",
                )
        elif param_type == "enum":
            valid = isinstance(value, str) and value in schema[key]["enum"]
        else:
            raise AudacityMCPError(
                ErrorCode.INVALID_PARAMETER,
                f"Unsupported Audacity parameter type for {key}: {param_type}",
            )

        if not valid:
            expected = (
                f"one of {schema[key]['enum']!r}"
                if param_type == "enum"
                else param_type
            )
            raise AudacityMCPError(
                ErrorCode.INVALID_PARAMETER,
                f"Invalid value for parameter {key}; expected {expected}",
            )
        validated[key] = value
    return validated


def register(mcp: FastMCP):
    from audacity_mcp.main import client

    @mcp.tool()
    async def plugin_list(
        category: str = "all",
        query: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """List enabled Audacity plugin candidates.

        Includes built-in and third-party Effect, Generate, Analyze, and Tool
        entries because Audacity does not expose plugin provenance. Tool entries
        are listed but require AUDACITY_MCP_ALLOWED_TOOL_PLUGINS authorization.

        Args:
            category: One of all, effect, generate, analyze, or tool.
            query: Optional case-insensitive search over plugin id and name.
            limit: Maximum results to return (1-200). Default: 50.
            offset: Zero-based pagination offset. Default: 0.
        """
        if category not in {"all", *_CATEGORY_SENTINELS.keys()}:
            raise AudacityMCPError(
                ErrorCode.INVALID_PARAMETER,
                "category must be one of: all, effect, generate, analyze, tool",
            )
        if not isinstance(query, str):
            raise AudacityMCPError(ErrorCode.INVALID_PARAMETER, "query must be a string")
        if len(query) > _MAX_QUERY_LENGTH:
            raise AudacityMCPError(
                ErrorCode.VALUE_OUT_OF_RANGE,
                f"query must not exceed {_MAX_QUERY_LENGTH} characters",
            )
        if has_dangerous_chars(query):
            raise AudacityMCPError(
                ErrorCode.INJECTION_DETECTED,
                "query contains illegal control characters",
            )
        if type(limit) is not int or not 1 <= limit <= _MAX_PAGE_SIZE:
            raise AudacityMCPError(
                ErrorCode.VALUE_OUT_OF_RANGE,
                f"limit must be an integer from 1 to {_MAX_PAGE_SIZE}",
            )
        if type(offset) is not int or offset < 0:
            raise AudacityMCPError(
                ErrorCode.VALUE_OUT_OF_RANGE,
                "offset must be a non-negative integer",
            )

        plugins = await _discover_plugins(client)
        query_folded = query.casefold()
        filtered = [
            plugin
            for plugin in plugins.values()
            if (category == "all" or category in plugin["categories"])
            and (
                not query_folded
                or query_folded in plugin["id"].casefold()
                or query_folded in plugin["name"].casefold()
            )
        ]
        filtered.sort(
            key=lambda plugin: (plugin["name"].casefold(), plugin["id"].casefold())
        )
        page = filtered[offset : offset + limit]
        entries = []
        for plugin in page:
            entry = {
                "id": plugin["id"],
                "name": plugin["name"],
                "categories": plugin["categories"],
                "parameter_count": len(plugin["params"]),
                "authorized": plugin["authorized"],
            }
            if "blocked_reason" in plugin:
                entry["blocked_reason"] = plugin["blocked_reason"]
            entries.append(entry)

        next_offset = offset + len(entries)
        return {
            "plugins": entries,
            "total": len(filtered),
            "offset": offset,
            "limit": limit,
            "next_offset": next_offset if next_offset < len(filtered) else None,
        }

    @mcp.tool()
    async def plugin_get(plugin_id: str) -> dict:
        """Get full metadata and parameter schema for an Audacity plugin candidate.

        Args:
            plugin_id: Exact scripting id returned by plugin_list.
        """
        _validate_plugin_id(plugin_id)
        plugins = await _discover_plugins(client)
        return _find_plugin(plugins, plugin_id)

    @mcp.tool()
    async def plugin_apply(plugin_id: str, parameters: dict | None = None) -> dict:
        """Apply an enabled Audacity plugin candidate using validated parameters.

        Effect plugins process the current selection, Generate plugins create
        audio, and Analyze plugins may create labels or analysis output. Installed
        plugins are trusted local code and cannot be sandboxed by AudacityMCP.
        Tool entries must be explicitly authorized with the comma-separated
        AUDACITY_MCP_ALLOWED_TOOL_PLUGINS environment variable.

        Args:
            plugin_id: Exact scripting id returned by plugin_list.
            parameters: Optional parameter object using keys from plugin_get.
        """
        _validate_plugin_id(plugin_id)
        plugins = await _discover_plugins(client)
        plugin = _find_plugin(plugins, plugin_id)
        if not plugin["authorized"]:
            raise AudacityMCPError(
                ErrorCode.COMMAND_REJECTED,
                plugin.get("blocked_reason", "Plugin execution is not authorized"),
            )
        validated = _validate_parameters(plugin, parameters)
        return await client.execute_long(plugin_id, extra_params=validated)
