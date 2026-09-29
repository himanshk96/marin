# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A pinned Docker workspace exposed through a chat shell tool."""

import json
import re
import shlex
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

from harbor.environments.docker.docker import DockerEnvironment

from taskcompendium.harbor.snapshot import download_snapshot
from taskcompendium.resources import validate_resource_path

ACTION_INTERFACE = "docker_shell:v1"
PROVIDER_REVISION = "docker_shell:v1"
IMAGE_DIGEST = re.compile(r"^[^\s@]+@sha256:([0-9a-f]{64})$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
TOOL_DEFINITIONS = (
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command in the task workspace.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
)


def _workdir(path: str | None) -> PurePosixPath:
    if path is None:
        raise ValueError("Docker shell tasks require an explicit workdir")
    normalized = PurePosixPath(path)
    if not normalized.is_absolute() or str(normalized) != path or ".." in normalized.parts or path == "/":
        raise ValueError("Docker shell workdir must be a normalized absolute directory")
    return normalized


class DockerShellEnvironment(DockerEnvironment):
    """Use Harbor's isolated Docker lifecycle for one persistent chat workspace."""

    ACTION_INTERFACE = ACTION_INTERFACE
    PROVIDER_REVISION = PROVIDER_REVISION
    TOOL_DEFINITIONS = TOOL_DEFINITIONS
    SUPPORTS_AGENT_FILES = True

    def __init__(
        self,
        *args: Any,
        seed_sha256: str,
        action_interface: str = ACTION_INTERFACE,
        **kwargs: Any,
    ) -> None:
        if action_interface != ACTION_INTERFACE:
            raise ValueError("Docker shell action interface differs from its binding")
        self.seed_sha256 = seed_sha256
        super().__init__(*args, **kwargs)

    def _validate_definition(self) -> None:
        super()._validate_definition()
        image = self.task_env_config.docker_image
        match = IMAGE_DIGEST.fullmatch(image or "")
        if match is None or match.group(1) != self.seed_sha256:
            raise ValueError("Docker task requires its SHA256-pinned image as the declared seed")
        _workdir(self.task_env_config.workdir)
        if (self.environment_dir / "Dockerfile").exists() or (self.environment_dir / "docker-compose.yaml").exists():
            raise ValueError("Pinned Docker shell tasks cannot override the image with a build or compose file")
        if self.extra_docker_compose_paths:
            raise ValueError("Pinned Docker shell tasks cannot use extra compose files")

    async def _run_docker_compose_command(self, command: list[str], check: bool = True, timeout_sec: int | None = None):
        if command and command[0] == "up":
            command = ["up", "--pull", "never", *command[1:]]
        return await super()._run_docker_compose_command(command, check=check, timeout_sec=timeout_sec)

    async def _upload_environment_dir_after_start(self) -> None:
        workdir = _workdir(self.task_env_config.workdir)
        result = await self.exec(f"mkdir -p {shlex.quote(str(workdir))}", cwd="/", user="root")
        if result.return_code != 0:
            raise RuntimeError(f"Cannot create Docker workspace: {result.stderr or result.stdout}")
        await super()._upload_environment_dir_after_start()

    async def native_tool_definitions(self) -> list[dict[str, Any]]:
        return deepcopy(list(self.TOOL_DEFINITIONS))

    async def dispatch_action(self, name: str, arguments: str, call_id: str) -> str:
        """Run one valid shell call; command failures remain tool observations."""
        try:
            payload = json.loads(arguments)
            if name != "shell" or not isinstance(payload, dict) or set(payload) != {"command"}:
                raise ValueError("Shell requires exactly one command string")
            command = payload["command"]
            if not isinstance(command, str):
                raise ValueError("Shell command must be a string")
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            observation = json.dumps({"error": str(error)}, separators=(",", ":"))
        else:
            result = await self.exec(command)
            observation = result.model_dump_json(exclude_none=True)
        return observation

    async def grade_state_async(self, expected_state_json: str) -> float:
        """Compare selected workspace files with a private SHA256 manifest."""
        expected = json.loads(expected_state_json)
        if not isinstance(expected, dict) or set(expected) != {"files"} or not isinstance(expected["files"], dict):
            raise ValueError("Docker expected state must be a file digest manifest")
        files = expected["files"]
        if not files:
            raise ValueError("Docker expected state must name at least one file")
        workdir = _workdir(self.task_env_config.workdir)
        for relative, digest in files.items():
            if not isinstance(relative, str) or not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
                raise ValueError("Docker expected state contains an invalid path or SHA256")
            path = validate_resource_path(relative)
            # Refuse symlinks in every path component before inspecting contents.
            target = workdir
            for index, component in enumerate(path.parts):
                target = target / component
                quoted = shlex.quote(str(target))
                kind = "-f" if index == len(path.parts) - 1 else "-e"
                result = await self.exec(
                    f"if [ -L {quoted} ]; then printf 'LINK'; "
                    f"elif [ ! {kind} {quoted} ]; then printf 'MISSING'; "
                    f"else printf 'PRESENT'; fi",
                    cwd="/",
                )
                if result.return_code != 0:
                    raise RuntimeError(
                        f"Docker state inspection failed for {relative!r}: {result.stderr or result.stdout}"
                    )
                if (result.stdout or "").strip() != "PRESENT":
                    return 0.0
            result = await self.exec(f"sha256sum {shlex.quote(str(target))}", cwd="/")
            if result.return_code != 0:
                raise RuntimeError(f"Docker state hash failed for {relative!r}: {result.stderr or result.stdout}")
            actual = (result.stdout or "").split(maxsplit=1)[0]
            if SHA256.fullmatch(actual) is None:
                raise RuntimeError(f"Docker state hash was malformed for {relative!r}")
            if actual != digest:
                return 0.0
        return 1.0

    async def snapshot_workspace(self, destination: Path, excluded_paths: tuple[str, ...] = ()) -> None:
        """Copy a bounded, link-free view of the live workspace for a private grader."""
        exclusions = tuple(str(validate_resource_path(path)) for path in excluded_paths)
        result = await self._run_docker_compose_command(["ps", "-q", "main"])
        container_id = (result.stdout or "").strip()
        if not container_id or len(container_id.splitlines()) != 1:
            raise RuntimeError("Expected one running Docker workspace container")
        await download_snapshot(container_id, str(_workdir(self.task_env_config.workdir)), destination, exclusions)

    async def stop(self, delete: bool) -> None:
        """Clean trial containers and volumes while retaining the shared pinned image."""
        try:
            await self.prepare_logs_for_host()
            if self._keep_containers:
                await self._run_docker_compose_command(["stop"])
            elif delete:
                await self._run_docker_compose_command(["down", "--volumes", "--remove-orphans"])
            else:
                await self._run_docker_compose_command(["down"])
        finally:
            self._cleanup_mounts_compose_file()
            self._cleanup_resources_compose_file()
