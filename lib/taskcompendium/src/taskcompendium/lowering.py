# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export a TaskSpec submission as a Harbor task package."""

import hashlib
import importlib
import json
import re
import stat
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, model_validator

from taskcompendium.models import SCHEMA_VERSION, AnswerType, TaskSpec, VerifierKind
from taskcompendium.resources import (
    MAX_RESOURCE_BYTES,
    MAX_TOTAL_RESOURCE_BYTES,
    ResourceResolver,
    ResourceVisibility,
    decode_base64_content,
    materialize_resources,
    validate_resource_path,
    validate_resources,
)
from taskcompendium.submission import SubmissionConvention, render_instruction
from taskcompendium.verifier_registry import resolve_verifier, validate_verifier
from taskcompendium.verifiers.private_command import PrivateCommandVerifier

SPECIFICATION_FILE = "specification.json"
SUBMISSION_CONVENTION_FILE = "submission_convention.json"
ENVIRONMENT_CONFIG_FILE = "environment_config.json"
AGENT_RESOURCES_DIR = "inputs"
PRIVATE_RESOURCES_DIR = "private_resources"
DOCKER_SHELL_PROVIDER = "docker_shell:v1"
REGISTERED_PROVIDERS: Mapping[str, str] = MappingProxyType(
    {DOCKER_SHELL_PROVIDER: "taskcompendium.harbor.docker:DockerShellEnvironment"}
)
DOCKER_IMAGE_PATTERN = re.compile(r"[^\s@]+@sha256:([0-9a-f]{64})\Z")


class DockerRuntimeBinding(BaseModel):
    """An immutable container image and workspace for one Harbor trial."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    image: str
    workdir: str

    @model_validator(mode="after")
    def validate_runtime(self) -> "DockerRuntimeBinding":
        if DOCKER_IMAGE_PATTERN.fullmatch(self.image) is None:
            raise ValueError("Docker image must be pinned by SHA256 digest")
        path = PurePosixPath(self.workdir)
        if not path.is_absolute() or path.as_posix() != self.workdir or ".." in path.parts or self.workdir == "/":
            raise ValueError("Docker workdir must be a normalized absolute path below root")
        return self

    @property
    def image_sha256(self) -> str:
        match = DOCKER_IMAGE_PATTERN.fullmatch(self.image)
        assert match is not None
        return match.group(1)


class ToolBinding(BaseModel):
    """A pinned chat tool surface and its registered Harbor provider."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action_interface: str
    seed_sha256: str
    provider: str
    provider_revision: str
    tools: tuple[str, ...]
    tools_sha256: str

    @model_validator(mode="after")
    def validate_binding(self) -> "ToolBinding":
        if self.provider not in REGISTERED_PROVIDERS:
            raise ValueError("Unknown tool provider")
        if not self.action_interface or not self.provider_revision or not self.tools:
            raise ValueError("Tool binding requires interface, revision, and tools")
        if len(set(self.tools)) != len(self.tools) or any(not name for name in self.tools):
            raise ValueError("Tool binding requires unique nonempty tool names")
        for digest in (self.seed_sha256, self.tools_sha256):
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("Tool binding requires lowercase SHA256 digests")
        return self


class HarborEnvironmentConfig(BaseModel):
    """A chat environment with independent tool and container bindings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_binding: ToolBinding | None = None
    docker_runtime: DockerRuntimeBinding | None = None


def provider_class(binding: ToolBinding) -> type:
    """Resolve the registered implementation named by a tool binding."""
    return provider_class_for_name(binding.provider)


def provider_class_for_name(name: str) -> type:
    """Resolve a registered provider without accepting arbitrary import paths."""
    module_name, class_name = REGISTERED_PROVIDERS[name].split(":")
    return getattr(importlib.import_module(module_name), class_name)


def validate_provider_surface(binding: ToolBinding, docker_runtime: DockerRuntimeBinding | None = None) -> None:
    """Check provider identity and action schemas before an export or launch."""
    provider = provider_class(binding)
    if binding.provider == DOCKER_SHELL_PROVIDER:
        if docker_runtime is None or binding.seed_sha256 != docker_runtime.image_sha256:
            raise ValueError("Docker tool seed differs from pinned image")
    elif docker_runtime is not None:
        raise ValueError("Provider does not support a Docker runtime")
    elif provider.SEED_SHA256 != binding.seed_sha256:
        raise ValueError("Provider SEED_SHA256 differs from Harbor binding")
    for field, expected in (
        ("ACTION_INTERFACE", binding.action_interface),
        ("PROVIDER_REVISION", binding.provider_revision),
    ):
        if getattr(provider, field) != expected:
            raise ValueError(f"Provider {field} differs from Harbor binding")
    definitions = provider.TOOL_DEFINITIONS
    digest = hashlib.sha256(
        json.dumps(definitions, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    if digest != binding.tools_sha256:
        raise ValueError("Provider tool schemas differ from Harbor binding")
    names = tuple(definition["function"]["name"] for definition in definitions)
    if names != binding.tools or len(set(names)) != len(names):
        raise ValueError("Provider tool names differ from Harbor binding")


def docker_shell_binding(runtime: DockerRuntimeBinding) -> ToolBinding:
    """Bind the registered shell tool surface to one pinned Docker image."""
    provider = provider_class_for_name(DOCKER_SHELL_PROVIDER)
    definitions = provider.TOOL_DEFINITIONS
    digest = hashlib.sha256(
        json.dumps(definitions, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return ToolBinding(
        action_interface=provider.ACTION_INTERFACE,
        seed_sha256=runtime.image_sha256,
        provider=DOCKER_SHELL_PROVIDER,
        provider_revision=provider.PROVIDER_REVISION,
        tools=tuple(definition["function"]["name"] for definition in definitions),
        tools_sha256=digest,
    )


@dataclass(frozen=True)
class LoweringCandidate:
    """A compatible submission convention and Harbor environment configuration."""

    convention: SubmissionConvention
    environment_config: HarborEnvironmentConfig


class SelectionPolicy(StrEnum):
    """How a caller chooses from compatible lowerings."""

    ALL = "all"
    FIRST = "first"
    SAMPLE = "sample"


def compatible_lowerings(
    specification: TaskSpec,
    convention_library: Sequence[SubmissionConvention],
    environment_configs: Sequence[HarborEnvironmentConfig],
) -> tuple[LoweringCandidate, ...]:
    """Enumerate conventions and environments that preserve this task's contract."""
    return tuple(
        LoweringCandidate(convention, environment_config)
        for convention in convention_library
        if convention.supports(specification.answer_type)
        for environment_config in environment_configs
        if _is_compatible(specification, environment_config)
    )


def _is_compatible(specification: TaskSpec, environment_config: HarborEnvironmentConfig) -> bool:
    try:
        validate_environment_config(specification, environment_config)
    except ValueError:
        return False
    return True


def select_lowerings(
    candidates: Sequence[LoweringCandidate],
    policy: SelectionPolicy,
    *,
    rng_key: int | None = None,
) -> tuple[LoweringCandidate, ...]:
    """Select from the caller's compatible lowering candidates."""
    if not candidates:
        raise ValueError("No compatible lowerings")
    if policy == SelectionPolicy.SAMPLE:
        if rng_key is None:
            raise ValueError("Sample selection requires an RNG key")
        digest = hashlib.sha256(str(rng_key).encode()).digest()
        return (candidates[int.from_bytes(digest, "big") % len(candidates)],)
    if rng_key is not None:
        raise ValueError("An RNG key is only used by sample selection")
    if policy == SelectionPolicy.ALL:
        return tuple(candidates)
    if policy == SelectionPolicy.FIRST:
        return (candidates[0],)
    raise ValueError(f"Unknown selection policy: {policy}")


def validate_environment_config(specification: TaskSpec, environment_config: HarborEnvironmentConfig) -> None:
    """Require the selected binding to satisfy the task's semantic requirements."""
    state_verifier = specification.verifier.kind in {VerifierKind.STATE_MATCH, VerifierKind.PRIVATE_COMMAND}
    if (specification.answer_type == AnswerType.STATE) != state_verifier:
        raise ValueError("State result requires state verifier")
    binding = environment_config.tool_binding
    docker_runtime = environment_config.docker_runtime
    requirements = specification.requirements
    if specification.verifier.kind == VerifierKind.PRIVATE_COMMAND:
        if docker_runtime is None or binding is None or binding.provider != DOCKER_SHELL_PROVIDER:
            raise ValueError("Private command grading requires a Docker shell binding")
        verifier = resolve_verifier(specification.verifier)
        assert isinstance(verifier, PrivateCommandVerifier)
        if not any(
            resource.path == verifier.script_path
            and resource.visibility == ResourceVisibility.VERIFIER
            and resource.executable
            for resource in specification.resources
        ):
            raise ValueError("Private command script must be an executable verifier resource")
    if docker_runtime is not None:
        if requirements.seed_sha256 != docker_runtime.image_sha256:
            raise ValueError("Docker image digest differs from task seed")
        if set(requirements.capabilities) - {"filesystem", "shell", "process"}:
            raise ValueError("Docker runtime lacks required capabilities")
    elif requirements.capabilities:
        raise ValueError("Environment cannot satisfy required capabilities")
    if binding is None:
        if requirements.action_interfaces or (requirements.seed_sha256 is not None and docker_runtime is None):
            raise ValueError("Chat without tools cannot satisfy action-interface or seed requirements")
        if specification.answer_type == AnswerType.STATE and docker_runtime is None:
            raise ValueError("Chat without tools cannot grade environment state")
        if any(resource.visibility == ResourceVisibility.AGENT for resource in specification.resources):
            raise ValueError("Chat without tools cannot expose agent-visible files")
        return
    if requirements.action_interfaces != (binding.action_interface,):
        raise ValueError("Tool binding does not satisfy action-interface requirements")
    if requirements.seed_sha256 != binding.seed_sha256:
        raise ValueError("Tool binding seed differs from task seed")
    validate_provider_surface(binding, docker_runtime)
    if any(resource.visibility == ResourceVisibility.AGENT for resource in specification.resources):
        if not provider_class(binding).SUPPORTS_AGENT_FILES:
            raise ValueError("Tool provider cannot expose agent-visible files")


def read_specification(path: Path) -> TaskSpec:
    data = json.loads(path.read_text())
    if data["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported TaskSpec schema: {data['schema_version']}")
    specification = TaskSpec.model_validate(data)
    validate_verifier(specification.verifier)
    return specification


def read_environment_config(path: Path) -> HarborEnvironmentConfig:
    return HarborEnvironmentConfig.model_validate_json(path.read_text())


def read_submission_convention(path: Path) -> SubmissionConvention:
    return SubmissionConvention.model_validate_json(path.read_text())


def validate_exported_resources(specification: TaskSpec, task_dir: Path) -> None:
    """Recheck pinned payloads and path safety in the exported task before launch."""
    total_bytes = 0
    for resource in specification.resources:
        root = (
            task_dir / "environment" / AGENT_RESOURCES_DIR
            if resource.visibility == ResourceVisibility.AGENT
            else task_dir / PRIVATE_RESOURCES_DIR
        )
        relative = validate_resource_path(resource.path)
        if root.is_symlink() or root.parent.is_symlink():
            raise ValueError(f"Exported resource root is a symlink: {root}")
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"Exported resource symlink: {current}")
        if not current.is_file():
            raise ValueError(f"Missing exported resource: {resource.path}")
        if current.stat().st_size > MAX_RESOURCE_BYTES:
            raise ValueError("Exported resources exceed size limits")
        payload = current.read_bytes()
        total_bytes += len(payload)
        if len(payload) > MAX_RESOURCE_BYTES or total_bytes > MAX_TOTAL_RESOURCE_BYTES:
            raise ValueError("Exported resources exceed size limits")
        if resource.reference is not None:
            digest = resource.reference.sha256
        else:
            if resource.content is not None:
                source_content = resource.content.encode()
            else:
                assert resource.content_base64 is not None
                source_content = decode_base64_content(resource.content_base64)
            digest = hashlib.sha256(source_content).hexdigest()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"Exported resource digest mismatch: {resource.path}")
        if bool(current.stat().st_mode & stat.S_IXUSR) != resource.executable:
            raise ValueError(f"Exported resource executable bit differs: {resource.path}")


def validate_exported_docker_runtime(
    specification: TaskSpec, environment_config: HarborEnvironmentConfig, task_dir: Path
) -> None:
    """Reject a modified Docker launch definition or undeclared container inputs."""
    runtime = environment_config.docker_runtime
    if runtime is None:
        return
    config = tomllib.loads((task_dir / "task.toml").read_text())
    if config != {
        "version": "1.0",
        "environment": {
            "allow_internet": False,
            "docker_image": runtime.image,
            "workdir": runtime.workdir,
        },
        "verifier": {"environment_mode": "shared"},
    }:
        raise ValueError("Exported Docker task configuration differs from binding")
    environment_dir = task_dir / "environment"
    declared = {
        (Path(AGENT_RESOURCES_DIR) / validate_resource_path(resource.path)).as_posix()
        for resource in specification.resources
        if resource.visibility == ResourceVisibility.AGENT
    }
    if environment_dir.is_symlink() or not environment_dir.is_dir():
        raise ValueError("Exported Docker environment directory is invalid")
    actual: set[str] = set()
    for path in environment_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Exported Docker environment contains a symlink: {path}")
        if path.is_file():
            actual.add(path.relative_to(environment_dir).as_posix())
    if actual != declared:
        raise ValueError("Exported Docker environment contains undeclared files")


def lower_to_harbor(
    specification: TaskSpec,
    convention: SubmissionConvention,
    environment_config: HarborEnvironmentConfig,
    destination: Path,
    *,
    trusted_resolver: ResourceResolver | None = None,
) -> Path:
    """Write one custom-verifier task; launch agent selection remains separate."""
    validate_environment_config(specification, environment_config)
    validate_verifier(specification.verifier)
    validate_resources(specification.resources, trusted_resolver=trusted_resolver)
    instruction = render_instruction(specification, convention)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "environment").mkdir()
    (destination / "instruction.md").write_text(instruction)
    task_toml = 'version = "1.0"\n\n[environment]\nallow_internet = false\n'
    if environment_config.docker_runtime is not None:
        runtime = environment_config.docker_runtime
        task_toml += f"docker_image = {json.dumps(runtime.image)}\nworkdir = {json.dumps(runtime.workdir)}\n"
    task_toml += '\n[verifier]\nenvironment_mode = "shared"\n'
    (destination / "task.toml").write_text(task_toml)
    # Harbor's pinned revision requires a test script even when a custom verifier runs.
    tests_dir = destination / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    (destination / SPECIFICATION_FILE).write_text(specification.model_dump_json(indent=2) + "\n")
    (destination / ENVIRONMENT_CONFIG_FILE).write_text(environment_config.model_dump_json(indent=2) + "\n")
    (destination / SUBMISSION_CONVENTION_FILE).write_text(convention.model_dump_json(indent=2) + "\n")
    if any(resource.visibility == ResourceVisibility.AGENT for resource in specification.resources):
        materialize_resources(
            specification.resources,
            destination / "environment" / AGENT_RESOURCES_DIR,
            visibility=ResourceVisibility.AGENT,
            trusted_resolver=trusted_resolver,
        )
    if any(resource.visibility != ResourceVisibility.AGENT for resource in specification.resources):
        materialize_resources(
            specification.resources,
            destination / PRIVATE_RESOURCES_DIR,
            visibility=frozenset({ResourceVisibility.VERIFIER, ResourceVisibility.ORACLE}),
            trusted_resolver=trusted_resolver,
        )
    return destination
