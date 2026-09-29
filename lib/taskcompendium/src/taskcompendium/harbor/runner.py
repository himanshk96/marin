# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a launch separately from a task-owned Harbor environment configuration."""

from enum import StrEnum
from pathlib import Path
from typing import Any

from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.trial import Trial
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from taskcompendium.lowering import (
    DOCKER_SHELL_PROVIDER,
    ENVIRONMENT_CONFIG_FILE,
    SPECIFICATION_FILE,
    SUBMISSION_CONVENTION_FILE,
    HarborEnvironmentConfig,
    provider_class,
    provider_class_for_name,
    read_environment_config,
    read_specification,
    read_submission_convention,
    validate_environment_config,
    validate_exported_docker_runtime,
    validate_exported_resources,
)

DEFAULT_CHAT_TIMEOUT = 120


class AgentStrategy(StrEnum):
    DIRECT_CHAT = "direct_chat"
    CHAT_TOOLS = "chat_tools"


class ReplayLaunch(BaseModel):
    """A fixed response for exercising the Harbor trial path."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    response: str


class ChatLaunch(BaseModel):
    """A model and endpoint selected when a lowered task is run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(min_length=1)
    api_base: str = Field(min_length=1)
    api_key_env: str | None = Field(default=None, min_length=1)
    strategy: AgentStrategy = AgentStrategy.DIRECT_CHAT
    request_timeout: float = Field(default=DEFAULT_CHAT_TIMEOUT, gt=0, allow_inf_nan=False)
    max_turns: int = Field(default=16, gt=0)
    trial_timeout: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @property
    def agent_kwargs(self) -> dict[str, Any]:
        values = self.model_dump(exclude={"model", "trial_timeout", "strategy"})
        if "max_turns" in values:
            values.pop("max_turns")
        return values


async def run_trial(
    task_dir: Path,
    environment_config: HarborEnvironmentConfig,
    launch: ReplayLaunch | ChatLaunch,
    trials_dir: Path,
    trial_name: str,
) -> TrialResult:
    """Run a lowered task and return Harbor's trial result."""
    if environment_config != read_environment_config(task_dir / ENVIRONMENT_CONFIG_FILE):
        raise ValueError("Launch environment configuration differs from the exported task")
    specification = read_specification(task_dir / SPECIFICATION_FILE)
    validate_environment_config(specification, environment_config)
    validate_exported_resources(specification, task_dir)
    validate_exported_docker_runtime(specification, environment_config, task_dir)
    try:
        convention = read_submission_convention(task_dir / SUBMISSION_CONVENTION_FILE)
    except (ValueError, ValidationError):
        if not isinstance(launch, ReplayLaunch) or environment_config.tool_binding is not None:
            raise
        convention = None  # The verifier records invalid private metadata as an ungraded outcome.
    tool_binding = environment_config.tool_binding
    if convention is not None and not convention.supports(specification.answer_type):
        raise ValueError("Submission convention differs from answer type")
    if isinstance(launch, ReplayLaunch):
        if tool_binding is not None:
            raise ValueError("Tool tasks require a model endpoint")
        agent: dict[str, Any] = {
            "import_path": "taskcompendium.harbor.adapter:ReplayAgent",
            "kwargs": launch.model_dump(),
        }
    else:
        expected_strategy = AgentStrategy.CHAT_TOOLS if tool_binding is not None else AgentStrategy.DIRECT_CHAT
        if launch.strategy != expected_strategy:
            raise ValueError("Agent strategy differs from Harbor environment binding")
        agent_path = "taskcompendium.harbor.adapter:DirectChatAgent"
        kwargs = launch.agent_kwargs
        if tool_binding is not None:
            agent_path = "taskcompendium.harbor.adapter:ChatToolAgent"
            kwargs["max_turns"] = launch.max_turns
        agent = {
            "import_path": agent_path,
            "model_name": launch.model,
            "kwargs": kwargs,
        }
    if environment_config.docker_runtime is not None:
        provider = provider_class_for_name(DOCKER_SHELL_PROVIDER)
        environment = {
            "import_path": f"{provider.__module__}:{provider.__name__}",
            "kwargs": {
                "seed_sha256": environment_config.docker_runtime.image_sha256,
                "action_interface": provider.ACTION_INTERFACE,
            },
        }
    elif tool_binding is None:
        environment = {"import_path": "taskcompendium.harbor.adapter:NoToolEnvironment"}
    else:
        environment = {
            "import_path": f"{provider_class(tool_binding).__module__}:{provider_class(tool_binding).__name__}",
            "kwargs": {
                "seed_sha256": tool_binding.seed_sha256,
                "action_interface": tool_binding.action_interface,
            },
        }
    config = TrialConfig.model_validate(
        {
            "task": {"path": str(task_dir.resolve())},
            "trials_dir": str(trials_dir.resolve()),
            "trial_name": trial_name,
            "environment": environment,
            "agent": agent,
            "verifier": {"import_path": "taskcompendium.harbor.adapter:SemanticVerifier"},
            **(
                {"trial_attempt_timeout_sec": launch.trial_timeout}
                if isinstance(launch, ChatLaunch) and launch.trial_timeout
                else {}
            ),
        }
    )
    trial = await Trial.create(config)
    return await trial.run()
