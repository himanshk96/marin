# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned Docker workspaces through TaskCompendium and Harbor."""

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread

import pytest

from taskcompendium.grading import exact_answer, state_match
from taskcompendium.harbor.runner import AgentStrategy, ChatLaunch, run_trial
from taskcompendium.lowering import DockerRuntimeBinding, HarborEnvironmentConfig, docker_shell_binding, lower_to_harbor
from taskcompendium.models import AnswerType, Source, TaskRequirements, TaskSpec
from taskcompendium.resources import ResourceVisibility, TaskResource
from taskcompendium.submission import AnswerFormat, SubmissionConvention
from taskcompendium.verifiers.private_command import private_command


def _source() -> Source:
    return Source(dataset="docker-contract", revision="1", row="workspace", importer_revision="1")


def _state_specification(runtime: DockerRuntimeBinding, resources: tuple[TaskResource, ...] = ()) -> TaskSpec:
    return TaskSpec(
        id="docker-workspace",
        instructions="Use the workspace to complete the task.",
        verifier=state_match(json.dumps({"files": {"answer.txt": hashlib.sha256(b"good\n").hexdigest()}})),
        source=_source(),
        requirements=TaskRequirements(
            capabilities=("filesystem", "shell"),
            action_interfaces=("docker_shell:v1",),
            seed_sha256=runtime.image_sha256,
        ),
        answer_type=AnswerType.STATE,
        resources=resources,
    )


def _direct_specification(runtime: DockerRuntimeBinding, resources: tuple[TaskResource, ...] = ()) -> TaskSpec:
    return TaskSpec(
        id="docker-workspace",
        instructions="Use the workspace to complete the task.",
        verifier=exact_answer("12"),
        source=_source(),
        requirements=TaskRequirements(capabilities=("filesystem", "shell"), seed_sha256=runtime.image_sha256),
        answer_type=AnswerType.NUMBER,
        resources=resources,
    )


def _docker_image() -> DockerRuntimeBinding:
    image = os.environ.get("TASKCOMPENDIUM_DOCKER_TEST_IMAGE")
    if not image:
        pytest.skip("Set TASKCOMPENDIUM_DOCKER_TEST_IMAGE to a locally available repository@sha256:digest image")
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI is unavailable")
    result = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        pytest.skip(f"Pinned Docker test image is unavailable: {image}")
    return DockerRuntimeBinding(image=image, workdir="/app")


def test_docker_image_and_tool_surface_must_match_task_before_export(tmp_path):
    first = DockerRuntimeBinding(image=f"example/a@sha256:{'a' * 64}", workdir="/app")
    second = DockerRuntimeBinding(image=f"example/b@sha256:{'b' * 64}", workdir="/app")
    specification = _state_specification(first)
    environment_config = HarborEnvironmentConfig(tool_binding=docker_shell_binding(first), docker_runtime=second)

    with pytest.raises(ValueError, match=r"seed|digest"):
        lower_to_harbor(
            specification,
            SubmissionConvention(id="state", answer_format=AnswerFormat.STATE),
            environment_config,
            tmp_path / "task",
        )
    assert not (tmp_path / "task").exists()


def test_direct_chat_docker_still_rejects_agent_files(tmp_path):
    runtime = DockerRuntimeBinding(image=f"example/a@sha256:{'a' * 64}", workdir="/app")
    specification = _direct_specification(
        runtime,
        resources=(TaskResource(path="input.txt", visibility=ResourceVisibility.AGENT, content="hidden"),),
    )

    with pytest.raises(ValueError, match="cannot expose agent-visible files"):
        lower_to_harbor(
            specification,
            SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN),
            HarborEnvironmentConfig(docker_runtime=runtime),
            tmp_path / "task",
        )
    assert not (tmp_path / "task").exists()


@pytest.mark.parametrize("tamper", ["image", "undeclared_file", "symlink"])
async def test_docker_launch_rechecks_exported_runtime_before_start(tmp_path, tamper):
    runtime = DockerRuntimeBinding(image=f"example/a@sha256:{'a' * 64}", workdir="/app")
    specification = _state_specification(runtime)
    environment_config = HarborEnvironmentConfig(tool_binding=docker_shell_binding(runtime), docker_runtime=runtime)
    task = lower_to_harbor(
        specification,
        SubmissionConvention(id="state", answer_format=AnswerFormat.STATE),
        environment_config,
        tmp_path / "task",
    )
    if tamper == "image":
        task_file = task / "task.toml"
        task_file.write_text(task_file.read_text().replace("allow_internet = false", "allow_internet = true"))
    elif tamper == "undeclared_file":
        (task / "environment/Dockerfile").write_text("FROM scratch\n")
    else:
        (task / "environment/escape").symlink_to(task / "private_resources")

    with pytest.raises(ValueError, match=r"configuration differs|undeclared files|symlink"):
        await run_trial(
            task,
            environment_config,
            ChatLaunch(model="fixture", api_base="http://127.0.0.1:1/v1", strategy=AgentStrategy.CHAT_TOOLS),
            tmp_path / "trials",
            "blocked",
        )
    assert not (tmp_path / "trials").exists()


@pytest.fixture
def scripted_endpoint():
    requests: list[dict] = []
    turns: dict[str, int] = {}
    lock = Lock()

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                requests.append(request)
                model = request["model"]
                turns[model] = turns.get(model, 0) + 1
                turn = turns[model]
            command = {
                "good": (
                    'test "$(cat /app/inputs/seed.txt)" = seed && '
                    "test ! -e /app/private.txt && printf 'good\\n' > /app/answer.txt"
                ),
                "wrong": "printf 'wrong\\n' > /app/answer.txt",
                "noop": "test ! -e /app/answer.txt",
                "recover": "printf 'good\\n' > /app/answer.txt",
                "private_good": (
                    'test "$(cat /app/inputs/seed.txt)" = seed && '
                    "test ! -e /app/private.txt && test ! -e /app/grade.sh && "
                    "printf 'good\\n' > /app/answer.txt"
                ),
                "private_wrong": "printf 'wrong\\n' > /app/answer.txt",
                "private_noop": "test ! -e /app/answer.txt",
                "private_infra": "printf 'good\\n' > /app/answer.txt",
            }.get(model)
            if command is not None and turn <= (2 if model == "recover" else 1):
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"{model}-{turn}",
                            "type": "function",
                            "function": {
                                "name": "shell",
                                "arguments": (
                                    "{invalid" if model == "recover" and turn == 1 else json.dumps({"command": command})
                                ),
                            },
                        }
                    ],
                }
            else:
                message = {"role": "assistant", "content": "12" if model == "direct" else "Done."}
            body = json.dumps({"choices": [{"message": message}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.docker
@pytest.mark.timeout(180)
async def test_docker_direct_chat_uses_runtime_without_tools(tmp_path, scripted_endpoint):
    runtime = _docker_image()
    specification = _direct_specification(runtime)
    environment_config = HarborEnvironmentConfig(docker_runtime=runtime)
    task = lower_to_harbor(
        specification,
        SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN),
        environment_config,
        tmp_path / "task",
    )
    api_base, requests = scripted_endpoint

    result = await run_trial(
        task,
        environment_config,
        ChatLaunch(model="direct", api_base=api_base, strategy=AgentStrategy.DIRECT_CHAT),
        tmp_path / "trials",
        "direct",
    )

    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": 1.0}
    assert requests[0]["model"] == "direct"
    assert "tools" not in requests[0]


@pytest.mark.docker
@pytest.mark.timeout(240)
async def test_docker_shell_grades_live_state_and_isolates_concurrent_trials(tmp_path, scripted_endpoint):
    runtime = _docker_image()
    resources = (
        TaskResource(path="seed.txt", visibility=ResourceVisibility.AGENT, content="seed\n"),
        TaskResource(path="private.txt", visibility=ResourceVisibility.VERIFIER, content="gold\n"),
    )
    specification = _state_specification(runtime, resources=resources)
    environment_config = HarborEnvironmentConfig(tool_binding=docker_shell_binding(runtime), docker_runtime=runtime)
    task = lower_to_harbor(
        specification,
        SubmissionConvention(id="state", answer_format=AnswerFormat.STATE),
        environment_config,
        tmp_path / "task",
    )
    api_base, requests = scripted_endpoint

    async def trial(model: str):
        return await run_trial(
            task,
            environment_config,
            ChatLaunch(model=model, api_base=api_base, strategy=AgentStrategy.CHAT_TOOLS, max_turns=4),
            tmp_path / "trials",
            model,
        )

    good, wrong, noop, recover = await asyncio.gather(trial("good"), trial("wrong"), trial("noop"), trial("recover"))

    assert all(result.exception_info is None for result in (good, wrong, noop, recover))
    assert [result.verifier_result.rewards for result in (good, wrong, noop, recover)] == [
        {"reward": 1.0},
        {"reward": 0.0},
        {"reward": 0.0},
        {"reward": 1.0},
    ]
    assert (task / "private_resources/private.txt").read_text() == "gold\n"
    assert not (task / "environment/private.txt").exists()
    good_observation = next(
        request for request in requests if request["model"] == "good" and len(request["messages"]) > 1
    )
    assert json.loads(good_observation["messages"][-1]["content"])["return_code"] == 0
    recovery = recover.agent_result.metadata["tools"]
    assert [action["call_id"] for action in recovery] == ["recover-1", "recover-2"]
    assert "error" in json.loads(recovery[0]["observation"])
    assert json.loads(recovery[1]["observation"])["return_code"] == 0


def _private_command_specification(runtime: DockerRuntimeBinding, script: str) -> TaskSpec:
    return TaskSpec(
        id="private-docker-grade",
        instructions="Write the correct answer into answer.txt.",
        verifier=private_command(runtime.image, "grade.sh"),
        source=_source(),
        requirements=TaskRequirements(
            capabilities=("filesystem", "shell"),
            action_interfaces=("docker_shell:v1",),
            seed_sha256=runtime.image_sha256,
        ),
        answer_type=AnswerType.STATE,
        resources=(
            TaskResource(path="seed.txt", visibility=ResourceVisibility.AGENT, content="seed\n"),
            TaskResource(path="private.txt", visibility=ResourceVisibility.VERIFIER, content="gold\n"),
            TaskResource(path="grade.sh", visibility=ResourceVisibility.VERIFIER, executable=True, content=script),
        ),
    )


@pytest.mark.docker
@pytest.mark.timeout(240)
async def test_private_docker_command_grades_isolated_workspace_and_retains_infra_trace(tmp_path, scripted_endpoint):
    runtime = _docker_image()
    script = (
        "#!/bin/sh\n"
        "set -eu\n"
        "test ! -e /tmp/taskcompendium-private-verifier-marker || exit 7\n"
        "touch /tmp/taskcompendium-private-verifier-marker\n"
        'test "$(cat /private/private.txt)" = gold || exit 7\n'
        "test ! -e /workspace/grade.sh || exit 7\n"
        "test ! -e /workspace/private.txt || exit 7\n"
        'test "$(cat /workspace/inputs/seed.txt)" = seed || exit 7\n'
        "test -f /workspace/answer.txt || exit 1\n"
        'test "$(cat /workspace/answer.txt)" = good\n'
    )
    environment_config = HarborEnvironmentConfig(tool_binding=docker_shell_binding(runtime), docker_runtime=runtime)
    task = lower_to_harbor(
        _private_command_specification(runtime, script),
        SubmissionConvention(id="state", answer_format=AnswerFormat.STATE),
        environment_config,
        tmp_path / "task",
    )
    api_base, _ = scripted_endpoint

    async def trial(model: str, trial_task=task):
        return await run_trial(
            trial_task,
            environment_config,
            ChatLaunch(model=model, api_base=api_base, strategy=AgentStrategy.CHAT_TOOLS, max_turns=2),
            tmp_path / "trials",
            model,
        )

    good, wrong, noop = await asyncio.gather(trial("private_good"), trial("private_wrong"), trial("private_noop"))
    assert all(result.exception_info is None for result in (good, wrong, noop))
    assert [result.verifier_result.rewards for result in (good, wrong, noop)] == [
        {"reward": 1.0},
        {"reward": 0.0},
        {"reward": 0.0},
    ]

    broken = lower_to_harbor(
        _private_command_specification(runtime, "#!/bin/sh\nexit 7\n"),
        SubmissionConvention(id="state", answer_format=AnswerFormat.STATE),
        environment_config,
        tmp_path / "broken-task",
    )
    failed = await trial("private_infra", broken)
    outcome = json.loads((tmp_path / "trials/private_infra/verifier/taskcompendium-result.json").read_text())
    assert outcome["status"] == "infra_error"
    assert outcome["reward"] is None
    assert failed.verifier_result is None
    assert [action["call_id"] for action in failed.agent_result.metadata["tools"]] == ["private_infra-1"]
