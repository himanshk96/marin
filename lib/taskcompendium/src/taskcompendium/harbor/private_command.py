# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a Docker workspace with a private command in a separate container."""

import asyncio
import json
import subprocess
import tempfile
import uuid
from pathlib import Path

from taskcompendium.grading import GradeResult, Outcome
from taskcompendium.harbor.docker import DockerShellEnvironment
from taskcompendium.lowering import PRIVATE_RESOURCES_DIR
from taskcompendium.models import TaskSpec, VerifierKind
from taskcompendium.resources import ResourceVisibility
from taskcompendium.verifier_registry import resolve_verifier
from taskcompendium.verifiers.private_command import PrivateCommandVerifier


def _run_verifier(verifier: PrivateCommandVerifier, workspace: Path, private_resources: Path) -> GradeResult:
    script = f"/private/{verifier.script_path}"
    name = f"taskcompendium-private-verifier-{uuid.uuid4().hex}"
    command = [
        "docker",
        "run",
        "--name",
        name,
        "--pull",
        "never",
        "--network",
        "none",
        "--read-only",
        "--pids-limit",
        "128",
        "--memory",
        "1g",
        "--cpus",
        "2",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        "/tmp:rw,exec,nosuid,size=268435456",
        "--volume",
        f"{workspace}:/workspace:rw",
        "--volume",
        f"{private_resources}:/private:ro",
        "--workdir",
        "/workspace",
        verifier.image,
        script,
        *verifier.args,
    ]
    result: GradeResult
    try:
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=verifier.timeout, check=False)
        except subprocess.TimeoutExpired:
            result = GradeResult(Outcome.INFRA_ERROR, None, "Private verifier timed out")
        except OSError as error:
            result = GradeResult(Outcome.INFRA_ERROR, None, f"Private verifier launch failed: {type(error).__name__}")
        else:
            inspection = subprocess.run(
                ["docker", "inspect", "--format", "{{json .State}}", name],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if inspection.returncode != 0:
                result = GradeResult(Outcome.INFRA_ERROR, None, "Private verifier container did not start")
            else:
                state = json.loads(inspection.stdout)
                if state.get("Status") != "exited" or state.get("Error") or state.get("OOMKilled"):
                    result = GradeResult(Outcome.INFRA_ERROR, None, "Private verifier container failed")
                elif state.get("ExitCode") != completed.returncode:
                    result = GradeResult(Outcome.INFRA_ERROR, None, "Private verifier exit status was inconsistent")
                elif completed.returncode == 0:
                    result = GradeResult(Outcome.GRADED, 1.0)
                elif completed.returncode == 1:
                    result = GradeResult(Outcome.GRADED, 0.0)
                else:
                    result = GradeResult(Outcome.INFRA_ERROR, None, f"Private verifier exited {completed.returncode}")
    finally:
        cleanup = subprocess.run(
            ["docker", "rm", "--force", "--volumes", name], capture_output=True, text=True, check=False
        )
        if cleanup.returncode != 0 and "No such container" not in cleanup.stderr:
            raise RuntimeError("Could not remove private verifier container")
    return result


async def grade_private_command(
    specification: TaskSpec, environment: DockerShellEnvironment, task_dir: Path
) -> GradeResult:
    """Grade submitted Docker state without exposing private checks to the agent."""
    if specification.verifier.kind != VerifierKind.PRIVATE_COMMAND:
        raise ValueError("Task does not select a private command verifier")
    verifier = resolve_verifier(specification.verifier)
    assert isinstance(verifier, PrivateCommandVerifier)
    scripts = tuple(
        resource
        for resource in specification.resources
        if resource.path == verifier.script_path
        and resource.visibility == ResourceVisibility.VERIFIER
        and resource.executable
    )
    if len(scripts) != 1:
        raise ValueError("Private command must name one executable verifier resource")
    private_resources = task_dir / PRIVATE_RESOURCES_DIR
    executable = private_resources / verifier.script_path
    current = private_resources
    if current.is_symlink():
        raise ValueError("Private verifier executable is missing or unsafe")
    for part in Path(verifier.script_path).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Private verifier executable is missing or unsafe")
    if not executable.is_file():
        raise ValueError("Private verifier executable is missing or unsafe")
    with tempfile.TemporaryDirectory(prefix="taskcompendium-verifier-") as temporary:
        workspace = Path(temporary) / "workspace"
        await environment.snapshot_workspace(workspace, verifier.snapshot_exclusions)
        # The snapshot is a disposable copy. Allow build tools to write in it
        # while keeping the agent's live container and private resources separate.
        for path in (workspace, *workspace.rglob("*")):
            if path.is_symlink():
                raise ValueError("Verifier workspace contains a symlink")
            mode = path.stat().st_mode
            path.chmod((0o777 if path.is_dir() else 0o666) | (mode & 0o111))
        return await asyncio.to_thread(_run_verifier, verifier, workspace.resolve(), private_resources.resolve())
