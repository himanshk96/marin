# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded, link-free copies of a candidate Docker workspace."""

import asyncio
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from taskcompendium.resources import validate_resource_path

MAX_SNAPSHOT_BYTES = 256 * 1024 * 1024
MAX_SNAPSHOT_ENTRIES = 100_000
SNAPSHOT_TIMEOUT = 120


def extract_snapshot(archive_path: Path, destination: Path, excluded_paths: tuple[str, ...] = ()) -> None:
    """Extract only bounded regular files and directories beneath a fresh root."""
    for excluded in excluded_paths:
        validate_resource_path(excluded)
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise ValueError("Workspace snapshot destination must be a real directory")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Workspace snapshot destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    total = 0
    seen: set[str] = set()
    with tarfile.open(archive_path, mode="r:") as archive:
        for index, member in enumerate(archive):
            name = member.name.removeprefix("./")
            if member.isdir() and name in {"", "."}:
                continue
            path = validate_resource_path(name)
            if (
                index >= MAX_SNAPSHOT_ENTRIES
                or not (member.isfile() or member.isdir())
                or any(path == PurePosixPath(excluded) or path.is_relative_to(excluded) for excluded in excluded_paths)
                or name in seen
            ):
                raise ValueError(f"Unsafe workspace archive member: {member.name}")
            seen.add(name)
            total += member.size
            if total > MAX_SNAPSHOT_BYTES:
                raise ValueError("Workspace snapshot exceeds its byte budget")
            target = destination.joinpath(*path.parts)
            if target.is_symlink() or not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Workspace archive escapes its destination")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            assert source is not None
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(0o755 if member.mode & 0o111 else 0o644)


async def download_snapshot(
    container_id: str, workdir: str, destination: Path, excluded_paths: tuple[str, ...] = ()
) -> None:
    """Stream a Docker archive with explicit limits, without host execution."""
    for excluded in excluded_paths:
        validate_resource_path(excluded)
    command = ["docker", "exec", "--user", "root", "--workdir", "/", container_id, "/bin/tar", "-C", workdir]
    command.extend(["--no-wildcards", *[f"--exclude=./{path}" for path in excluded_paths], "-cf", "-", "."])
    with tempfile.TemporaryDirectory(prefix="taskcompendium-snapshot-") as temporary:
        archive_path = Path(temporary) / "workspace.tar"
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            stdin=asyncio.subprocess.DEVNULL,
        )
        assert process.stdout is not None
        try:
            async with asyncio.timeout(SNAPSHOT_TIMEOUT):
                total = 0
                with archive_path.open("wb") as archive:
                    while data := await process.stdout.read(65536):
                        total += len(data)
                        if total > MAX_SNAPSHOT_BYTES:
                            raise ValueError("Workspace archive exceeds its byte budget")
                        archive.write(data)
                if await process.wait() != 0:
                    raise RuntimeError("Docker workspace export failed")
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        extract_snapshot(archive_path, destination, excluded_paths)
