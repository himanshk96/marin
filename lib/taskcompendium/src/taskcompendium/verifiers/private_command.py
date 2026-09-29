# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned private command for grading a submitted Docker workspace."""

import re

from pydantic import Field, model_validator

from taskcompendium.grading import GradeResult, GradingAttempt, Verifier
from taskcompendium.models import VerifierKind, VerifierSpec
from taskcompendium.resources import validate_resource_path

PINNED_IMAGE = re.compile(r"(?:[^\s@]+@sha256:|sha256:)[0-9a-f]{64}\Z")


class PrivateCommandVerifier(Verifier):
    """Run a private executable in a separate pinned verifier image.

    Exit code zero accepts the workspace, one rejects it, and any other exit
    leaves the trial ungraded. The image must already be available locally.
    """

    image: str
    script_path: str
    args: tuple[str, ...] = ()
    snapshot_exclusions: tuple[str, ...] = ()
    timeout: float = Field(default=120.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_contract(self) -> "PrivateCommandVerifier":
        if not PINNED_IMAGE.fullmatch(self.image):
            raise ValueError("Verifier image requires an immutable SHA256 digest")
        validate_resource_path(self.script_path)
        for path in self.snapshot_exclusions:
            validate_resource_path(path)
        if len(set(self.snapshot_exclusions)) != len(self.snapshot_exclusions):
            raise ValueError("Snapshot exclusions must be unique")
        if any("\x00" in arg for arg in self.args):
            raise ValueError("Verifier arguments cannot contain NUL")
        return self

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        raise RuntimeError("Private command grading requires an isolated Docker verifier")


def private_command(
    image: str,
    script_path: str,
    *,
    args: tuple[str, ...] = (),
    snapshot_exclusions: tuple[str, ...] = (),
    timeout: float = 120.0,
) -> VerifierSpec:
    """Construct a private executable verifier descriptor."""
    verifier = PrivateCommandVerifier(
        image=image, script_path=script_path, args=args, snapshot_exclusions=snapshot_exclusions, timeout=timeout
    )
    return VerifierSpec(kind=VerifierKind.PRIVATE_COMMAND, parameters_json=verifier.model_dump_json())
