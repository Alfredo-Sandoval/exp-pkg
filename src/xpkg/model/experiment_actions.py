"""Governed state transitions for experiment ontology objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from xpkg.model.experiment import (
    Experiment,
    ExperimentalCondition,
    ExperimentSessionLink,
    Protocol,
    Subject,
)
from xpkg.model.metadata import DatasetShareMetadata
from xpkg.model.session import RecordingSession


class InvalidExperimentTransitionError(ValueError):
    """Raised when an experiment action violates its preconditions."""


def add_experiment_session(
    experiment: Experiment,
    session: RecordingSession,
    *,
    link_metadata: Mapping[str, Any] | None = None,
) -> Experiment:
    """Add one uniquely identified recording session to an experiment."""
    if session.session_id in experiment.session_ids:
        raise InvalidExperimentTransitionError(
            f"Experiment {experiment.experiment_id!r} already has session "
            f"{session.session_id!r}."
        )
    link = ExperimentSessionLink(session=session, metadata=dict(link_metadata or {}))
    return replace(experiment, session_links=(*experiment.session_links, link))


def replace_experiment_session(
    experiment: Experiment,
    session: RecordingSession,
) -> Experiment:
    """Replace one recording session while preserving its relationship metadata."""
    if session.session_id not in experiment.session_ids:
        raise InvalidExperimentTransitionError(
            f"Experiment {experiment.experiment_id!r} has no session "
            f"{session.session_id!r}."
        )
    links = tuple(
        replace(link, session=session) if link.session_id == session.session_id else link
        for link in experiment.session_links
    )
    return replace(experiment, session_links=links)


def replace_experiment_session_link(
    experiment: Experiment,
    link: ExperimentSessionLink,
) -> Experiment:
    """Replace one complete session relationship while preserving order."""
    if link.session_id not in experiment.session_ids:
        raise InvalidExperimentTransitionError(
            f"Experiment {experiment.experiment_id!r} has no session {link.session_id!r}."
        )
    links = tuple(
        link if current.session_id == link.session_id else current
        for current in experiment.session_links
    )
    return replace(experiment, session_links=links)


def add_experiment_subject(experiment: Experiment, subject: Subject) -> Experiment:
    """Register one new biological subject with an experiment."""
    if any(current.subject_id == subject.subject_id for current in experiment.subjects):
        raise InvalidExperimentTransitionError(
            f"Experiment {experiment.experiment_id!r} already has subject {subject.subject_id!r}."
        )
    return replace(experiment, subjects=(*experiment.subjects, subject))


def add_experiment_protocol(experiment: Experiment, protocol: Protocol) -> Experiment:
    """Register one new protocol with an experiment."""
    if any(current.protocol_id == protocol.protocol_id for current in experiment.protocols):
        raise InvalidExperimentTransitionError(
            f"Experiment {experiment.experiment_id!r} already has protocol "
            f"{protocol.protocol_id!r}."
        )
    return replace(experiment, protocols=(*experiment.protocols, protocol))


def add_experiment_condition(
    experiment: Experiment, condition: ExperimentalCondition
) -> Experiment:
    """Register one new experimental condition with an experiment."""
    if any(
        current.condition_id == condition.condition_id for current in experiment.conditions
    ):
        raise InvalidExperimentTransitionError(
            f"Experiment {experiment.experiment_id!r} already has condition "
            f"{condition.condition_id!r}."
        )
    return replace(experiment, conditions=(*experiment.conditions, condition))


def replace_experiment_dataset_share(
    experiment: Experiment,
    dataset_share: DatasetShareMetadata | None,
) -> Experiment:
    """Replace the experiment's dataset-sharing record."""
    if dataset_share is not None and not isinstance(dataset_share, DatasetShareMetadata):
        raise TypeError("dataset_share must be DatasetShareMetadata or None.")
    return replace(experiment, dataset_share=dataset_share)


def replace_experiment_metadata(
    experiment: Experiment,
    metadata: Mapping[str, Any] | None,
) -> Experiment:
    """Replace experiment metadata through the governed action layer."""
    return replace(experiment, metadata=dict(metadata or {}))


__all__ = [
    "InvalidExperimentTransitionError",
    "add_experiment_condition",
    "add_experiment_protocol",
    "add_experiment_session",
    "add_experiment_subject",
    "replace_experiment_dataset_share",
    "replace_experiment_metadata",
    "replace_experiment_session",
    "replace_experiment_session_link",
]
