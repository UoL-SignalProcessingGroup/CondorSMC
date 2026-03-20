"""
Symbolic names for the integer codes used in the database queue layer.

Using IntEnum means the values remain drop-in replacements for bare integers
everywhere the DB API expects an int, while giving readable names in logs,
comparisons, and IDE autocomplete.
"""
from __future__ import annotations
from enum import IntEnum


class JobType(IntEnum):
    """Job types exchanged between coordinator, managers, and followers."""
    IMPORTANCE_SAMPLING        = 0  # Coordinator/manager to follower: sample
    IMPORTANCE_SAMPLING_RESULT = 1  # Follower to coordinator/manager: result
    MANAGER_SAMPLING           = 2  # Coordinator to manager: run sub-round
    MANAGER_SAMPLING_RESULT    = 3  # Manager to coordinator: aggregated result


class DaemonRole(IntEnum):
    """Roles assigned when a node registers as a daemon."""
    COORDINATOR = 0
    MANAGER     = 1
    FOLLOWER    = 2


class DaemonStatus(IntEnum):
    """Status values set on daemon pool entries."""
    IDLE       = 0  # Waiting for a job
    ACTIVE     = 1  # Polling loop running
    COMPLETE   = 2  # Session finished cleanly
    TERMINATED = 3  # Session ended / evicted
    BUSY       = 4  # Executing a job
    PROCESSING = 5  # Coordinator aggregating results


class JobStatus(IntEnum):
    """Status values set on job_queue entries."""
    RUNNING    = 1  # Claimed and in progress
    COMPLETE   = 2  # Finished successfully
    FAILED     = 3  # Aborted or errored
    SUPERSEDED = 4  # Replaced by a new round; pending cleanup


# Mapping: result job type to the originating job type it is a response to.
RESULT_TO_ORIGIN: dict[JobType, JobType] = {
    JobType.IMPORTANCE_SAMPLING_RESULT: JobType.IMPORTANCE_SAMPLING,
    JobType.MANAGER_SAMPLING_RESULT:    JobType.MANAGER_SAMPLING,
}

# Set of job types that carry results back to the coordinator/manager.
RESULT_JOB_TYPES: frozenset[JobType] = frozenset(RESULT_TO_ORIGIN)
