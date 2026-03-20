import json
import logging
import uuid
from time import time

from .job import Job


class Daemon:
    def __init__(
        self,
        db,
        session_id: str,
        role: int,
        node_id: str = None,
        created_at: float = None,
    ):
        self.db = db
        self.session_id = session_id
        self.role = role

        self.node_id = str(uuid.uuid4()) if node_id is None else node_id
        self.created_at = time() if created_at is None else created_at
        self.last_seen = created_at
        self.status_code = 0
        self.job_queue = []  # type: ignore

    def join(self, payload=json.dumps({})):
        logging.info(f"{self.node_id} joining pool with session id: {self.session_id}")
        self.last_seen = time()
        node = self.db.select_one(
            "pool", "node_id", f"`session_id`='{self.session_id}' AND `node_id`='{self.node_id}'"
        )
        if node is None:
            self.db.insert(
                "pool",
                "(`session_id`, `node_id`, `role`, `created_at`, `last_seen`, `status_code`, `payload`)",
                (
                    self.session_id,
                    self.node_id,
                    self.role,
                    self.created_at,
                    self.last_seen,
                    1,
                    payload,
                ),
            )
        else:
            self.db.update(
                table="pool",
                set_values=f"`status_code` = 1, `last_seen` = '{self.last_seen}'",
                where_clause=f"session_id = '{self.session_id}' AND `node_id` = '{self.node_id}'",
            )
        logging.info(f"{self.node_id} joined pool with session id: {self.session_id}")
        self.status_code = 1

    def status(self):
        logging.info(
            f"getting status of {self.node_id} with session id: {self.session_id}"
        )
        status = self.db.select_one(
            "pool", "status_code", f"`session_id`='{self.session_id}'"
        )
        logging.info(f"got status of {self.node_id} with session id: {self.session_id}")
        self.status_code = status[0]
        return status[0]

    def set_status(self, status=1):
        logging.info(f"{self.node_id} updating pool with session id: {self.session_id}")
        self.last_seen = time()
        self.db.update(
            table="pool",
            set_values=f"`status_code` = '{status}', `last_seen` = '{self.last_seen}'",
            where_clause=f"session_id = '{self.session_id}' AND `node_id` = '{self.node_id}'",
        )
        logging.info(f"{self.node_id} updated pool with session id: {self.session_id}")
        self.status_code = status

    def fetch_job(self, round_id=None, job_type=None, job_id=None):
        logging.info(
            f"{self.node_id} fetching latest job with session id: {self.session_id}"
        )

        where_clause = f"`session_id`='{self.session_id}' AND `to_id`='{self.node_id}' AND `status_code`=0"

        if round_id is not None:
            where_clause += f" AND `round_id`='{round_id}'"

        if job_id is not None:
            where_clause += f" AND `job_id`='{job_id}'"

        if job_type is not None:
            where_clause += f" AND `type`='{job_type}'"

        job = self.db.select_one(
            "job_queue",
            "`id`, `session_id`, `job_id`, `round_id`, `to_id`, `from_id`, `type`, `created_at`, `deadline`",
            where_clause,
            orderby="created_at DESC",
        )

        logging.info(
            f"{self.node_id} fetched latest job with session id: {self.session_id}"
        )

        if job is not None:
            return Job(
                db=self.db,
                session_id=job[1],
                to_id=job[4],
                from_id=job[5],
                type=job[6],
                job_id=job[2],
                round_id=job[3],
                created_at=job[7],
                deadline=job[8],
            )

        return None

    def fetch_all_jobs(self, round_id=None, job_type=None):
        logging.info(
            f"{self.node_id} fetching all jobs with session id: {self.session_id}"
        )
        where_clause = f"`session_id`='{self.session_id}' AND `to_id`='{self.node_id}' AND `status_code`=0"

        if round_id is not None:
            where_clause += f" AND `round_id`='{round_id}'"

        if job_type is not None:
            where_clause += f" AND `type`='{job_type}'"

        jobs = self.db.select(
            "job_queue",
            "`id`, `session_id`, `job_id`, `round_id`, `to_id`, `from_id`, `type`, `created_at`, `deadline`",
            where_clause,
            orderby="created_at DESC",
        )
        logging.info(
            f"{self.node_id} fetched all jobs with session id: {self.session_id}"
        )
        if jobs is not None:
            return [
                Job(
                    db=self.db,
                    session_id=job[1],
                    to_id=job[4],
                    from_id=job[5],
                    type=job[6],
                    job_id=job[2],
                    round_id=job[3],
                    created_at=job[7],
                    deadline=job[8],
                )
                for job in jobs
            ]
        return []

    def fetch_global_jobs(self):
        """
        Fetches all jobs from the job queue that are not assigned to any node.
        Should use locking to prevent other nodes from fetching the same jobs.
        """
        raise NotImplementedError

    def leave(self):
        logging.info(f"{self.node_id} leaving pool with session id: {self.session_id}")
        self.last_seen = time()
        self.db.update(
            table="pool",
            set_values=f"`status_code` = 0, `last_seen` = '{self.last_seen}'",
            where_clause=f"session_id = '{self.session_id}' AND `node_id` = '{self.node_id}'",
        )
        logging.info(f"{self.node_id} left pool with session id: {self.session_id}")
