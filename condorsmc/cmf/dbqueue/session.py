import json
import logging
import uuid
from time import time

from .job import Job


class Session:
    def __init__(
        self,
        db,
        session_id: str = None,
        created_at: float = None,
        deadline: float = None,
        runtime: float = 86400.0,
        payload=json.dumps({}),
    ):
        self.db = db
        self.payload = payload

        self.session_id = str(uuid.uuid4()) if session_id is None else session_id
        self.created_at = time() if created_at is None else created_at
        if deadline is None:
            self.deadline = self.created_at + runtime
        else:
            self.deadline = deadline
        self.active = False

    def create(self, active=False):
        logging.info(f"Creating session with id: {self.session_id}")
        self.db.insert(
            "session",
            "(`session_id`, `created_at`, `active`, `deadline`, `payload`)",
            (self.session_id, self.created_at, active, self.deadline, self.payload),
        )
        logging.info(f"Session created with id {self.session_id}")
        self.active = active

    def status(self):
        logging.info(f"Getting status of session with id: {self.session_id}")
        status = self.db.select_one(
            "session", "active", f"`session_id`='{self.session_id}'"
        )
        logging.info(f"Got status of session with id: {self.session_id}")
        self.active = status[0]
        return status[0]

    def set_status(self, active=False):
        logging.info(f"Setting session status to: {active}")
        self.db.update(
            "session", f"`active`='{active}'", f"`session_id`='{self.session_id}'"
        )
        logging.info(f"Session status set to: {active}")
        self.active = active

    def set_payload(self, payload):
        logging.info(f"Updating session payload")
        payload = json.dumps(payload)
        self.db.update(
            "session", f"`payload`='{payload}'", f"`session_id`='{self.session_id}'"
        )
        self.payload = payload
        logging.info(f"Updated session payload")

    def delete(self):
        logging.info(f"Deleting session with id: {self.session_id}")
        self.db.delete("session", f"`session_id`='{self.session_id}'")
        logging.info(f"Session deleted with id: {self.session_id}")

    def pool_status(self, id: str = None):
        """
        Return the status of each daemon associated with `session_id`.
        If `id` is specified, return the status of the daemon with that `id`.

        The result is a dictionary of the form:
        {
            "session_id": "session_id",
            "status": {
                "node_id": {
                    "role": "role",
                    "status_code": "status_code",
                    "payload": "payload"
                }
            }

        TO DO
        - Handle polling errors
        - Add maximum number of attempts
        """

        logging.info(f"Getting pool status with session id: {self.session_id}")
        if id is None:
            result = self.db.select(
                "pool",
                "`node_id`, `role`, `status_code`, `payload`",
                f"`session_id`='{self.session_id}'",
            )
        else:
            result = self.db.select(
                "pool",
                "`node_id`, `role`, `status_code`, `payload`",
                f"`session_id`='{self.session_id}' AND `node_id`='{id}'",
            )
        logging.info(f"Pool status retrieved with session id: {self.session_id}")

        if id is None:
            status = {}
            for row in result:
                status[row[0]] = {
                    "role": row[1],
                    "status_code": row[2],
                }
            return {"session_id": self.session_id, "status": status}
        else:
            for row in result:
                if row[0] == id:
                    return {
                        "session_id": self.session_id,
                        "status": {
                            row[0]: {
                                "role": row[1],
                                "status_code": row[2],
                            }
                        },
                    }
            return {
                "session_id": self.session_id,
                "status": {
                    id: {
                        "role": None,
                        "status_code": None,
                    }
                },
            }

    def n_active_daemons(self, role=2, ids=[]):
        """
        Return the number of active daemons of type `type` or `ids` associated with
         `session_id` where `status_code` is not 1 or 2.
        """
        logging.info(
            f"Getting number of active daemons with session id: {self.session_id}"
        )
        if ids:
            ids = tuple(ids)
            result = self.db.select(
                "pool",
                "`node_id`",
                f"`session_id`='{self.session_id}' AND `role`='{role}' AND `status_code` NOT IN (2,3) AND `node_id` IN {ids}",
            )
            logging.info(
                f"Got number of active daemons with session id: {self.session_id}"
            )

            if result is None:
                return 0

            return len(result)
        
        if role is None:
            result = self.db.select(
                "pool",
                "`node_id`",
                f"`session_id`='{self.session_id}' AND `status_code` NOT IN (2,3)",
            )
            logging.info(
                f"Got number of active daemons with session id: {self.session_id}"
            )

            if result is None:
                return 0

            return len(result)

        result = self.db.select(
            "pool",
            "`node_id`",
            f"`session_id`='{self.session_id}' AND `role`='{role}' AND `status_code` NOT IN (2,3)",
        )
        logging.info(f"Got number of active daemons with session id: {self.session_id}")

        if result is None:
            return 0

        return len(result)

    def job_queue(self, status_code=None):
        """
        Returns the jobs that are in the queue associated with `session_id`.
        If `status_code` is specified, only return jobs with that status code.
        """
        logging.info(f"Getting job queue with session id: {self.session_id}")
        if status_code is None:
            self.db.select("job_queue", "*", f"`session_id`='{self.session_id}'")
        else:
            self.db.select( 
                "job_queue",
                "*",
                f"`session_id`='{self.session_id}' AND `status_code`='{status_code}'",
            )

        logging.info(f"Got job queue with session id: {self.session_id}")
        return result

    def n_active_jobs(self, round_id=None, job_type=None, from_id=None, to_id=None):
        """
        Return the number of active jobs of type `type` associated with `session_id` where
        `status_code` is not 0 or 3.
        """
        logging.info(
            f"Getting number of active jobs with session id: {self.session_id}"
        )
        if job_type is not None:
            qry = f"`session_id`='{self.session_id}' AND `type`='{job_type}' AND `status_code` NOT IN (0,3,4)"
        else:
            qry = f"`session_id`='{self.session_id}' AND `status_code` NOT IN (0,3,4)"
        if round_id is not None:
            qry += f" AND `round_id`='{round_id}'"
        if from_id is not None:
            qry += f" AND `from_id`='{from_id}'"
        if to_id is not None:
            qry += f" AND `to_id`='{to_id}'"
        result = self.db.select(
            "job_queue",
            "`job_id`",
            qry
        )
        logging.info(f"Got number of active jobs with session id: {self.session_id}")
        if result is None:
            return 0
        return len(result)

    def fetch_latest_job(self, return_query=False):
        logging.info(f"Pulling the latest inactive job from the queue")
        job = self.db.select_one(
            "job_queue",
            "`id`, `session_id`, `job_id`, `to_id`, `from_id`, `type`,  `created_at`, `deadline`",
            f"`status_code` NOT IN (0, 3, 4) AND `active` = 0 AND `to_id` = '' ORDER BY `created_at` DESC LIMIT 1 FOR UPDATE",
        )
        if job:
            self.db.update("jobs", "`active`=1", f"`job_id`='{job['job_id']}'")
        if job is not None:
            if return_query:
                return jobs
            return Job(
                db=self.db,
                session_id=job[1],
                to_id=job[3],
                from_id=job[4],
                type=job[5],
                job_id=job[2],
                created_at=job[6],
                deadline=job[7],
            )
        return None

    def fetch_jobs(self, job_type=None, active=False, return_query=False):
        """
        Pull all jobs from the queue. If job_type is specified, we only pull jobs of that type.
        If active is True, we pull all jobs that do not have a status of (0 or 3)
        """
        logging.info(f"Fetching jobs with session id: {self.session_id}")
        if job_type is not None:
            if active:
                jobs = self.db.select(
                    "job_queue",
                    "`id`, `session_id`, `job_id`, `to_id`, `from_id`, `type`,  `created_at`, `deadline`, `status_code`",
                    f"`session_id`='{self.session_id}' AND `type`='{job_type}' AND `status_code`, `status_code` NOT IN (0,3,4)",
                )
            else:
                jobs = self.db.select(
                    "job_queue",
                    "`id`, `session_id`, `job_id`, `to_id`, `from_id`, `type`,  `created_at`, `deadline`, `status_code`",
                    f"`session_id`='{self.session_id}' AND `type`='{job_type}' AND `status_code` NOT IN (2)",
                )
        else:
            if active:
                jobs = self.db.select(
                    "job_queue",
                    "`id`, `session_id`, `job_id`, `to_id`, `from_id`, `type`,  `created_at`, `deadline`, `status_code`",
                    f"`session_id`='{self.session_id}' AND `status_code` NOT IN (0,3,4)",
                )
            else:
                jobs = self.db.select(
                    "job_queue",
                    "`id`, `session_id`, `job_id`, `to_id`, `from_id`, `type`,  `created_at`, `deadline`, `status_code`",
                    f"`session_id`='{self.session_id}' AND `status_code` NOT IN (2)",
                )
        logging.info(f"Fetched jobs with session id: {self.session_id}")
        if jobs is not None:
            if return_query:
                return jobs
            return [
                Job(
                    db=self.db,
                    session_id=job[1],
                    to_id=job[3],
                    from_id=job[4],
                    type=job[5],
                    job_id=job[2],
                    created_at=job[6],
                    deadline=job[7],
                )
                for job in jobs
            ]
        return []

    def clean_stale_jobs(self, job_type=None, round_id=None, from_id=None, deadline=None, check_deadline=True, clear_running=False):
        """
        Pull all jobs from the job queue associated with `session_id` that are
        older than the associated `deadline` that do not have a status code of (1,2,3)
        and set their status to 3.
        """

        logging.info(f"Tidying stale jobs with session id: {self.session_id}")

        if clear_running:
            check_statuses = "(2,3,4)"
        else:
            check_statuses = "(1,2,3,4)"

        # Connect to the database
        self.db.connect()
        conn = self.db.connection

        try:
            # Set isolation level to REPEATABLE READ
            conn.autocommit(False)
            conn.begin()

            cursor = conn.cursor()

            if job_type is None:
                qry = f"`session_id`='{self.session_id}' AND `status_code` NOT IN {check_statuses}"
                if check_deadline:
                    qry += f" AND `deadline` < {time()}"
                elif deadline:
                    qry += f" AND `deadline` < {deadline}"
                if from_id:
                    qry += f" AND `from_id`='{from_id}'"
                if round_id:
                    qry += f" AND `round_id`='{round_id}'"

                # Acquire row-level locks using SELECT ... FOR UPDATE
                select_query = f"SELECT * FROM job_queue WHERE {qry} FOR UPDATE"
                cursor.execute(select_query)

                # Update the rows
                update_query = f"UPDATE job_queue SET `status_code`=3 WHERE {qry}"
                cursor.execute(update_query)

            else:
                qry = f"`session_id`='{self.session_id}' AND `type`='{job_type}' AND `status_code` NOT IN {check_statuses}"
                if check_deadline:
                    qry += f" AND `deadline` < {time()}"
                elif deadline:
                    qry += f" AND `deadline` < {deadline}"
                if from_id:
                    qry += f" AND `from_id`='{from_id}'"
                if round_id:
                    qry += f" AND `round_id`='{round_id}'"

                # Acquire row-level locks using SELECT ... FOR UPDATE
                select_query = f"SELECT * FROM job_queue WHERE {qry} FOR UPDATE"
                cursor.execute(select_query)

                # Update the rows
                update_query = f"UPDATE job_queue SET `status_code`=3 WHERE {qry}"
                cursor.execute(update_query)

            # Commit the transaction
            conn.commit()

        except Exception as e:
            # Rollback the transaction if an error occurs
            conn.rollback()
            raise e

        finally:
            # Restore autocommit and close the cursor and connection
            conn.autocommit(True)
            cursor.close()
            conn.close()

        logging.info(f"Tidied stale jobs with session id: {self.session_id}")

    def count_stale_jobs(self, job_type=None, from_id=None):
        """
        Return the number of stale jobs associated with `session_id` that do not have a status code of (1,2,3).
        """
        logging.info(f"Counting stale jobs with session id: {self.session_id}")
        if job_type is None:
            where_clause = f"`session_id`='{self.session_id}' AND `deadline` < NOW() AND `status_code` NOT IN (1,2,3)"
        else:
            where_clause = f"`session_id`='{self.session_id}' AND `type`='{job_type}' AND `deadline` < NOW() AND `status_code`=3"
        if from_id:
            where_clause += f" AND `from_id`='{from_id}'"
        result = self.db.select(
            "job_queue",
            "`job_id`",
            where_clause,
        )
        logging.info(f"Counted stale jobs with session id: {self.session_id}")
        return len(result)

    def fetch_stale_jobs(self, round_id=None, job_type=None, from_id=None):
        """
        Fetch stale jobs from the job queue. If job_type is specified, only fetch jobs of that type.
        Return a list of Job objects.
        """
        logging.info(f"Fetching stale jobs with session id: {self.session_id}")
        if job_type is None:
            where_clause = f"`session_id`='{self.session_id}' AND `status_code`=3"
        else:
            where_clause = f"`session_id`='{self.session_id}' AND `type`='{job_type}' AND `status_code`=3"

        if from_id:
            where_clause += f" AND `from_id`='{from_id}'"
        
        if round_id:
            where_clause += f" AND `round_id`='{round_id}'"

        stale_jobs = self.db.select(
            "job_queue",
            "`id`, `session_id`, `job_id`, `to_id`, `from_id`, `type`, `created_at`, `deadline`",
            where_clause,
        )

        logging.info(f"Fetched stale jobs with session id: {self.session_id}")
        if stale_jobs is not None:
            jobs = [
                Job(
                    db=self.db,
                    session_id=job[1],
                    to_id=job[3],
                    from_id=job[4],
                    type=job[5],
                    job_id=job[2],
                    created_at=job[6],
                    deadline=job[7],
                )
                for job in stale_jobs
            ]

            return jobs, len(jobs)
        else:
            return [], 0

    def clean_complete_jobs(self, from_id=None, to_id=None):
        """
        Remove all jobs from the job queue associated with `session_id` that have a status code of 4.
        """
        logging.info(f"Cleaning complete jobs with session id: {self.session_id}")
        if from_id:
            where_clause = f"`session_id`='{self.session_id}' AND `status_code`=4 AND `from_id`='{from_id}'"
        elif to_id:
            where_clause = f"`session_id`='{self.session_id}' AND `status_code`=4 AND `to_id`='{to_id}'"
        else:
            where_clause = f"`session_id`='{self.session_id}' AND `status_code`=4"

        self.db.delete(
            "job_queue",
            where_clause
        )

        logging.info(f"Cleaned complete jobs with session id: {self.session_id}")

    def clean_stale_daemons(self, ids=None, role=None, timeout=60):
        """
        Check for workers that have not checked in for longer than the specified timeout.
        """
        logging.info(f"Checking for stale workers with session id: {self.session_id}")
        where_query = f"`session_id`='{self.session_id}' AND `status_code` NOT IN (2,3) AND `last_seen` < {time() - timeout}"
        if role:
            where_query += f" AND `role`='{role}'"
        if ids:
            ids_str = ", ".join([f"'{id}'" for id in ids])
            where_query += f" AND `node_id` IN ({ids_str})"
        result = self.db.select(
            "pool",
            "`id`, `session_id`, `node_id`, `status_code`, `last_seen`",
            # f"`session_id`='{self.session_id}' AND `status_code` NOT IN (0,1,3) AND `last_seen` < {time() - timeout}",
            where_query,
        )
        logging.info(f"Checked for stale workers with session id: {self.session_id}")

        if result is not None:
            logging.info(f"Found stale workers with session id: {self.session_id}")
            for worker in result:
                # Set worker status to 2 (Held)
                self.db.update(
                    "pool",
                    "`status_code`=2",
                    f"`session_id`='{self.session_id}' AND `node_id`='{worker[2]}'",
                    queue_query=True,
                )

                # Set any jobs allocated to them as stale
                self.db.update(
                    "job_queue",
                    "`status_code`=3",
                    f"`session_id`='{self.session_id}' AND `to_id`='{worker[2]}'",
                    queue_query=True,
                )
            self.db._execute_query_queue()

        logging.info(f"Checked for stale workers with session id: {self.session_id}")

    def clean_jobs(self, round_id):
        """
        Remove all jobs from the job queue associated with `session_id` that have a round id of `round_id`.
        """
        logging.info(f"Cleaning jobs with session id: {self.session_id}")
        self.db.delete(
            "job_queue",
            f"`session_id`='{self.session_id}' AND `round_id`='{round_id}'"
        )
        logging.info(f"Cleaned jobs with session id: {self.session_id}")

    def clear_session(self, clear_results=False):
        logging.info(f"Clearing session with session id: {self.session_id}", queue_query=True)
        self.db.delete("job_queue", f"`session_id`='{self.session_id}'", queue_query=True)
        self.db.delete("pool", f"`session_id`='{self.session_id}'", queue_query=True)
        self.db.delete("session", f"`session_id`='{self.session_id}'", queue_query=True)
        self.db.delete("checkpoint", f"`session_id`='{self.session_id}'", queue_query=True)
        if clear_results:
            self.db.delete("results", f"`session_id`='{self.session_id}'", queue_query=True)
        self.db._execute_query_queue()

        logging.info(f"Session cleared with session id: {self.session_id}")
