import json
import logging
import uuid
from pathlib import Path

from .. import definitions
from . import utils


class Job:
    def __init__(
        self,
        executable,
        universe="vanilla",
        arguments=[],
        input_files=[],
        output_files=[],
        requests=[],
        requirements=None,
        notification="never",
        custom_args=[],
        n_jobs=1,
        payload={},
        session_id=None,
        job_id=None,
    ):
        self.executable = executable
        self.universe = universe
        self.arguments = arguments
        self.input_files = [str(f) for f in input_files]
        self.output_files = output_files
        self.requests = requests
        self.requirements = requirements
        self.notification = notification
        self.custom_args = custom_args
        self.n_jobs = n_jobs
        self.payload = payload
        self.session_id = session_id if session_id else str(uuid.uuid4())

        self.job_id = job_id if job_id else str(uuid.uuid4())

        self._cluster_id = None
        self._description = None
        if self.session_id:
            self._cache_dir = Path(
                definitions.SESSION_CACHE_DIR(self.session_id), "scheduler", self.job_id
            )
            self._log_dir = Path(
                definitions.SESSION_LOG_DIR(self.session_id), "scheduler", self.job_id
            )
        else:
            self._cache_dir = Path(
                definitions.PACKAGE_CACHE_DIR, "scheduler", self.job_id
            )
            self._log_dir = Path(definitions.PACKAGE_LOG_DIR, "scheduler", self.job_id)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._desc_file = Path(self._cache_dir, "job.sub")
        self._data_file = Path(self._cache_dir, "job.json")
        self._stdout_file = Path(self._log_dir, "stdout.log")
        self._stderr_file = Path(self._log_dir, "stderr.log")
        self._log_file = Path(self._log_dir, "log.log")

    @property
    def cluster_id(self):
        if self._cluster_id:
            return self._cluster_id
        print("Job not submitted")

    @property
    def status(self):
        if self._cluster_id:
            return utils.get_job_status(log_file=self._log_file)
        print("Job not submitted")

    @property
    def description(self):
        if self._description:
            return self._description
        print("Job not built")

    def add_input_files(self, input_files):
        input_files = [str(f) for f in input_files]
        self.input_files.extend(input_files)

    def build(self):
        logging.info(f"Building job {self.job_id}")

        job_description = ""
        job_description += f"universe = {self.universe}\n"
        job_description += f"executable = {self.executable}\n"

        if len(self.arguments) > 0:
            job_description += f"arguments = {' '.join(self.arguments)}\n"

        if len(self.input_files) > 0 or self.payload:
            job_description += f"transfer_input_files = "
            if len(self.input_files) > 0:
                job_description += f"{','.join(self.input_files)}"
            if self.payload:
                job_description += f",{self._data_file}"
            job_description += "\n"

        if len(self.output_files) > 0:
            job_description += (
                f"transfer_output_files = {','.join(self.output_files)}\n"
            )

        if len(self.output_files) > 0 or self.payload:
            job_description += "should_transfer_files = YES\n"
            job_description += "when_to_transfer_output = ON_EXIT_OR_EVICT\n"

        if len(self.requests) > 0:
            job_description += f"{self.requests[0][0]} = {self.requests[0][1]}\n"

        if self.requirements:
            job_description += f"requirements = {self.requirements}\n"

        job_description += f"output = {self._stdout_file}\n"
        job_description += f"error = {Path(self._stderr_file)}\n"
        job_description += f"log = {Path(self._log_file)}\n"
        job_description += f"notification = {self.notification}\n"

        if len(self.custom_args) > 0:
            for arg in self.custom_args:
                job_description += f"{arg}\n"

        job_description += f"queue {self.n_jobs}\n"

        with open(self._desc_file, "w") as f:
            f.write(job_description)

        self._description = job_description

        if self.payload:
            logging.info(f"Dumping job data to {self._data_file}")
            with open(self._data_file, "w") as f:
                json.dump(self.payload, f)

    def submit(self):
        if not self._desc_file.exists():
            raise FileNotFoundError(
                "Could not find job description, have you ran build() yet?"
            )
        logging.info(f"Submitting job {self.job_id}")
        self._cluster_id = utils.condor_submit(self._desc_file)

    def wait(self, target_status=3):
        if self._cluster_id:
            utils.wait_all_jobs_match_status(
                self._log_file, self._cluster_id, target_status=target_status
            )
        raise RuntimeError("Job not submitted")

    def finished(self, target_status=3):
        if self._cluster_id:
            return utils.all_jobs_match_status(
                [self._cluster_id],
                target_status=target_status,
                job_statuses=[self.status],
            )
        raise RuntimeError("Job not submitted")

    def remove(self):
        if self._cluster_id:
            utils.condor_rm(self._cluster_id)
        raise RuntimeError("Job not submitted")

    def clean(self):
        if self._cluster_id:
            if self._cache_dir.exists():
                utils.rm_rf(self._cache_dir)
            if self._log_dir.exists():
                utils.rm_rf(self._log_dir)
            if definitions.SESSION_CACHE_DIR(self.session_id).exists():
                utils.rm_rf(definitions.SESSION_CACHE_DIR(self.session_id))
            if definitions.SESSION_LOG_DIR(self.session_id).exists():
                utils.rm_rf(definitions.SESSION_LOG_DIR(self.session_id))
            return True
        raise RuntimeError("Job not submitted")
