import re
import shutil
import subprocess
from time import sleep

from . import parser


def convert_job_status(status):
    status_map = {
        "U": "0",  # Unexpected
        "I": "1",  # Idle
        "R": "2",  # Running
        "X": "3",  # Removed
        "C": "4",  # Completed
        "H": "5",  # Held
        "E": "6",  # Submission error
    }
    return status_map.get(status, -1)


def map_event_code_to_status(event_code):
    status_mapping = {
        "000": "I",  # Job submitted from host
        "001": "R",  # Execute
        "002": "E",  # Executable error
        "003": "R",  # Checkpointed
        "004": "C",  # Job evicted
        "005": "X",  # Job terminated
        "006": "R",  # Image size
        "007": "X",  # Shadow exception
        "009": "X",  # Job aborted
        "010": "H",  # Job suspended
        "011": "R",  # Job unsuspended
        "012": "H",  # Job held
        "013": "R",  # Job released
        "014": "R",  # Node execute
        "015": "X",  # Node terminated
        "016": "X",  # Post script terminated
        "021": "X",  # Remote error
        "022": "X",  # Job disconnected
        "023": "X",  # Job reconnected
        "024": "X",  # Job reconnect failed
        "025": "R",  # Grid resource up
        "026": "X",  # Grid resource down
        "027": "R",  # Grid submit
        "028": "I",  # Job ClassAd attribute values added to event log
        "029": "R",  # Job status unknown
        "030": "R",  # Job status known
        "031": "R",  # Grid job stage in
        "032": "R",  # Grid job stage out
        "033": "I",  # Job classad attribute update
        "034": "U",  # DAGMan PRE_SKIP defined
        "035": "I",  # Cluster submitted
        "036": "X",  # Cluster removed
        "037": "R",  # Factory paused
        "038": "R",  # Factory resumed
        "039": "U",  # No event could be returned
        "040": "R",  # File transfer
    }
    return status_mapping.get(event_code, "U")


def map_event_code_to_status_code(event_code):
    status_mapping = {
        "000": "1",  # Job submitted from host
        "001": "2",  # Execute
        "002": "6",  # Executable error
        "003": "2",  # Checkpointed
        "004": "3",  # Job evicted
        "005": "3",  # Job terminated
        "006": "2",  # Image size
        "007": "3",  # Shadow exception
        "009": "3",  # Job aborted
        "010": "5",  # Job suspended
        "011": "2",  # Job unsuspended
        "012": "5",  # Job held
        "013": "2",  # Job released
        "014": "2",  # Node execute
        "015": "3",  # Node terminated
        "016": "3",  # Post script terminated
        "021": "3",  # Remote error
        "022": "3",  # Job disconnected
        "023": "3",  # Job reconnected
        "024": "3",  # Job reconnect failed
        "025": "2",  # Grid resource up
        "026": "3",  # Grid resource down
        "027": "2",  # Grid submit
        "028": "1",  # Job ClassAd attribute values added to event log
        "029": "2",  # Job status unknown
        "030": "2",  # Job status known
        "031": "2",  # Grid job stage in
        "032": "2",  # Grid job stage out
        "033": "1",  # Job classad attribute update
        "034": "0",  # DAGMan PRE_SKIP defined
        "035": "1",  # Cluster submitted
        "036": "3",  # Cluster removed
        "037": "2",  # Factory paused
        "038": "2",  # Factory resumed
        "039": "0",  # No event could be returned
        "040": "2",  # File transfer
    }
    return status_mapping.get(event_code, "0")


def execute(command, args):
    """
    Executes a command with arguments.

    Args:
        command (str): The command to be executed.
        args (list): List of arguments to be passed to the command.

    Returns:
        int: Return code of the executed command.
    """
    full_command = [command] + args
    process = subprocess.Popen(full_command)
    process.wait()
    return process.returncode


def condor_status():
    """
    Retrieves status information from HTCondor.

    Returns:
        str: Output of the condor_status command.
    """
    try:
        status_output = subprocess.check_output(
            ["condor_status"], universal_newlines=True
        )
        return status_output
    except subprocess.CalledProcessError as e:
        print("Error retrieving HTCondor status:", e)
        return None


def get_job_status(cluster_id=None, log_file=None):
    job_statuses = {}
    if cluster_id:
        status_output = subprocess.check_output(
            ["condor_q", "-nobatch", cluster_id], universal_newlines=True
        )
        cluster_id_pattern = r"(\d{4})\.(\d+)\s+"  # Matches the cluster ID and job ID
        status_pattern = r"(H|R|I|C|X|S)\s+"  # Matches the job status
        cluster_id_regex = re.compile(cluster_id_pattern)
        status_regex = re.compile(status_pattern)
        matches = re.findall(
            rf"{cluster_id_pattern}.*?{status_pattern}", status_output, re.MULTILINE
        )
        for match in matches:
            cluster_id = match[0]
            job_id = match[1].zfill(3)
            status = convert_job_status(match[2])

            if cluster_id not in job_statuses:
                job_statuses[cluster_id] = {}

            job_statuses[cluster_id][job_id] = status
    elif log_file:
        parsed_log = parser.parse_log_file(log_file)
        job_statuses = {
            cluster_id: {
                job_id: map_event_code_to_status_code(job_info["event_code"])
                for job_id, job_info in job_dict.items()
            }
            for cluster_id, job_dict in parsed_log.items()
        }
    else:
        raise ValueError("Must provide either a cluster ID or a log file.")

    return job_statuses


def condor_submit(submit_file):
    """
    Submits a job to HTCondor using a submit file.

    Args:
        submit_file (str): Path to the HTCondor submit file.

    Returns:
        int: Job ID assigned by HTCondor.
    """
    try:
        submit_output = subprocess.check_output(
            ["condor_submit", submit_file], universal_newlines=True
        )

        if "ERROR" in submit_output:
            print("Error submitting Condor job:", submit_output)
            return None

        # Extract the job ID from the submit output
        job_id = submit_output.strip().split()[-1].strip(".")
        return job_id
    except subprocess.CalledProcessError as e:
        print("Error submitting Condor job:", e)
        return None


def condor_wait(log_file, max_wait=3600):
    """
    Waits for a job to finish.

    Args:
        log_file (str): Path to the HTCondor log file.
        max_wait (int): Maximum number of seconds to wait for the job to finish.

    Returns:
        bool: True if the job finished successfully, False otherwise.
    """
    wait_output = subprocess.check_output(
        ["condor_wait", log_file, "-wait", str(max_wait)]
    )
    return (
        wait_output.strip() == b"Job terminated."
    )  # Check if the job terminated successfully


def all_jobs_match_status(
    cluster_ids,
    target_status,
    job_statuses=[],
    log_dirs=[],
):
    if len(job_statuses) == 0 and len(log_dirs) == 0:
        job_statuses = [get_job_status(cluster_id=cluster_id) for cluster_id in cluster_ids]
    if len(job_statuses) < len(log_dirs):
        print(get_job_status(log_file=log_dirs[0]))
        job_statuses = [get_job_status(log_file=log_dir) for log_dir in log_dirs]
        if not all(len(job_status) > 0 for job_status in job_statuses):
            raise ValueError("Could not retrieve job statuses from all log files.")
    jobs_matched = [False] * len(cluster_ids)
    for i in range(len(cluster_ids)):
        jobs_matched[i] = all(
            value == str(target_status)
            for job_status in job_statuses[i].values()
            for value in job_status.values()
        )
    return all(jobs_matched)


def wait_all_jobs_match_status(
    cluster_ids,
    target_status,
    job_statuses=[],
    log_dirs=[],
):
    while not all_jobs_match_status(cluster_ids, target_status, job_statuses, log_dirs):
        sleep(5)


def condor_hold(job_id):
    """
    Holds a specific HTCondor job.

    Args:
        job_id (int): Job ID assigned by HTCondor.

    Returns:
        bool: True if the job was successfully put on hold, False otherwise.
    """
    hold_output = subprocess.check_output(["condor_hold", str(job_id)])
    return hold_output.strip() == b"Job is on hold."  # Check if the job is on hold


def condor_remove(job_id):
    """
    Removes a specific HTCondor job from the queue.

    Args:
        job_id (int): Job ID assigned by HTCondor.

    Returns:
        bool: True if the job was successfully removed, False otherwise.
    """
    remove_output = subprocess.check_output(["condor_rm", str(job_id)])
    return (
        remove_output.strip() == b"Job removed."
    )  # Check if the job was removed successfully


def rm_rf(path):
    """
    Recursively deletes a directory.

    Args:
        path (str): Path to the directory to be deleted.
    """
    shutil.rmtree(path, ignore_errors=True)
