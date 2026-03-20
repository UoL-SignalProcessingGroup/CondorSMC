import re
from pathlib import Path


def parse_log_file(log):
    if isinstance(log, Path):
        log = log.read_text()

    event_code_pattern = (
        r"(\d{3})"  # Matches three-digit status code at the beginning of a line
    )
    job_id_pattern = r"\((\d+)\.(\d+)\.\d+\)"  # Matches job ID and cluster ID in the format "(000.000.000)"
    datetime_pattern = r"(\d{2}/\d{2} \d{2}:\d{2}:\d{2})"  # Matches date and time in the format "MM/DD HH:MM:SS"
    status_pattern = r"(Job submitted|Job executing|Job was evicted|Job was held|Job was aborted|Job terminated)"  # Matches job status

    event_code_regex = re.compile(event_code_pattern)
    job_id_regex = re.compile(job_id_pattern)
    datetime_regex = re.compile(datetime_pattern)
    status_regex = re.compile(status_pattern)

    matches = re.findall(
        rf"{event_code_pattern}.*?{job_id_pattern}.*?{datetime_pattern}.*?{status_pattern}",
        log,
        re.MULTILINE,
    )

    log_dict = {}
    for match in matches:
        event_code = match[0]
        cluster_id = match[1]
        job_id = match[2]
        datetime = match[3]
        status = match[4]

        if cluster_id not in log_dict:
            log_dict[cluster_id] = {}

        log_dict[cluster_id][job_id] = {
            "status": status,
            "event_code": event_code,
            "last_updated": datetime,
        }

    return log_dict
