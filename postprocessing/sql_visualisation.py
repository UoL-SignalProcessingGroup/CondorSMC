import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns

from condorsmc import definitions

try:
    from condorcmf.dbqueue.connector.mysql import MySQLConnector as DBQConnector  # type: ignore
except:
    from condorcmf.dbqueue.connector.pymysql import PyMySQLConnector as DBQConnector  # type: ignore

sns.set_style("whitegrid")


def plot_active_nodes(architecture="cf", session_id="test", coordinator_id="test_coordinator"):
    # Create the database connection
    dbq_db = DBQConnector(
        host=definitions.MYSQL_HOST,
        user=definitions.MYSQL_USER,
        password=definitions.MYSQL_PASSWORD,
        database=definitions.MYSQL_DATABASE,
        poll_delay=definitions.MYSQL_POLL_DELAY,
    )

    if architecture == "cf":
        where_clause = f"`session_id`='{session_id}' AND `node_id`='{coordinator_id}'"

        results = dbq_db.select(
            "results",
            "`id`, `session_id`, `node_id`, `role`, `job_id`, `attributes`",
            where_clause,
            "id ASC",
        )

        sampling_iters = []
        active_followers = []
        for result in results:
            attributes = json.loads(result[5])
            sampling_iters.append(attributes["sampling_iter"])
            active_followers.append(attributes["active_followers"])

        # Save results to a json file
        with open("n_active_followers.json", "w") as f:
            json.dump(
                {
                    "sampling_iters": sampling_iters,
                    "active_followers": active_followers,
                },
                f,
            )

        # Plot results as a bar chart using matplotlib
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(sampling_iters, active_followers, color="blue")
        ax.set_xlabel("Sampling Iteration")
        ax.set_ylabel("Active Followers")

        plt.tight_layout()

        plt.savefig(
            f"{session_id}_cf_active_followers.pdf",
            bbox_inches="tight",
        )
    else:
        raise NotImplementedError




if __name__ == "__main__":
    plot_active_nodes()