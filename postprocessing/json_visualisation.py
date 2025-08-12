import re
import json
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

from pathlib import Path

sns.set_style("whitegrid")

# LOG_PATH = Path("/home/sgmcart3/.cache/condorcmf/UNKNOWN/log/test/scheduler")
LOG_PATH = Path("/home/sgmcart3/.cache/condorcmf/UNKNOWN/log/test_old/scheduler")
N_FOLLOWERS = 256
N_MANAGERS = 4
N_GLOBAL_MANAGER_ITERS = 7


def create_follower_heatmap(architecture="cmf"):
    follower_iters = {}

    if architecture == "cmf":
        # Extract the number of sampling iterations for each follower for each global manager iteration
        ess_per_follower = []
        for i in range(N_MANAGERS):
            manager_out_path = Path(LOG_PATH, f"test_manager_{i}", "stdout.log")
            try:
                with open(manager_out_path, "r") as f:
                    lines = f.readlines()
            except:
                print(f"Could not find manager {i}")
                continue

            follower_iters[f"manager_{i}"] = {}

            for j in range(N_FOLLOWERS // N_MANAGERS):
                follower_out_path = Path(LOG_PATH, f"test_manager_{i}_follower_{j}", "stdout.log")
                try:
                    with open(follower_out_path, "r") as f:
                        lines = f.readlines()
                except:
                    print(f"Could not find follower {j} for manager {i}")
                    continue

                follower_iters[f"manager_{i}"][f"follower_{j}"] = []


                for line in lines:
                    match = re.search(r"Sampling complete after (\d+) iterations", line)

                    if match:
                        follower_iters[f"manager_{i}"][f"follower_{j}"].append(int(match.group(1)))

                    ess_match = re.search(r"ESS: (\d*\.\d+)", line)
                    if ess_match:
                        ess_per_follower.append(float(ess_match.group(1)) / 50 * 100)
        print(np.mean(ess_per_follower), np.std(ess_per_follower))

        # Pad the follower_iters dictionary with zeros
        for manager in follower_iters.keys():
            for follower in follower_iters[manager].keys():
                follower_iters[manager][follower] = [0] * (N_GLOBAL_MANAGER_ITERS - len(follower_iters[manager][follower])) + follower_iters[manager][follower]

        min_iters = -np.inf
        max_iters = np.inf

        follower_iters_avg = {}
        follower_iters_std = {}
        for manager in follower_iters.keys():
            follower_iters_avg[manager] = {}
            follower_iters_std[manager] = {}

            for follower in follower_iters[manager].keys():
                follower_iters_avg[manager][follower] = []
                follower_iters_std[manager][follower] = []

                for i in range(N_GLOBAL_MANAGER_ITERS):
                    follower_iters_avg[manager][follower].append(np.mean(follower_iters[manager][follower][:i+1]))
                    follower_iters_std[manager][follower].append(np.std(follower_iters[manager][follower][:i+1]))

                    if follower_iters_avg[manager][follower][-1] < max_iters:
                        max_iters = follower_iters_avg[manager][follower][-1]

                    if follower_iters_avg[manager][follower][-1] > min_iters:
                        min_iters = follower_iters_avg[manager][follower][-1]

        with open("follower_iters.json", "w") as f:
            json.dump(
                {
                    "follower_iters": follower_iters,
                    "follower_iters_avg": follower_iters_avg,
                    "follower_iters_std": follower_iters_std,
                },
                f,
            )

        # Plot the number of sampling iterations for each follower for each global manager iteration using imshow
        for manager in follower_iters.keys():
            follower_iters_array = np.array(list(follower_iters[manager].values()))

            plt.figure(figsize=(4, 4))

            plt.imshow(
                follower_iters_array,
                cmap="viridis",
                aspect="auto",
                interpolation="nearest",
                vmin=min_iters,
                vmax=max_iters,
            )

            plt.xlabel(r"Global Manager Iteration $G^{g_{m}}_{f}$")
            plt.ylabel("Follower ID")
            plt.colorbar(label=r"Number of Local Iteration $K_{f}$")

            plt.grid(False)

            plt.tight_layout()
            plt.savefig(f"{manager}_follower_iters.pdf", bbox_inches="tight")
    else:
        raise NotImplementedError

def count_total_samples(architecture="cmf"):
    total_samples = 0

    if architecture == "cmf":
        # Extract the number of sampling iterations for each follower for each global manager iteration
        for i in range(N_MANAGERS):
            manager_out_path = Path(LOG_PATH, f"test_manager_{i}", "stdout.log")
            try:
                with open(manager_out_path, "r") as f:
                    lines = f.readlines()
            except:
                print(f"Could not find manager {i}")
                continue

            for j in range(64):
                follower_out_path = Path(LOG_PATH, f"test_manager_{i}_follower_{j}", "stdout.log")
                try:
                    with open(follower_out_path, "r") as f:
                        lines = f.readlines()
                except:
                    print(f"Could not find follower {j} for manager {i}")
                    continue

                for line in lines:
                    match = re.search(r"Sampling complete after (\d+) iterations \((\d+) samples\)", line)

                    if match:
                        total_samples += int(match.group(2))
                        print(f"Found {match.group(2)} samples for follower {j} of manager {i}")

        print(f"Total samples: {total_samples}")

    else:
        raise NotImplementedError


if __name__ == "__main__":
    create_follower_heatmap()
    count_total_samples()
