import re
import numpy as np

from pathlib import Path

LOG_PATH = Path("/home/sgmcart3/.cache/condorcmf/UNKNOWN/log/test/scheduler")
N_FOLLOWERS = 200
N_MANAGERS = 0


def main(): 
    follower_init_time = []
    manager_init_time = []

    if N_MANAGERS > 0:
        for i in range(N_MANAGERS):
            manager_out_path = Path(LOG_PATH, f"test_manager_{i}", "stdout.log")
            try:
                with open(manager_out_path, "r") as f:
                    lines = f.readlines()
            except:
                print(f"Could not find manager {i}")
                continue

            pattern = re.compile(r"Node took (\d+\.\d+) seconds to initialise")

            for line in lines:
                match = pattern.match(line)
                if match:
                    manager_init_time.append(float(match.group(1)))

            for j in range(N_FOLLOWERS // N_MANAGERS):
                follower_out_path = Path(LOG_PATH, f"test_manager_{i}_follower_{j}", "stdout.log")
                try:
                    with open(follower_out_path, "r") as f:
                        lines = f.readlines()
                except:
                    print(f"Could not find follower {j} for manager {i}")
                    continue

                pattern = re.compile(r"Node took (\d+\.\d+) seconds to initialise")

                for line in lines:
                    match = pattern.match(line)
                    if match:
                        follower_init_time.append(float(match.group(1)))

        manager_init_time = np.array(manager_init_time)
        print("Manager init time")
        print(f"mean: {np.mean(manager_init_time)}; sd: {np.std(manager_init_time)}")

    else:
        for i in range(N_FOLLOWERS):
            follower_out_path = Path(LOG_PATH, f"test_follower_{i:04d}", "stdout.log")
            try:
                with open(follower_out_path, "r") as f:
                    lines = f.readlines()
            except:
                print(f"Could not find follower {i}")
                continue

            pattern = re.compile(r"Node took (\d+\.\d+) seconds to initialise")

            for line in lines:
                match = pattern.match(line)
                if match:
                    follower_init_time.append(float(match.group(1)))

    follower_init_time = np.array(follower_init_time)
    print("Follower init time")
    print(f"mean: {np.mean(follower_init_time)}; sd: {np.std(follower_init_time)}")


if __name__ == "__main__":
    main()
