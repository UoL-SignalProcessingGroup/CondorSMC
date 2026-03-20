![GitHub repo size](https://img.shields.io/github/repo-size/UoL-SignalProcessingGroup/CondorSMC)
![GitHub contributors](https://img.shields.io/github/contributors/UoL-SignalProcessingGroup/CondorSMC)
![GitHub stars](https://img.shields.io/github/stars/UoL-SignalProcessingGroup/CondorSMC?style=social)
![GitHub forks](https://img.shields.io/github/forks/UoL-SignalProcessingGroup/CondorSMC?style=social)

# CondorSMC
### An opportunistic Sequential Monte Carlo Sampler on HTCondor

CondorSMC is a Python package that enables users to sample from target densities using an opportunistic Sequential Monte Carlo sampler distributed on HTCondor. 

## Installing CondorSMC

```
pip install git+https://github.com/UoL-SignalProcessingGroup/CondorSMC
```

## How to Run

### 1. Create configuration files

CondorSMC uses YAML (or TOML/INI) config files to supply MySQL connection details and tuning parameters. Copy `condorsmc.example.yaml` from the repository root and fill in your database credentials.

In **distributed mode**, three config files are needed — one per node role — because each role typically connects to the database with different credentials:

| File | Role |
|---|---|
| `condorsmc.yaml` | Coordinator (read from the working directory) |
| `follower_config.cfg` | Follower nodes (transferred to workers by HTCondor) |
| `manager_config.cfg` | Manager nodes (transferred to managers by HTCondor) |

A minimal config file looks like:

```yaml
mysql:
  host: my-db-server     # required
  user: myuser           # required
  password: secret
  database: condorsmc    # required

htcondor:
  python_env: /path/to/env.tar.gz   # required (transferred to all HTCondor nodes)
```

`mysql.host`, `mysql.user`, `mysql.database`, and `htcondor.python_env` are validated at startup in distributed mode — CondorSMC will exit with a clear error if any are missing. All settings can alternatively be supplied as `CONDORSMC_*` environment variables — see `condorsmc.example.yaml` for the full list.

**Where to place the files**

CondorSMC searches for config files in this order:

1. Path set by the `CONDORSMC_CONFIG` environment variable
2. `condorsmc.yaml` / `.yml` / `.toml` / `.ini` in the **current working directory**
3. `config.yaml` / `.yml` / `.toml` / `.ini` in the user config directory:
   ```
   python -c "from condorsmc.paths import get_package_dirs; print(get_package_dirs().config_dir)"
   ```

The follower and manager config files are looked up in `./config/` in the working directory first, then the user config directory.

### 2. Define a target model

Create a Python file containing a `Target` class with `dim`, `logpdf`, and `logpdfgrad`. For example, `normal5d.py`:

```python
import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd.scipy import stats as AutoStats

class Target:
    def __init__(self, data={}):
        self.dim = 5
        self.mean = np.array([-4, -2, 0, 2, 4])
        self.cov = np.eye(5)

    def logpdf(self, x):
        return AutoStats.multivariate_normal.logpdf(x, mean=self.mean, cov=self.cov)

    def logpdfgrad(self, x):
        return egrad(self.logpdf)(x)
```

If the model requires data, place it in a JSON file with the same stem (e.g. `normal5d.json`) alongside the model file — it will be loaded automatically. Stan models (`.stan`) are also supported.

### 3. Execute CondorSMC

**Sequential mode** — runs locally, no HTCondor or database required:

```
python -m condorsmc sequential --model normal5d --nsamples 1000 --niters 100
```

**Distributed mode** — run the coordinator on the submit node; worker jobs are submitted to HTCondor automatically:

```
python -m condorsmc distributed --role coordinator \
    --model normal5d --nsamples 500 \
    --nfollowers 50 --follower-runtime 30 \
    --coordinator-runtime 600
```

Pass `--debug` to either mode to enable verbose logging to stderr.

A number of worked examples are provided in the `examples/` folder.

## Contributing to CondorSMC
To contribute to CondorSMC, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## Contact
If you want to contact me you can reach me at <matthew.carter (at) liverpool (dot) ac (dot) uk>.

## Citation
We appreciate citations as they let us discover what people have been doing with the software. 

To cite CondorSMC in publications use:

Carter, M., Devlin, L., Phillips, A., Pyzer-Knapp, K., Spirakis, P. and Maskell, S. (2025) CondorSMC (1.0.0). https://github.com/UoL-SignalProcessingGroup/CondorSMC

Or use the following BibTeX entry:

```
@misc{CondorSMC,
  title = {CondorSMC (1.0.0)},
  author = {Carter, Matthew and Devlin, Lee and Phillips, Alexander and Pyzer-Knapp, Edward and Spirakis, Paul and Maskell, Simon},
  year = {2025},
  month = may,
  howpublished = {GitHub},
  url = {https://github.com/UoL-SignalProcessingGroup/CondorSMC}
}
```
