# CondorSMC
### An opportunistic Sequential Monte Carlo Sampler on HTCondor

<!-- tempate https://github.com/scottydocs/README-template.md/blob/master/README.md -->
![GitHub repo size](https://img.shields.io/github/repo-size/UoL-SignalProcessingGroup/CondorSMC)
![GitHub contributors](https://img.shields.io/github/contributors/UoL-SignalProcessingGroup/CondorSMC)
![GitHub stars](https://img.shields.io/github/stars/UoL-SignalProcessingGroup/CondorSMC?style=social)
![GitHub forks](https://img.shields.io/github/forks/UoL-SignalProcessingGroup/CondorSMC?style=social)

CondorSMC is a Python package that enables users to sample from target densities using an opportunistic Sequential Monte Carlo sampler distributed on HTCondor. 

## Installing CondorSMC
To install CondorSMC, follow these steps:

1. To begin, install the CondorCMF package

```
pip install pip@git+https://github.com/mjcarter95/CondorCMF.git
```

2. Install the CondorSMC package

```
pip install pip@git+https://github.com/mjcarter95/CondorSMC.git
```

3. Set up configuration files
Modify the configuration file in `bin/config.example`, to run in distributed mode, there should be a config file for the:
* Coordinator `config.cfg`
* Manager `manager_config.cfg`
* Follower `follower_config.cfg`

These should be placed in the SMC-Stan cache directory `~/.config/condorscmstan/0.1.0` or the working directory which you are launching CondorSMC from `working_directory/config/*.cfg`.

## Using CondorSMC
A number of example problems are provided in the `examples` folder.

### Sampling from a 5-Dimensional Multivariate Gaussian Distribution
Suppose we want to generate samples from the distribution

$$\pi(x) = N(x; [-4, -2, 0, 2, 4], I_{5})$$

We can define this target distribution, `normal5d.py`, as:

```
import autograd.numpy as np  # type: ignore
from autograd import elementwise_grad as egrad  # type: ignore
from autograd.scipy import stats as AutoStats  # type: ignore
from scipy.stats import multivariate_normal  # type: ignore


class Target:
    def __init__(self, data={}):
        self.data = data
        self.dim = 5
        self.mean = np.array([-4, 2, 0, 2, 4])
        self.cov = np.eye(5)

    def logpdf(self, x):
        return AutoStats.multivariate_normal.logpdf(x, mean=self.mean, cov=self.cov)

    def logpdfgrad(self, x):
        grad = egrad(self.logpdf)
        return grad(x)

```

We can then generate samples from this target distribution using CondorSMC in either sequential mode

```
python3 -m condorsmc --session-id test --model normal5d --verbose
```

or in distributed mode

```
python3 -m condorsmc --session-id test --node-id test_coordinator --mode distributed --nfollowers 500 --follower-runtime 15 --model normal5d --niters 200 --verbose
```

A target class must have only `data` as an argument to the constructor, have both `data` and `dim` as attributes, and contain methods to calculate the log probability (`logpdf`) and log gradient (`logpdfgrad`) of the target distribution. All data should be stored in a JSON file which has the same name as the targets `.py` file, e.g. `normal5d.json`. The data will be automatically loaded and passed to the target constructor when launching CondorSMC. 

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

Carter, M., Devlin, L., Philips, A., Pyzer-Knapp, K., Spirakis, P. Maskell, S.,(2025). CondorSMC (1.0.0). https://github.com/UoL-SignalProcessingGroup/CondorSMC

Or use the following BibTeX entry:

```
@misc{CondorSMC,
  title = {CondorSMC (1.0.0)},
  author = {Carter, Matthew and Devlin, Lee and Philips, Alexander and Pyzer-Knapp, Edward and Spirakis, Paul and Maskell, Simon},
  year = {2025},
  month = may,
  howpublished = {GitHub},
  url = {https://github.com/UoL-SignalProcessingGroup/CondorSMC}
}
```
