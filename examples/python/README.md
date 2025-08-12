# Summary of CMFMPI examples
This README file summarises the example problems implemented for the CondorSMCStan package. To run one of the example problems in sequential mode, cd into the model directory and execute the following command

```
python3 -m condorsmcstan --model modelname --session-id test --niters 100 --nsamples 200 --proposal hmc --lkernel pseudo --recycling ess --verbose
```

To execute in distributed mode, use the following


```
python3 -m condorsmcstan --model modelname --session-id test --node-id test_coordinator --mode distributed --nfollowers 500 --follower-runtime 15 --niters 200 --recycling ess --resampling centralised --verbose
```

Where modelname is the name of the Python file that contains the target class.

Table of Contents:
* `normal_5d`: Target a 5-dimensional multivariate Gaussian distribution
* `student_t`: Target a 5-dimensional multivariate Student-T distribution