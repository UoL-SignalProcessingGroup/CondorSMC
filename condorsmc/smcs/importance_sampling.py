import autograd.numpy as np  # type: ignore
from scipy.special import logsumexp

from condorcmf.dbqueue.checkpoint import Checkpoint as DBQCheckpoint


def normalise_weights(logw):
    """
    Normalises the sample weights

    Args:
        logw: A list of sample weights on the log scale

    Returns:
        A list of normalised weights

    """

    logw = np.nan_to_num(logw, nan=-np.inf)

    index = ~np.isneginf(logw)

    log_likelihood = logsumexp(logw[index])

    # Normalise the weights
    wn = np.zeros_like(logw)
    wn[index] = np.exp(logw[index] - log_likelihood)

    return wn, log_likelihood


def calculate_ess(wn):
    """
    Calculate the effective sample size using the normalised
    sample weights.

    Args:
        wn: A list of normalised sample weights

    Return:
        The effective sample size
    """

    return 1 / np.sum(np.square(wn))


def resample(x, wn, log_likelihood):
    """
    Resamples samples and their weights from the specified indexes

    Args:
        x: A list of samples to resample
        wn: A list of normalise sample weights to resample
        indexes: A list of the indexes of samples and weights to resample

    Returns:
        x_new: A list of resampled samples
        logw_new: A list of resampled weights
    """

    i = np.linspace(0, x.shape[0] - 1, x.shape[0], dtype=int)
    i_new = np.random.choice(i, x.shape[0], p=wn)
    wn_new = np.ones(x.shape[0]) / x.shape[0]

    x_new = x[i_new]
    logw_new = np.log(np.ones(x.shape[0]) / x.shape[0])

    return x_new, logw_new


def node_resample(x, logw, nm_log_likelihoods, log_likelihood):
    """
    Resamples samples and their weights from the specified indexes

    Args:
        x: A list of samples to resample
        wn: A list of normalise sample weights to resample
        indexes: A list of the indexes of samples and weights to resample

    Returns:
        x_new: A list of resampled samples
        logw_new: A list of resampled weights
    """

    i = np.linspace(0, x.shape[0] - 1, x.shape[0], dtype=int)
    i_new = np.random.choice(i, x.shape[0], p=nm_log_likelihoods)
    wn_new = np.ones(x.shape[0]) / x.shape[0]

    x_new = x[i_new]
    logw_new = (np.ones(x.shape[0], x.shape[1]) / x.shape[0] * x.shape[1])

    return x_new, logw_new


def checkpoint_resample(args, dbq_db, active_nodes):
    """
    Resamples nodes from their checkpointed state.

    Args:
        args: The arguments passed to CondorSMCStan
        active_nodes: A list of active nodes
    """

    x = []
    logw = []
    node_log_likehoods = []

    for node in active_nodes:
        dbq_checkpoint = DBQCheckpoint(
            dbq_db,
            args.session_id,
            node,
            2,
        )

        checkpoint_payload = dbq_checkpoint.get(type=0)
        x.append(checkpoint_payload["x"])
        logw.append(checkpoint_payload["logw"])
        log_likelihood = logsumexp(checkpoint_payload["logw"])
        node_log_likehoods.append(log_likelihood)

    _logw = np.hstack(logw)
    _wn, _log_likelihood = normalise_weights(_logw)
    _ess = calculate_ess(_wn)
    print(_ess / len(x) * 100)

    if args.resampling == "centralised":
        # Stack samples and weights
        x = np.vstack(x)
        logw = np.hstack(logw)

        # Normalise weights and resample
        wn, log_likelihood = normalise_weights(logw)
        ess = calculate_ess(wn)
        _ess = ess / len(x) * 100
        # if args.verbose:
        print(f"Global effective sample size: ({_ess:.2f}%)")
        if ess < len(x) / 2:
            print("resample")
            x_new, logw_new = resample(x, wn, log_likelihood)

            # Split samples and weights
            x_new = np.split(x_new, len(active_nodes))
            logw_new = np.split(logw_new, len(active_nodes))

            # Update checkpointed states
            for i, node in enumerate(active_nodes):
                dbq_checkpoint = DBQCheckpoint(
                    dbq_db,
                    args.session_id,
                    node,
                    2,
                )

                payload = {
                    "x": x_new[i],
                    "logw": logw_new[i],
                }

                dbq_checkpoint.update(
                    type=0,
                    payload=payload,
                )

    elif args.resampling == "centralised_ews":
        # Stack samples, weights and node log likelihoods
        x = np.vstack(x)
        logw = np.hstack(logw)
        node_log_likehoods = np.hstack(node_log_likehoods)

        # Normalise weights and resample
        node_wn, node_log_likelihood = normalise_weights(node_log_likehoods)
        ews = calculate_ess(node_wn)
        _ews = ews / len(active_nodes) * 100
        if args.verbose:
            print(f"Global effective worker size: ({_ews:.2f}%)")
        if ews < len(active_nodes) / 2:
            wn, log_likelihood = normalise_weights(logw)
            x_new, logw_new = resample(x, wn, log_likelihood)

            # Split samples and weights
            x_new = np.split(x_new, len(active_nodes))
            logw_new = np.split(logw_new, len(active_nodes))

            # Update checkpointed states
            for i, node in enumerate(active_nodes):
                dbq_checkpoint = DBQCheckpoint(
                    dbq_db,
                    args.session_id,
                    node,
                    2,
                )

                payload = {
                    "x": x_new[i],
                    "logw": logw_new[i],
                }

                dbq_checkpoint.update(
                    type=0,
                    payload=payload,
                )

    elif args.resampling == "centralised_nodes":
        # Stack samples, weights and node log likelihoods
        x = np.vstack(x)
        logw = np.hstack(logw)
        node_log_likehoods = np.hstack(node_log_likehoods)

        # Normalise weights and resample
        node_wn, node_log_likelihood = normalise_weights(node_log_likehoods)
        ews = calculate_ess(node_wn)
        _ews = ews / len(active_nodes) * 100
        if args.verbose:
            print(f"Global effective worker size: ({_ews:.2f}%)")
        if ews < len(active_nodes) / 2:
            x_new, logw_new = resample(x, wn, log_likelihood)

            # Split samples and weights
            x_new = np.split(x_new, len(active_nodes))
            logw_new = np.split(logw_new, len(active_nodes))

            # Update checkpointed states
            for i, node in enumerate(active_nodes):
                dbq_checkpoint = DBQCheckpoint(
                    dbq_db,
                    args.session_id,
                    node,
                    2,
                )

                payload = {
                    "x": x_new[i],
                    "logw": logw_new[i],
                }

                dbq_checkpoint.update(
                    type=0,
                    payload=payload,
                )


def estimate(x, wn, target):
    """
    Description:
        Importance sampling estimate of the mean and variance of the
        target distribution.

    Args:
        x: Particle positions.
        wn: Normalised importance weights.

    Returns:
        mean_estimate: Estimated mean of the target distribution.
        variance_estimate: Estimated variance of the target distribution.
    """

    if hasattr(target, "constrain"):
        _x = target.constrain(x)
    else:
        _x = x.copy()

    mean = wn.T @ _x

    x_shift = _x - mean
    if x.shape[1] == 1:
        var = wn.T @ np.square(x_shift)
    else:
        var = np.zeros((_x.shape[1], _x.shape[1]))
        for i in range(_x.shape[0]):
            xv = x_shift[i][None, :]
            var += wn[i] * xv.T @ xv
        var = np.diagonal(var)

    return mean, var


def estimate_nodes(means, variances, weights, logs=True):
    if logs:
        weights = np.exp(weights)
    z = np.sum(weights)
    mean = np.sum(weights[:, np.newaxis] * means, axis=0) / z
    variance = np.sum(weights[:, np.newaxis] * (variances + (means - mean)), axis=0) / z
    return mean, variance, z
