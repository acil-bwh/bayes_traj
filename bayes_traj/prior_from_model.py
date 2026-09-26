"""Generate a compact hyperprior from a fitted bayes_traj model.

This helper predates :mod:`bayes_traj.generate_prior` and is retained for
backward compatibility. Structured-DP models use posterior-reportable
components rather than ``sig_trajs_`` because all truncation components remain
computationally alive by design.
"""

import pickle
from argparse import ArgumentParser

import numpy as np
from provenance_tools.write_provenance_data import write_provenance_data


def _reportable_ids(mm):
    if hasattr(mm, 'get_reportable_trajectory_ids'):
        return np.asarray(mm.get_reportable_trajectory_ids(), dtype=int)
    sig = mm.sig_trajs_.detach().cpu().numpy() \
        if hasattr(mm.sig_trajs_, 'detach') else np.asarray(mm.sig_trajs_)
    return np.where(sig)[0]


def prior_from_model(mm):
    """Compute a hyperprior from a fitted model.

    The trajectory-specific posterior parameters are sampled in proportion to
    posterior subject-level trajectory mass. Non-reportable truncation slots in
    structured DP models are ignored.
    """
    traj_ids = _reportable_ids(mm)
    if traj_ids.size == 0:
        raise ValueError('No reportable trajectories in fitted model')

    M = mm.M_
    D = mm.D_
    w_mu0_post = np.zeros([M, D])
    w_var0_post = np.ones([M, D])

    traj_probs = np.asarray(mm.get_traj_probs(), dtype=float)
    selected_probs = traj_probs[traj_ids]
    selected_probs = selected_probs / np.sum(selected_probs)
    num_selected_samples = np.random.multinomial(10000, selected_probs)
    num_traj_samples = np.zeros(mm.K_, dtype=int)
    num_traj_samples[traj_ids] = num_selected_samples

    w_mu = mm.w_mu_.detach().cpu().numpy() if hasattr(mm.w_mu_, 'detach') else np.asarray(mm.w_mu_)
    w_var = mm.w_var_.detach().cpu().numpy() if hasattr(mm.w_var_, 'detach') else np.asarray(mm.w_var_)
    lambda_a = mm.lambda_a_.detach().cpu().numpy() if hasattr(mm.lambda_a_, 'detach') else np.asarray(mm.lambda_a_)
    lambda_b = mm.lambda_b_.detach().cpu().numpy() if hasattr(mm.lambda_b_, 'detach') else np.asarray(mm.lambda_b_)

    for m in range(M):
        for d in range(D):
            samples = []
            for t in traj_ids:
                n = num_traj_samples[t]
                if n == 0:
                    continue
                samples.append(
                    w_mu[m, d, t] + np.sqrt(w_var[m, d, t]) * np.random.randn(n))
            sample = np.hstack(samples)
            w_mu0_post[m, d] = np.mean(sample)
            w_var0_post[m, d] = np.var(sample)

    lambda_a0_post = np.ones(D)
    lambda_b0_post = np.ones(D)
    for d in range(D):
        samples = []
        for t in traj_ids:
            n = num_traj_samples[t]
            if n == 0:
                continue
            samples.append(np.random.gamma(
                lambda_a[d, t], 1.0 / lambda_b[d, t], n))
        sample = np.hstack(samples)
        mu = np.mean(sample)
        var = np.var(sample)
        lambda_a0_post[d] = mu ** 2 / var
        lambda_b0_post[d] = mu / var

    prior_traj_probs = np.zeros(mm.K_, dtype=float)
    prior_traj_probs[traj_ids] = selected_probs
    prior = {
        'w_mu0': w_mu0_post,
        'w_var0': w_var0_post,
        'lambda_a0': lambda_a0_post,
        'lambda_b0': lambda_b0_post,
        'traj_probs': prior_traj_probs,
        'alpha': mm.alpha_,
        'source_trajectory_ids': traj_ids,
    }
    return prior


def main():
    parser = ArgumentParser(
        description='Generate a compact prior from a fitted bayes_traj model')
    parser.add_argument('--model', required=True)
    parser.add_argument('--prior', required=True)
    op = parser.parse_args()

    with open(op.model, 'rb') as f:
        mm = pickle.load(f)['MultDPRegression']
    prior = prior_from_model(mm)
    with open(op.prior, 'wb') as f:
        pickle.dump(prior, f)
    write_provenance_data(op.prior, generator_args=op, desc='')


if __name__ == '__main__':
    main()
