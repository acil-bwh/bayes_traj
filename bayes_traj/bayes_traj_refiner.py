#!/usr/bin/env python
"""Continue a saved bayes_traj structured fit.

This command is retained for compatibility with historical workflows.  New
workflows may equivalently use ``bayes_traj_main --resume_model``.  The old
refiner re-entered the obsolete mean-field ``fit`` API and could inadvertently
reinitialize model state; this implementation performs a true structured-state
continuation instead.
"""

from argparse import ArgumentParser
import pickle

from provenance_tools.provenance_tracker import write_provenance_data


def main():
    parser = ArgumentParser(
        description='Continue variational inference from a saved structured '
                    'MultDPRegression model without reinitialization.')
    parser.add_argument('--in_p', required=True,
        help='Input pickle containing MultDPRegression')
    parser.add_argument('--out_file', required=True,
        help='Output pickle for the continued model')
    parser.add_argument('--iters', type=int, default=100,
        help='Maximum number of additional iterations')
    parser.add_argument('--ranef_cov_mode', choices=['fixed', 'estimate', 'staged'],
        default=None, help='Optional covariance strategy override for continuation')
    parser.add_argument('--structured_tol_elbo_rel', type=float, default=None)
    parser.add_argument('--structured_tol_r', type=float, default=None)
    parser.add_argument('--structured_tol_w', type=float, default=None)
    parser.add_argument('--structured_tol_lambda', type=float, default=None)
    parser.add_argument('--structured_tol_ranef_cov', type=float, default=None)
    parser.add_argument('--structured_min_iters', type=int, default=None)
    parser.add_argument('--ranef_cov_warmup_iters', type=int, default=None)
    parser.add_argument('--verbose', action='store_true')
    op = parser.parse_args()

    with open(op.in_p, 'rb') as f:
        mm = pickle.load(f)['MultDPRegression']

    if getattr(mm, 'ranef_factorization_', 'mean_field') != 'structured':
        raise RuntimeError(
            'bayes_traj_refiner now supports corrected structured inference only. '
            'Historical mean-field refinement is intentionally not maintained.')

    mm.continue_structured_fit(
        iters=op.iters,
        verbose=op.verbose,
        ranef_cov_mode=op.ranef_cov_mode,
        structured_tol_elbo_rel=op.structured_tol_elbo_rel,
        structured_tol_r=op.structured_tol_r,
        structured_tol_w=op.structured_tol_w,
        structured_tol_lambda=op.structured_tol_lambda,
        structured_tol_ranef_cov=op.structured_tol_ranef_cov,
        structured_min_iters=op.structured_min_iters,
        ranef_cov_warmup_iters=op.ranef_cov_warmup_iters)

    with open(op.out_file, 'wb') as f:
        pickle.dump({'MultDPRegression': mm}, f)
    write_provenance_data(op.out_file, generator_args=op)


if __name__ == '__main__':
    main()
