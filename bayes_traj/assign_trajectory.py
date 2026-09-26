#!/usr/bin/env python

from argparse import ArgumentParser
import pandas as pd
import numpy as np
from provenance_tools.write_provenance_data import write_provenance_data
import pickle


def _augment_from_R(df, R, traj_ids):
    out = df.copy()
    out['traj'] = np.argmax(R, axis=1).astype(int)
    for kk in traj_ids:
        out[f'traj_{int(kk)}'] = R[:, int(kk)]
    return out


def main():
    desc = """Assigns individuals to trajectory subgroups using a fitted model.

For structured random-effect models, external subjects are classified using
fresh q(b_i|z_i=k) local random-effect posteriors. By default all trained global
quantities, including trajectory prevalence, are frozen. An optional prevalence-
adaptation mode can re-estimate only the new cohort's DP stick weights while
keeping trajectory coefficients, residual precisions, and D fixed.
"""

    args = ArgumentParser(description=desc)
    args.add_argument('--in_csv', help='Input csv data file. Individuals may be '
        'different from those used to fit the model.', default=None, type=str)
    args.add_argument('--groupby', help='Subject identifier column name in the '
        'input data file to use for grouping.', required=False, type=str,
        default=None)
    args.add_argument('--model', help='Pickled trajectory model to use for '
        'assigning data instances to trajectories', type=str, required=True)
    args.add_argument('--out_csv', help='Output csv file with trajectory MAP '
        'assignment and posterior-probability columns.', type=str, default=None)
    args.add_argument('--inference_mode', choices=['strict', 'adapt_prevalence'],
        default='strict', help='Structured external-inference mode. strict '
        'freezes all trained global quantities. adapt_prevalence additionally '
        're-estimates the new cohort DP stick weights.')
    args.add_argument('--prevalence_max_iters', type=int, default=100,
        help='Maximum q(z)/q(v) iterations for --inference_mode adapt_prevalence.')
    args.add_argument('--prevalence_tol', type=float, default=1e-8,
        help='Responsibility-change tolerance for prevalence adaptation.')
    args.add_argument('--out_ranef_npz', type=str, default=None,
        help='Optional .npz output containing structured new-subject '
        'class-conditional random-effect means/covariances and group-level '
        'posterior probabilities.')
    args.add_argument('--traj_map', help='Mapping from native trajectory IDs to '
        'desired output numbering, e.g. 3-1,18-2,7-3. Native trajectories not '
        'listed are mapped to NaN.', type=str, default=None)

    op = args.parse_args()

    print('Reading model...')
    mm = pd.read_pickle(op.model)['MultDPRegression']

    if op.in_csv is None:
        if op.inference_mode != 'strict':
            raise ValueError('--inference_mode only applies when --in_csv is supplied')
        df_out = mm.to_df().copy()
        inference_result = None
    else:
        print('Reading data...')
        df = pd.read_csv(op.in_csv)
        if getattr(mm, 'ranef_factorization_', 'mean_field') == 'structured':
            print(f'Assigning with structured {op.inference_mode} inference...')
            inference_result = mm.infer_new_data(
                df, gb_col=op.groupby, mode=op.inference_mode,
                max_iters=op.prevalence_max_iters, tol=op.prevalence_tol,
                return_random_effects=(op.out_ranef_npz is not None))
            if op.inference_mode == 'adapt_prevalence' and \
               not inference_result['converged']:
                print(
                    'Warning: prevalence adaptation reached max iterations '
                    f"({inference_result['iterations']}) with max dR="
                    f"{inference_result['max_responsibility_change']:.3e}.")
            R = inference_result['R'].detach().cpu().numpy()
            # Preserve the full posterior vector over the truncation. Empty
            # components are cheap to retain and this avoids silently discarding
            # posterior information needed by downstream analyses.
            df_out = _augment_from_R(df, R, np.arange(mm.K_))
        else:
            if op.inference_mode != 'strict':
                raise ValueError(
                    'adapt_prevalence is only implemented for structured models')
            inference_result = None
            df_out = mm.augment_df_with_traj_info(
                df, op.groupby, test_data=True)

    all_ids = [int(x) for x in range(mm.K_)]
    if op.traj_map is not None:
        traj_map = {}
        for item in op.traj_map.split(','):
            src, dst = item.split('-')
            traj_map[int(src)] = int(dst)
        unknown = sorted(set(traj_map) - set(all_ids))
        if unknown:
            raise ValueError(f'--traj_map contains unknown trajectory IDs: {unknown}')
        if len(set(traj_map.values())) != len(traj_map):
            raise ValueError('--traj_map destination IDs must be unique')

        native_map = df_out['traj'].astype(int).to_numpy()
        df_out['traj'] = [traj_map.get(int(x), np.nan) for x in native_map]
        # Unmapped posterior columns are intentionally dropped rather than
        # being renamed to repeated ``traj_nan`` columns.
        prob_cols = [f'traj_{ii}' for ii in all_ids if f'traj_{ii}' in df_out.columns]
        keep_prob = {}
        for src, dst in traj_map.items():
            col = f'traj_{src}'
            if col in df_out.columns:
                keep_prob[col] = f'traj_{dst}'
        df_out.drop(columns=[c for c in prob_cols if c not in keep_prob],
                    inplace=True)
        df_out.rename(columns=keep_prob, inplace=True)
    # With no map, native IDs and the full posterior vector are preserved.

    if op.out_csv is not None:
        print('Saving data with trajectory info...')
        df_out.to_csv(op.out_csv, index=False)
        print('Saving data file provenance info...')
        write_provenance_data(op.out_csv, generator_args=op, desc='')

    if op.out_ranef_npz is not None:
        if inference_result is None or 'u_mu' not in inference_result:
            raise ValueError('--out_ranef_npz requires structured external inference')
        print('Saving new-subject random-effect posterior info...')
        np.savez_compressed(
            op.out_ranef_npz,
            group_labels=np.asarray(inference_result['group_labels']),
            R_group=inference_result['R_group'].detach().cpu().numpy(),
            u_mu=inference_result['u_mu'].detach().cpu().numpy(),
            u_Sig=inference_result['u_Sig'].detach().cpu().numpy(),
            u_mu_marginal=inference_result['u_mu_marginal'].detach().cpu().numpy(),
            u_Sig_marginal=inference_result['u_Sig_marginal'].detach().cpu().numpy(),
            ranef_indices=np.asarray(inference_result['ranef_indices'], dtype=int),
            ranef_predictor_names=np.asarray(
                inference_result['ranef_predictor_names'], dtype=str),
            v_a=inference_result['v_a'].detach().cpu().numpy(),
            v_b=inference_result['v_b'].detach().cpu().numpy(),
            inference_mode=np.asarray(inference_result['mode']),
            iterations=np.asarray(inference_result['iterations']),
            converged=np.asarray(inference_result['converged']),
            max_responsibility_change=np.asarray(
                inference_result['max_responsibility_change']),
        )
        write_provenance_data(op.out_ranef_npz, generator_args=op, desc='')

    print('DONE.')


if __name__ == '__main__':
    main()
