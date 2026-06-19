#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from argparse import ArgumentParser
try:
    from provenance_tools.write_provenance_data import write_provenance_data
except ImportError:
    write_provenance_data = None


def _get_id_col(mm, requested_id_col):
    """Resolve the subject-ID column used when the model was fit."""
    if getattr(mm, 'gb_', None) is None:
        raise ValueError(
            "Subject overlays require a model fit with groupby=<subject-ID>."
        )

    if requested_id_col is not None:
        if requested_id_col not in mm.df_.columns:
            raise ValueError(f"--id_col '{requested_id_col}' is not in model data")
        return requested_id_col

    id_col = mm.gb_.keys
    if not isinstance(id_col, str):
        raise ValueError(
            "Could not infer one subject-ID column from the model. "
            "Please specify --id_col."
        )
    if id_col not in mm.df_.columns:
        raise ValueError(
            f"The fitted groupby column '{id_col}' was not retained in df_; "
            "please specify --id_col."
        )
    return id_col


def _get_subject_table(mm, df_traj, id_col):
    """Return one row per fitted subject, in stable model-data order."""
    group_first = getattr(mm, 'group_first_index_', None)
    if group_first is None:
        if getattr(mm, 'gb_', None) is None:
            group_first = np.ones(df_traj.shape[0], dtype=bool)
        else:
            mm._set_group_first_index(mm.df_, mm.gb_)
            group_first = mm.group_first_index_

    group_first = np.asarray(group_first, dtype=bool)
    first_rows = np.flatnonzero(group_first)
    if first_rows.size == 0:
        raise ValueError("The fitted model does not contain any subject groups")

    subject_df = df_traj.iloc[first_rows][[id_col, 'traj']].copy()
    subject_df['_row_index'] = first_rows

    group_map = getattr(mm, 'N_to_G_index_map_', None)
    if group_map is None:
        mm._set_N_to_G_index_map()
        group_map = mm.N_to_G_index_map_
    subject_df['_group_index'] = np.asarray(group_map)[first_rows]
    subject_df['_subject_index'] = np.arange(subject_df.shape[0])
    return subject_df.reset_index(drop=True)


def _select_subjects(subject_df, op):
    """Select explicit subjects or a random MAP-trajectory sample."""
    if op.subject_ids is not None:
        requested = [item.strip() for item in op.subject_ids.split(',')
                     if item.strip()]
        available = subject_df.iloc[:, 0].astype(str)
        missing = [item for item in requested if item not in set(available)]
        if missing:
            raise ValueError(f"Subject IDs not found in fitted data: {missing}")
        # Preserve command-line order, which is useful when comparing cases.
        return pd.concat(
            [subject_df.loc[available == item] for item in requested],
            ignore_index=True
        )

    if op.subject_indices is not None:
        requested = [int(item.strip()) for item in op.subject_indices.split(',')
                     if item.strip()]
        if len(requested) != len(set(requested)):
            raise ValueError("--subject_indices contains duplicate indices")
        invalid = [ii for ii in requested
                   if ii < 0 or ii >= subject_df.shape[0]]
        if invalid:
            raise ValueError(
                f"Subject indices out of range [0, {subject_df.shape[0] - 1}]: "
                f"{invalid}"
            )
        return subject_df.iloc[requested].copy()

    if op.sample_n is None and op.sample_pct is None:
        return None
    if op.sample_traj is None:
        raise ValueError(
            "--sample_traj is required with --sample_n or --sample_pct"
        )

    eligible = subject_df.loc[subject_df['traj'] == op.sample_traj].copy()
    if eligible.empty:
        raise ValueError(
            f"No subjects have MAP assignment to trajectory {op.sample_traj}"
        )

    if op.sample_n is not None:
        n_sample = op.sample_n
    else:
        if op.sample_pct <= 0 or op.sample_pct > 100:
            raise ValueError("--sample_pct must be in (0, 100]")
        n_sample = int(np.ceil(eligible.shape[0] * op.sample_pct / 100.0))

    if n_sample <= 0:
        raise ValueError("--sample_n must be a positive integer")
    if n_sample > eligible.shape[0]:
        raise ValueError(
            f"Requested {n_sample} subjects but trajectory {op.sample_traj} "
            f"contains only {eligible.shape[0]} subjects"
        )

    rng = np.random.default_rng(op.random_seed)
    sample_rows = rng.choice(eligible.shape[0], size=n_sample, replace=False)
    return eligible.iloc[sample_rows].copy()


def _as_torch_2d(X, dtype=torch.float64):
    """Convert a design matrix to a detached 2-D torch tensor."""
    if torch.is_tensor(X):
        X_t = X.detach().clone().to(dtype=dtype)
    else:
        X_t = torch.as_tensor(X, dtype=dtype)
    if X_t.ndim != 2:
        raise ValueError("X must be a 2-D design matrix")
    return X_t


def _target_type(mm, target_index):
    """Return target type with a conservative Gaussian default."""
    target_type = getattr(mm, 'target_type_', {})
    return target_type.get(target_index, 'gaussian')


def _get_gaussian_fixed_mean(mm, X, target_index, traj_id):
    """Return fixed-effect Gaussian mean for one trajectory.

    This mirrors the model's Gaussian mean convention without requiring a
    subject-level prediction method on MultDPRegression.
    """
    has_shared = (
        hasattr(mm, 'num_shared_preds_') and
        getattr(mm, 'num_shared_preds_', 0) > 0
    )

    if has_shared:
        shared = torch.matmul(
            X[:, mm.shared_indices_],
            mm.w_mu_shared_[:, target_index]
        )
        traj = torch.matmul(
            X[:, mm.traj_indices_],
            mm.w_mu_[mm.traj_indices_, target_index, traj_id]
        )
        return shared + traj

    return torch.matmul(X, mm.w_mu_[:, target_index, traj_id])


def _get_subject_posterior_mean(mm, X, target_index, traj_id, group_index,
                                row_index=None):
    """Return a subject-specific conditional mean for plotting.

    The fixed-effect part is conditional on ``traj_id``. If Gaussian random
    effects are present, their contribution is the posterior-membership-
    weighted mean for the subject:

        X @ sum_k R_ik * u_ik

    where ``R_ik`` is the subject's posterior probability of trajectory k and
    ``u_ik`` is that subject's random-effect vector under trajectory k.

    Parameters
    ----------
    mm : MultDPRegression
        Fitted model object.
    X : torch.Tensor or numpy.ndarray, shape (n_rows, M)
        Design rows at which to calculate the conditional mean.
    target_index : int
        Index of the target variable to predict.
    traj_id : int
        Trajectory used for the fixed-effect contribution.
    group_index : int
        Internal subject/group index used by the random-effect tensors.
    row_index : int, optional
        A representative fitted-data row for the subject. If supplied, this is
        used to retrieve the subject's posterior trajectory probabilities.

    Returns
    -------
    torch.Tensor, shape (n_rows,)
        Conditional mean for a Gaussian target, or conditional probability for
        a binary target.
    """
    X = _as_torch_2d(X)

    if X.shape[1] != mm.M_:
        raise ValueError("X must have shape (n_rows, M_)")
    if target_index < 0 or target_index >= mm.D_:
        raise ValueError("Invalid target_index")
    if traj_id < 0 or traj_id >= mm.K_:
        raise ValueError("Invalid traj_id")
    if group_index < 0 or group_index >= mm.G_:
        raise ValueError("Invalid group_index")

    if _target_type(mm, target_index) == 'gaussian':
        fixed_mean = _get_gaussian_fixed_mean(mm, X, target_index, traj_id)

        has_ranefs = (
            getattr(mm, 'ranef_indices_', None) is not None and
            np.any(mm.ranef_indices_) and
            hasattr(mm, 'u_mu_') and mm.u_mu_ is not None
        )
        if not has_ranefs:
            return fixed_mean

        R = mm.R_
        if not torch.is_tensor(R):
            R = torch.as_tensor(R, dtype=torch.float64)
        else:
            R = R.detach().clone().to(dtype=torch.float64)

        if row_index is None:
            group_first = np.asarray(mm.group_first_index_, dtype=bool)
            group_rows = np.flatnonzero(group_first)
            if group_index >= group_rows.size:
                raise ValueError("Could not map group_index to R_ row")
            row_index = group_rows[group_index]

        subject_probs = R[int(row_index), :]
        prob_sum = torch.sum(subject_probs)
        if prob_sum <= 0:
            raise ValueError("Subject posterior trajectory probabilities sum to zero")
        subject_probs = subject_probs / prob_sum

        # u_mu_ has zeros in non-random-effect coordinates, so using the full
        # M-vector is clear and preserves compatibility with model internals.
        weighted_u = torch.sum(
            subject_probs[:, None] * mm.u_mu_[group_index, target_index, :, :],
            dim=0
        ).to(dtype=X.dtype)
        return fixed_mean + torch.matmul(X, weighted_u)

    # Binary target support is included for completeness, although subject-level
    # random-effect trend lines are currently drawn only for Gaussian targets.
    linear_predictor = torch.sum(
        X * mm.w_mu_[:, target_index, traj_id].unsqueeze(0), dim=1
    )
    return 1.0 / (1.0 + torch.exp(-linear_predictor))


def _add_subject_overlays(ax, mm, df_traj, selected_subjects, id_col,
                          x_axis, y_axis, traj_map, hide_subject_fit):
    """Overlay selected subjects' data and posterior-weighted RE trends."""
    target_index = np.where(np.asarray(mm.target_names_) == y_axis)[0][0]
    has_ranefs = (
        getattr(mm, 'ranef_indices_', None) is not None and
        np.any(mm.ranef_indices_) and
        _target_type(mm, target_index) == 'gaussian' and
        hasattr(mm, 'u_mu_') and mm.u_mu_ is not None
    )
    subject_cmap = plt.get_cmap('tab20')

    for ii, (_, subject) in enumerate(selected_subjects.iterrows()):
        subject_id = subject[id_col]
        # String matching supports command-line IDs for numeric and string
        # identifiers without coercing the stored identifier.
        row_ids = np.flatnonzero(
            df_traj[id_col].astype(str).to_numpy() == str(subject_id)
        )
        if row_ids.size == 0:
            continue

        x_values = df_traj.iloc[row_ids][x_axis].to_numpy(dtype=float)
        y_values = df_traj.iloc[row_ids][y_axis].to_numpy(dtype=float)
        finite = np.isfinite(x_values) & np.isfinite(y_values)
        row_ids = row_ids[finite]
        x_values = x_values[finite]
        y_values = y_values[finite]
        if row_ids.size == 0:
            continue

        order = np.argsort(x_values, kind='stable')
        row_ids = row_ids[order]
        x_values = x_values[order]
        y_values = y_values[order]
        color = subject_cmap(ii % subject_cmap.N)
        display_traj = int(subject['traj'])
        if traj_map is not None and display_traj in traj_map:
            display_traj = traj_map[display_traj]
        label = f"Subject {subject_id} (Traj {display_traj})"

        ax.scatter(x_values, y_values, color=color, edgecolor='k',
                   linewidth=0.5, s=36, zorder=5)
        ax.plot(x_values, y_values, color=color, linewidth=1.4, alpha=0.9,
                label=label, zorder=4)

        if has_ranefs and not hide_subject_fit:
            X_subject = mm.X_[row_ids, :]
            expected = _get_subject_posterior_mean(
                mm, X_subject, target_index, int(subject['traj']),
                int(subject['_group_index']), row_index=int(subject['_row_index'])
            ).detach().cpu().numpy()
            ax.plot(x_values, expected, color=color, linewidth=2.3,
                    linestyle='--', label=f"{label}: expected", zorder=6)


def _parse_set_vals(set_vals_arg):
    set_vals = {}
    if set_vals_arg is not None:
        for item in set_vals_arg.split(','):
            item = item.strip()
            if item == '':
                continue
            if '=' not in item:
                raise ValueError("--set_vals entries must have the form predictor=value")
            pred, val = item.split('=', 1)
            pred = pred.strip()
            val = float(val.strip())
            set_vals[pred] = val
    return set_vals


def _parse_traj_map(traj_map_arg):
    traj_map = None
    if traj_map_arg is not None:
        traj_map = {}
        for item in traj_map_arg.split(','):
            item = item.strip()
            if item == '':
                continue
            src, dst = item.split('-', 1)
            traj_map[int(src)] = int(dst)
    return traj_map


def main():
    desc = """Visualize fitted bayes_traj trajectory models."""

    parser = ArgumentParser(description=desc)
    parser.add_argument('--model', help='Model containing trajectories to visualize',
        type=str, required=True)
    parser.add_argument('--y_axis', help='Name of the target variable that will '
        'be plotted on the y-axis', type=str, required=True)
    parser.add_argument('--y_label', help='Label to display on y-axis. If none '
        'given, the variable name specified with the y_axis flag will be used.',
        type=str, default=None)
    parser.add_argument('--x_axis', help='Name of the predictor variable that will '
        'be plotted on the x-axis', type=str, required=True)
    parser.add_argument('--x_label', help='Label to display on x-axis. If none '
        'given, the variable name specified with the x_axis flag will be used.',
        type=str, default=None)
    parser.add_argument('--set_vals', help='Comma-separated predictor=value '
        'pairs to pin predictors to fixed values when plotting trajectories. '
        'Example: "cohort=1,sex=0". Predictors not specified here and not '
        'used for the x-axis default to their sample means.',
        type=str, default=None)
    parser.add_argument('--trajs', help='Comma-separated list of trajectories to '
        'plot. If none specified, all trajectories will be plotted.', type=str,
        default=None)
    parser.add_argument('--min_traj_prob', help='The probability of a given '
        'trajectory must be at least this value in order to be rendered. Value '
        'should be between 0 and 1 inclusive.', type=float, default=0)
    parser.add_argument('--max_traj_prob', help='The probability of a given '
        'trajectory can not be larger than this value in order to be rendered. '
        'Value should be between 0 and 1 inclusive.', type=float, default=1.01)
    parser.add_argument('--fig_file', help='If specified, will save the figure to '
        'file.', type=str, default=None)
    parser.add_argument('--traj_map', help='The default trajectory numbering '
        'scheme is somewhat arbitrary. Use this flag to provide a mapping '
        'between the defualt trajectory numbers and a desired numbering scheme. '
        'Provide as a comma-separated list of hyphenated mappings. '
        'E.g.: 3-1,18-2,7-3 would indicate a mapping from 3 to 1, from 18 to 2, '
        'and from 7 to 3. Only the default trajectories in the mapping will be '
        'plotted. If this flag is specified, it will override --trajs', type=str,
        default=None)
    parser.add_argument('--xlim', help='Comma-separated tuple to set the '
        'limits of display for the x-axis', type=str, default=None)
    parser.add_argument('--ylim', help='Comma-separated tuple to set the '
        'limits of display for the y-axis', type=str, default=None)
    parser.add_argument('--hs', help='This flag will hide the data scatter '
        'plot', action="store_true")
    parser.add_argument('--htd', help='This flag will hide trajectory legend '
        'details (can reduce clutter)', action="store_true")
    parser.add_argument('--traj_markers', help='Comma-separated list of '
        'markers to use for each trajectory. The number of markers should match '
        'the number of trajectories to renders. See matplotlib documentation '
        'for marker options', default=None)
    parser.add_argument('--traj_colors', help='Comma-separated list of '
        'colors to use for each trajectory. The number of colors should match '
        'the number of trajectories to renders. See matplotlib documentation '
        'for color options', default=None)
    parser.add_argument('--fill_alpha', help='Value between 0 and 1 that '
        'controls opacity of each trajectorys fill region (which indicates '
        '+\\- 2 residual standard deviations about the mean)', default=0.3,
        type=float)

    subject_select = parser.add_mutually_exclusive_group()
    subject_select.add_argument('--subject_ids', help='Comma-separated subject '
        'IDs to overlay. IDs are matched as strings so this works for numeric '
        'or string identifiers.', type=str, default=None)
    subject_select.add_argument('--subject_indices', help='Comma-separated '
        'zero-based subject indices to overlay. Indices refer to the order of '
        'subjects in the fitted data.', type=str, default=None)
    subject_select.add_argument('--sample_n', help='Randomly overlay this many '
        'subjects with MAP assignment to --sample_traj.', type=int, default=None)
    subject_select.add_argument('--sample_pct', help='Randomly overlay this '
        'percentage of subjects with MAP assignment to --sample_traj. '
        'Must be in (0, 100].', type=float, default=None)
    parser.add_argument('--sample_traj', help='Trajectory from which random '
        'subjects will be sampled, based on MAP assignment.', type=int,
        default=None)
    parser.add_argument('--id_col', help='Subject-ID column. By default the '
        'fitted groupby column is used.', type=str, default=None)
    parser.add_argument('--random_seed', help='Optional seed for reproducible '
        'random subject sampling.', type=int, default=None)
    parser.add_argument('--hide_subject_fit', help='Hide the dashed expected '
        'subject-specific trend. This trend is shown only for Gaussian targets '
        'with fitted random effects.', action='store_true')

    op = parser.parse_args()

    set_vals = _parse_set_vals(op.set_vals)
    traj_map = _parse_traj_map(op.traj_map)

    with open(op.model, 'rb') as f:
        mm = pd.read_pickle(f)['MultDPRegression']
        assert op.x_axis in mm.predictor_names_, \
            'x-axis variable not among model predictor variables'
        assert op.y_axis in mm.target_names_, \
            'y-axis variable not among model target variables'

        for pred in set_vals:
            assert pred in mm.predictor_names_, \
                f"{pred} from --set_vals not among model predictors"

        subject_requested = any([
            op.subject_ids is not None,
            op.subject_indices is not None,
            op.sample_n is not None,
            op.sample_pct is not None,
        ])
        selected_subjects = None
        id_col = None
        df_traj = None
        if subject_requested:
            id_col = _get_id_col(mm, op.id_col)
            df_traj = mm.to_df()
            subject_df = _get_subject_table(mm, df_traj, id_col)
            selected_subjects = _select_subjects(subject_df, op)

        # Subject overlays must be drawn before showing an interactive figure,
        # so request an Axes object from mm.plot in that case.
        show = op.fig_file is None and not subject_requested

        traj_markers = None
        if op.traj_markers is not None:
            traj_markers = op.traj_markers.split(',')

        traj_colors = None
        if op.traj_colors is not None:
            traj_colors = op.traj_colors.split(',')

        which_trajs = None
        if op.trajs is not None:
            which_trajs = np.array(op.trajs.split(','), dtype=int)

        ax = mm.plot(op.x_axis, op.y_axis, op.x_label, op.y_label,
                     which_trajs=which_trajs,
                     show=show, min_traj_prob=op.min_traj_prob,
                     max_traj_prob=op.max_traj_prob, traj_map=traj_map,
                     hide_scatter=op.hs, hide_traj_details=op.htd,
                     traj_markers=traj_markers, traj_colors=traj_colors,
                     fill_alpha=op.fill_alpha, set_vals=set_vals)

        if selected_subjects is not None:
            if df_traj is None:
                df_traj = mm.to_df()
            _add_subject_overlays(
                ax, mm, df_traj, selected_subjects, id_col,
                op.x_axis, op.y_axis, traj_map, op.hide_subject_fit
            )
            # mm.plot() creates its legend before subject overlays are added.
            ax.legend()

        if op.ylim is not None:
            plt.ylim(float(op.ylim.strip('--').split(',')[0]),
                     float(op.ylim.strip('--').split(',')[1]))
        if op.xlim is not None:
            plt.xlim(float(op.xlim.strip('--').split(',')[0]),
                     float(op.xlim.strip('--').split(',')[1]))

        if subject_requested and op.fig_file is None:
            plt.show()

        if op.fig_file is not None:
            print("Saving figure...")
            plt.savefig(op.fig_file)
            if write_provenance_data is not None:
                print("Writing provenance info...")
                write_provenance_data(op.fig_file, generator_args=op, desc=""" """,
                                      module_name='bayes_traj')
            else:
                print("Skipping provenance info; provenance_tools is not installed.")
            print("DONE.")


if __name__ == "__main__":
    main()
