import sys
import pickle
import numpy as np
import pandas as pd
import pytest

from bayes_traj.bayes_traj_main import main

def test_bayes_traj_main_old_prior_backward_compatible(tmp_path, monkeypatch):
    import sys
    import pickle
    import numpy as np
    import pandas as pd

    from bayes_traj.bayes_traj_main import main

    # ------------------------------------------------------------------
    # Create tiny data set
    # ------------------------------------------------------------------
    df = pd.DataFrame({
        'id': ['a', 'a', 'b', 'b'],
        'intercept': [1.0, 1.0, 1.0, 1.0],
        'time': [0.0, 1.0, 0.0, 1.0],
        'y': [0.0, 1.0, 0.5, 1.5]
    })

    in_csv = tmp_path / "data.csv"
    df.to_csv(in_csv, index=False)

    # ------------------------------------------------------------------
    # Old-style prior file: no shared_predictors block
    # ------------------------------------------------------------------
    prior_info = {
        'alpha': 1.0,
        'lambda_a0': {'y': 1.0},
        'lambda_b0': {'y': 1.0},
        'w_mu0': {'y': {'intercept': 0.0, 'time': 0.0}},
        'w_var0': {'y': {'intercept': 1.0, 'time': 1.0}},
        'w_mu': None,
        'w_var': None,
        'lambda_a': None,
        'lambda_b': None,
        'traj_probs': None,
        'v_a': None,
        'v_b': None,
        'Sig0': None,
        'ranefs': None,
        'ranef_indices': None
    }

    prior_file = tmp_path / "prior_old.pkl"
    with open(prior_file, "wb") as f:
        pickle.dump(prior_info, f)

    out_model = tmp_path / "model.pkl"

    argv = [
        "bayes_traj_main.py",
        "--in_csv", str(in_csv),
        "--targets", "y",
        "--groupby", "id",
        "--prior", str(prior_file),
        "--out_model", str(out_model),
        "--iters", "2",
        "-k", "4",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    main()

    assert out_model.exists()

    with open(out_model, "rb") as f:
        saved = pickle.load(f)

    mm = saved['MultDPRegression']

    assert mm.predictor_names_ == ['intercept', 'time']
    assert mm.shared_predictor_names_ == []
    assert mm.num_shared_preds_ == 0
    assert mm.num_traj_preds_ == 2

def test_bayes_traj_main_reads_shared_predictor_prior(tmp_path, monkeypatch):
    import sys
    import pickle
    import numpy as np
    import pandas as pd

    from bayes_traj.bayes_traj_main import main

    # ------------------------------------------------------------------
    # Create tiny data set
    # ------------------------------------------------------------------
    df = pd.DataFrame({
        'id': ['a', 'a', 'b', 'b'],
        'intercept': [1.0, 1.0, 1.0, 1.0],
        'time': [0.0, 1.0, 0.0, 1.0],
        'cohort': [0.0, 0.0, 1.0, 1.0],
        'y': [0.0, 1.0, 2.5, 3.5]
    })

    in_csv = tmp_path / "data.csv"
    df.to_csv(in_csv, index=False)

    # ------------------------------------------------------------------
    # New-style prior file with shared predictor
    # ------------------------------------------------------------------
    prior_info = {
        'alpha': 1.0,
        'lambda_a0': {'y': 1.0},
        'lambda_b0': {'y': 1.0},
        'w_mu0': {'y': {'intercept': 0.0, 'time': 0.0}},
        'w_var0': {'y': {'intercept': 1.0, 'time': 1.0}},
        'shared_predictors': ['cohort'],
        'w_mu0_shared': {'y': {'cohort': 1.5}},
        'w_var0_shared': {'y': {'cohort': 0.25}},
        'w_mu': None,
        'w_var': None,
        'lambda_a': None,
        'lambda_b': None,
        'traj_probs': None,
        'v_a': None,
        'v_b': None,
        'Sig0': None,
        'ranefs': None,
        'ranef_indices': None
    }

    prior_file = tmp_path / "prior_shared.pkl"
    with open(prior_file, "wb") as f:
        pickle.dump(prior_info, f)

    out_model = tmp_path / "model.pkl"

    argv = [
        "bayes_traj_main.py",
        "--in_csv", str(in_csv),
        "--targets", "y",
        "--groupby", "id",
        "--prior", str(prior_file),
        "--out_model", str(out_model),
        "--iters", "2",
        "-k", "4",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    main()

    assert out_model.exists()

    with open(out_model, "rb") as f:
        saved = pickle.load(f)

    mm = saved['MultDPRegression']

    assert mm.predictor_names_ == ['intercept', 'time', 'cohort']
    assert mm.shared_predictor_names_ == ['cohort']
    assert mm.num_shared_preds_ == 1
    assert mm.num_traj_preds_ == 2

    assert np.array_equal(mm.shared_indices_, np.array([2]))
    assert np.array_equal(mm.traj_indices_, np.array([0, 1]))

    assert hasattr(mm, 'w_mu0_shared_')
    assert hasattr(mm, 'w_var0_shared_')
    assert mm.w_mu0_shared_.shape == (1, 1)
    assert mm.w_var0_shared_.shape == (1, 1)

    assert np.isclose(mm.w_mu0_shared_[0, 0].item(), 1.5)
    assert np.isclose(mm.w_var0_shared_[0, 0].item(), 0.25)

def test_bayes_traj_main_shared_predictor_ordering(tmp_path, monkeypatch):
    import sys
    import pickle
    import pandas as pd

    from bayes_traj.bayes_traj_main import main

    df = pd.DataFrame({
        'id': ['a', 'a'],
        'intercept': [1.0, 1.0],
        'time': [0.0, 1.0],
        'cohort': [1.0, 1.0],
        'y': [2.0, 3.0]
    })

    in_csv = tmp_path / "data.csv"
    df.to_csv(in_csv, index=False)

    prior_info = {
        'alpha': 1.0,
        'lambda_a0': {'y': 1.0},
        'lambda_b0': {'y': 1.0},
        'w_mu0': {'y': {'intercept': 0.0, 'time': 0.0}},
        'w_var0': {'y': {'intercept': 1.0, 'time': 1.0}},
        'shared_predictors': ['cohort'],
        'w_mu0_shared': {'y': {'cohort': 2.0}},
        'w_var0_shared': {'y': {'cohort': 0.5}},
        'w_mu': None,
        'w_var': None,
        'lambda_a': None,
        'lambda_b': None,
        'traj_probs': None,
        'v_a': None,
        'v_b': None,
        'Sig0': None,
        'ranefs': None,
        'ranef_indices': None
    }

    prior_file = tmp_path / "prior.pkl"
    with open(prior_file, "wb") as f:
        pickle.dump(prior_info, f)

    out_model = tmp_path / "model.pkl"

    argv = [
        "bayes_traj_main.py",
        "--in_csv", str(in_csv),
        "--targets", "y",
        "--groupby", "id",
        "--prior", str(prior_file),
        "--out_model", str(out_model),
        "--iters", "1",
        "-k", "3",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    main()

    with open(out_model, "rb") as f:
        saved = pickle.load(f)

    mm = saved['MultDPRegression']

    assert mm.predictor_names_[-1] == 'cohort'
    assert mm.shared_indices_[0] == len(mm.predictor_names_) - 1

def test_bayes_traj_main_old_prior_has_empty_shared_block(tmp_path, monkeypatch):
    import sys
    import pickle
    import pandas as pd

    from bayes_traj.bayes_traj_main import main

    df = pd.DataFrame({
        'id': ['a', 'a'],
        'intercept': [1.0, 1.0],
        'time': [0.0, 1.0],
        'y': [0.0, 1.0]
    })

    in_csv = tmp_path / "data.csv"
    df.to_csv(in_csv, index=False)

    prior_info = {
        'alpha': 1.0,
        'lambda_a0': {'y': 1.0},
        'lambda_b0': {'y': 1.0},
        'w_mu0': {'y': {'intercept': 0.0, 'time': 0.0}},
        'w_var0': {'y': {'intercept': 1.0, 'time': 1.0}},
        'w_mu': None,
        'w_var': None,
        'lambda_a': None,
        'lambda_b': None,
        'traj_probs': None,
        'v_a': None,
        'v_b': None,
        'Sig0': None,
        'ranefs': None,
        'ranef_indices': None
    }

    prior_file = tmp_path / "prior.pkl"
    with open(prior_file, "wb") as f:
        pickle.dump(prior_info, f)

    out_model = tmp_path / "model.pkl"

    argv = [
        "bayes_traj_main.py",
        "--in_csv", str(in_csv),
        "--targets", "y",
        "--groupby", "id",
        "--prior", str(prior_file),
        "--out_model", str(out_model),
        "--iters", "1",
        "-k", "3",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    main()

    with open(out_model, "rb") as f:
        saved = pickle.load(f)

    mm = saved['MultDPRegression']

    assert mm.shared_predictor_names_ == []
    assert mm.num_shared_preds_ == 0    
