"""Block-bootstrap CIs and Diebold-Mariano test on the 2025 holdout."""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
N_BOOT = 2000
RNG = np.random.default_rng(42)


def load_all_predictions() -> pd.DataFrame:
    param_path = ROOT / 'src/data/forecasting_parameters/holdout_predictions_detail.csv'
    gam_path = ROOT / 'src/data/forecasting_gam/gam_holdout_predictions.csv'

    dfs = []
    if param_path.exists():
        dfs.append(pd.read_csv(param_path))
    if gam_path.exists():
        dfs.append(pd.read_csv(gam_path))

    if not dfs:
        raise FileNotFoundError(
            f"Neither {param_path} nor {gam_path} exists. "
            "Please run parameter_forecaster.py first to generate holdout predictions."
        )

    df = pd.concat(dfs, ignore_index=True)
    df['sq_err'] = (df['Actual_WQI'] - df['Predicted_WQI']) ** 2
    df['abs_err'] = (df['Actual_WQI'] - df['Predicted_WQI']).abs()
    return df


def block_bootstrap_rmse_ci(df: pd.DataFrame, model: str, n_boot: int = N_BOOT) -> dict:
    """
    Resample WHOLE LOCATIONS with replacement (not individual rows) so
    that a location's correlated errors move together -- this is the
    'blocked by location' resampling required for spatial panels.
    Vectorized: precompute each location's sq_err array once, then just
    concatenate numpy arrays per bootstrap draw (the naive version re-filtered
    the DataFrame ~n_boot * n_locations times and effectively never finished).
    """
    sub = df[df['model'] == model]
    locations = sub['Location'].unique()
    by_loc = {loc: sub.loc[sub['Location'] == loc, 'sq_err'].to_numpy() for loc in locations}
    boot_rmses = np.empty(n_boot)
    for i in range(n_boot):
        sampled_locs = RNG.choice(locations, size=len(locations), replace=True)
        pooled = np.concatenate([by_loc[loc] for loc in sampled_locs])
        boot_rmses[i] = np.sqrt(pooled.mean())
    return {
        'model': model,
        'point_rmse': float(np.sqrt(sub['sq_err'].mean())),
        'ci_2.5': float(np.percentile(boot_rmses, 2.5)),
        'ci_97.5': float(np.percentile(boot_rmses, 97.5)),
        'n_locations': len(locations),
        'n_rows': len(sub),
    }


def diebold_mariano(df: pd.DataFrame, model_a: str, model_b: str, h: int = 1) -> dict:
    """
    DM test on squared-error loss differential, aligned by (Date, Location).
    Uses small-sample correction from Harvey, Leybourne & Newbold (1997).
    """
    a = df[df['model'] == model_a].set_index(['Date', 'Location'])['sq_err']
    b = df[df['model'] == model_b].set_index(['Date', 'Location'])['sq_err']
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return {'model_a': model_a, 'model_b': model_b, 'n_pairs': 0, 'dm_stat': np.nan, 'p_value': np.nan}

    d = (a.loc[common] - b.loc[common]).values
    n = len(d)
    d_bar = d.mean()
    gamma0 = np.var(d, ddof=0)
    if gamma0 == 0:
        return {'model_a': model_a, 'model_b': model_b, 'n_pairs': n, 'dm_stat': 0.0, 'p_value': 1.0}

    dm_stat = d_bar / np.sqrt(gamma0 / n)
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat_corrected = dm_stat * correction
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat_corrected), df=n - 1))
    return {
        'model_a': model_a,
        'model_b': model_b,
        'n_pairs': n,
        'dm_stat': float(dm_stat_corrected),
        'p_value': float(p_value),
    }


def main():
    out_dir = ROOT / 'reproducibility_results'
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_predictions()
    models = df['model'].unique().tolist()
    ci_rows = [block_bootstrap_rmse_ci(df, m) for m in models]
    ci_df = pd.DataFrame(ci_rows)
    ci_path = out_dir / 'holdout_bootstrap_ci.csv'
    ci_df.to_csv(ci_path, index=False)
    print(f"Saved bootstrap CIs to {ci_path}")
    print(ci_df.to_string(index=False))

    baseline = 'Persistence'
    if baseline in models:
        dm_rows = [diebold_mariano(df, baseline, m) for m in models if m != baseline]
        dm_df = pd.DataFrame(dm_rows)
        dm_path = out_dir / 'holdout_dm_test_vs_persistence.csv'
        dm_df.to_csv(dm_path, index=False)
        print(f"\nSaved Diebold-Mariano tests vs Persistence to {dm_path}")
        print(dm_df.to_string(index=False))


if __name__ == '__main__':
    main()
