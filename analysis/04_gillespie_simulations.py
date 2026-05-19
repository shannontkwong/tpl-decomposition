import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def gillespie_logistic_stationary(K, lam=1.2, mu=0.2, t_burn=500.0,
                                   t_sample=2000.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    K   = max(float(K), 1.0)
    r   = lam - mu
    N   = max(1, int(round(K)))
    t   = 0.0
    t_target = t_burn + rng.uniform(0.0, t_sample)

    while True:
        if N == 0:
            N = max(1, int(round(K)))

        birth_rate = lam * N
        death_rate = mu * N + r * N * (N - 1) / K
        total_rate = birth_rate + death_rate

        if total_rate <= 0.0:
            break

        dt = rng.exponential(1.0 / total_rate)

        if t + dt >= t_target:
            return N

        t += dt
        if rng.random() < birth_rate / total_rate:
            N += 1
        else:
            N = max(0, N - 1)

    return N


def gillespie_batch(K_array, lam=1.2, mu=0.2, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    K_arr = np.asarray(K_array, dtype=float)
    N_out = np.zeros(len(K_arr), dtype=float)
    r     = lam - mu
    B     = lam + mu + r
    c1    = B / (2.0 * r)

    exact  = K_arr < 50
    approx = ~exact

    for i in np.where(exact)[0]:
        N_out[i] = gillespie_logistic_stationary(
            K_arr[i], lam=lam, mu=mu, rng=rng)

    if approx.any():
        Kv = K_arr[approx]
        N_out[approx] = np.maximum(rng.normal(Kv, np.sqrt(Kv * c1)), 0.0)

    return N_out


def validate_gaussian_approximation(K_vals=(5, 10, 20, 50, 100, 200, 500),
                                     n_draws=2000, lam=1.2, mu=0.2, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    r  = lam - mu
    B  = lam + mu + r
    c1 = B / (2.0 * r)

    print(f"c1_theory = {c1:.4f}")
    print(f"{'K':>6}  {'E[N]_sim':>10}  {'Var_theory':>12}  {'Var_sim':>10}  {'ratio':>7}")

    records = []
    for K in K_vals:
        samples = np.array([
            gillespie_logistic_stationary(K, lam=lam, mu=mu, rng=rng)
            for _ in range(n_draws)
        ])
        E_sim   = samples.mean()
        Var_sim = samples.var(ddof=1)
        Var_th  = c1 * K
        ratio   = Var_sim / Var_th if Var_th > 0 else np.nan
        print(f"{K:>6}  {E_sim:>10.2f}  {Var_th:>12.4f}  {Var_sim:>10.4f}  {ratio:>7.3f}")
        records.append({'K': K, 'E_sim': E_sim,
                        'Var_theory': Var_th, 'Var_sim': Var_sim, 'ratio': ratio})

    return pd.DataFrame(records)


def simulate_theorem2_validation(sigma_L_vals, K_bar_vals,
                                  n_taxa=60, n_sites=100,
                                  lam=1.2, mu=0.2, n_reps=8, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    r         = lam - mu
    B         = lam + mu + r
    c1_theory = B / (2.0 * r)

    records = []

    for sigma_L in sigma_L_vals:
        sigma2_true = sigma_L ** 2

        for K_bar in K_bar_vals:
            for rep in range(n_reps):

                K_taxa = np.logspace(
                    np.log10(K_bar) - 0.5,
                    np.log10(K_bar) + 0.5,
                    n_taxa)

                taxon_means       = []
                taxon_vars        = []
                sigma2_L_obs_list = []

                for Kb in K_taxa:
                    K_sites = rng.lognormal(np.log(Kb), sigma_L, n_sites)
                    K_sites = np.clip(K_sites, 0.5, 50_000)
                    N_obs   = gillespie_batch(K_sites, lam=lam, mu=mu, rng=rng)
                    N_obs   = N_obs[N_obs > 0]
                    if len(N_obs) < 5:
                        continue
                    taxon_means.append(np.mean(N_obs))
                    taxon_vars.append(np.var(N_obs, ddof=1))
                    sigma2_L_obs_list.append(np.var(np.log(N_obs), ddof=1))

                if len(taxon_means) < 8:
                    continue

                tm = np.array(taxon_means)
                tv = np.array(taxon_vars)

                y    = tv / tm
                X    = np.column_stack([np.ones(len(tm)), tm])
                mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
                coef, _ = nnls(X[mask], y[mask])
                c1_fit, c2_fit = coef[0], coef[1]

                y_hat = c1_fit + c2_fit * tm
                ss_r  = np.sum((y - y_hat) ** 2)
                ss_t  = np.sum((y - np.mean(y)) ** 2)
                R2    = 1 - ss_r / ss_t if ss_t > 0 else 0.0

                records.append({
                    'sigma_L'     : sigma_L,
                    'sigma2_true' : sigma2_true,
                    'K_bar'       : K_bar,
                    'rep'         : rep,
                    'c1_fit'      : c1_fit,
                    'c1_theory'   : c1_theory,
                    'c2_fit'      : c2_fit,
                    'R2'          : R2,
                    'c2_ratio'    : (c2_fit / sigma2_true if sigma2_true > 0 else np.nan),
                    'sigma2_L_obs': float(np.nanmean(sigma2_L_obs_list)),
                })

    df    = pd.DataFrame(records)
    valid = df[df['c2_fit'].notna() & (df['c2_fit'] > 0)]
    sl, _, rv, pv, se = stats.linregress(valid['sigma2_true'], valid['c2_fit'])
    print(f"\nTheorem 2: slope={sl:.4f}+/-{se:.4f}, R^2={rv**2:.4f}, p={pv:.2e}")
    print(f"Mean c2/sigma2_L = {valid['c2_ratio'].mean():.3f}")
    return df


def simulate_aggregation_fixedpoint(sigma_L=0.5, n_taxa=80, n_sites=300,
                                     m_values=None, lam=1.2, mu=0.2,
                                     K_bar=100, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    if m_values is None:
        m_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    r  = lam - mu
    B  = lam + mu + r
    c1 = B / (2 * r)

    m_max  = max(m_values)
    K_bars = np.logspace(np.log10(K_bar) - 1, np.log10(K_bar) + 1, n_taxa)

    N_patches = np.zeros((n_taxa, n_sites, m_max))
    for ti, Kb in enumerate(K_bars):
        patch_K = rng.lognormal(np.log(max(Kb / m_max, 0.5)),
                                sigma_L, (n_sites, m_max))
        patch_K = np.clip(patch_K, 0.5, 50_000)
        for si in range(n_sites):
            N_patches[ti, si, :] = gillespie_batch(
                patch_K[si, :], lam=lam, mu=mu, rng=rng)

    records = []
    for m in m_values:
        N_super     = N_patches[:, :, :m].sum(axis=2)
        taxon_means = N_super.mean(axis=1)
        taxon_vars  = N_super.var(axis=1, ddof=1)

        mask = ((taxon_means > 0) & (taxon_vars > 0) &
                np.isfinite(taxon_means) & np.isfinite(taxon_vars))
        if mask.sum() < 5:
            continue

        sl, _, rv, _, se = stats.linregress(
            np.log(taxon_means[mask]), np.log(taxon_vars[mask]))
        records.append({'m': m, 'b': sl, 'se_b': se, 'R2': rv ** 2})
        print(f"  m={m:4d}: b={sl:.4f}+/-{se:.4f}  R^2={rv**2:.4f}")

    return pd.DataFrame(records)


if __name__ == '__main__':
    import os

    OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(OUT_DIR, exist_ok=True)

    RNG = np.random.default_rng(seed=42)

    print("=== Gaussian approximation validation ===")
    val_df = validate_gaussian_approximation(
        K_vals=(5, 10, 20, 50, 100, 200, 500),
        n_draws=2000, lam=1.2, mu=0.2, rng=RNG)
    val_df.to_csv(
        os.path.join(OUT_DIR, 'gaussian_approx_validation.csv'), index=False)

    print("\n=== Theorem 2 simulation ===")
    sigma_L_vals = np.array([0.3, 0.5, 0.6, 0.8, 1.0])
    K_bar_vals   = np.logspace(0.5, 3.0, 10)
    sim_df = simulate_theorem2_validation(
        sigma_L_vals, K_bar_vals,
        n_taxa=60, n_sites=100, n_reps=8,
        lam=1.2, mu=0.2, rng=RNG)
    sim_df.to_csv(
        os.path.join(OUT_DIR, 'theorem2_simulation.csv'), index=False)

    print("\n=== Theorem 3 simulation: aggregation fixed point ===")
    agg_df = simulate_aggregation_fixedpoint(
        sigma_L=0.5, n_taxa=80, n_sites=300, K_bar=100,
        m_values=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        lam=1.2, mu=0.2, rng=RNG)
    agg_df.to_csv(
        os.path.join(OUT_DIR, 'aggregation_fixedpoint.csv'), index=False)
