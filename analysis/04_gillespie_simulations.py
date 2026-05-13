"""
Gillespie Simulation — Theorem 2 and Aggregation Fixed Point (Theorem 3)
=========================================================================
Shannon T. Wong (2026)

Validates the analytical derivations with individual-based simulations:
  - Theorem 2: fitted c2 recovers true σ²_L across parameter space
  - Theorem 3: b(m) → 2 as aggregation level m increases

These simulations require no external data.
"""

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. Stationary-distribution sampler (vectorised Gillespie approximation)
# =============================================================================

def gillespie_batch(K_array, lam=1.2, mu=0.2, rng=None):
    """
    Sample from the stationary distribution of the logistic birth-death process.

    Uses the van Kampen approximation:
      N | K ~ Normal(K, K*c1)  for K >= 5
      N | K ~ Poisson(K)       for K < 5

    where c1 = (λ + μ + r) / (2r) exactly from Theorem 1.

    Parameters
    ----------
    K_array : array-like
        Carrying capacities per site.
    lam, mu : float
        Birth and death rates (r = lam - mu).
    rng : np.random.Generator, optional

    Returns
    -------
    N_out : ndarray  sampled abundances (clipped to >= 0)
    """
    if rng is None:
        rng = np.random.default_rng()

    K_arr  = np.clip(np.asarray(K_array, dtype=float), 0.5, np.inf)
    r      = lam - mu
    B      = lam + mu + r          # = 2*lam
    c1     = B / (2.0 * r)        # Theorem 1 coefficient

    large  = K_arr >= 5.0
    N_out  = np.zeros(len(K_arr))

    if large.any():
        N_out[large] = rng.normal(K_arr[large],
                                   np.sqrt(K_arr[large] * c1))
    if (~large).any():
        N_out[~large] = rng.poisson(K_arr[~large])

    return np.maximum(N_out, 0.0)


# =============================================================================
# 2. Theorem 2 validation
# =============================================================================

def simulate_theorem2_validation(sigma_L_vals, K_bar_vals,
                                  n_taxa=60, n_sites=100,
                                  lam=1.2, mu=0.2, n_reps=8):
    """
    Validate Theorem 2: fitted c2 should equal σ²_L.

    For each (σ_L, K̄) combination:
      - Generate n_taxa species with K ~ LogNormal(log(K̄), σ_L)
      - For each species, n_sites sites with K_i ~ LogNormal(log(K_b), σ_L)
      - Sample N_i from stationary distribution
      - Fit two-term V/M = c1 + c2*M across species
      - Compare c2 to σ²_L = σ_L²

    Returns
    -------
    DataFrame with columns:
        sigma_L, sigma2_true, K_bar, rep, c1_fit, c2_fit, R2,
        c2_ratio (= c2_fit / sigma2_true)
    """
    r         = lam - mu
    B         = lam + mu + r
    c1_theory = B / (2.0 * r)

    print(f"  Theorem 1: c1_theory = {c1_theory:.4f}  "
          f"(lam={lam}, mu={mu}, r={r})")
    print(f"  Testing σ_L ∈ {sigma_L_vals}")
    print(f"  Testing K̄ ∈ [{K_bar_vals[0]:.1f}, {K_bar_vals[-1]:.1f}]")

    records = []
    rng2    = np.random.default_rng(42)

    for sigma_L in sigma_L_vals:
        sigma2_true = sigma_L**2
        print(f"\n  σ_L = {sigma_L:.1f}  (σ²_L = {sigma2_true:.2f}):")

        for K_bar in K_bar_vals:
            for rep in range(n_reps):
                # Species span ~1 order of magnitude in K̄
                K_taxa = np.logspace(
                    np.log10(K_bar) - 0.5,
                    np.log10(K_bar) + 0.5,
                    n_taxa)

                taxon_means, taxon_vars = [], []
                sigma2_L_obs_list       = []

                for Kb in K_taxa:
                    K_sites = rng2.lognormal(np.log(Kb), sigma_L, n_sites)
                    K_sites = np.clip(K_sites, 0.5, 50_000)
                    N_obs   = gillespie_batch(K_sites, lam, mu, rng=rng2)
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

                # Two-term NNLS fit: V/M = c1 + c2*M
                y    = tv / tm
                X    = np.column_stack([np.ones(len(tm)), tm])
                mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
                coef, _ = nnls(X[mask], y[mask])
                c1_fit, c2_fit = coef[0], coef[1]

                y_hat = c1_fit + c2_fit * tm
                ss_r  = np.sum((y - y_hat)**2)
                ss_t  = np.sum((y - np.mean(y))**2)
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
                    'c2_ratio'    : (c2_fit / sigma2_true
                                    if sigma2_true > 0 else np.nan),
                    'sigma2_L_obs': np.nanmean(sigma2_L_obs_list),
                })

    df = pd.DataFrame(records)
    print(f"\n  Simulation complete: {len(df)} runs")

    # Summary
    valid = df[df['c2_fit'].notna() & (df['c2_fit'] > 0)]
    sl, _, rv, pv, se = stats.linregress(valid['sigma2_true'],
                                          valid['c2_fit'])
    print(f"  Overall: slope={sl:.4f}±{se:.4f}, R²={rv**2:.4f}, p={pv:.2e}")
    print(f"  Mean c2/σ²_L = {valid['c2_ratio'].mean():.3f}  (target: 1.0)")

    return df


# =============================================================================
# 3. Aggregation fixed point (Theorem 3)
# =============================================================================

def simulate_aggregation_fixedpoint(sigma_L=0.5, n_taxa=80, n_sites=300,
                                     m_values=None, lam=1.2, mu=0.2,
                                     K_bar=100):
    """
    Theorem 3: b(m) → 2 under spatial aggregation for any ρ > 0.

    For each aggregation level m:
      - Combine m patches into one super-site
      - Compute Taylor exponent across taxa
      - Verify convergence to b=2

    Returns
    -------
    DataFrame with columns: m, b, se_b, R2
    """
    if m_values is None:
        m_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    print(f"\n  Aggregation fixed point (σ_L={sigma_L}, K̄={K_bar}):")

    r  = lam - mu
    B  = lam + mu + r
    c1 = B / (2 * r)

    K_bars  = np.logspace(np.log10(K_bar) - 1, np.log10(K_bar) + 1, n_taxa)
    records = []
    rng2    = np.random.default_rng(42)

    for m in m_values:
        taxon_means, taxon_vars = [], []

        for Kb in K_bars:
            # Each super-site = aggregate of m patches
            K_patch = rng2.lognormal(
                np.log(max(Kb / m, 0.5)), sigma_L, (n_sites, m))
            K_patch = np.clip(K_patch, 0.5, 50_000)

            # Abundance per patch (Gaussian approximation)
            N_patch = (K_patch +
                       np.sqrt(np.maximum(K_patch, 0)) *
                       rng2.standard_normal((n_sites, m)) * np.sqrt(c1))
            N_patch = np.maximum(N_patch, 0)
            N_site  = N_patch.sum(axis=1)

            taxon_means.append(np.mean(N_site))
            taxon_vars.append(np.var(N_site, ddof=1))

        tm   = np.array(taxon_means)
        tv   = np.array(taxon_vars)
        mask = (tm > 0) & (tv > 0) & np.isfinite(tm) & np.isfinite(tv)
        if mask.sum() < 5:
            continue

        sl, ic, rv, _, se = stats.linregress(np.log(tm[mask]),
                                              np.log(tv[mask]))
        records.append({'m': m, 'b': sl, 'se_b': se, 'R2': rv**2})
        print(f"    m={m:4d}: b = {sl:.4f} ± {se:.4f}  R²={rv**2:.4f}")

    df = pd.DataFrame(records)
    print(f"\n  b at m=1:   {df['b'].iloc[0]:.4f}")
    print(f"  b at m={m}: {df['b'].iloc[-1]:.4f}  (theory: → 2.0)")
    return df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import os
    OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Theorem 2 simulation ──────────────────────────────────────
    print("=" * 65)
    print("THEOREM 2 SIMULATION: c2 ≈ σ²_L")
    print("=" * 65)

    sigma_L_vals = np.array([0.3, 0.5, 0.6, 0.8, 1.0])
    K_bar_vals   = np.logspace(0.5, 3.0, 10)

    sim_df = simulate_theorem2_validation(
        sigma_L_vals, K_bar_vals,
        n_taxa=60, n_sites=100, n_reps=8)

    sim_df.to_csv(os.path.join(OUT_DIR, 'theorem2_simulation.csv'),
                 index=False)
    print(f"\nSaved: theorem2_simulation.csv")

    # ── Theorem 3 simulation ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("THEOREM 3 SIMULATION: AGGREGATION FIXED POINT")
    print("=" * 65)

    agg_df = simulate_aggregation_fixedpoint(
        sigma_L=0.5, n_taxa=80, n_sites=300, K_bar=100,
        m_values=[1, 2, 4, 8, 16, 32, 64, 128, 256])

    agg_df.to_csv(os.path.join(OUT_DIR, 'aggregation_fixedpoint.csv'),
                 index=False)
    print(f"Saved: aggregation_fixedpoint.csv")
