import os
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR  = os.path.join(os.path.dirname(__file__), '..')
OUT_DIR   = os.path.join(ROOT_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

EXTINCTION_CSV = os.path.join(OUT_DIR, 'prospective_extinction.csv')
FITS_CSV       = os.path.join(OUT_DIR, 'study_fits.csv')
OUTPUT_CSV     = os.path.join(OUT_DIR, 'kc1_sensitivity.csv')


def run_extinction_test(df, label=''):
    """
    Given a dataframe with columns [went_extinct, mean_N, sigma2_L],
    residualise sigma2_L against mean_N (abundance control), split
    at median residual, run Mann-Whitney. Returns dict of results.
    """
    sub = df[(df['mean_N'] > 0) & (df['sigma2_L'] > 0)].copy()

    slope, intercept, *_ = stats.linregress(
        np.log(sub['mean_N']), np.log(sub['sigma2_L']))
    sub['resid'] = (np.log(sub['sigma2_L']) -
                    (slope * np.log(sub['mean_N']) + intercept))

    low  = sub[sub['resid'] <  sub['resid'].median()]
    high = sub[sub['resid'] >= sub['resid'].median()]

    _, p = stats.mannwhitneyu(
        low['went_extinct'].astype(int),
        high['went_extinct'].astype(int),
        alternative='two-sided')

    rate_low  = low['went_extinct'].mean()
    rate_high = high['went_extinct'].mean()
    rel_red   = (rate_low - rate_high) / rate_low if rate_low > 0 else np.nan

    if label:
        print(f"\n  {label}")
        print(f"    n pairs:        {len(sub):,}")
        print(f"    extinctions:    {sub['went_extinct'].sum():,}")
        print(f"    rate low:       {rate_low:.3f}")
        print(f"    rate high:      {rate_high:.3f}")
        print(f"    rel reduction:  {rel_red:.1%}")
        print(f"    Mann-Whitney p: {p:.4f}")

    return {
        'n_pairs'      : len(sub),
        'extinctions'  : int(sub['went_extinct'].sum()),
        'rate_low'     : round(rate_low, 3),
        'rate_high'    : round(rate_high, 3),
        'rel_reduction': round(rel_red, 3),
        'p'            : round(p, 4),
    }


def main():
    if not os.path.exists(EXTINCTION_CSV):
        raise FileNotFoundError(
            f"{EXTINCTION_CSV} not found. "
            "Run 02_local_extinction.py first.")

    if not os.path.exists(FITS_CSV):
        raise FileNotFoundError(
            f"{FITS_CSV} not found. "
            "Run 01_core_analysis.py first.")

    print("Loading prospective extinction dataset...")
    ext_df = pd.read_csv(EXTINCTION_CSV)
    print(f"  {len(ext_df):,} species-study pairs loaded")

    print("\nLoading study fits (c1 estimates)...")
    fits = pd.read_csv(FITS_CSV)
    fits_valid = fits[
        fits['c1'].notna() &
        (fits['c1'] > 0) &
        (fits['c1'] < 1e4)
    ][['STUDY_ID', 'c1']].copy()
    print(f"  {len(fits_valid):,} studies with valid c1 estimates")

    print("\n── Full sample (baseline) ───────────────────────────────────────────")
    baseline = run_extinction_test(ext_df, label='Full prospective sample')

    print("\n── Merging c1 into extinction pairs ─────────────────────────────────")
    ext_df2 = ext_df.merge(fits_valid, on='STUDY_ID', how='inner')

    ext_df2['K_over_c1'] = ext_df2['mean_N'] / ext_df2['c1']

    print(f"  Pairs with c1 match:      {len(ext_df2):,}")
    print(f"  Median K/c1:              {ext_df2['K_over_c1'].median():.2f}")
    print(f"  % pairs with K/c1 >= 2:  "
          f"{(ext_df2['K_over_c1'] >= 2).mean()*100:.1f}%")
    print(f"  % pairs with K/c1 >= 5:  "
          f"{(ext_df2['K_over_c1'] >= 5).mean()*100:.1f}%")
    print(f"  % pairs with K/c1 >= 10: "
          f"{(ext_df2['K_over_c1'] >= 10).mean()*100:.1f}%")

    print(f"\n  epsilon_rel = c1/(2K) at median K/c1 = "
          f"{ext_df2['K_over_c1'].median():.2f}: "
          f"{1/(2*ext_df2['K_over_c1'].median()):.1%}")

    print("\n── K >> c1 Sensitivity Analysis ─────────────────────────────────────")

    thresholds = [
        (2,  '<=25%'),
        (5,  '<=10%'),
        (10, '<=5%'),
    ]

    rows = []

    # Add full-sample row first
    rows.append({
        'kappa'        : 'Full sample',
        'eps_rel'      : '---',
        **baseline,
    })

    for kappa, eps_label in thresholds:
        sub = ext_df2[ext_df2['K_over_c1'] >= kappa].copy()

        if len(sub) < 50:
            print(f"\n  kappa={kappa}: insufficient pairs ({len(sub)}), skipping")
            continue

        result = run_extinction_test(
            sub,
            label=f"kappa >= {kappa}  (epsilon_rel {eps_label})"
        )

        rows.append({
            'kappa'  : f'>= {kappa}',
            'eps_rel': eps_label,
            **result,
        })

    sensitivity_df = pd.DataFrame(rows)
    sensitivity_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n── Summary table ────────────────────────────────────────────────────")
    print(sensitivity_df.to_string(index=False))
    print(f"\nSaved: {OUTPUT_CSV}")

    print("\n── Interpretation ───────────────────────────────────────────────────")
    full_red = baseline['rel_reduction']
    max_red  = max(r['rel_reduction'] for r in rows[1:] if r['rel_reduction'])
    print(f"  Full sample relative reduction:     {full_red:.1%}")
    print(f"  Peak reduction (restricted subset): {max_red:.1%}")

    if max_red > full_red:
        print(f"\n  Signal AMPLIFIES under restriction — consistent with a genuine")
        print(f"  mechanistic relationship. Noise attenuates; mechanism amplifies.")
    else:
        print(f"\n  Signal does not amplify under restriction.")

    all_sig = all(r['p'] < 0.05 for r in rows[1:])
    if all_sig:
        print(f"\n  All restricted subsets significant at p < 0.05.")
        print(f"  The main result does not depend on low-K/c1 pairs where")
        print(f"  the van Kampen approximation is least accurate.")


if __name__ == '__main__':
    main()
