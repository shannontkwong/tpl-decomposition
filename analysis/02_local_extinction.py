import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import nnls
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')


def build_extinction_dataset(bt, min_years=8, min_sites=3, sigma2_L_floor=0.01):
    col_map = {c.upper(): c for c in bt.columns}
    bt = bt.rename(columns={v: k for k, v in col_map.items()
                             if k in ['STUDY_ID', 'VALID_NAME', 'SAMPLE_DESC',
                                      'ABUNDANCE', 'YEAR', 'TAXON']})
    if 'VALID_NAME' not in bt.columns and 'valid_name' in bt.columns:
        bt = bt.rename(columns={'valid_name': 'VALID_NAME'})

    records = []

    for sid, grp in bt.groupby('STUDY_ID'):
        years = sorted(grp['YEAR'].unique())
        if len(years) < min_years:
            continue

        n           = len(years)
        early_years = years[:n // 3]
        late_years  = years[2 * n // 3:]

        early_sp = set(grp[grp['YEAR'].isin(early_years)]['VALID_NAME'].unique())
        late_sp  = set(grp[grp['YEAR'].isin(late_years)]['VALID_NAME'].unique())
        gone     = early_sp - late_sp
        stayed   = early_sp & late_sp

        if len(gone) < 2 or len(stayed) < 4:
            continue

        realm = (grp['TAXON'].mode()[0] if 'TAXON' in grp.columns else 'Unknown')

        for sp in list(gone) + list(stayed):
            sp_early = grp[grp['VALID_NAME'].eq(sp) & grp['YEAR'].isin(early_years)]
            sites    = sp_early.groupby('SAMPLE_DESC')['ABUNDANCE'].mean()
            sites    = sites[sites > 0]

            if len(sites) < min_sites:
                continue

            sigma2_L = np.var(np.log(sites.values), ddof=1)
            if sigma2_L <= sigma2_L_floor:
                continue

            records.append({
                'STUDY_ID'    : sid,
                'species'     : sp,
                'went_extinct': sp in gone,
                'mean_N'      : float(sites.mean()),
                'sigma2_L'    : float(sigma2_L),
                'n_sites'     : len(sites),
                'realm'       : realm,
            })

    ext_df = pd.DataFrame(records).dropna()
    print(f"  Species-study pairs: {len(ext_df):,}")
    print(f"  Observed local extinctions: {ext_df['went_extinct'].sum():,}")
    return ext_df


def abundance_control(ext_df):
    df    = ext_df.copy()
    df    = df[(df['mean_N'] > 0) & (df['sigma2_L'] > 0)].copy()
    log_s = np.log(df['sigma2_L'])
    log_n = np.log(df['mean_N'])
    slope, intercept, *_ = stats.linregress(log_n, log_s)
    df['sigma2_L_resid']  = log_s - (slope * log_n + intercept)
    print(f"  OLS slope log(σ²_L) ~ log(N̄): {slope:.3f}")
    return df, slope


def main_extinction_test(ext_df):
    print("\n=== MAIN EXTINCTION RESULT ===")
    median_resid = ext_df['sigma2_L_resid'].median()
    low  = ext_df[ext_df['sigma2_L_resid'] <  median_resid]
    high = ext_df[ext_df['sigma2_L_resid'] >= median_resid]

    rate_low  = low['went_extinct'].mean()
    rate_high = high['went_extinct'].mean()

    U, p = stats.mannwhitneyu(
        low['went_extinct'].astype(int),
        high['went_extinct'].astype(int),
        alternative='two-sided')

    print(f"  n pairs total:       {len(ext_df):,}")
    print(f"  Extinctions total:   {ext_df['went_extinct'].sum():,}")
    print(f"  Extinction rate low  σ²_L: {rate_low:.3f}  ({rate_low*100:.1f}%)")
    print(f"  Extinction rate high σ²_L: {rate_high:.3f}  ({rate_high*100:.1f}%)")
    print(f"  Relative reduction:  {(1 - rate_high/rate_low)*100:.1f}%")
    print(f"  Mann-Whitney p = {p:.4f}")
    return p


def robustness_extinction_definitions(bt, min_sites=3, sigma2_L_floor=0.01):
    print("\n=== TABLE 1: ROBUSTNESS ACROSS EXTINCTION DEFINITIONS ===")

    definitions = [
        ('Original (first/last thirds)',
         lambda years: (years[:len(years)//3], years[2*len(years)//3:])),
        ('Strict (last half absent)',
         lambda years: (years[:len(years)//3], years[len(years)//2:])),
        ('Conservative (last quarter)',
         lambda years: (years[:len(years)//4], years[3*len(years)//4:])),
    ]

    if 'VALID_NAME' not in bt.columns and 'valid_name' in bt.columns:
        bt = bt.rename(columns={'valid_name': 'VALID_NAME'})

    print(f"  {'Definition':<35} {'n pairs':>8} {'Extinctions':>12} {'p':>8}")
    print("  " + "-" * 65)

    for def_name, split_fn in definitions:
        records = []
        for sid, grp in bt.groupby('STUDY_ID'):
            years = sorted(grp['YEAR'].unique())
            if len(years) < 8:
                continue
            early_years, late_years = split_fn(years)

            early_sp = set(grp[grp['YEAR'].isin(early_years)]['VALID_NAME'].unique())
            late_sp  = set(grp[grp['YEAR'].isin(late_years)]['VALID_NAME'].unique())
            gone     = early_sp - late_sp
            stayed   = early_sp & late_sp

            if len(gone) < 2 or len(stayed) < 4:
                continue

            for sp in list(gone) + list(stayed):
                sp_early = grp[grp['VALID_NAME'].eq(sp) & grp['YEAR'].isin(early_years)]
                sites    = sp_early.groupby('SAMPLE_DESC')['ABUNDANCE'].mean()
                sites    = sites[sites > 0]
                if len(sites) < min_sites:
                    continue
                sigma2_L = np.var(np.log(sites.values), ddof=1)
                if sigma2_L <= sigma2_L_floor:
                    continue
                records.append({
                    'went_extinct': sp in gone,
                    'mean_N'      : float(sites.mean()),
                    'sigma2_L'    : float(sigma2_L),
                })

        df    = pd.DataFrame(records).dropna()
        df    = df[(df['mean_N'] > 0) & (df['sigma2_L'] > 0)]
        log_s = np.log(df['sigma2_L'])
        log_n = np.log(df['mean_N'])
        slope, intercept, *_ = stats.linregress(log_n, log_s)
        df['resid'] = log_s - (slope * log_n + intercept)

        low  = df[df['resid'] <  df['resid'].median()]
        high = df[df['resid'] >= df['resid'].median()]
        _, p = stats.mannwhitneyu(
            low['went_extinct'].astype(int),
            high['went_extinct'].astype(int),
            alternative='two-sided')

        print(f"  {def_name:<35} {len(df):>8,} {df['went_extinct'].sum():>12,} {p:>8.4f}")


def jackknife_stability(ext_df):
    print("\n=== JACKKNIFE STABILITY ===")

    study_counts = ext_df['STUDY_ID'].value_counts()
    top20        = study_counts.head(20).index.tolist()
    p_values     = []

    for sid in top20:
        sub = ext_df[ext_df['STUDY_ID'] != sid].copy()
        if len(sub) < 100:
            continue

        log_s = np.log(sub['sigma2_L'])
        log_n = np.log(sub['mean_N'])
        slope, intercept, *_ = stats.linregress(log_n, log_s)
        sub['resid'] = log_s - (slope * log_n + intercept)

        low  = sub[sub['resid'] <  sub['resid'].median()]
        high = sub[sub['resid'] >= sub['resid'].median()]
        _, p = stats.mannwhitneyu(
            low['went_extinct'].astype(int),
            high['went_extinct'].astype(int),
            alternative='two-sided')
        p_values.append(p)

    p_values = np.array(p_values)
    print(f"  Replicates: {len(p_values)}")
    print(f"  Median p:   {np.median(p_values):.4f}")
    print(f"  Max p:      {np.max(p_values):.4f}")
    print(f"  All p<0.05: {(p_values < 0.05).all()}")
    return p_values


def mixed_effects_test(ext_df):
    print("\n=== MIXED-EFFECTS LOGISTIC REGRESSION ===")
    df = ext_df.copy()
    df['sigma2_L_resid_z'] = stats.zscore(df['sigma2_L_resid'])

    try:
        model  = smf.mixedlm("went_extinct ~ sigma2_L_resid_z",
                              df, groups=df["STUDY_ID"])
        result = model.fit(reml=False)
        coef   = result.params['sigma2_L_resid_z']
        z_val  = result.tvalues['sigma2_L_resid_z']
        p_val  = result.pvalues['sigma2_L_resid_z']
        print(f"  β = {coef:.3f}, z = {z_val:.2f}, p = {p_val:.4f}")
    except Exception as e:
        print(f"  Model failed: {e}")


def auc_by_realm(ext_df):
    print("\n=== AUC BY REALM ===")
    if 'realm' not in ext_df.columns:
        print("  (no realm column)")
        return

    print(f"  {'Realm':<45} {'AUC':>6} {'n':>6} {'ext_rate':>10}")
    print("  " + "-" * 70)

    for realm, grp in ext_df.groupby('realm'):
        if len(grp) < 30 or grp['went_extinct'].sum() < 5:
            continue
        X  = grp[['sigma2_L_resid']].values
        y  = grp['went_extinct'].astype(int).values
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []
        for train, test in cv.split(X, y):
            clf  = LogisticRegression(max_iter=1000)
            clf.fit(X[train], y[train])
            prob = clf.predict_proba(X[test])[:, 1]
            if len(np.unique(y[test])) > 1:
                aucs.append(roc_auc_score(y[test], prob))
        if aucs:
            ext_rate = grp['went_extinct'].mean()
            print(f"  {realm:<45} {np.mean(aucs):6.3f} {len(grp):>6} {ext_rate:>10.3f}")


def geographic_robustness(ext_df, bt):
    print("\n=== GEOGRAPHIC ROBUSTNESS ===")

    if 'CEN_LAT' not in ext_df.columns:
        lat_map = bt.groupby('STUDY_ID')['LATITUDE'].mean().to_dict()
        ext_df  = ext_df.copy()
        ext_df['CEN_LAT'] = ext_df['STUDY_ID'].map(lat_map)

    regions = [
        ('Northern hemisphere', ext_df['CEN_LAT'] > 23.5),
        ('Southern hemisphere', ext_df['CEN_LAT'] < -23.5),
        ('Tropics',             ext_df['CEN_LAT'].abs() <= 23.5),
    ]

    for region, mask in regions:
        sub = ext_df[mask].dropna(subset=['sigma2_L_resid'])
        if len(sub) < 50:
            continue
        low  = sub[sub['sigma2_L_resid'] <  sub['sigma2_L_resid'].median()]
        high = sub[sub['sigma2_L_resid'] >= sub['sigma2_L_resid'].median()]
        _, p = stats.mannwhitneyu(
            low['went_extinct'].astype(int),
            high['went_extinct'].astype(int),
            alternative='two-sided')
        print(f"  {region}: n={len(sub):,}, p={p:.4f}")


if __name__ == '__main__':
    import sys
    import os

    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    OUT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        import pyreadr
        result = pyreadr.read_r(os.path.join(DATA_DIR, 'biotime_v2_full_2025.rds'))
        bt     = result[None] if None in result else list(result.values())[0]
        bt.columns = bt.columns.str.upper()
        bt = bt[bt['ABUNDANCE'] > 0].copy()
        if 'VALID_NAME' not in bt.columns and 'valid_name' in bt.columns:
            bt = bt.rename(columns={'valid_name': 'VALID_NAME'})
        if 'SAMPLE_DESC' not in bt.columns and 'sample_desc' in bt.columns:
            bt = bt.rename(columns={'sample_desc': 'SAMPLE_DESC'})
    except Exception as e:
        print(f"Could not load BioTIME: {e}")
        sys.exit(1)

    print("Building local extinction dataset...")
    ext_df = build_extinction_dataset(bt)
    ext_df.to_csv(os.path.join(OUT_DIR, 'extinction_pairs_raw.csv'), index=False)

    ext_df, ols_slope = abundance_control(ext_df)
    ext_df.to_csv(os.path.join(OUT_DIR, 'extinction_pairs.csv'), index=False)

    main_extinction_test(ext_df)
    robustness_extinction_definitions(bt)
    jack_ps = jackknife_stability(ext_df)
    mixed_effects_test(ext_df)
    auc_by_realm(ext_df)
    geographic_robustness(ext_df, bt)

    print("\nDone. See outputs/extinction_pairs.csv")
