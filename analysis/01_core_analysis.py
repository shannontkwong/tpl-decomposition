import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def fit_two_term_nnls(means, variances, min_pts=8, n_boot=500):
    mask = ((means > 0) & (variances > 0) &
            np.isfinite(means) & np.isfinite(variances))
    M = means[mask]
    V = variances[mask]
    if mask.sum() < min_pts:
        return np.nan, np.nan, np.nan, np.nan, np.nan, 0

    y = V / M
    X = np.column_stack([np.ones(len(M)), M])
    coef, _ = nnls(X, y)
    c1, c2  = coef[0], coef[1]

    y_hat  = c1 + c2 * M
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    R2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    rng2 = np.random.default_rng(seed=99)
    c1_b, c2_b = [], []
    for _ in range(n_boot):
        idx = rng2.integers(0, len(M), len(M))
        cb, _ = nnls(X[idx], y[idx])
        c1_b.append(cb[0])
        c2_b.append(cb[1])

    return (float(c1), float(c2), float(R2),
            float(np.std(c1_b)), float(np.std(c2_b)), int(mask.sum()))


def compute_sigma2_L(site_sp_df, min_sites=5, min_species=3):
    sp_vals = {}
    for sp, grp in site_sp_df.groupby('SPECIES_ID'):
        ab = grp['ABUND'].values
        ab = ab[ab > 0]
        if len(ab) >= min_sites:
            sp_vals[sp] = np.var(np.log(ab), ddof=1)

    if len(sp_vals) < min_species:
        return np.nan, np.nan

    sigma2_L = float(np.mean(list(sp_vals.values())))

    keys = np.array(list(sp_vals.keys()))
    if len(keys) < 6:
        return sigma2_L, sigma2_L

    rng2  = np.random.default_rng(seed=42)
    idx_a = rng2.choice(len(keys), len(keys) // 2, replace=False)
    idx_b = np.setdiff1d(np.arange(len(keys)), idx_a)
    vals_a = [sp_vals[keys[i]] for i in idx_a]
    vals_b = [sp_vals[keys[i]] for i in idx_b]
    sigma2_L_cv = np.mean([np.mean(vals_a), np.mean(vals_b)])

    return sigma2_L, float(sigma2_L_cv)


def compute_sigma2_SAD(snap_df, min_species=5, min_sites=3):
    sad_vals = []
    for site, grp in snap_df.groupby('SITE'):
        ab = grp.groupby('SPECIES_ID')['ABUNDANCE'].mean().values
        ab = ab[ab > 0]
        if len(ab) >= min_species:
            sad_vals.append(np.var(np.log(ab), ddof=1))

    if len(sad_vals) < min_sites:
        return np.nan
    return float(np.mean(sad_vals))


def crossover_scale(c1, sigma2_L):
    if sigma2_L > 0 and c1 > 0:
        return c1 / sigma2_L
    return np.nan


def compute_study(study_df, meta_row, min_sites=10, min_species=12):
    sdf = study_df.copy()

    year_sites = sdf.groupby('YEAR')['SITE'].nunique()
    best_year  = year_sites.idxmax()
    snap       = sdf[sdf['YEAR'] == best_year].copy()

    site_sp = (snap.groupby(['SPECIES_ID', 'SITE'])['ABUNDANCE']
               .mean().reset_index())
    site_sp.columns = ['SPECIES_ID', 'SITE', 'ABUND']
    site_sp = site_sp[site_sp['ABUND'] > 0]

    n_sites   = site_sp['SITE'].nunique()
    n_species = site_sp['SPECIES_ID'].nunique()
    if n_sites < min_sites or n_species < min_species:
        return None

    td = (site_sp.groupby('SPECIES_ID')['ABUND']
          .agg(['mean', 'var', 'count']).reset_index())
    td.columns = ['SPECIES_ID', 'MEAN', 'VAR', 'N_SITES']
    td = td[td['N_SITES'] >= 5].copy()
    if len(td) < min_species:
        return None

    M_arr = td['MEAN'].values
    V_arr = td['VAR'].values

    M_range_log = (np.log10(M_arr.max() / M_arr.min())
                   if M_arr.min() > 0 else 0.0)

    c1, c2, R2, se_c1, se_c2, n_pts = fit_two_term_nnls(M_arr, V_arr)
    c2_valid = np.isfinite(c2) and c2 > 0.01

    sigma2_L, sigma2_L_cv = compute_sigma2_L(site_sp)
    sigma2_SAD             = compute_sigma2_SAD(snap)

    M_bar_study = float(np.mean(M_arr))

    ratio_c2_sigmaL = (c2 / sigma2_L
                       if (c2_valid and np.isfinite(sigma2_L) and sigma2_L > 0)
                       else np.nan)
    ratio_SAD_sigmaL = (sigma2_L / sigma2_SAD
                        if (np.isfinite(sigma2_SAD) and sigma2_SAD > 0
                            and np.isfinite(sigma2_L) and sigma2_L > 0)
                        else np.nan)

    return {
        'n_sites'         : n_sites,
        'n_species'       : n_species,
        'n_years'         : sdf['YEAR'].nunique(),
        'best_year'       : best_year,
        'M_bar'           : M_bar_study,
        'M_range_log'     : M_range_log,
        'c1'              : c1,
        'c2'              : c2 if c2_valid else np.nan,
        'se_c1'           : se_c1,
        'se_c2'           : se_c2 if c2_valid else np.nan,
        'R2_two_term'     : R2,
        'sigma2_L'        : sigma2_L,
        'sigma2_L_cv'     : sigma2_L_cv,
        'sigma2_SAD'      : sigma2_SAD,
        'ratio_c2_sigmaL' : ratio_c2_sigmaL,
        'ratio_SAD_sigmaL': ratio_SAD_sigmaL,
        'N_star'          : crossover_scale(c1, c2),
        'n_pts_fit'       : n_pts,
    }


def run_all_studies(df_raw, meta, min_sites=10, min_species=12):
    records   = []
    study_ids = df_raw['STUDY_ID'].unique()

    print(f"  Running {len(study_ids)} studies...")
    for i, sid in enumerate(study_ids):
        if (i + 1) % 30 == 0:
            print(f"    {i+1}/{len(study_ids)} done...")

        sdf_rows = meta[meta['STUDY_ID'] == sid]
        if len(sdf_rows) == 0:
            continue
        meta_row = sdf_rows.iloc[0]

        sdf = df_raw[df_raw['STUDY_ID'] == sid]
        r   = compute_study(sdf, meta_row, min_sites, min_species)
        if r is None:
            continue

        r['STUDY_ID'] = sid
        r['TAXA']     = meta_row.get('TAXA', 'Unknown')
        r['REALM']    = meta_row.get('REALM', 'Unknown')
        r['CEN_LAT']  = meta_row.get('CEN_LATITUDE',
                        meta_row.get('CENT_LAT', np.nan))
        r['CEN_LON']  = meta_row.get('CEN_LONGITUDE',
                        meta_row.get('CENT_LONG', np.nan))
        records.append(r)

    results = pd.DataFrame(records)
    print(f"  Complete: {len(results)} studies fitted")
    return results


def validate_theorem2(results_df):
    print("\n=== THEOREM 2 VALIDATION: c2 ~ sigma2_L ===")
    valid = results_df[
        results_df['c2'].notna() &
        results_df['sigma2_L'].notna() &
        (results_df['c2'] > 0) &
        (results_df['sigma2_L'] > 0)
    ].copy()

    print(f"  Studies with valid c2 and sigma2_L: {len(valid)}")
    for label, thresh in [('All studies',  0.0),
                           ('>1 decade',    1.0),
                           ('>1.5 decades', 1.5)]:
        sub = valid[valid['M_range_log'] >= thresh]
        if len(sub) < 5:
            continue
        r, p = stats.spearmanr(np.log(sub['sigma2_L']), np.log(sub['c2']))
        sl, ic, _, _, se = stats.linregress(
            np.log(sub['sigma2_L']), np.log(sub['c2']))
        print(f"  {label}: n={len(sub)}, r={r:.3f}, p={p:.2e}, "
              f"slope={sl:.3f}+/-{se:.3f}")

    return valid


def validate_theorem4(results_df):
    print("\n=== THEOREM 4 VALIDATION: sigma2_L / sigma2_SAD < 1 ===")
    sub = results_df[
        results_df['ratio_SAD_sigmaL'].notna() &
        np.isfinite(results_df['ratio_SAD_sigmaL']) &
        (results_df['ratio_SAD_sigmaL'] > 0)
    ].copy()

    frac_lt1 = (sub['ratio_SAD_sigmaL'] < 1).mean()
    median_r = sub['ratio_SAD_sigmaL'].median()

    print(f"  Studies: {len(sub)}")
    print(f"  Fraction < 1:  {frac_lt1:.3f}")
    print(f"  Median ratio:  {median_r:.3f}")
    print(f"  IQR:           [{sub['ratio_SAD_sigmaL'].quantile(0.25):.3f}, "
          f"{sub['ratio_SAD_sigmaL'].quantile(0.75):.3f}]")

    return sub


def validate_c1_lifehistory(results_df):
    print("\n=== c1 LIFE-HISTORY VALIDATION ===")

    gen_time = {
        'Chromista'                            : 0.1,
        'Marine/freshwater invertebrates'      : 1.0,
        'Fish'                                 : 3.0,
        'Invertebrates'                        : 1.0,
        'Terrestrial/freshwater invertebrates' : 1.0,
        'Plants'                               : 10.0,
        'Birds'                                : 8.0,
        'Mammals'                              : 12.0,
        'Amphibians & reptiles'                : 5.0,
    }

    sub = results_df[
        results_df['c1'].notna() &
        (results_df['c1'] >= 1e-8) &
        (results_df['c1'] <= 1e4)
    ].copy()

    sub['gen_time'] = sub['TAXA'].map(gen_time)

    groups = [g['c1'].values for _, g in sub.groupby('TAXA') if len(g) >= 5]
    if len(groups) >= 2:
        H, p = stats.kruskal(*groups)
        print(f"  Kruskal-Wallis: H={H:.2f}, p={p:.4f}, n_studies={len(sub)}")

    sub_gt = sub.dropna(subset=['gen_time'])
    if len(sub_gt) >= 10:
        r, p = stats.spearmanr(sub_gt['gen_time'], sub_gt['c1'])
        print(f"  Gen. time vs c1: r={r:.3f}, p={p:.4f}, n={len(sub_gt)}")

    print("\n  Median c1 by taxonomic realm:")
    realm_summary = (sub.groupby('TAXA')['c1']
                        .median().sort_values(ascending=False))
    for realm, val in realm_summary.items():
        n = len(sub[sub['TAXA'] == realm])
        if n >= 3:
            print(f"    {realm:<45} median c1 = {val:6.1f}  (n={n})")

    return sub


def load_biotime(rds_path, meta_path):
    import pyreadr
    print("Loading BioTIME v2...")
    result  = pyreadr.read_r(rds_path)
    df_raw  = result[None] if None in result else list(result.values())[0]
    meta    = pd.read_csv(meta_path, low_memory=False)

    df_raw.columns = df_raw.columns.str.upper()
    meta.columns   = meta.columns.str.upper()

    print(f"  Raw: {df_raw.shape[0]:,} rows | {meta.shape[0]} studies")

    good_studies = meta[
        (meta['ABUNDANCE_TYPE'].isin(['Count', 'Density', 'MeanCount'])) &
        (meta['NUMBER_LAT_LONG'] >= 10) &
        (meta['NUMBER_OF_SPECIES'] >= 10) &
        (meta['TOTAL'] >= 100)
    ]['STUDY_ID'].tolist()

    df_raw = df_raw[df_raw['STUDY_ID'].isin(good_studies)].copy()
    df_raw = df_raw[df_raw['ABUNDANCE'] > 0].copy()

    if 'SPECIES_ID' not in df_raw.columns:
        df_raw['SPECIES_ID'] = (df_raw['NEWID']
                                if 'NEWID' in df_raw.columns
                                else df_raw['ID_SPECIES'])
    df_raw['SITE'] = (df_raw['LATITUDE'].round(4).astype(str) + '_' +
                      df_raw['LONGITUDE'].round(4).astype(str))

    print(f"  Filtered: {df_raw.shape[0]:,} records | "
          f"{len(good_studies)} qualifying studies")
    return df_raw, meta


if __name__ == '__main__':
    import os

    DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data')
    RDS_PATH  = os.path.join(DATA_DIR, 'biotime_v2_full_2025.rds')
    META_PATH = os.path.join(DATA_DIR, 'biotime_v2_metadata_2025.csv')
    OUT_DIR   = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(OUT_DIR, exist_ok=True)

    df_raw, meta = load_biotime(RDS_PATH, META_PATH)

    print("\nFitting two-term Taylor decomposition per study...")
    results_df = run_all_studies(df_raw, meta)
    results_df.to_csv(os.path.join(OUT_DIR, 'study_fits.csv'), index=False)
    print(f"  Saved: study_fits.csv")

    validate_theorem2(results_df)
    validate_theorem4(results_df)
    validate_c1_lifehistory(results_df)

    print("\nDone. Run 02_local_extinction.py next.")
