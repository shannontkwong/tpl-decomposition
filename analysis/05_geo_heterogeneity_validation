import numpy as np
import pandas as pd
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')


def mean_dist_km_fast(lats, lons, max_sites=200):
    if len(lats) > max_sites:
        idx  = np.random.choice(len(lats), max_sites, replace=False)
        lats = lats[idx]
        lons = lons[idx]
    R    = 6371.0
    lats = np.radians(lats)
    lons = np.radians(lons)
    dlat = lats[:, None] - lats[None, :]
    dlon = lons[:, None] - lons[None, :]
    a    = (np.sin(dlat/2)**2 +
            np.cos(lats[:,None]) * np.cos(lats[None,:]) * np.sin(dlon/2)**2)
    d    = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    upper = d[np.triu_indices(len(lats), k=1)]
    return float(np.mean(upper)) if len(upper) > 0 else np.nan


def compute_geographic_dispersion(bt, min_sites=8):
    bt = bt.copy()
    bt.columns = bt.columns.str.upper()
    records = []
    groups  = list(bt.groupby('STUDY_ID'))
    total   = len(groups)

    for i, (sid, grp) in enumerate(groups):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{total} ...")
        sites = (grp.groupby('SAMPLE_DESC')[['LATITUDE', 'LONGITUDE']]
                 .first().dropna())
        if len(sites) < min_sites:
            continue
        lats = sites['LATITUDE'].values
        lons = sites['LONGITUDE'].values
        if lats.std() < 1e-6 and lons.std() < 1e-6:
            continue
        mean_d = mean_dist_km_fast(lats, lons)
        if np.isnan(mean_d):
            continue
        records.append({
            'STUDY_ID'     : sid,
            'n_sites_geo'  : len(sites),
            'mean_dist_km' : mean_d,
            'log_mean_dist': float(np.log1p(mean_d)),
        })
    return pd.DataFrame(records)


def run_validation(study_fits_path, bt, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    print("Loading study_fits.csv ...")
    fits = pd.read_csv(study_fits_path)
    print(f"  {len(fits)} studies")

    print("Computing geographic dispersion ...")
    np.random.seed(42)
    geo_df = compute_geographic_dispersion(bt)
    print(f"  {len(geo_df)} studies with valid geographic data")

    merged = fits.merge(geo_df, on='STUDY_ID', how='inner')
    merged = merged[
        merged['sigma2_L'].notna() &
        merged['log_mean_dist'].notna() &
        (merged['sigma2_L'] > 0) &
        (merged['mean_dist_km'] > 0)
    ].copy()
    print(f"  {len(merged)} studies after merging")

    merged['log_sigma2_L'] = np.log(merged['sigma2_L'])
    merged['log_M_bar']    = np.log(merged['M_bar'].clip(lower=0.01))

    print("\n=== RAW CORRELATION ===")
    r_raw, p_raw = stats.spearmanr(merged['log_sigma2_L'], merged['log_mean_dist'])
    print(f"  Spearman r = {r_raw:.3f},  p = {p_raw:.4f},  n = {len(merged)}")

    print("\n=== ABUNDANCE-CONTROLLED CORRELATION ===")
    slope, intercept, *_ = stats.linregress(merged['log_M_bar'], merged['log_sigma2_L'])
    merged['sigma2_L_resid'] = (merged['log_sigma2_L'] -
                                 (slope * merged['log_M_bar'] + intercept))
    r_ctrl, p_ctrl = stats.spearmanr(merged['sigma2_L_resid'], merged['log_mean_dist'])
    print(f"  Spearman r = {r_ctrl:.3f},  p = {p_ctrl:.4f},  n = {len(merged)}")

    if 'REALM' in merged.columns:
        print("\n=== BY REALM ===")
        for realm, grp in merged.groupby('REALM'):
            if len(grp) < 8:
                continue
            r_r, p_r = stats.spearmanr(grp['sigma2_L_resid'], grp['log_mean_dist'])
            print(f"  {realm:<35} r={r_r:.3f}  p={p_r:.4f}  n={len(grp)}")

    out_path = os.path.join(out_dir, 'sigma2L_geo_validation.csv')
    merged.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"\n=== SUMMARY ===")
    print(f"  Raw r:                {r_raw:.3f}  p={p_raw:.4f}")
    print(f"  Abundance-controlled: {r_ctrl:.3f}  p={p_ctrl:.4f}")

    return merged


if __name__ == '__main__':
    import os

    DATA_DIR        = os.path.join(os.path.dirname(__file__), '..', 'data')
    OUT_DIR         = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    STUDY_FITS_PATH = os.path.join(OUT_DIR, 'study_fits.csv')
    RDS_PATH        = os.path.join(DATA_DIR, 'biotime_v2_full_2025.rds')

    if not os.path.exists(STUDY_FITS_PATH):
        print("study_fits.csv not found. Run 01_core_analysis.py first.")
        import sys; sys.exit(1)

    import pyreadr
    result = pyreadr.read_r(RDS_PATH)
    bt     = result[None] if None in result else list(result.values())[0]
    bt.columns = bt.columns.str.upper()
    bt = bt[bt['ABUNDANCE'] > 0].copy()
    print(f"Loaded: {len(bt):,} records")

    run_validation(STUDY_FITS_PATH, bt, OUT_DIR)
