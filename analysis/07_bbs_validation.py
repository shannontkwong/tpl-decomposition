import os
import glob
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import nnls
from sklearn.utils import resample

warnings.filterwarnings('ignore')

BBS_DIR     = 'data/bbs/50-StopData/'
SPP_PATH    = 'data/bbs/SpeciesList.csv'
WEATHER_PATH= 'data/bbs/Weather.csv'
OUT_DIR     = 'outputs/'
os.makedirs(OUT_DIR, exist_ok=True)

EARLY = set(range(1997, 2006))           # 1997-2005
LATE  = set(range(2006, 2025)) - {2020}  # 2006-2024, no COVID year
STOP_COLS = [f'Stop{i}' for i in range(1, 51)]

print("Loading species list...")
spp = pd.read_csv(SPP_PATH, encoding='latin-1')

exclude = (
    spp['English_Common_Name'].str.contains(
        r'unid\.|hybrid|/|sp\.', case=False, na=False) |
    spp['Species'].isna() |
    spp['Genus'].isna()
)
clean_aou = set(spp[~exclude]['AOU'].values)
aou_to_name = spp.set_index('AOU')['English_Common_Name'].to_dict()
print(f"  Clean species: {len(clean_aou):,} of {len(spp):,}")

print("Loading Weather.csv to identify continuously-surveyed routes...")
weather = pd.read_csv(WEATHER_PATH, encoding='latin-1')
weather['ROUTE_ID'] = (
    weather['CountryNum'].astype(str) + '_' +
    weather['StateNum'].astype(str) + '_' +
    weather['Route'].astype(str)
)
early_routes = set(weather[weather['Year'].isin(EARLY)]['ROUTE_ID'])
late_routes  = set(weather[weather['Year'].isin(LATE)]['ROUTE_ID'])
both_routes  = early_routes & late_routes
print(f"  Routes active in early period: {len(early_routes):,}")
print(f"  Routes active in late period:  {len(late_routes):,}")
print(f"  Routes active in both:         {len(both_routes):,}")

print("\nComputing sigma2_L from 50-stop data (file by file)...")
files = sorted(glob.glob(os.path.join(BBS_DIR, 'Fifty*.csv')))
print(f"  Found {len(files)} files")

records = []

for fpath in files:
    fname = os.path.basename(fpath)
    df = pd.read_csv(fpath)
    df['ROUTE_ID'] = (
        df['CountryNum'].astype(str) + '_' +
        df['StateNum'].astype(str) + '_' +
        df['Route'].astype(str)
    )
    # filter to clean species and continuously-surveyed routes
    df = df[df['AOU'].isin(clean_aou) & df['ROUTE_ID'].isin(both_routes)]

    early_df = df[df['Year'].isin(EARLY)][['ROUTE_ID', 'AOU'] + STOP_COLS]
    late_df  = df[df['Year'].isin(LATE)][['ROUTE_ID', 'AOU'] + STOP_COLS]

    # mean count per stop across years
    early_mean = early_df.groupby(['ROUTE_ID', 'AOU'])[STOP_COLS].mean()
    late_sum   = late_df.groupby(['ROUTE_ID', 'AOU'])[STOP_COLS].sum()

    for (route, aou), row in early_mean.iterrows():
        vals = row.values
        vals = vals[vals > 0]

        if len(vals) < 3:
            continue
        sigma2_L = float(np.var(np.log(vals), ddof=1))
        if sigma2_L <= 0.01:
            continue
        mean_N = float(vals.mean())

        # extinction: zero detections in late period on this route
        if (route, aou) in late_sum.index:
            went_extinct = int(late_sum.loc[(route, aou)].sum() == 0)
        else:
            went_extinct = 1

        records.append({
            'ROUTE_ID'    : route,
            'AOU'         : aou,
            'common_name' : aou_to_name.get(aou, ''),
            'sigma2_L'    : sigma2_L,
            'mean_N'      : mean_N,
            'went_extinct': went_extinct,
            'n_stops'     : len(vals),
        })

    print(f"  {fname}: pairs so far = {len(records):,}")
    del df, early_df, late_df, early_mean, late_sum

result = pd.DataFrame(records)
print(f"\nTotal route-species pairs: {len(result):,}")
print(f"Routes:                    {result['ROUTE_ID'].nunique():,}")
print(f"Species:                   {result['AOU'].nunique():,}")
print(f"Extinctions:               {result['went_extinct'].sum():,}")
print(f"Extinction rate:           {result['went_extinct'].mean():.3f}")

print("\n── Main prospective test ─────────────────────────────────────────")
slope, intercept, *_ = stats.linregress(
    np.log(result['mean_N']), np.log(result['sigma2_L']))
result['resid'] = (
    np.log(result['sigma2_L']) -
    (slope * np.log(result['mean_N']) + intercept)
)

low  = result[result['resid'] <  result['resid'].median()]
high = result[result['resid'] >= result['resid'].median()]

_, p_main = stats.mannwhitneyu(
    low['went_extinct'].astype(int),
    high['went_extinct'].astype(int),
    alternative='two-sided'
)
rel_main = (
    (low['went_extinct'].mean() - high['went_extinct'].mean()) /
    low['went_extinct'].mean()
)

print(f"Rate low  sigma2_L: {low['went_extinct'].mean():.3f}")
print(f"Rate high sigma2_L: {high['went_extinct'].mean():.3f}")
print(f"Rel. reduction:     {rel_main:.1%}")
print(f"Mann-Whitney p:     {p_main:.4e}")

print("\n── Jackknife stability (20 replicates) ──────────────────────────")
jack_ps = []
for i in range(20):
    routes_sub = resample(
        result['ROUTE_ID'].unique(),
        n_samples=int(result['ROUTE_ID'].nunique() * 0.9),
        random_state=i
    )
    sub = result[result['ROUTE_ID'].isin(routes_sub)].copy()
    sl, si, *_ = stats.linregress(
        np.log(sub['mean_N']), np.log(sub['sigma2_L']))
    sub['resid_j'] = (
        np.log(sub['sigma2_L']) - (sl * np.log(sub['mean_N']) + si))
    l = sub[sub['resid_j'] <  sub['resid_j'].median()]
    h = sub[sub['resid_j'] >= sub['resid_j'].median()]
    _, pj = stats.mannwhitneyu(
        l['went_extinct'].astype(int),
        h['went_extinct'].astype(int),
        alternative='two-sided'
    )
    jack_ps.append(pj)

print(f"All 20 significant: {all(p < 0.05 for p in jack_ps)}")
print(f"Min p: {min(jack_ps):.2e}  Max p: {max(jack_ps):.2e}  "
      f"Median p: {np.median(jack_ps):.2e}")

print("\n── Estimating c1 from year-level variance per stop ──────────────")
c1_records = []

for fpath in files:
    df = pd.read_csv(fpath)
    df['ROUTE_ID'] = (
        df['CountryNum'].astype(str) + '_' +
        df['StateNum'].astype(str) + '_' +
        df['Route'].astype(str)
    )
    df = df[
        df['AOU'].isin(clean_aou) &
        df['Year'].isin(EARLY) &
        df['ROUTE_ID'].isin(both_routes)
    ]
    if len(df) == 0:
        continue

    for (route, aou), grp in df.groupby(['ROUTE_ID', 'AOU']):
        if len(grp) < 4:
            continue
        stop_means = grp[STOP_COLS].mean().values
        stop_vars  = grp[STOP_COLS].var(ddof=1).values
        mask = stop_means > 0
        if mask.sum() < 5:
            continue
        M = stop_means[mask]
        V = stop_vars[mask]
        A = np.column_stack([M, M**2])
        coeffs, _ = nnls(A, V)
        c1_est = float(coeffs[0])
        if c1_est <= 0:
            continue
        c1_records.append({
            'ROUTE_ID': route,
            'AOU'     : aou,
            'c1'      : c1_est,
        })

c1_df = pd.DataFrame(c1_records)
print(f"c1 estimates: {len(c1_df):,}")

result_c1 = result.merge(c1_df, on=['ROUTE_ID', 'AOU'], how='inner')
print(f"Pairs with c1: {len(result_c1):,}")
print(f"\nc1 distribution:")
print(result_c1['c1'].describe())

print("\n── P4: c1 subgroup test ──────────────────────────────────────────")
q25 = result_c1['c1'].quantile(0.25)
q75 = result_c1['c1'].quantile(0.75)
print(f"c1 Q25: {q25:.3f}  Q75: {q75:.3f}")

subgroups = {
    'Low c1 (env. dominated)' : result_c1[result_c1['c1'] <= q25],
    'High c1 (demo. dominated)': result_c1[result_c1['c1'] >= q75],
}

c1_summary = []
for label, sub in subgroups.items():
    sl, si, *_ = stats.linregress(
        np.log(sub['mean_N']), np.log(sub['sigma2_L']))
    sub = sub.copy()
    sub['resid_c1'] = (
        np.log(sub['sigma2_L']) - (sl * np.log(sub['mean_N']) + si))
    l = sub[sub['resid_c1'] <  sub['resid_c1'].median()]
    h = sub[sub['resid_c1'] >= sub['resid_c1'].median()]
    _, p_sub = stats.mannwhitneyu(
        l['went_extinct'].astype(int),
        h['went_extinct'].astype(int),
        alternative='two-sided'
    )
    rel_sub = (
        (l['went_extinct'].mean() - h['went_extinct'].mean()) /
        l['went_extinct'].mean()
        if l['went_extinct'].mean() > 0 else np.nan
    )
    print(f"\n  {label}")
    print(f"  n pairs:        {len(sub):,}")
    print(f"  rate low  sigma2_L:  {l['went_extinct'].mean():.3f}")
    print(f"  rate high sigma2_L:  {h['went_extinct'].mean():.3f}")
    print(f"  rel. reduction: {rel_sub:.1%}")
    print(f"  p:              {p_sub:.4e}")

    print(f"\n  Most common species:")
    print(sub['common_name'].value_counts().head(8).to_string())

    c1_summary.append({
        'group'        : label,
        'n_pairs'      : len(sub),
        'rate_low'     : l['went_extinct'].mean(),
        'rate_high'    : h['went_extinct'].mean(),
        'rel_reduction': rel_sub,
        'p'            : p_sub,
    })

print("\n── Extinction rate by n_stops ────────────────────────────────────")
result['stop_bin'] = pd.cut(result['n_stops'], bins=[2, 5, 10, 20, 50])
print(result.groupby('stop_bin', observed=False)['went_extinct']
      .agg(['mean', 'count']).rename(
          columns={'mean': 'ext_rate', 'count': 'n_pairs'}))

result.to_csv(os.path.join(OUT_DIR, 'bbs_prospective_extinction.csv'),
              index=False)
result_c1.to_csv(os.path.join(OUT_DIR, 'bbs_c1_subgroup.csv'),
                 index=False)

summary_lines = [
    "═══ BBS VALIDATION SUMMARY ═══════════════════════════════════════",
    f"Dataset:        BBS 50-stop, 1997-2024, continuously-surveyed routes",
    f"Pairs:          {len(result):,}",
    f"Routes:         {result['ROUTE_ID'].nunique():,}",
    f"Species:        {result['AOU'].nunique():,}",
    f"Extinctions:    {result['went_extinct'].sum():,} "
    f"({result['went_extinct'].mean():.1%})",
    "",
    f"Rate low  sigma2_L: {low['went_extinct'].mean():.3f}",
    f"Rate high sigma2_L: {high['went_extinct'].mean():.3f}",
    f"Rel. reduction:     {rel_main:.1%}",
    f"Mann-Whitney p:     {p_main:.2e}",
    "",
    f"Jackknife (20x90% routes): all significant={all(p<0.05 for p in jack_ps)}",
    f"  min p={min(jack_ps):.2e}  max p={max(jack_ps):.2e}",
    "",
    "P4 c1 subgroup:",
]
for row in c1_summary:
    summary_lines.append(
        f"  {row['group']}: n={row['n_pairs']:,}, "
        f"rel={row['rel_reduction']:.1%}, p={row['p']:.2e}"
    )
summary_lines.append(
    "═══════════════════════════════════════════════════════════════════"
)
summary_text = '\n'.join(summary_lines)
print('\n' + summary_text)

with open(os.path.join(OUT_DIR, 'bbs_validation_summary.txt'), 'w') as f:
    f.write(summary_text + '\n')

print("\nSaved:")
print("  outputs/bbs_prospective_extinction.csv")
print("  outputs/bbs_c1_subgroup.csv")
print("  outputs/bbs_validation_summary.txt")
