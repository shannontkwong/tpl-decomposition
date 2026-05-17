import numpy as np
import pandas as pd
import requests
import time
import json
import os
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import roc_auc_score
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

BASE_URL   = "https://apiv3.iucnredlist.org/api/v3"
RATE_LIMIT = 28


def compute_per_species_sigma2L(bt, min_sites=5):
    if 'VALID_NAME' in bt.columns:
        bt = bt.rename(columns={'VALID_NAME': 'species_name'})
    elif 'valid_name' in bt.columns:
        bt = bt.rename(columns={'valid_name': 'species_name'})

    bt['species_name'] = (bt['species_name'].astype(str)
                          .str.strip()
                          .str.replace(r'\s+', ' ', regex=True))

    junk = r'(\bsp\.?\s*$|\bspp\.?\s*$|\bcf\b|\baff\b|\bindet\b|^nan$|^\s*$)'
    mask = (bt['species_name'].str.contains(r'\s', na=False) &
            ~bt['species_name'].str.contains(junk, case=False, na=True))
    bt   = bt[mask & (bt['ABUNDANCE'] > 0)].copy()

    print(f"  Records after cleaning:  {len(bt):,}")
    print(f"  Unique species:          {bt['species_name'].nunique():,}")

    site_means = (bt.groupby(['STUDY_ID', 'SAMPLE_DESC', 'species_name'],
                              observed=True)['ABUNDANCE']
                  .mean().reset_index()
                  .rename(columns={'ABUNDANCE': 'site_abund'}))

    def sigma_L_stats(group):
        a = group['site_abund'].dropna().values
        a = a[a > 0]
        n = len(a)
        if n < min_sites:
            return pd.Series({'sigma_L2': np.nan, 'mean_abund': np.nan, 'n_sites': n})
        return pd.Series({
            'sigma_L2'  : float(np.var(np.log(a), ddof=1)),
            'mean_abund': float(np.mean(a)),
            'n_sites'   : n,
        })

    species_stats = (site_means
                     .groupby(['STUDY_ID', 'species_name'], observed=True)
                     .apply(sigma_L_stats)
                     .reset_index()
                     .dropna(subset=['sigma_L2']))

    per_species = (species_stats
                   .groupby('species_name')
                   .agg(
                       sigma_L2_mean   = ('sigma_L2',   'mean'),
                       sigma_L2_median = ('sigma_L2',   'median'),
                       sigma_L2_std    = ('sigma_L2',   'std'),
                       mean_abund      = ('mean_abund', 'mean'),
                       n_studies       = ('STUDY_ID',   'count'),
                       n_sites_mean    = ('n_sites',    'mean'),
                   )
                   .reset_index())

    per_species = per_species[per_species['n_sites_mean'] >= min_sites].copy()
    print(f"  Species passing filter (>={min_sites} sites): {len(per_species):,}")
    return per_species, species_stats


def load_cache(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    return {}


def save_cache(cache, cache_file):
    with open(cache_file, 'w') as f:
        json.dump(cache, f)


def query_iucn(species_name, token, retries=3):
    parts = str(species_name).strip().split()
    if len(parts) < 2:
        return {'category': 'NE', 'population_trend': 'Unknown', 'taxonid': None}

    genus, species = parts[0].capitalize(), parts[1].lower()
    url = f"{BASE_URL}/species/{genus}%20{species}?token={token}"

    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('result'):
                    r = data['result'][0]
                    return {
                        'category'        : r.get('category', 'NE'),
                        'population_trend': r.get('population_trend', 'Unknown'),
                        'taxonid'         : r.get('taxonid'),
                        'scientific_name' : r.get('scientific_name', species_name),
                    }
                return {'category': 'NE', 'population_trend': 'Unknown', 'taxonid': None}
            elif resp.status_code == 404:
                return {'category': 'NE', 'population_trend': 'Unknown', 'taxonid': None}
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)

    return {'category': 'NE', 'population_trend': 'Unknown', 'taxonid': None}


def query_iucn_synonym(species_name, token):
    parts = str(species_name).strip().split()
    if len(parts) < 2:
        return None
    genus, species = parts[0].capitalize(), parts[1].lower()
    url = f"{BASE_URL}/species/synonym/{genus}%20{species}?token={token}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('result'):
                return data['result'][0].get('accepted_name')
    except Exception:
        pass
    return None


def batch_query_iucn(unique_species, token, cache_file, rate_limit=28):
    cache        = load_cache(cache_file)
    species_todo = [s for s in unique_species if s not in cache]
    delay        = 60.0 / rate_limit

    print(f"  Total species:  {len(unique_species):,}")
    print(f"  Cached:         {len(cache):,}")
    print(f"  To query:       {len(species_todo):,}")
    print(f"  Est. time:      {len(species_todo) * delay / 3600:.1f} hours")

    for i, sp in enumerate(tqdm(species_todo, desc='IUCN')):
        res = query_iucn(sp, token)

        if res['category'] == 'NE':
            accepted = query_iucn_synonym(sp, token)
            if accepted and accepted.lower() != sp.lower():
                r2 = query_iucn(accepted.lower(), token)
                if r2 and r2['category'] not in ('NE', None):
                    res = r2
                    res['synonym_of'] = sp

        cache[sp] = res
        if i % 100 == 0:
            save_cache(cache, cache_file)
        time.sleep(delay)

    save_cache(cache, cache_file)
    print(f"\nDone. Cache: {len(cache):,} entries.")
    return cache


def build_matched_dataset(per_species, cache):
    iucn_df = pd.DataFrame([
        {'species_name': sp, **res}
        for sp, res in cache.items() if res
    ])

    merged   = per_species.merge(iucn_df, on='species_name', how='inner')
    assessed = merged[merged['category'].isin(['LC', 'NT', 'VU', 'EN', 'CR'])].copy()
    assessed['threatened'] = assessed['category'].isin({'VU', 'EN', 'CR'}).astype(int)

    print(f"\n  Matched + assessed: {len(assessed):,}")
    print(f"\n  Category breakdown:")
    print(assessed['category'].value_counts()
          .reindex(['LC', 'NT', 'VU', 'EN', 'CR']).to_string())

    return assessed


def run_iucn_statistics(assessed):
    print("\n=== IUCN RESULTS ===")

    df    = assessed.copy()
    df    = df[(df['sigma_L2_mean'] > 0) & (df['mean_abund'] > 0)].copy()
    log_s = np.log(df['sigma_L2_mean'])
    log_n = np.log(df['mean_abund'])
    slope, intercept, *_ = stats.linregress(log_n, log_s)
    df['sigma2_L_resid'] = log_s - (slope * log_n + intercept)

    groups = [df[df['category'] == c]['sigma2_L_resid'].values
              for c in ['LC', 'NT', 'VU', 'EN', 'CR']
              if (df['category'] == c).sum() > 0]
    H, p_kw = stats.kruskal(*groups)
    print(f"\n  Kruskal-Wallis H={H:.2f}, p={p_kw:.2e}")

    lc = df[df['category'] == 'LC']['sigma2_L_resid']
    print(f"\n  Pairwise Mann-Whitney vs LC:")
    for cat in ['NT', 'VU', 'EN', 'CR']:
        sub = df[df['category'] == cat]['sigma2_L_resid']
        if len(sub) < 5:
            continue
        _, p = stats.mannwhitneyu(sub, lc, alternative='two-sided')
        print(f"    {cat} vs LC: p={p:.4f}  (n={len(sub)})")

    auc = roc_auc_score(df['threatened'], df['sigma2_L_resid'])
    print(f"\n  AUC (σ²_L_resid → threatened): {auc:.3f}")

    lm = smf.ols("threatened ~ sigma2_L_resid + mean_abund", data=df).fit()
    print(f"\n  OLS controlling for abundance:")
    print(lm.summary().tables[1])

    return df


if __name__ == '__main__':
    import sys

    IUCN_TOKEN = os.environ.get('IUCN_TOKEN', 'YOUR_TOKEN_HERE')
    if IUCN_TOKEN == 'YOUR_TOKEN_HERE':
        print("Set IUCN_TOKEN environment variable.")
        print("Get a free token at: https://apiv3.iucnredlist.org/api/v3/token")
        sys.exit(1)

    DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
    OUT_DIR    = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    CACHE_FILE = os.path.join(DATA_DIR, 'iucn_cache.json')
    os.makedirs(OUT_DIR, exist_ok=True)

    sigma_path = os.path.join(OUT_DIR, 'per_species_sigmaL2.csv')
    if os.path.exists(sigma_path):
        print("Loading pre-computed per-species σ²_L...")
        per_species = pd.read_csv(sigma_path)
        print(f"  {len(per_species):,} species loaded")
    else:
        print("Computing per-species σ²_L from BioTIME...")
        import pyreadr
        result      = pyreadr.read_r(os.path.join(DATA_DIR, 'biotime_v2_full_2025.rds'))
        bt          = result[None] if None in result else list(result.values())[0]
        bt.columns  = bt.columns.str.upper()
        per_species, species_stats = compute_per_species_sigma2L(bt)
        per_species.to_csv(sigma_path, index=False)
        species_stats.to_csv(os.path.join(OUT_DIR, 'species_study_stats.csv'), index=False)

    unique_species = per_species['species_name'].tolist()
    cache = batch_query_iucn(unique_species, IUCN_TOKEN, CACHE_FILE, RATE_LIMIT)

    assessed = build_matched_dataset(per_species, cache)
    assessed.to_csv(os.path.join(OUT_DIR, 'biotime_iucn_matched.csv'), index=False)

    run_iucn_statistics(assessed)
