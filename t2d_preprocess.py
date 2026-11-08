import os
import pandas as pd
import numpy as np

# cohort palette
THESIS_PALETTE = {
    'healthy':   '#7FB069',
    'ifg':       '#E6AA68',
    't2d':       '#D8657D',
    'slate':     '#5B8FA8',
    'grey':      '#7C7C7C',
    'aubergine': '#3D2645',
    'text':      '#2B2B2B',
    'grid':      '#EAEAEA',
    'bg':        '#FFFFFF',
}

import sys as _sys, os as _os

class _Tee:
    def __init__(self, path):
        _os.makedirs(_os.path.dirname(path), exist_ok=True)
        self._f = open(path, 'w', buffering=1)
        self._orig = _sys.stdout
    def write(self, x): self._orig.write(x); self._f.write(x)
    def flush(self): self._orig.flush(); self._f.flush()
    def close(self): self._f.close()
_tee = _Tee('runtime/pre/pre_output.txt')
_sys.stdout = _tee

import sys as _sys, os as _os
_os.makedirs('output', exist_ok=True)
class _Tee:
    def __init__(self, path):
        import os as _mkdir_os
        _mkdir_os.makedirs(_mkdir_os.path.dirname(path), exist_ok=True)
        self._f = open(path, 'w', buffering=1)
        self._orig = _sys.stdout
    def write(self, x): self._orig.write(x); self._f.write(x)
    def flush(self): self._orig.flush(); self._f.flush()

os.makedirs('Training Data', exist_ok=True)

# Original Data/ houses the real UKB export - swap in Synthetic Data/ to run without UKB access
# everything downstream is identical either way
df_raw   = pd.read_csv('Original Data/ukb_cohort_diabetic.csv')
df_h_raw = pd.read_csv('Original Data/ukb_cohort_healthy.csv')

print(f'Diabetic raw: {len(df_raw):,} rows')
print(f'Healthy raw:  {len(df_h_raw):,} rows')

# ── shared column definitions

stress_cols = [
    'p20510', 'p20507', 'p20519', 'p20514',
    'p20511', 'p20513', 'p20508', 'p20518',
    'p20505', 'p20512', 'p20506', 'p20509',
    'p20516', 'p20515', 'p20520', 'p20517'
]

# minimum biomarker columns needed to initialize the ODE state vector
# CRP (p30710_i0) included - a single NaN silently kills all DeepONet branch weights
biomarker_cols = ['p30740_i0', 'p21001_i0', 'p31', 'p21022', 'p30710_i0']

# 8 medication slots per instance - NaN means the slot is empty, not missing data
med_cols_i0 = [f'p20003_i0_a{i}' for i in range(8)]
med_cols_i1 = [f'p20003_i1_a{i}' for i in range(8)]

# glucose-lowering drug codes from UKB Field 20003 (Data Coding 4)
# every code below was verified verbatim against the coding4 dictionary
# do not add codes from memory - verify against coding4 dictionary first
# GLP-1 agonists, SGLT2 inhibitors, DPP-4 inhibitors are not in coding4 at all -
# reflects UKB recruitment 2006-2010 / Instance 1 2012-2013, before these classes existed at scale
treated_codes_by_class = {
    'biguanide': [
        1140884600,  # metformin
        1140874686,  # glucophage 500mg tablet (metformin brand)
        1140874690,  # orabet 500mg tablet (metformin brand)
    ],
    'sulfonylurea': [
        1140874744,  # gliclazide
        1140874746,  # diamicron 80mg tablet (gliclazide brand)
        1140874718,  # glibenclamide
        1140874724,  # daonil 5mg tablet (glibenclamide brand)
        1140874726,  # semi-daonil 2.5mg tablet
        1140874728,  # euglucon 2.5mg tablet
        1140874732,  # malix 2.5mg tablet
        1140874736,  # diabetamide 2.5mg tablet
        1140874740,  # calabren 2.5mg tablet
        1140874646,  # glipizide
        1140874650,  # glibenese 5mg tablet (glipizide brand)
        1140874652,  # minodiab 2.5mg tablet (glipizide brand)
        1140874658,  # gliquidone
        1140874660,  # glurenorm 30mg tablet (gliquidone brand)
        1140874664,  # tolazamide
        1140874666,  # tolanase 100mg tablet (tolazamide brand)
        1140874674,  # tolbutamide
        1140874678,  # glyconon 500mg tablet (tolbutamide brand)
        1140874680,  # rastinon 500mg tablet (tolbutamide brand)
        1140857506,  # pramidex 500mg tablet (tolbutamide brand)
        1140874706,  # chlorpropamide
        1140874712,  # diabinese 100mg tablet (chlorpropamide brand)
        1140874716,  # glymese 250mg tablet (chlorpropamide brand)
        1140857584,  # acetohexamide
        1140857586,  # dimelor 500mg tablet (acetohexamide brand)
        1140857494,  # glibornuride
        1140857496,  # glutril 25mg tablet (glibornuride brand)
        1140857500,  # glymidine
        1140857502,  # gondafon 500mg tablet (glymidine brand)
        1141152590,  # glimepiride
        1141156846,  # amaryl 1mg tablet (glimepiride brand)
    ],
    'thiazolidinedione': [
        1141171646,  # pioglitazone
        1141171652,  # actos 15mg tablet (pioglitazone brand)
        1141177600,  # rosiglitazone
        1141177606,  # avandia 4mg tablet (rosiglitazone brand)
        1141189090,  # rosiglitazone 1mg / metformin 500mg combo
        1141189094,  # avandamet 1mg/500mg tablet (combo brand)
    ],
    'meglitinide': [
        1141168660,  # repaglinide
        1141168668,  # novonorm 0.5mg tablet (repaglinide brand)
        1141173882,  # nateglinide
        1141173786,  # starlix 60mg tablet (nateglinide brand)
    ],
    'alpha_glucosidase_inhibitor': [
        1140868902,  # acarbose
        1140868908,  # glucobay 50mg tablet (acarbose brand)
    ],
    'insulin': [
        1140883066,  # insulin product
    ],
}

treated_codes = [code for codes in treated_codes_by_class.values() for code in codes]

print(f'Loaded {len(treated_codes)} drug codes across {len(treated_codes_by_class)} classes')
for cls, codes in treated_codes_by_class.items():
    print(f'  {cls:30s} {len(codes):3d} codes')

# ── diabetic pipeline

df_raw.columns = [c.replace('participant.', '') for c in df_raw.columns]
df_raw['p31'] = df_raw['p31'].map({'Female': 0, 'Male': 1}).fillna(df_raw['p31'])

# fill empty medication slots before dropna - empty slot != missing data
df_raw[med_cols_i0] = df_raw[med_cols_i0].fillna(0)
df_raw[med_cols_i1] = df_raw[med_cols_i1].fillna(0)

df_d = df_raw.dropna(subset=biomarker_cols + stress_cols)

for col in stress_cols:
    df_d = df_d[df_d[col] >= 1]  # valid 1-4 range only; -818 = prefer not to answer, gets dropped here

df_d = df_d.copy()
df_d['S_input']   = (df_d[stress_cols] - 1).sum(axis=1) / 48.0  # normalized 0-1 stress score
df_d['G_mgdl']    = df_d['p30740_i0'] * 18.0182
df_d['G_i1_mgdl'] = df_d['p30740_i1'] * 18.0182

df_d = df_d[df_d['G_mgdl'] >= 126]

# medication wall: exclude anyone on glucose-lowering agents at either instance
is_treated = (
    df_d[med_cols_i0].isin(treated_codes).any(axis=1) |
    df_d[med_cols_i1].isin(treated_codes).any(axis=1)
)
df_d = df_d[~is_treated].copy()

df_diabetic_all = df_d.copy()

# put longitudinal patients at top so v8's blind validation finds them easily
has_longitudinal = df_diabetic_all['p30740_i1'].notna() & df_diabetic_all['p53_i1'].notna()
df_with_longval    = df_diabetic_all[has_longitudinal]
df_without_longval = df_diabetic_all[~has_longitudinal]

n_longval = min(len(df_with_longval), 50)
n_fill    = 50 - n_longval

df_diabetic_final = pd.concat([
    df_with_longval.head(n_longval),
    df_without_longval.sample(n=min(n_fill, len(df_without_longval)), random_state=28)
], ignore_index=True)

print(f'Diabetic pipeline complete. Total available: {len(df_diabetic_all)}')
print(f'  With longitudinal data: {len(df_with_longval)}')
print(f'  Selected with longval: {n_longval}, filled without: {n_fill}')
print(f'Final Diabetic N: {len(df_diabetic_final)}')

df_diabetic_final.to_csv('Training Data/t2d_pilot_50_diabetics.csv', index=False)
print('Saved: t2d_pilot_50_diabetics.csv')

# ── healthy + IFG pipeline
# both come from the same healthy file, split after filtering by measured glucose

df_h_raw.columns = [c.replace('participant.', '') for c in df_h_raw.columns]
df_h_raw['p31'] = df_h_raw['p31'].map({'Female': 0, 'Male': 1}).fillna(df_h_raw['p31'])

df_h_raw[med_cols_i0] = df_h_raw[med_cols_i0].fillna(0)
df_h_raw[med_cols_i1] = df_h_raw[med_cols_i1].fillna(0)

df_h = df_h_raw.dropna(subset=biomarker_cols + stress_cols).copy()

# healthy file came via Table Exporter so stress fields are text labels, not numbers
# diabetic file came via terminal so it's already numeric - hence this only runs on df_h
stress_label_map = {
    'Not at all': 1, 'Several days': 2,
    'More than half the days': 3, 'Nearly every day': 4
}
for col in stress_cols:
    if df_h[col].dtype == object:
        df_h[col] = df_h[col].map(stress_label_map)

for col in stress_cols:
    df_h = df_h[df_h[col] >= 1]

df_h = df_h.copy()  # re-copy after row filtering to drop view
df_h['S_input']   = (df_h[stress_cols] - 1).sum(axis=1) / 48.0
df_h['G_mgdl']    = df_h['p30740_i0'] * 18.0182
df_h['G_i1_mgdl'] = df_h['p30740_i1'] * 18.0182

is_treated_h = (
    df_h[med_cols_i0].isin(treated_codes).any(axis=1) |
    df_h[med_cols_i1].isin(treated_codes).any(axis=1)
)
df_h = df_h[~is_treated_h].copy()

df_ifg_all     = df_h[(df_h['G_mgdl'] >= 100) & (df_h['G_mgdl'] < 126)].copy()
df_healthy_all = df_h[(df_h['G_mgdl'] >= 70)  & (df_h['G_mgdl'] < 100)].copy()

print(f'Healthy pipeline complete.')
print(f'IFG available: {len(df_ifg_all)}, Healthy available: {len(df_healthy_all)}')

df_ifg_final     = df_ifg_all.sample(n=min(50, len(df_ifg_all)),         random_state=28)
df_healthy_final = df_healthy_all.sample(n=min(50, len(df_healthy_all)), random_state=28)

df_healthy_final.to_csv('Training Data/t2d_pilot_50_healthy.csv', index=False)
df_ifg_final.to_csv('Training Data/t2d_pilot_50_ifg.csv',         index=False)

print('Saved: t2d_pilot_50_healthy.csv')
print('Saved: t2d_pilot_50_ifg.csv')
print()
print('Glucose range check (mg/dL):')
print(f'  Diabetic: {df_diabetic_final["G_mgdl"].min():.1f} - {df_diabetic_final["G_mgdl"].max():.1f}  (expected >= 126)')
print(f'  IFG:      {df_ifg_final["G_mgdl"].min():.1f} - {df_ifg_final["G_mgdl"].max():.1f}  (expected 100-126)')
print(f'  Healthy:  {df_healthy_final["G_mgdl"].min():.1f} - {df_healthy_final["G_mgdl"].max():.1f}  (expected 70-100)')

# ── full cohort outputs

df_diabetic_full = pd.concat([df_with_longval, df_without_longval], ignore_index=True)
df_diabetic_full.to_csv('Training Data/ukb_full_diabetics.csv', index=False)
print(f'Saved: ukb_full_diabetics.csv  ({len(df_diabetic_full)} patients, {df_diabetic_full["G_i1_mgdl"].notna().sum()} with Instance 1)')

N_HEALTHY = 1000
N_IFG     = 500

def prioritised_sample(df_all, n, olink_eids=None, seed=28):
    """two-tier priority: longitudinal first, then Olink-covered, then random fill"""
    has_i1 = df_all['p30740_i1'].notna() & df_all['p53_i1'].notna()
    with_longval    = df_all[has_i1]
    without_longval = df_all[~has_i1]

    n_longval   = min(len(with_longval), n)
    remaining_n = n - n_longval

    if remaining_n == 0:
        return with_longval.head(n_longval).reset_index(drop=True)

    if olink_eids is not None:
        has_olink = without_longval['eid'].isin(olink_eids)
        with_olink    = without_longval[has_olink]
        without_olink = without_longval[~has_olink]
        n_olink = min(len(with_olink), remaining_n)
        n_fill  = remaining_n - n_olink
        fill = pd.concat([
            with_olink.sample(n=n_olink, random_state=seed) if n_olink > 0 else pd.DataFrame(),
            without_olink.sample(n=min(n_fill, len(without_olink)), random_state=seed)
        ], ignore_index=True)
    else:
        fill = without_longval.sample(n=min(remaining_n, len(without_longval)), random_state=seed)

    return pd.concat([with_longval.head(n_longval), fill], ignore_index=True)

try:
    _od = pd.read_csv('Original Data/ukb_olink_i0_diabetic.csv')[['eid']]
    _oh = pd.read_csv('Original Data/ukb_olink_i0_healthy.csv')[['eid']]
    olink_eids = set(pd.concat([_od, _oh])['eid'])
    print(f'Olink eids loaded for prioritization: {len(olink_eids)}')
except FileNotFoundError:
    olink_eids = None
    print('Olink CSVs not found - falling back to random sampling')

df_healthy_full = prioritised_sample(df_healthy_all, N_HEALTHY, olink_eids=olink_eids)
df_ifg_full     = prioritised_sample(df_ifg_all,     N_IFG,     olink_eids=olink_eids)

df_healthy_full.to_csv('Training Data/ukb_full_healthy.csv', index=False)
df_ifg_full.to_csv('Training Data/ukb_full_ifg.csv',         index=False)

print(f'Saved: ukb_full_healthy.csv  ({len(df_healthy_full)} patients)')
print(f'Saved: ukb_full_ifg.csv      ({len(df_ifg_full)} patients)')
total = len(df_diabetic_full) + len(df_healthy_full) + len(df_ifg_full)
print(f'Total cohort: {total}')

# ── olink merge (in-place)
# merges Olink NPX values into existing cohort CSVs - no new files, safe to rerun

OLINK_PROTEINS = ['il6', 'tnf', 'igf2bp3', 'adipoq', 'lep', 'gdf15']

def load_and_clean_olink(path):
    df = pd.read_csv(path)
    for col in OLINK_PROTEINS:
        if col not in df.columns: continue
        mu, sd = df[col].mean(), df[col].std()
        df[col] = df[col].clip(lower=mu - 5*sd, upper=mu + 5*sd)
    return df[['eid'] + [c for c in OLINK_PROTEINS if c in df.columns]]

def merge_olink_inplace(csv_path, olink_df):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in OLINK_PROTEINS if c in df.columns], errors='ignore')
    df = df.merge(olink_df, on='eid', how='left')
    n_with = df[OLINK_PROTEINS[0]].notna().sum()
    df.to_csv(csv_path, index=False)
    print(f'  {csv_path}: {len(df)} rows, {n_with} have Olink ({100*n_with/len(df):.1f}%)')

print('Loading Olink CSVs...')
df_olink_d   = load_and_clean_olink('Original Data/ukb_olink_i0_diabetic.csv')
df_olink_h   = load_and_clean_olink('Original Data/ukb_olink_i0_healthy.csv')
df_olink_all = pd.concat([df_olink_d, df_olink_h], ignore_index=True).drop_duplicates('eid')
print(f'Combined Olink: {len(df_olink_all)} unique eids')
print()

for path, olink in [
    ('Training Data/ukb_full_diabetics.csv',      df_olink_d),
    ('Training Data/ukb_full_healthy.csv',         df_olink_h),
    ('Training Data/ukb_full_ifg.csv',             df_olink_all),
    ('Training Data/t2d_pilot_50_diabetics.csv',   df_olink_d),
    ('Training Data/t2d_pilot_50_healthy.csv',     df_olink_h),
    ('Training Data/t2d_pilot_50_ifg.csv',         df_olink_all),
]:
    merge_olink_inplace(path, olink)

print('\nDone. Olink columns merged in-place, no new files.')

# ── cohort counts
print('\ncohort counts:')

OLINK_COLS = ['il6', 'tnf', 'igf2bp3', 'adipoq', 'lep', 'gdf15']
SEX_COL    = 'p31'   # 1=male, 0=female (UKB coding 9)

cohort_files = {
    'T2D':     'Training Data/ukb_full_diabetics.csv',
    'IFG':     'Training Data/ukb_full_ifg.csv',
    'Healthy': 'Training Data/ukb_full_healthy.csv',
}

sankey_rows = []
for cohort, fpath in cohort_files.items():
    df_c = pd.read_csv(fpath)
    n    = len(df_c)

    n_male   = int((df_c[SEX_COL] == 1).sum()) if SEX_COL in df_c.columns else -1
    n_female = int((df_c[SEX_COL] == 0).sum()) if SEX_COL in df_c.columns else -1

    present = [c for c in OLINK_COLS if c in df_c.columns]
    n_olink = int(df_c[present].notna().any(axis=1).sum()) if present else 0

    print(f'\n{cohort} (N={n}):')
    if n_male >= 0:
        print(f'  sex    male: {n_male} ({100*n_male/n:.1f}%)  |  female: {n_female} ({100*n_female/n:.1f}%)')
    print(f'  olink  any non-NaN: {n_olink}/{n} ({100*n_olink/n:.1f}%)')
    sankey_rows.append((cohort, n, n_male, n_female))

print('\nsex split (for title page Sankey):')
total = sum(r[1] for r in sankey_rows)
print(f'  total N={total}')
for cohort, n, nm, nf in sankey_rows:
    print(f'  {cohort:8s}: N={n}  male={nm} ({100*nm/n:.1f}%)  female={nf} ({100*nf/n:.1f}%)')

# ── some stat checks
# baseline characteristics per cohort - rough Table 1 to eyeball before writing it up
char_cols = {
    'age':     'p21022',
    'bmi':     'p21001_i0',
    'glucose': 'G_mgdl',
    'hba1c':   'p30750_i0',   # mmol/mol
    'crp':     'p30710_i0',
}

print('\nbaseline characteristics (mean +/- sd):')
for cohort, fpath in cohort_files.items():
    df_c = pd.read_csv(fpath)
    print(f'\n{cohort} (N={len(df_c)}):')
    for name, col in char_cols.items():
        if col not in df_c.columns:
            print(f'  {name:8s} -- column missing')
            continue
        x  = pd.to_numeric(df_c[col], errors='coerce')
        dp = 2 if name == 'crp' else 1
        print(f'  {name:8s} {x.mean():.{dp}f} +/- {x.std():.{dp}f}  (n={int(x.notna().sum())})')