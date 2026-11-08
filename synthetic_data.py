# synthetic_data - UKB-shaped cohorts so the pipeline runs without biobank access

import pandas as pd
import numpy as np
import os

os.makedirs('Original Data', exist_ok=True)
os.makedirs('Training Data', exist_ok=True)
os.makedirs('cache', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# medication codes that pass the exclusion wall
SAFE_MED_CODES = [1140875408, 1140879802, 1140879778, 1140916356, 1140861958]

TREATED_CODES = [
    # diabetes codes that get excluded - subset of preprocess's treated_codes_by_class
    1140884600,
    1140874746,
    1140883066,
    1140874686,
    1140874744,
    1140874690,
    1140874718,
    1140874724,
    1141189090,
    1140874726,
    1141171646,
    1141168660,
    1140874728,
    1140874732,
    1140874736,
    1140874740,
    1140874646,
]

MH_FIELDS = [
    'p20510', 'p20507', 'p20519', 'p20514', 'p20511', 'p20513',
    'p20508', 'p20518', 'p20505', 'p20512', 'p20506', 'p20509',
    'p20516', 'p20515', 'p20520', 'p20517'
]


def generate_raw_cohort(n, cohort_type, seed, start_eid,
                        n_medicated=0, n_longitudinal=0):
    """
    Raw UKB-format cohort with participant. prefix, matching DNAnexus export structure.
    cohort_type: 'diabetic' or 'healthy' (healthy includes IFG rows mixed in).
    n_medicated:   first n_medicated rows get treated drug codes in slot p20003_i0_a0.
    n_longitudinal: first n_longitudinal rows get Instance 1 glucose + date.
    """
    np.random.seed(seed)
    data = {}

    data['participant.eid']    = list(range(start_eid, start_eid + n))
    data['participant.p31']    = np.random.choice([0, 1], n)
    data['participant.p21022'] = np.random.randint(45, 75, n)

    heights = np.random.normal(172, 9, n)

    if cohort_type == 'diabetic':
        bmis         = np.random.uniform(26.0, 42.0, n)
        glucose_mmol = np.random.uniform(7.2, 10.5, n)
        hba1c_i0     = np.random.uniform(48.0, 95.0, n)
        crp_i0       = np.round(np.random.lognormal(1.5, 0.6, n), 2)
        igf1         = np.round(np.random.normal(14, 3, n), 2)
        sys_bp       = np.random.randint(135, 165, n)
        dia_bp       = np.random.randint(85, 105, n)
        mh_probs     = [0.25, 0.35, 0.25, 0.15]
    else:
        # healthy file mixes normoglycemic (~67%) and IFG (~33%)
        is_ifg       = np.random.random(n) < 0.33
        bmis         = np.where(is_ifg,
                                np.random.uniform(22.0, 32.0, n),
                                np.random.uniform(18.5, 27.0, n))
        glucose_mmol = np.where(is_ifg,
                                np.random.uniform(5.56, 6.94, n),
                                np.random.uniform(3.9, 5.4, n))
        hba1c_i0     = np.where(is_ifg,
                                np.random.uniform(39.0, 47.0, n),
                                np.random.uniform(28.0, 41.0, n))
        crp_i0       = np.round(np.where(is_ifg,
                                         np.random.lognormal(0.5, 0.5, n),
                                         np.random.lognormal(0, 0.5, n) * 0.5), 2)
        igf1         = np.round(np.where(is_ifg,
                                         np.random.normal(18, 3, n),
                                         np.random.normal(22, 3, n)), 2)
        sys_bp       = np.where(is_ifg,
                                np.random.randint(120, 145, n),
                                np.random.randint(110, 130, n))
        dia_bp       = np.where(is_ifg,
                                np.random.randint(78, 92, n),
                                np.random.randint(70, 85, n))
        mh_probs     = [0.65, 0.22, 0.10, 0.03]

    weights  = bmis * (heights / 100) ** 2
    hba1c_i1 = hba1c_i0 + np.random.normal(0.5, 2.0, n)
    crp_i1   = np.round(crp_i0 * np.random.uniform(0.9, 1.1, n), 2)
    base_dates = pd.date_range('2007-01-01', '2010-12-31', periods=n)

    p53_i1 = [None] * n
    g_i1   = np.full(n, np.nan)
    for j in range(min(n_longitudinal, n)):
        g_i1[j]   = round(float(glucose_mmol[j]) * np.random.uniform(1.04, 1.18), 2)
        p53_i1[j] = (pd.Timestamp(base_dates[j]) +
                     pd.DateOffset(years=int(np.random.randint(3, 6)))).strftime('%Y-%m-%d')

    data['participant.p53_i0']    = [d.strftime('%Y-%m-%d') for d in base_dates]
    data['participant.p53_i1']    = p53_i1
    data['participant.p53_i2']    = [None] * n
    data['participant.p53_i3']    = [None] * n
    data['participant.p50_i0']    = np.round(heights, 1)
    data['participant.p21002_i0'] = np.round(weights, 1)
    data['participant.p21002_i1'] = np.round(weights * np.random.uniform(0.97, 1.03, n), 1)
    data['participant.p21001_i0'] = np.round(bmis, 1)
    data['participant.p21001_i1'] = np.round(bmis * np.random.uniform(0.97, 1.03, n), 1)
    data['participant.p30740_i0'] = np.round(glucose_mmol, 2)
    data['participant.p30740_i1'] = g_i1
    data['participant.p30750_i0'] = np.round(hba1c_i0, 1)
    data['participant.p30750_i1'] = np.round(hba1c_i1, 1)
    data['participant.p30710_i0'] = crp_i0
    data['participant.p30710_i1'] = crp_i1
    data['participant.p30770_i0'] = igf1
    data['participant.p4080_i0_a0'] = sys_bp
    data['participant.p4080_i0_a1'] = sys_bp + np.random.randint(-2, 3, n)
    data['participant.p4079_i0_a0'] = dia_bp
    data['participant.p4079_i0_a1'] = dia_bp + np.random.randint(-2, 3, n)

    for j in range(8):
        col_i0 = np.random.choice(SAFE_MED_CODES, n)
        col_i1 = np.random.choice(SAFE_MED_CODES, n)
        if n_medicated > 0 and j == 0:
            col_i0[:n_medicated] = np.random.choice(TREATED_CODES, n_medicated)
        data[f'participant.p20003_i0_a{j}'] = col_i0
        data[f'participant.p20003_i1_a{j}'] = col_i1

    for field in MH_FIELDS:
        data[f'participant.{field}'] = np.random.choice([1, 2, 3, 4], n, p=mh_probs)

    return pd.DataFrame(data)


def generate_olink(eids, cohort_type, seed):
    np.random.seed(seed)
    n = len(eids)
    if cohort_type == 'diabetic':
        il6_mu, adipoq_mu, lep_mu = 3.0, 2.2, 3.5
    else:
        il6_mu, adipoq_mu, lep_mu = 1.8, 3.0, 2.2
    return pd.DataFrame({
        'eid':     eids,
        'il6':     np.round(np.random.normal(il6_mu, 0.6, n), 4),
        'tnf':     np.round(np.random.normal(2.0,    0.5, n), 4),
        'igf2bp3': np.round(np.random.normal(1.8,    0.4, n), 4),
        'adipoq':  np.round(np.random.normal(adipoq_mu, 0.5, n), 4),
        'lep':     np.round(np.random.normal(lep_mu, 0.5, n), 4),
        'gdf15':   np.round(np.random.normal(2.5,    0.6, n), 4),
    })


print('generator defined')

# diabetic cohort: 232 unmedicated + 110 medicated
# medicated rows are first so preprocess can split them cleanly
df_diab = generate_raw_cohort(342, 'diabetic', seed=24, start_eid=10000,
                               n_medicated=110, n_longitudinal=4)

df_diab.to_csv('Original Data/ukb_cohort_diabetic.csv', index=False)
print(f'synth_cohort_diabetic.csv: {len(df_diab)} rows')

# healthy cohort: normoglycemic + IFG mixed, enough to sample 1000 + 500 after filtering
df_healthy = generate_raw_cohort(2500, 'healthy', seed=42, start_eid=20000)

df_healthy.to_csv('Original Data/ukb_cohort_healthy.csv', index=False)

glucose_mgdl = df_healthy['participant.p30740_i0'] * 18.0182
print(f'synth_cohort_healthy.csv: {len(df_healthy)} rows')
print(f'  IFG range (100-126):  {((glucose_mgdl >= 100) & (glucose_mgdl < 126)).sum()}')
print(f'  healthy (<100):       {(glucose_mgdl < 100).sum()}')

# Olink proteomics: eid + 6 NPX proteins per cohort
diab_eids    = df_diab['participant.eid'].values
healthy_eids = df_healthy['participant.eid'].values

generate_olink(diab_eids,    'diabetic', seed=77).to_csv(
    'Original Data/ukb_olink_i0_diabetic.csv', index=False)
generate_olink(healthy_eids, 'healthy',  seed=88).to_csv(
    'Original Data/ukb_olink_i0_healthy.csv',  index=False)

print(f'synth_olink_i0_diabetic.csv:  {len(diab_eids)} rows')
print(f'synth_olink_i0_healthy.csv:   {len(healthy_eids)} rows')
print('\nRun t2d_preprocess.py next.')

np.random.seed(7)

G0_VALS  = np.linspace(70, 320, 50)
S_VALS   = np.linspace(0, 0.56, 30)
SI_VALS  = np.linspace(0.8, 2.8, 20)
BMI_VALS = [22, 25, 27, 30]

# SI threshold above which a (BMI, SI_target) combo is two-level. gives 10/14/15/16 two-level per BMI = 55/80 total, matching Snellius
_twolevel_si = {22: SI_VALS[10], 25: SI_VALS[6], 27: SI_VALS[5], 30: SI_VALS[4]}

rows = []
for bmi in BMI_VALS:
    for si in SI_VALS:
        two_level = si >= _twolevel_si[bmi] - 1e-9

        # attractor positions calibrated to Snellius fixed-point output
        fp_h    = 95.0 - 7.4 * (si - SI_VALS[10])  + 0.4 * (bmi - 22) / 8
        fp_d    = 145.0 + 8.0 * (si - 1.0)         + 5.0 * (bmi - 22) / 8
        fp_mono = 148.0 - 5.5 * (si - 0.8)         + 3.0 * (bmi - 22) / 8
        boundary = 128.0

        for s in S_VALS:
            for g0 in G0_VALS:
                # very low SI: solver diverges, no fixed point (8 combinations)
                if si < SI_VALS[2] - 1e-9:
                    rows.append({'G0': round(g0, 4), 'S': round(s, 6),
                                 'SI_target': round(si, 6), 'BMI': bmi,
                                 'G_end': np.nan, 'delta_G': np.nan, 'class': 'crash'})
                    continue

                fp    = (fp_h if g0 <= boundary else fp_d) if two_level else fp_mono
                g_end = round(fp + (g0 - fp) * np.exp(-2.0) + np.random.normal(0, 0.8), 2)
                dg    = round(g_end - g0, 2)
                cls   = 'stable' if abs(dg) <= 5 else ('progressive' if dg > 5 else 'recovering')

                rows.append({'G0': round(g0, 4), 'S': round(s, 6),
                             'SI_target': round(si, 6), 'BMI': bmi,
                             'G_end': g_end, 'delta_G': dg, 'class': cls})

df_sweep = pd.DataFrame(rows)
df_sweep.to_csv('synthetic_sweep_results_full.csv', index=False)   # own name so it never overwrites the real one
print(f'synthetic_sweep_results_full.csv: {len(df_sweep):,} rows')
print(df_sweep['class'].value_counts())