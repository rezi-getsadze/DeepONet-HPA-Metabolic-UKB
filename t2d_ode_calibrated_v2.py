import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import string
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp as sivp
from numpy import sin, cos, exp, sqrt, mod, heaviside

# figure palette
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
_os.makedirs('output', exist_ok=True)
class _Tee:
    def __init__(self, path):
        import os as _mkdir_os
        _mkdir_os.makedirs(_mkdir_os.path.dirname(path), exist_ok=True)
        self._f = open(path, 'w', buffering=1)
        self._orig = _sys.stdout
    def write(self, x): self._orig.write(x); self._f.write(x)
    def flush(self): self._orig.flush(); self._f.flush()

plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Inter', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':          9,
    'axes.titlesize':    10,
    'axes.titleweight':  'normal',
    'axes.titlepad':      6,
    'axes.labelsize':     9,
    'axes.labelpad':      4,
    'axes.labelcolor':   THESIS_PALETTE['text'],
    'axes.edgecolor':    THESIS_PALETTE['text'],
    'axes.linewidth':    0.8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.facecolor':    THESIS_PALETTE['bg'],
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'xtick.color':       THESIS_PALETTE['text'],
    'ytick.color':       THESIS_PALETTE['text'],
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'legend.fontsize':       8,
    'legend.frameon':        False,
    'legend.handlelength':  1.5,
    'legend.handletextpad': 0.5,
    'lines.linewidth':  1.2,
    'lines.markersize': 4,
    'grid.color':     THESIS_PALETTE['grid'],
    'grid.linewidth': 0.5,
    'grid.linestyle': '-',
    'figure.facecolor':   THESIS_PALETTE['bg'],
    'figure.dpi':         100,
    'savefig.dpi':        300,
    'savefig.facecolor':  THESIS_PALETTE['bg'],
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
    'savefig.format':     'svg',
    'svg.fonttype': 'none',  # keep text editable when i open in inkscape
    'mathtext.fontset': 'custom',
    'mathtext.rm':      'Inter',
    'mathtext.it':      'Inter:italic',
    'mathtext.bf':      'Inter:bold',
    'axes.prop_cycle': plt.cycler(color=[
        THESIS_PALETTE['slate'],
        THESIS_PALETTE['t2d'],
        THESIS_PALETTE['healthy'],
        THESIS_PALETTE['aubergine'],
        THESIS_PALETTE['ifg'],
        THESIS_PALETTE['grey'],
    ]),
})

COL_WIDTH  = 6.85
HALF_WIDTH = 3.35
import sys as _sys, os as _os

class _Tee:
    def __init__(self, path):
        _os.makedirs(_os.path.dirname(path), exist_ok=True)
        self._f = open(path, 'w', buffering=1)
        self._orig = _sys.stdout
    def write(self, x): self._orig.write(x); self._f.write(x)
    def flush(self): self._orig.flush(); self._f.flush()
    def close(self): self._f.close()
_tee = _Tee('runtime/cal/cal_output.txt')
_sys.stdout = _tee


_os.makedirs('runtime/cal', exist_ok=True)
_os.makedirs('output', exist_ok=True)
_vcal_n      = [0]
_vcal_saved  = [False]
_vcal_orig_sf  = plt.savefig
_vcal_orig_cl  = plt.close
_vcal_orig_fig = plt.figure
_vcal_orig_sp  = plt.subplots

def _vcal_sf(*a, **kw):
    _vcal_saved[0] = True
    return _vcal_orig_sf(*a, **kw)

def _vcal_cl(*a, **kw):
    if not _vcal_saved[0] and plt.get_fignums():
        _vcal_n[0] += 1
        _vcal_orig_sf(f'runtime/cal/cal_fig{_vcal_n[0]}.svg', bbox_inches='tight')
    _vcal_saved[0] = False
    return _vcal_orig_cl(*a, **kw)

def _vcal_figure(*a, **kw):
    _vcal_saved[0] = False
    return _vcal_orig_fig(*a, **kw)

def _vcal_subplots(*a, **kw):
    _vcal_saved[0] = False
    return _vcal_orig_sp(*a, **kw)

plt.savefig  = _vcal_sf
plt.close    = _vcal_cl
plt.figure   = _vcal_figure
plt.subplots = _vcal_subplots

# calibrated ODE - full cohort (N=1,798)
# key result: individualized scaling separates phenotypes without group-level refitting
# IFG mean delta G = -10 to -11 mg/dL, healthy 0 crashes, diabetics ~34-39% crash
# contrast: uncalibrated collapses everyone to ~88 mg/dL

# ── parameters
# same as uncalibrated except racor=0.08 (reduced from 0.54 to prevent NaN at cort>2.85)
# calibration constants defined below - not in the pars dict, added separately

pars = {}
inf = np.inf

pars['w1']=244.8; pars['w2']=50.4; pars['w3']=12.384; pars['kgr']=4; pars['n']=3
pars['whpa']=0.02083; pars['lam']=1; pars['wcrhe']=23.04; pars['wc']=0.02; pars['wa']=0.01
pars['wr']=0.023; pars['wcr']=0; pars['ub']=1; pars['uon']=0; pars['uoff']=3600
pars['uinp']=1; pars['don']=0; pars['doff']=0; pars['dinp']=20; pars['per']=30
pars['slt']=20; pars['sw']=1; pars['eg0']=24.48; pars['k']=700; pars['bv']=5
pars['mmax']=1; pars['alpha_m']=140; pars['km']=2; pars['alpha_isr']=1.2; pars['kisr']=2
pars['pmax']=4.55; pars['kp']=4; pars['alpha_p']=35; pars['p_b']=0; pars['amax']=5
pars['alpha_a']=0.37; pars['ka']=6; pars['a_b']=0.9; pars['tau_b']=1800; pars['height']=1.66
pars['age_b']=64; pars['sex']=1; pars['cage']=0; pars['target_si']=1.8; pars['tau_si']=1
pars['bmi_h']=22; pars['mffa']=0.8; pars['ksi_infl']=4.0; pars['ksi_ffa']=400
pars['nsi_ffa']=6; pars['sw11']=1; pars['inc_i1']=0; pars['inc_i2']=0; pars['inc_i3']=0
pars['it1']=0; pars['it2']=inf; pars['it3']=inf; pars['kpc']=7700; pars['tau_w']=1
pars['k_infl']=6.0; pars['n_infl']=2; pars['infl_b']=0; pars['tau_infl']=1; pars['hgp_bas']=2000
pars['hepa_max']=3000; pars['hepa_sic']=1; pars['a_hgp']=4; pars['gcg']=0.1; pars['k_gcg']=0
pars['s']=0.0002; pars['png']=350; pars['sigma_b']=600; pars['sgmu']=1.5; pars['sgmd']=1
pars['sfm']=1.2; pars['sim']=0.25; pars['sgku']=81; pars['sgkd']=137; pars['sfk']=357
pars['sik']=0.6; pars['nsgku']=6; pars['nsgkd']=6; pars['nsfk']=6; pars['nsik']=4
pars['tau_sigma']=1; pars['sci']=1; pars['gclamp']=0; pars['iclamp']=0; pars['maxinfl']=1
pars['l0']=170; pars['l2']=8.1; pars['cf']=2; pars['ksif']=11; pars['aa']=2; pars['sif']=1
pars['ffa_ramax']=1.12; pars['max_w']=0.5; pars['kcort']=1; pars['ncort']=2; pars['kcortsi']=1.0
pars['k_fmass']=1.2; pars['n_fmass']=2; pars['max_cort']=0.5; pars['p_vf_base']=0.2
pars['p_vf_amp']=0.2; pars['k_p']=1; pars['n_p']=2; pars['maxinfl_vf']=0.9; pars['maxinfl_sf']=0.1
pars['max_insulin']=0.1; pars['k_i']=15; pars['n_i']=2; pars['intake_i']=2380
pars['kcrd']=61.9; pars['khgp']=16.1; pars['chgp']=0.027
pars['racor']=0.08  # reduced from 0.54 - original caused NaN when cort>2.85
pars['racor1']=0.03; pars['maxrd']=215; pars['nhgp']=2; pars['ncrd']=2; pars['ncsi']=2; pars['tata']=2

p_base = np.array(list(pars.values()))

# calibration constants - found by binary search (eg0_ref) and fine alpha sweep
G_REF     = 86.5      # healthy reference glucose anchoring the attractor
eg0_ref   = 56.2964   # eg0 that puts the healthy ODE fixed point at 86.5 mg/dL
ALPHA_CAL = 0.984     # per-patient eg0 scaling exponent
ALPHA_HGP = 0.3       # per-patient hgp_bas scaling exponent
EG0_FLOOR = 0.75      # kept from the pilot, never binds here

# ── ODE system
def odde(t, y, pars_npa):
	heav = lambda x: np.heaviside(x, 1)

	crh=y[0]; acth=y[1]; cort=y[2]; cmass=y[3]; amass=y[4]
	cr=y[5]; ext_crh=y[6]; g=y[7]; i=y[8]; ffa=y[9]
	si=y[10]; b=y[11]; sigma=y[12]; infl=y[13]; w=y[14]
	fmass=y[15]; vf=y[16]; sf=y[17]

	w1=pars_npa[0]; w2=pars_npa[1]; w3=pars_npa[2]; kgr=pars_npa[3]; n=pars_npa[4]
	whpa=pars_npa[5]; lam=pars_npa[6]; wcrhe=pars_npa[7]; wc=pars_npa[8]; wa=pars_npa[9]
	wr=pars_npa[10]; wcr=pars_npa[11]; ub=pars_npa[12]; uon=pars_npa[13]; uoff=pars_npa[14]
	uinp=pars_npa[15]; don=pars_npa[16]; doff=pars_npa[17]; dinp=pars_npa[18]
	per=pars_npa[19]; slt=pars_npa[20]; sw=pars_npa[21]; eg0=pars_npa[22]; k=pars_npa[23]
	bv=pars_npa[24]; mmax=pars_npa[25]; alpha_m=pars_npa[26]; km=pars_npa[27]
	alpha_isr=pars_npa[28]; kisr=pars_npa[29]; pmax=pars_npa[30]; kp=pars_npa[31]
	alpha_p=pars_npa[32]; p_b=pars_npa[33]; amax=pars_npa[34]; alpha_a=pars_npa[35]
	ka=pars_npa[36]; a_b=pars_npa[37]; tau_b=pars_npa[38]; height=pars_npa[39]
	age_b=pars_npa[40]; sex=pars_npa[41]; cage=pars_npa[42]; target_si=pars_npa[43]
	tau_si=pars_npa[44]; bmi_h=pars_npa[45]; mffa=pars_npa[46]; ksi_infl=pars_npa[47]
	ksi_ffa=pars_npa[48]; nsi_ffa=pars_npa[49]; sw11=pars_npa[50]; inc_i1=pars_npa[51]
	inc_i2=pars_npa[52]; inc_i3=pars_npa[53]; it1=pars_npa[54]; it2=pars_npa[55]
	it3=pars_npa[56]; kpc=pars_npa[57]; tau_w=pars_npa[58]; k_infl=pars_npa[59]
	n_infl=pars_npa[60]; infl_b=pars_npa[61]; tau_infl=pars_npa[62]; hgp_bas=pars_npa[63]
	hepa_max=pars_npa[64]; hepa_sic=pars_npa[65]; a_hgp=pars_npa[66]; gcg=pars_npa[67]
	k_gcg=pars_npa[68]; s=pars_npa[69]; png=pars_npa[70]; sigma_b=pars_npa[71]
	sgmu=pars_npa[72]; sgmd=pars_npa[73]; sfm=pars_npa[74]; sim=pars_npa[75]
	sgku=pars_npa[76]; sgkd=pars_npa[77]; sfk=pars_npa[78]; sik=pars_npa[79]
	nsgku=pars_npa[80]; nsgkd=pars_npa[81]; nsfk=pars_npa[82]; nsik=pars_npa[83]
	tau_sigma=pars_npa[84]; sci=pars_npa[85]; gclamp=pars_npa[86]; iclamp=pars_npa[87]
	maxinfl=pars_npa[88]; l0=pars_npa[89]; l2=pars_npa[90]; cf=pars_npa[91]
	ksif=pars_npa[92]; aa=pars_npa[93]; sif=pars_npa[94]; ffa_ramax=pars_npa[95]
	max_w=pars_npa[96]; kcort=pars_npa[97]; ncort=pars_npa[98]; kcortsi=pars_npa[99]
	k_fmass=pars_npa[100]; n_fmass=pars_npa[101]; max_cort=pars_npa[102]
	p_vf_base=pars_npa[103]; p_vf_amp=pars_npa[104]; k_p=pars_npa[105]; n_p=pars_npa[106]
	maxinfl_vf=pars_npa[107]; maxinfl_sf=pars_npa[108]; max_insulin=pars_npa[109]
	k_i=pars_npa[110]; n_i=pars_npa[111]; intake_i=pars_npa[112]
	kcrd=pars_npa[113]; khgp=pars_npa[114]; chgp=pars_npa[115]
	racor=pars_npa[116]; racor1=pars_npa[117]; maxrd=pars_npa[118]
	nhgp=pars_npa[119]; ncrd=pars_npa[120]; ncsi=pars_npa[121]; tata=pars_npa[122]

	u1 = ub + (heav(t-uon) - heav(t-uoff)) * uinp
	d  = 0  + (heav(t-don) - heav(t-doff)) * dinp
	uw = ub + uinp * heav(mod(t, per) - slt)
	u  = sw * u1 + (1-sw) * uw

	stim_crh = crh + ext_crh
	gcort = 1 / (1 + (cort/kgr)**n); mcort = 1 / cort
	bmi = w / height**2; age = age_b + cage * t / 365
	tsi = (kcortsi**ncsi / (kcortsi**ncsi + (cort-1)**ncsi)) * target_si * \
	      (ksi_infl / (ksi_infl + infl)) * \
	      (1 - mffa * ffa**nsi_ffa / (ffa**nsi_ffa + ksi_ffa**nsi_ffa))
	inc_int = (0 + heav(t-it1)*heav(it2-t)*heav(it3-t)*inc_i1
	           + heav(t-it2)*heav(it3-t)*inc_i2 + heav(t-it3)*inc_i3)
	activity_level = 1.6
	bmr = sex * (
	    (age < 30)              * (15.057*w + 692.2) +
	    ((age >= 30) & (age < 60)) * (11.472*w + 873.1) +
	    (age >= 60)             * (11.711*w + 587.7)
	) + (1-sex) * (
	    (age < 30)              * (14.818*w + 486.6) +
	    ((age >= 30) & (age < 60)) * (8.126*w  + 845.6) +
	    (age >= 60)             * (9.082*w  + 658.5)
	)
	expd = activity_level * bmr
	k_w = 1 / kpc; m = mmax * g**km / (alpha_m**km + g**km)
	isr = sigma * m**kisr / (alpha_isr**kisr + m**kisr)
	p_isr = p_b + pmax * isr**kp / (alpha_p**kp + isr**kp)
	a_m = a_b + amax * m**ka / (alpha_a**ka + m**ka); p = p_isr; a = a_m
	ffm = w - fmass; inc = 1 + inc_int; hepa_si = hepa_sic * si
	hgp = hgp_bas + (1+chgp*(cort-1)) * hepa_max * khgp**nhgp / (khgp**nhgp + (hepa_si*i)**nhgp)
	intake = intake_i * inc
	s_glucu = sgmu*g**nsgku / (g**nsgku + sgku**nsgku)
	s_glucd = sgmd*g**nsgkd / (g**nsgkd + sgkd**nsgkd)
	s_ffa   = sfm*ffa**nsfk / (ffa**nsfk + sfk**nsfk)
	s_infl  = sim*infl**nsik / (infl**nsik + sik**nsik)
	s_inf   = sigma_b * (s_glucu - s_glucd*s_ffa - s_infl)
	siff = si * sif; cl0 = l0 * 24*60 / bv; cl2 = l2 * 24*60 / bv
	p_vf_base = sex*0.25 + (1-sex)*0.15
	p_vf = p_vf_base + p_vf_amp*(cort-1)**n_p / ((cort-1)**n_p + k_p**n_p); p_sf = 1.0 - p_vf
	racor_eff  = np.clip(racor  * (cort-1), 0.0, 0.92)
	racor1_eff = np.clip(racor1 * (cort-1), 0.0, 0.92)
	rd = (eg0 + sci*(si*i) + (1-racor1_eff)*maxrd*(si*i)**ncrd /
	      ((1-racor_eff)*kcrd**ncrd + (si*i)**ncrd)) * g

	dcrh=w1*(gcort*mcort*u - crh); dacth=w2*(cmass*gcort*stim_crh - acth)
	dcort=w3*(amass*acth - cort/cr); dcmass=wc*cmass*(stim_crh - 1); damass=wa*amass*(acth - 1)
	dcr=wcr*(cort - cr); dext_crh=wcrhe*(d - ext_crh)
	dg=gclamp + hgp - rd; di=iclamp + b*isr/bv - k*i
	dffa=(cl0 + cl2*fmass)*(ffa_ramax*ksif**aa / (ksif**aa + (siff*i)**aa)) - cf*w*ffa
	dsi=(tsi - si) / tau_si; db=(png + p*b - a*b - s*b**2) / tau_b; dsigma=(s_inf - sigma) / tau_sigma
	vf_term = vf**n_infl / (vf**n_infl + k_infl**n_infl)
	sf_term = sf**n_infl / (sf**n_infl + k_infl**n_infl)
	infl_drive = maxinfl_vf*vf_term + maxinfl_sf*sf_term; dinfl = infl_b + (infl_drive - infl) / tau_infl
	i_eff = max(0, i - 10.0); phi_insulin = max_insulin * i_eff**n_i / (k_i**n_i + i_eff**n_i)
	cortisol_factor = 1 + max_w*(cort-1)**ncort / (kcort**ncort + (cort-1)**ncort)
	phi = 0.5 * intake * k_w * phi_insulin * cortisol_factor
	dw = (intake*k_w - expd*k_w + phi) / tau_w; dfmass = (0.35*(intake*k_w - expd*k_w) + phi) / tau_w
	dvf = p_vf * dfmass; dsf = p_sf * dfmass
	cort = max(y[2], 0.001); g = max(y[7], 0.001)
	return np.array([dcrh,dacth,dcort,dcmass,damass,dcr,dext_crh,dg,di,dffa,dsi,db,dsigma,dinfl,dw,dfmass,dvf,dsf])

# ── helper with individualized scaling
# stress: UKB 0-1 score mapped to physiological 18.7-25.0 range
# eg0, hgp_bas: scaled per-patient based on ratio to healthy reference glucose
def run_twin_sim(patient, current_pars):
    y0 = [1,1,1,1,1,1,0, patient['G_mgdl'], 9,400,0.8,900,536,0.73,77,18.7,2.8,15.9]
    p_sim = current_pars.copy()
    p_sim[15] = 18.7 + patient['S_input'] * 6.3  # map UKB 0-1 to physiological 18.7-25
    ratio = patient['G_mgdl'] / G_REF
    eg0_i = eg0_ref * (1.0 / ratio) ** ALPHA_CAL
    if patient['G_mgdl'] > 126:
        eg0_i = min(eg0_i, EG0_FLOOR * eg0_ref)  # pilot floor, inactive at these thresholds
    p_sim[22] = eg0_i; p_sim[63] = p_base[63] * (ratio ** ALPHA_HGP)
    try:
        sol = sivp(odde, [0, 7200], y0, method='LSODA', args=[p_sim])
        return sol.y[7][-1]
    except: return patient['G_mgdl']

# ── cohort runs
df = pd.read_csv("Training Data/ukb_full_diabetics.csv")
final_results = []
for idx, patient in df.iterrows():
    y0 = [1,1,1,1,1,1,0, patient['G_mgdl'], 9,400,0.8,900,536,0.73,77,18.7,2.8,15.9]
    p_fail = p_base.copy(); p_fail[15] = 18.7 + patient['S_input'] * 6.3
    ratio = patient['G_mgdl'] / G_REF
    eg0_i = eg0_ref * (1.0 / ratio) ** ALPHA_CAL
    if patient['G_mgdl'] > 126: eg0_i = min(eg0_i, EG0_FLOOR * eg0_ref)
    p_fail[22] = eg0_i; p_fail[63] = p_base[63] * (ratio ** ALPHA_HGP)
    try:
        sol = sivp(odde, [0, 7200], y0, method='LSODA', args=[p_fail])
        g_start, g_end = sol.y[7][0], sol.y[7][-1]
        status = 'Crash' if g_end > (g_start + 5) else 'Stable'
        final_results.append({'eid': patient['eid'], 'Start': g_start, 'End': g_end, 'Delta': g_end - g_start, 'Status': status})
    except: continue

res_df = pd.DataFrame(final_results); print(res_df['Status'].value_counts())

df_h = pd.read_csv("Training Data/ukb_full_healthy.csv")
healthy_results = []
for idx, patient in df_h.iterrows():
    y0 = [1,1,1,1,1,1,0, patient['G_mgdl'], 9,400,0.8,900,536,0.73,77,18.7,2.8,15.9]
    p_fail = p_base.copy(); p_fail[15] = 18.7 + patient['S_input'] * 6.3
    ratio = patient['G_mgdl'] / G_REF
    eg0_i = eg0_ref * (1.0 / ratio) ** ALPHA_CAL
    if patient['G_mgdl'] > 126: eg0_i = min(eg0_i, EG0_FLOOR * eg0_ref)
    p_fail[22] = eg0_i; p_fail[63] = p_base[63] * (ratio ** ALPHA_HGP)
    try:
        sol = sivp(odde, [0, 7200], y0, method='LSODA', args=[p_fail])
        g_start, g_end = sol.y[7][0], sol.y[7][-1]
        status = 'Crash' if g_end > (g_start + 5) else 'Stable'
        healthy_results.append({'eid': patient['eid'], 'Start': g_start, 'End': g_end, 'Delta': g_end - g_start, 'Status': status})
    except: continue

res_df_healthy = pd.DataFrame(healthy_results)
print(f"Healthy: {res_df_healthy['Status'].value_counts().get('Crash', 0)} crashes out of {len(res_df_healthy)}")

df_ifg = pd.read_csv("Training Data/ukb_full_ifg.csv")
ifg_results = []
for idx, patient in df_ifg.iterrows():
    y0 = [1,1,1,1,1,1,0, patient['G_mgdl'], 9,400,0.8,900,536,0.73,77,18.7,2.8,15.9]
    p_fail = p_base.copy(); p_fail[15] = 18.7 + patient['S_input'] * 6.3
    ratio = patient['G_mgdl'] / G_REF
    eg0_i = eg0_ref * (1.0 / ratio) ** ALPHA_CAL
    # EG0_FLOOR doesn't activate for IFG (G0<126)
    # IFG dips to ~healthy by day 5 after the stress input stops, then drifts back over weeks
    if patient['G_mgdl'] > 126: eg0_i = min(eg0_i, EG0_FLOOR * eg0_ref)
    p_fail[22] = eg0_i; p_fail[63] = p_base[63] * (ratio ** ALPHA_HGP)
    try:
        sol = sivp(odde, [0, 7200], y0, method='LSODA', args=[p_fail])
        g_start, g_end = sol.y[7][0], sol.y[7][-1]
        status = 'Crash' if g_end > (g_start + 5) else 'Stable'
        ifg_results.append({'eid': patient['eid'], 'Start': g_start, 'End': g_end, 'Delta': g_end - g_start, 'Status': status})
    except: continue

res_df_ifg = pd.DataFrame(ifg_results)
print(f"IFG: {res_df_ifg['Status'].value_counts().get('Crash', 0)} crashes, mean delta G: {res_df_ifg['Delta'].mean():.2f} mg/dL")

# ── combined data prep
ukb_all = np.concatenate([df['G_mgdl'], df_h['G_mgdl'], df_ifg['G_mgdl']])
sim_all = np.concatenate([res_df['End'], res_df_healthy['End'], res_df_ifg['End']])
bins = [0, 100, 126, 1000]; labels = ['Normal', 'IFG', 'Diabetes']
print(f"Cohort delta G: diabetic={res_df['Delta'].mean():.2f}, IFG={res_df_ifg['Delta'].mean():.2f}, healthy={res_df_healthy['Delta'].mean():.2f} mg/dL")

plot_data = pd.DataFrame({'Glucose': np.concatenate([ukb_all, sim_all]),
                           'Source': ['UKB Actual']*len(ukb_all) + ['Model Simulated']*len(sim_all)})
ukb_sorted = np.sort(ukb_all); sim_sorted = np.sort(sim_all)
percs = np.linspace(0, 100, 100)
q_ukb = np.percentile(ukb_sorted, percs); q_sim = np.percentile(sim_sorted, percs)
counts_ukb = pd.Series(pd.cut(ukb_all, bins=bins, labels=labels)).value_counts(normalize=True).reindex(labels).values
counts_sim = pd.Series(pd.cut(sim_all, bins=bins, labels=labels)).value_counts(normalize=True).reindex(labels).values

# ── resilience scatter
fig, ax = plt.subplots(figsize=(12, 7))
cohort_cfg = [(res_df, '#e74c3c', '#3498db', 'o', 'T2D'), (res_df_ifg, '#e67e22', '#f39c12', 's', 'IFG'),
              (res_df_healthy, '#85C1E9', '#2ecc71', '^', 'Healthy')]
for rdf, c_crash, c_stable, marker, name in cohort_cfg:
    for status, color, label in [('Crash', c_crash, f'{name} - Crash'), ('Stable', c_stable, f'{name} - Stable')]:
        sub = rdf[rdf['Status'] == status]
        ax.scatter(sub['Start'], sub['Delta'], c=color, label=label, marker=marker, s=70, edgecolors='k', alpha=0.8)
ax.axhline(0, color='black', ls='--', lw=1.5, label='Stability baseline')
ax.axvline(100, color='orange', ls=':', lw=1, alpha=0.7); ax.axvline(126, color='red', ls=':', lw=1, alpha=0.7)
ax.set_xlabel("Baseline glucose (mg/dL)"); ax.set_ylabel("Delta G after 5 days")
ax.set_title("Resilience across three cohorts (calibrated - phenotypes separated)")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); ax.grid(True, alpha=0.3)
plt.tight_layout();

# ── 4-panel distributional figure
c_ukb = '#808080'; c_sim = '#0000FF'; c_ifg = '#FFD700'; c_dia = '#FF0000'
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
ax_qq, ax_ecdf, ax_dens, ax_violin = axes.flat

ax_qq.scatter(q_ukb, q_sim, color=c_sim, alpha=0.6, s=40, edgecolors='k', zorder=3)
ax_qq.plot([0, 400], [0, 400], color='green', ls='--', label='Perfect fit', lw=2)
ax_qq.set_title("A. Q-Q plot", loc='left')
ax_qq.set_xlabel('Clinical quantiles (mg/dL)'); ax_qq.set_ylabel('Simulated quantiles (mg/dL)')
ax_qq.set_xlim(50, 325); ax_qq.set_ylim(50, 325); ax_qq.set_box_aspect(1)
ax_qq.grid(True, alpha=0.3, ls='--'); ax_qq.legend()

ax_ecdf.step(ukb_sorted, np.linspace(0, 1, len(ukb_sorted)), color=c_ukb, lw=2.5, label='UKB Clinical')
ax_ecdf.step(sim_sorted, np.linspace(0, 1, len(sim_sorted)), color=c_sim, lw=2.5, label='Model Simulated')
ax_ecdf.axvline(100, color=c_ifg, ls='--', alpha=0.8, label='IFG (100)')
ax_ecdf.axvline(126, color=c_dia, ls='--', alpha=0.8, label='Diabetes (126)')
ax_ecdf.set_title("B. ECDF comparison", loc='left')
ax_ecdf.set_xlabel('Glucose (mg/dL)'); ax_ecdf.set_ylabel('Cumulative proportion')
ax_ecdf.set_box_aspect(1); ax_ecdf.legend(loc='lower right'); ax_ecdf.grid(True, alpha=0.2)

sns.kdeplot(ukb_all, ax=ax_dens, fill=True, color=c_ukb, alpha=0.8, bw_adjust=1.5, label='UKB Clinical')
sns.kdeplot(sim_all, ax=ax_dens, fill=True, color=c_sim, alpha=0.8, bw_adjust=0.5, label='Model Simulated')
ax_dens.axvline(100, color=c_ifg, ls='--', lw=2); ax_dens.axvline(126, color=c_dia, ls='--', lw=2)
ax_dens.set_title("C. Density overlay", loc='left')
ax_dens.set_xlabel('Glucose (mg/dL)'); ax_dens.set_ylabel('Density')
ax_dens.set_ylim(0, 0.03); ax_dens.set_xlim(60, 350); ax_dens.legend(); ax_dens.grid(True, alpha=0.2)

sns.violinplot(ax=ax_violin, x='Source', y='Glucose', data=plot_data, inner='quartile', hue='Source', palette={'UKB Actual': c_ukb, 'Model Simulated': c_sim}, legend=False)
ax_violin.axhline(126, color=c_dia, ls='--', alpha=0.8); ax_violin.axhline(100, color=c_ifg, ls='--', alpha=0.8)
ax_violin.set_title("D. Quartile distribution", loc='left')
ax_violin.set_ylabel("Glucose (mg/dL)")
plt.tight_layout();

# ── clinical proportions
x = np.arange(3); width = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, counts_ukb, width, label='UKB Actual',      color=c_ukb, edgecolor='black')
ax.bar(x + width/2, counts_sim, width, label='Model Simulated', color=c_sim, edgecolor='black', alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels(['Normal', 'IFG', 'Diabetes'])
ax.set_ylabel('Proportion'); ax.set_title('Clinical proportions: UKB vs simulation')
ax.set_ylim(0, 1.1); ax.grid(axis='y', alpha=0.3); ax.legend()
plt.tight_layout();

# ── sensitivity: kcortsi sweep
kcortsi_range = np.linspace(0.1, 2.5, 10); diabetes_proportions = []
kc_idx = list(pars.keys()).index('kcortsi')
print("starting kcortsi sweep...")
for kval in kcortsi_range:
    p_sweep = p_base.copy(); p_sweep[kc_idx] = kval
    sweep_results = [run_twin_sim(twin, p_sweep) for _, twin in df.iterrows()]
    diab_count = sum(1 for g in sweep_results if g >= 126)
    diabetes_proportions.append(diab_count / len(df))
    print(f"  kcortsi={kval:.2f} -> {diab_count/len(df)*100:.1f}% diabetic")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(kcortsi_range, diabetes_proportions, marker='o', lw=2, color='red')
ax.axhline(1.0, color='black', ls=':', label='100% diabetic')
ax.set_xlabel('kcortsi'); ax.set_ylabel('Fraction in diabetic range')
ax.set_title('Sensitivity to cortisol-insulin coupling')
ax.grid(True, alpha=0.3); ax.legend(); plt.tight_layout();

# ── 2D calibration heatmap
kc_range = np.linspace(0.5, 5.0, 10); si_range_sw = np.linspace(0.5, 1.8, 10)
results_grid = []
for kc in kc_range:
    row_results = []
    for si_target in si_range_sw:
        p_sweep = p_base.copy(); p_sweep[99] = kc; p_sweep[43] = si_target
        combo_glucose = [run_twin_sim(twin, p_sweep) for _, twin in df.iterrows()]
        row_results.append(np.mean(combo_glucose))
    results_grid.append(row_results); print(f"  finished kc={kc:.2f}")

sweep_df = pd.DataFrame(results_grid, index=np.round(kc_range, 2), columns=np.round(si_range_sw, 2))
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(sweep_df, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
ax.set_title('Calibration heatmap: mean T2D glucose (target range ~178 mg/dL)')
ax.set_xlabel('target_si'); ax.set_ylabel('kcortsi')
plt.tight_layout();

# ── sensitivity tornado
# +/-10% perturbation on 6 mechanistically relevant parameters
# first 10 patients is enough to get the ranking right

test_params = {'target_si': 43, 'hgp_bas': 63, 'eg0': 22, 'kcortsi': 99, 'racor': 116, 'k': 23}
_label_map = {'target_si': r'$SI_{target}$', 'hgp_bas': r'$HGP_{bas}$', 'eg0': r'$eg_0$',
              'kcortsi': r'$k_{cortsi}$', 'racor': r'$r_{acor}$', 'k': r'$k$'}

base_mean = np.mean([run_twin_sim(p, p_base) for _, p in df.head(10).iterrows()])
tornado_results = []
for name, idx in test_params.items():
    p_up = p_base.copy(); p_up[idx] *= 1.1
    p_down = p_base.copy(); p_down[idx] *= 0.9
    mu_up   = np.mean([run_twin_sim(p, p_up)   for _, p in df.head(10).iterrows()])
    mu_down = np.mean([run_twin_sim(p, p_down) for _, p in df.head(10).iterrows()])
    tornado_results.append((name, mu_up - base_mean, mu_down - base_mean))

tornado_results.sort(key=lambda x: abs(x[1]), reverse=True)
names_raw, ups_raw, downs_raw = zip(*tornado_results)
params_math = [_label_map[n] for n in names_raw]; ups = list(ups_raw); downs = list(downs_raw)

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(params_math, ups,   color='#e74c3c', alpha=0.7, label='+10%', edgecolor='k')
ax.barh(params_math, downs, color='#3498db', alpha=0.7, label='-10%', edgecolor='k')
ax.axvline(0, color='black', lw=1.5)
ax.set_xlabel('Delta glucose (mg/dL)'); ax.set_title('Sensitivity tornado')
ax.legend(); ax.grid(alpha=0.3, axis='x'); ax.invert_yaxis()
plt.tight_layout();

# ── alpha sweep - bimodality emergence
# shows how the bimodal population distribution emerges as alpha increases
# alpha=0: eg0 not scaled with G0
# alpha=0.984: calibrated, two clear peaks emerge
alphas_to_test = [0.1, 0.4, 0.7, 0.984]
sweep_results_list = []

fig, ax = plt.subplots(figsize=(12, 6))
for a in alphas_to_test:
    def sweep_sim(patient, alpha_val):
        p_sim = p_base.copy(); p_sim[15] = 18.7 + patient['S_input'] * 6.3
        ratio = patient['G_mgdl'] / 86.5
        eg0_i = 56.2964 * (1.0 / ratio) ** alpha_val
        if patient['G_mgdl'] > 126: eg0_i = min(eg0_i, 0.75 * 56.2964)
        p_sim[22] = eg0_i; p_sim[63] = 2000.0 * (ratio ** 0.3)
        y0 = [1,1,1,1,1,1,0, patient['G_mgdl'], 9,400,0.8,900,536,0.73,77,18.7,2.8,15.9]
        sol = sivp(odde, [0, 7200], y0, method='LSODA', args=[p_sim])
        return sol.y[7][-1]
    res = [sweep_sim(p, a) for _, p in pd.concat([df, df_h, df_ifg]).iterrows()]
    sweep_results_list.append(res)
    sns.kdeplot(res, ax=ax, label=f'alpha={a}', lw=2, fill=True, alpha=0.1)

ax.axvline(88,  color='green', ls='--', alpha=0.7, label='Healthy target')
ax.axvline(178, color='red',   ls='--', alpha=0.7, label='Diabetic target')
ax.set_title('Bimodality emerges as alpha increases')
ax.set_xlabel('Simulated glucose (mg/dL)'); ax.legend()
plt.tight_layout();

# ── combined panel: tornado + alpha sweep + proportions
fig3, axs3 = plt.subplots(1, 3, figsize=(18, 6))
plt.subplots_adjust(wspace=0.35)

axs3[0].barh(params_math, ups,   color=c_dia, alpha=0.6, label='+10%', edgecolor='k')
axs3[0].barh(params_math, downs, color=c_sim, alpha=0.6, label='-10%', edgecolor='k')
axs3[0].axvline(0, color='black', lw=1.5)
axs3[0].set_xlabel('Delta glucose (mg/dL)')
axs3[0].set_title('A. Sensitivity analysis', loc='left')
axs3[0].invert_yaxis(); axs3[0].legend(); axs3[0].grid(alpha=0.3, axis='x')

for res, a in zip(sweep_results_list, alphas_to_test):
    sns.kdeplot(res, ax=axs3[1], label=f'alpha={a}', lw=2, fill=True, alpha=0.1)
axs3[1].axvline(88,  color=c_ifg, ls='--', label='Healthy target')
axs3[1].axvline(178, color=c_dia, ls='--', label='Diabetic target')
axs3[1].set_title('B. Population distribution evolution', loc='left')
axs3[1].set_xlabel('Simulated glucose (mg/dL)'); axs3[1].set_ylabel('Density')
axs3[1].set_xlim(-50, 500); axs3[1].legend(loc='upper right'); axs3[1].grid(alpha=0.2)

x_pos = np.arange(3); width = 0.35
axs3[2].bar(x_pos - width/2, counts_ukb, width, label='UKB Actual',      color=c_ukb, edgecolor='black')
axs3[2].bar(x_pos + width/2, counts_sim, width, label='Model Simulated', color=c_sim, edgecolor='black', alpha=0.7)
axs3[2].set_xticks(x_pos); axs3[2].set_xticklabels(['Normal', 'IFG', 'Diabetes'])
axs3[2].set_title('C. Clinical proportions', loc='left')
axs3[2].set_ylabel('Proportion'); axs3[2].set_ylim(0, 1.1)
axs3[2].grid(axis='y', alpha=0.3); axs3[2].legend()
plt.tight_layout();