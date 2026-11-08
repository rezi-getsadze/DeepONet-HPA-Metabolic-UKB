import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp as sivp
from numpy import sin, cos, exp, sqrt, mod, heaviside
import string
import os

# v10 - full cohort (N=1,798) + Olink branch features
# extends v8 by adding 6 NPX proteins to the branch net (11 features vs 5, 25,921 params vs 25,537)
# primary result: Olink adds <0.1 mg/dL MAE improvement - null result at scale
# the pilot (v9) showed the same null on N=150. v10 confirms it on the full cohort.

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
_tee = _Tee('runtime/v10/v10_output.txt')
_sys.stdout = _tee


_os.makedirs('runtime/v10', exist_ok=True)
_os.makedirs('output', exist_ok=True)
_vv10_n      = [0]
_vv10_saved  = [False]
_vv10_orig_sf  = plt.savefig
_vv10_orig_cl  = plt.close
_vv10_orig_fig = plt.figure
_vv10_orig_sp  = plt.subplots

def _vv10_sf(*a, **kw):
    _vv10_saved[0] = True
    return _vv10_orig_sf(*a, **kw)

def _vv10_cl(*a, **kw):
    if not _vv10_saved[0] and plt.get_fignums():
        _vv10_n[0] += 1
        _vv10_orig_sf(f'runtime/v10/v10_fig{_vv10_n[0]}.svg', bbox_inches='tight')
    _vv10_saved[0] = False
    return _vv10_orig_cl(*a, **kw)

def _vv10_figure(*a, **kw):
    _vv10_saved[0] = False
    return _vv10_orig_fig(*a, **kw)

def _vv10_subplots(*a, **kw):
    _vv10_saved[0] = False
    return _vv10_orig_sp(*a, **kw)

plt.savefig  = _vv10_sf
plt.close    = _vv10_cl
plt.figure   = _vv10_figure
plt.subplots = _vv10_subplots

# ── parameters
pars={}
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
pars['racor']=0.08; pars['racor1']=0.03; pars['maxrd']=215
pars['nhgp']=2; pars['ncrd']=2; pars['ncsi']=2; pars['tata']=2

p_base = np.array(list(pars.values()))

G_REF     = 86.5
eg0_ref   = 56.2964
ALPHA_CAL = 0.984
ALPHA_HGP = 0.3
EG0_FLOOR = 0.75

# ── cache key
import hashlib as _hashlib, json as _json

def _ode_param_hash(extra=None):
    h = _hashlib.md5(p_base.tobytes())
    calib = {'G_REF': G_REF, 'eg0_ref': eg0_ref,
             'ALPHA_CAL': ALPHA_CAL, 'ALPHA_HGP': ALPHA_HGP, 'EG0_FLOOR': EG0_FLOOR}
    h.update(_json.dumps(calib, sort_keys=True).encode())
    if extra: h.update(_json.dumps(extra, sort_keys=True).encode())
    return h.hexdigest()[:8]

# ── ODE system
def odde(t,y,pars_npa):

	heav=lambda x: np.heaviside(x,1)

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

	u1 = ub+(heav(t-uon)-heav(t-uoff))*uinp
	d = 0+(heav(t-don)-heav(t-doff))*dinp
	uw = ub+uinp*heav(mod(t,per)-slt)
	u = sw*u1+(1-sw)*uw

	stim_crh = crh+ext_crh
	gcort = 1/(1+(cort/kgr)**n); mcort = 1/cort
	bmi = w/height**2; age = age_b+cage*t/365
	tsi = (kcortsi**ncsi/(kcortsi**ncsi+(cort-1)**ncsi))*target_si*(ksi_infl/(ksi_infl+infl))*(1-mffa*ffa**nsi_ffa/(ffa**nsi_ffa+ksi_ffa**nsi_ffa))
	inc_int = 0 + heav(t-it1)*heav(it2-t)*heav(it3-t)*inc_i1 + heav(t-it2)*heav(it3-t)*inc_i2 + heav(t-it3)*inc_i3
	activity_level = 1.6
	bmr = sex * (
		(age < 30) * (15.057 * w + 692.2) + ((age >= 30) & (age < 60)) * (11.472 * w + 873.1) + (age >= 60) * (11.711 * w + 587.7)
	) + (1 - sex) * (
		(age < 30) * (14.818 * w + 486.6) + ((age >= 30) & (age < 60)) * (8.126 * w + 845.6) + (age >= 60) * (9.082 * w + 658.5)
	)
	expd = activity_level * bmr
	k_w = 1/kpc; m = mmax*g**km/(alpha_m**km + g**km)
	isr = sigma*(m)**kisr/(alpha_isr**kisr + (m)**kisr)
	p_isr = p_b + pmax*isr**kp/(alpha_p**kp + isr**kp)
	a_m = a_b + amax*m**ka/(alpha_a**ka + m**ka); p = p_isr; a = a_m
	fmr = sex*(-0.12 + 0.013*bmi) + (1-sex)*(0.07 + 0.01*bmi); ffm = w-fmass
	inc = 1+inc_int; hepa_si = hepa_sic*si
	hgp = hgp_bas + (1+chgp*(cort-1))*hepa_max*khgp**nhgp/(khgp**nhgp+(hepa_si*i)**nhgp)
	intake = intake_i*inc
	s_glucu = sgmu*g**nsgku/(g**nsgku+sgku**nsgku); s_glucd = sgmd*g**nsgkd/(g**nsgkd+sgkd**nsgkd)
	s_ffa = sfm*ffa**nsfk/(ffa**nsfk+sfk**nsfk); s_infl = sim*infl**nsik/(infl**nsik+sik**nsik)
	s_inf = sigma_b*(s_glucu - s_glucd*s_ffa - s_infl)
	siff = si*sif; cl0 = l0*24*60/bv; cl2 = l2*24*60/bv; al0 = 2.45*24*60
	p_vf_base = sex * 0.25 + (1 - sex) * 0.15
	p_vf = p_vf_base + p_vf_amp * (cort-1)**n_p / ((cort-1)**n_p + k_p**n_p); p_sf = 1.0 - p_vf
	racor_eff  = np.clip(racor  * (cort - 1), 0.0, 0.92)
	racor1_eff = np.clip(racor1 * (cort - 1), 0.0, 0.92)
	rd=(eg0+sci*(si*i)+(1-racor1_eff)*maxrd*(si*i)**ncrd/((1-racor_eff)*kcrd**ncrd+(si*i)**ncrd))*g
	dcrh=w1*(gcort*mcort*u-crh); dacth=w2*(cmass*gcort*stim_crh-acth)
	dcort=w3*(amass*acth-cort/cr); dcmass=wc*cmass*(stim_crh-1); damass=wa*amass*(acth-1)
	dcr=wcr*(cort-cr); dext_crh=wcrhe*(d-ext_crh); dg=gclamp+hgp-rd; di=iclamp+b*isr/bv-k*i
	dffa=(cl0+cl2*fmass)*(ffa_ramax*(ksif**aa)/(ksif**aa+(siff*i)**aa))-cf*w*ffa
	dsi=(tsi-si)/tau_si; db=(png+p*b-a*b-s*b**2)/tau_b; dsigma=(s_inf-sigma)/tau_sigma
	vf_term = vf**n_infl/(vf**n_infl+k_infl**n_infl); sf_term = sf**n_infl/(sf**n_infl+k_infl**n_infl)
	infl_drive = maxinfl_vf*vf_term + maxinfl_sf*sf_term; dinfl = infl_b + (infl_drive - infl)/tau_infl
	i_thresh = 10.0; i_eff = max(0, i - i_thresh)
	phi_insulin = max_insulin * i_eff**n_i / (k_i**n_i + i_eff**n_i)
	cortisol_factor = 1 + max_w * (cort-1)**ncort / (kcort**ncort + (cort-1)**ncort)
	carb_frac = 0.5; phi = carb_frac * intake * k_w * phi_insulin * cortisol_factor
	dw = (intake * k_w - expd * k_w + phi) / tau_w
	p_fmass = 0.35; dfmass = (p_fmass * (intake * k_w - expd * k_w) + phi)/tau_w
	dvf=p_vf*dfmass; dsf=p_sf*dfmass
	cort = max(y[2], 0.001); g = max(y[7], 0.001)
	dy=np.array([dcrh,dacth,dcort,dcmass,damass,dcr,dext_crh,dg,di,dffa,dsi,db,dsigma,dinfl,dw,dfmass,dvf,dsf])
	return dy

# ── load cohorts
df   = pd.read_csv("Training Data/ukb_full_diabetics.csv")
df_h = pd.read_csv("Training Data/ukb_full_healthy.csv")

# ── ODE runs (diabetic + healthy)
def run_ode_patient(patient):
    y0 = [1,1,1,1,1,1,0, patient['G_mgdl'], 9,400,0.8,900,536,0.73,77,18.7,2.8,15.9]
    p_sim = p_base.copy(); p_sim[15] = 18.7 + patient['S_input'] * 6.3
    ratio = patient['G_mgdl'] / G_REF
    eg0_i = eg0_ref * (1.0 / ratio) ** ALPHA_CAL
    if patient['G_mgdl'] > 126: eg0_i = min(eg0_i, EG0_FLOOR * eg0_ref)
    p_sim[22] = eg0_i; p_sim[63] = p_base[63] * (ratio ** ALPHA_HGP)
    try:
        sol = sivp(odde, [0, 7200], y0, method='LSODA', args=[p_sim])
        g_start, g_end = sol.y[7][0], sol.y[7][-1]
        return g_start, g_end, 'Crash' if g_end > (g_start + 5) else 'Stable'
    except: return patient['G_mgdl'], np.nan, 'Crash'

final_results = []
for idx, patient in df.iterrows():
    g_start, g_end, status = run_ode_patient(patient)
    if not np.isnan(g_end):
        final_results.append({'eid': patient['eid'], 'Start': g_start, 'End': g_end, 'Delta': g_end - g_start, 'Status': status})
res_df = pd.DataFrame(final_results)
print(res_df['Status'].value_counts())

healthy_results = []
for idx, patient in df_h.iterrows():
    g_start, g_end, status = run_ode_patient(patient)
    if not np.isnan(g_end):
        healthy_results.append({'eid': patient['eid'], 'Start': g_start, 'End': g_end, 'Delta': g_end - g_start, 'Status': status})
res_df_healthy = pd.DataFrame(healthy_results)
print(f"Healthy: {res_df_healthy['Status'].value_counts().get('Crash', 0)} crashes out of {len(res_df_healthy)}")

# ── Olink-extended DeepONet architecture
# branch: 11 features = 5 clinical + 6 Olink NPX proteins
# trunk: same 2 features [t_norm, G0/200]
# comparing to v8 (5-feature branch) is the whole point of this script

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed(28)
os.makedirs('cache', exist_ok=True)

class _MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x): return self.net(x)

class DeepONet(nn.Module):
    """
    Branch: [S_input, BMI/40, age/80, sex, CRP/10, IL6, TNF, IGF2BP3, ADIPOQ, LEP, GDF15]
    Trunk:  [t_norm, G0/200]
    11 branch features vs 5 in v8. The extra 6 are Olink NPX proteins.
    """
    def __init__(self, basis_dim=64, hidden=64):
        super().__init__()
        self.branch = _MLP(11, hidden, basis_dim)  # 5 clinical + 6 Olink NPX
        self.trunk  = _MLP(2, hidden, basis_dim)
        self.bias   = nn.Parameter(torch.zeros(1))

    def forward(self, branch_in, trunk_in):
        b = self.branch(branch_in); t = self.trunk(trunk_in)
        return (b * t).sum(dim=-1, keepdim=True) + self.bias

# ── train/val split (same logic as v8)
df_ifg = pd.read_csv("Training Data/ukb_full_ifg.csv")
has_longval_d = (df['p30740_i1'].notna() & df['p53_i1'].notna())
longval_eids = set(df.loc[has_longval_d, 'eid'].values)
df_train     = df[~df['eid'].isin(longval_eids)].copy()
df_h_train   = df_h.copy()
df_ifg_train = df_ifg[~df_ifg['eid'].isin(longval_eids)].copy()
print(f"Training set: {len(df_train)} diabetic + {len(df_h_train)} healthy + {len(df_ifg_train)} IFG")

# ── training data builder (Olink-aware)
OLINK_COLS = ['il6', 'tnf', 'igf2bp3', 'adipoq', 'lep', 'gdf15']

def build_training_dataset(n_timepoints=40, cache_path=None):
    if cache_path is None:
        cache_path = f'cache/v10_5d_data_{_ode_param_hash({"n_tp": n_timepoints})}.npz'
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        # check feature dim - if cache was built without Olink, rebuild
        if d['branch_in'].shape[1] != 11:
            os.remove(cache_path)
            print(f'cache stale (feature dim mismatch) - rebuilding {cache_path}')
        else:
            print(f'loading cached dataset from {cache_path}')
            n_pat = int(d['n_patients'][0]) if 'n_patients' in d else len(np.unique(d['trunk_in'][:,1]))
            return d['branch_in'], d['trunk_in'], d['targets'], n_pat

    all_patients = pd.concat([df_train, df_h_train, df_ifg_train], ignore_index=True)
    all_patients['p30750_i0'] = all_patients['p30750_i0'].fillna(all_patients['p30750_i0'].median())
    all_patients['p30710_i0'] = all_patients['p30710_i0'].fillna(all_patients['p30710_i0'].median())

    # Olink mean imputation - NaN means patient has no proteomics coverage, not missing
    # filling with mean keeps those patients in the training set
    olink_means = {}
    for col in OLINK_COLS:
        if col in all_patients.columns:
            olink_means[col] = all_patients[col].mean(skipna=True)
            all_patients[col + '_imp'] = all_patients[col].fillna(olink_means[col])
        else:
            olink_means[col] = 0.0
            all_patients[col + '_imp'] = 0.0

    t_eval = np.linspace(0, 7200, n_timepoints)
    branch_rows, trunk_rows, target_rows = [], [], []
    failed = 0
    print(f'running ODE for {len(all_patients)} patients x {n_timepoints} timepoints...')

    for _, row in all_patients.iterrows():
        p_sim = p_base.copy(); p_sim[15] = 18.7 + row['S_input'] * 6.3
        ratio = row['G_mgdl'] / G_REF
        eg0_i = eg0_ref * (1.0 / ratio) ** ALPHA_CAL
        if row['G_mgdl'] > 126: eg0_i = min(eg0_i, EG0_FLOOR * eg0_ref)
        p_sim[22] = eg0_i; p_sim[63] = p_base[63] * (ratio ** ALPHA_HGP)
        y0 = [1,1,1,1,1,1,0, row['G_mgdl'], 9,400,0.8,900,536,0.73,77,18.7,2.8,15.9]
        try:
            sol = sivp(odde, [0,7200], y0, method='LSODA', args=[p_sim], t_eval=t_eval)
            if not sol.success: failed += 1; continue
        except: failed += 1; continue

        G0 = row['G_mgdl']; t_norm = t_eval / 7200.0
        branch_feat = [
            row['S_input'], row['p21001_i0']/40.0, row['p21022']/80.0, float(row['p31']),
            row['p30710_i0']/10.0,
            # Olink NPX (mean-imputed)
            float(row.get('il6_imp',     olink_means.get('il6',     0.0))),
            float(row.get('tnf_imp',     olink_means.get('tnf',     0.0))),
            float(row.get('igf2bp3_imp', olink_means.get('igf2bp3', 0.0))),
            float(row.get('adipoq_imp',  olink_means.get('adipoq',  0.0))),
            float(row.get('lep_imp',     olink_means.get('lep',     0.0))),
            float(row.get('gdf15_imp',   olink_means.get('gdf15',   0.0))),
        ]
        for t, g in zip(t_norm, sol.y[7]):
            branch_rows.append(branch_feat); trunk_rows.append([t, G0/200.0])
            target_rows.append([(g - G0) / 200.0])

    B = np.array(branch_rows, dtype=np.float32); T = np.array(trunk_rows, dtype=np.float32)
    Y = np.array(target_rows, dtype=np.float32); n_ok = len(B) // n_timepoints
    np.savez(cache_path, branch_in=B, trunk_in=T, targets=Y, n_patients=np.array([n_ok]))
    print(f'saved {len(B):,} rows ({failed} ODE failures) to {cache_path}')
    return B, T, Y, n_ok

# ── training loop
TANH_K = 5.0

def train_deeponet_v2(epochs=300, batch_size=256, lr=3e-4, cache_path=None):
    B, T, Y, n_patients_total = build_training_dataset(cache_path=cache_path)
    t_norms = T[:, 0]; keep = t_norms > 0
    B, T, Y = B[keep], T[keep], Y[keep]
    tanh_ramp = np.tanh(TANH_K * T[:, 0:1]).astype(np.float32); Y = Y / tanh_ramp

    n_patients = n_patients_total; n_timepoints_per = len(B) // n_patients
    patient_idx = np.arange(n_patients); np.random.seed(28); np.random.shuffle(patient_idx)
    split = int(0.8 * n_patients)
    train_patients = patient_idx[:split]; val_patients = patient_idx[split:]
    train_rows = np.concatenate([np.arange(i*n_timepoints_per, (i+1)*n_timepoints_per) for i in train_patients])
    val_rows   = np.concatenate([np.arange(i*n_timepoints_per, (i+1)*n_timepoints_per) for i in val_patients])

    Bt, Tt, Yt = torch.tensor(B), torch.tensor(T), torch.tensor(Y)
    ones = torch.ones(len(B))
    train_ds = TensorDataset(Bt[train_rows], Tt[train_rows], Yt[train_rows], ones[train_rows])
    val_ds   = TensorDataset(Bt[val_rows],   Tt[val_rows],   Yt[val_rows],   ones[val_rows])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    model = DeepONet()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=80, gamma=0.4)
    train_losses, val_losses = [], []
    n_params = sum(p.numel() for p in model.parameters())
    print(f'training Olink-extended DeepONet - {n_params:,} params')

    for epoch in range(1, epochs + 1):
        model.train(); epoch_loss = 0.0
        for b_b, t_b, y_b, _ in train_loader:
            optimiser.zero_grad()
            loss = nn.functional.mse_loss(model(b_b, t_b), y_b)
            loss.backward(); optimiser.step(); epoch_loss += loss.item() * len(b_b)
        train_losses.append(epoch_loss / len(train_ds))
        model.eval()
        with torch.no_grad():
            vl = sum(nn.functional.mse_loss(model(b_b, t_b), y_b).item() * len(b_b) for b_b, t_b, y_b, _ in val_loader)
        val_losses.append(vl / len(val_ds)); scheduler.step()
        if epoch % 50 == 0:
            print(f'  epoch {epoch:3d} | train {train_losses[-1]:.6f} | val {val_losses[-1]:.6f}')

    torch.save(model.state_dict(), 'cache/v10_5d_model.pt')
    return model, train_losses, val_losses, val_patients.tolist()

# ── prediction function (Olink-aware)
def predict_trajectory_tanh(model, patient, n_timepoints=100):
    G0 = patient['G_mgdl']
    t_norm = np.linspace(0, 1, n_timepoints).astype(np.float32)
    # known issue: missing Olink falls back to 0 here, not the training means.
    # the paper's Olink numbers came from this version - fix once UKB reopens
    olink_vals = [
        float(patient.get(c, patient.get(c+'_imp', 0.0))
              if pd.notna(patient.get(c, float('nan'))) else 0.0)
        for c in OLINK_COLS
    ]
    branch_feat = np.array([
        patient['S_input'], patient['p21001_i0']/40.0, patient['p21022']/80.0,
        float(patient['p31']), patient['p30710_i0']/10.0,
    ] + olink_vals, dtype=np.float32)
    branch_in = np.tile(branch_feat, (n_timepoints, 1))
    trunk_in  = np.column_stack([t_norm, np.full(n_timepoints, G0/200.0)]).astype(np.float32)
    model.eval()
    with torch.no_grad():
        raw = model(torch.tensor(branch_in), torch.tensor(trunk_in)).numpy().flatten()
    ramp = np.tanh(TANH_K * t_norm)
    return t_norm * 7200.0, G0 + ramp * raw * 200.0

print('training 5-day Olink DeepONet...')
model_tanh, tl_tanh, vl_tanh, val_idx_tanh = train_deeponet_v2()
best_model = model_tanh; best_predict = predict_trajectory_tanh
print('done')

# ── temporal MAE
def compute_temporal_mae(model, predict_fn, val_indices, n_timepoints=40):
    all_patients = pd.concat([df_train.assign(_cohort='T2D'), df_h_train.assign(_cohort='Healthy'),
                              df_ifg_train.assign(_cohort='IFG')], ignore_index=True)
    all_patients['p30750_i0'] = all_patients['p30750_i0'].fillna(all_patients['p30750_i0'].median())
    all_patients['p30710_i0'] = all_patients['p30710_i0'].fillna(all_patients['p30710_i0'].median())
    eval_pts = all_patients.iloc[val_indices].reset_index(drop=True)
    t_eval = np.linspace(0, 7200, n_timepoints)

    _key = _ode_param_hash({'T_END': 7200, 'n_tp': n_timepoints, 'val': sorted(map(int, val_indices))})
    traj_cache = f'cache/v10_5d_val_ode_{_key}.npz'
    if os.path.exists(traj_cache):
        print(f'loading cached ode trajs from {traj_cache}')
        ode_trajs = list(np.load(traj_cache)['ode_trajs'])
    else:
        ode_trajs = []
        for _, row in eval_pts.iterrows():
            p_sim = p_base.copy(); p_sim[15] = 18.7 + row['S_input'] * 6.3
            ratio = row['G_mgdl'] / G_REF
            eg0_i = eg0_ref * (1.0 / ratio) ** ALPHA_CAL
            if row['G_mgdl'] > 126: eg0_i = min(eg0_i, EG0_FLOOR * eg0_ref)
            p_sim[22] = eg0_i; p_sim[63] = p_base[63] * (ratio ** ALPHA_HGP)
            y0 = [1,1,1,1,1,1,0, row['G_mgdl'], 9,400,0.8,900,536,0.73,77,18.7,2.8,15.9]
            try:
                sol = sivp(odde, [0,7200], y0, method='LSODA', args=[p_sim], t_eval=t_eval)
                ode_traj = sol.y[7] if sol.success else np.full(n_timepoints, np.nan)
            except: ode_traj = np.full(n_timepoints, np.nan)
            ode_trajs.append(ode_traj)
        np.savez(traj_cache, ode_trajs=np.array(ode_trajs))

    don_trajs = [predict_fn(model, row, n_timepoints=n_timepoints)[1] for _, row in eval_pts.iterrows()]
    ode_trajs = np.array(ode_trajs); don_trajs = np.array(don_trajs)
    mae_t = np.nanmean(np.abs(ode_trajs - don_trajs), axis=0)
    std_t = np.nanstd(np.abs(ode_trajs - don_trajs), axis=0)
    return mae_t, std_t, ode_trajs, don_trajs

n_tp = 40
t_days = np.linspace(0, 7200, n_tp) / 1440.0
mae_tanh, std_tanh, ode_t, don_t = compute_temporal_mae(model_tanh, predict_trajectory_tanh, val_idx_tanh, n_tp)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.plot(tl_tanh, label='train', color='#e74c3c', alpha=0.8)
ax.plot(vl_tanh, label='val',   color='#e74c3c', ls='--')
ax.set_xlabel('epoch'); ax.set_ylabel('MSE loss')
ax.set_title('A. Training convergence (Olink branch, tanh IC)', loc='left')
ax.legend(); ax.grid(alpha=0.3); ax.set_yscale('log')

ax2 = axes[1]
ax2.plot(t_days, mae_tanh, color='#e74c3c', lw=2.5)
ax2.fill_between(t_days, mae_tanh - std_tanh, mae_tanh + std_tanh, color='#e74c3c', alpha=0.15)
ax2.set_xlabel('Time (days)'); ax2.set_ylabel('MAE (mg/dL)')
ax2.set_title(f'B. Temporal MAE - held-out patients (n={len(val_idx_tanh)})', loc='left')
ax2.set_ylim(0, max(1.0, (mae_tanh + std_tanh).max() * 1.25))
ax2.grid(alpha=0.3)
plt.tight_layout();
print(f'Olink MAE: t=0: {mae_tanh[0]:.2f}  mid: {mae_tanh[n_tp//2]:.2f}  end: {mae_tanh[-1]:.2f} mg/dL')

# ── stability sweep (approach A)
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

def classify_endpoint(g_start, g_end, tol=5.0):
    delta = g_end - g_start
    if delta > tol: return 'progressive'
    elif delta < -tol: return 'recovering'
    else: return 'stable'

def endpoint_stability_sweep(model, predict_fn, n_G0=80, n_S=60, bmi_values=[22, 27, 32]):
    all_pts = pd.concat([df_train, df_h_train, df_ifg_train], ignore_index=True)
    age_med = all_pts['p21022'].median(); sex_med = float(all_pts['p31'].mode()[0]); crp_med = all_pts['p30710_i0'].median()
    G0_range = np.linspace(65, 400, n_G0); S_range = np.linspace(0.0, 1.0, n_S)
    results = {}
    for bmi in bmi_values:
        landscape = np.full((n_S, n_G0), np.nan); classes = np.empty((n_S, n_G0), dtype='U15')
        for i, S in enumerate(S_range):
            for j, G0 in enumerate(G0_range):
                # zero-fill Olink for sweep - we're testing whether Olink changes the attractor map
                # it shouldn't, since Olink doesn't enter the ODE itself, only the surrogate
                fake_patient = {'S_input': S, 'p21001_i0': bmi, 'p21022': age_med,
                                'p31': sex_med, 'p30710_i0': crp_med, 'G_mgdl': G0,
                                **{c: 0.0 for c in OLINK_COLS}}
                _, g_traj = predict_fn(model, fake_patient, n_timepoints=50)
                landscape[i, j] = g_traj[-1]; classes[i, j] = classify_endpoint(G0, g_traj[-1])
        results[bmi] = {'landscape': landscape, 'classes': classes, 'G0_range': G0_range, 'S_range': S_range}
        print(f'  BMI={bmi}: done')
    return results

sweep_A = endpoint_stability_sweep(best_model, best_predict)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
cat_colors_A = {'recovering': '#3498db', 'stable': '#2ecc71', 'progressive': '#e74c3c'}
for ax, bmi in zip(axes, [22, 27, 32]):
    res = sweep_A[bmi]
    cat_map = {'recovering': 0, 'stable': 1, 'progressive': 2}
    numeric = np.vectorize(lambda c: cat_map.get(c, -1))(res['classes'])
    cmap = ListedColormap([cat_colors_A[k] for k in ['recovering', 'stable', 'progressive']])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.pcolormesh(res['G0_range'], res['S_range'], numeric, cmap=cmap, norm=norm, shading='nearest')
    ax.axvline(100, color='orange', ls='--', lw=1.5); ax.axvline(126, color='red', ls='--', lw=1.5)
    ax.set_xlabel('Initial glucose G0 (mg/dL)'); ax.set_ylabel('Stress score S')
    ax.set_title(f'BMI={bmi}')
plt.suptitle('Self-consistency map: Olink-extended model (compare to v8 for Olink effect on the landscape)', y=1.01)
plt.tight_layout();

# ── IFG ODE run
df_ifg = pd.read_csv("Training Data/ukb_full_ifg.csv")
ifg_results = []
for idx, patient in df_ifg.iterrows():
    g_start, g_end, status = run_ode_patient(patient)
    if not np.isnan(g_end):
        ifg_results.append({'eid': patient['eid'], 'Start': g_start, 'End': g_end, 'Delta': g_end - g_start, 'Status': status})
res_df_ifg = pd.DataFrame(ifg_results)
print(f"IFG: mean delta G = {res_df_ifg['Delta'].mean():.2f} mg/dL")

# ── HbA1c validation
from scipy import stats

def get_hba1c_matched(df_raw, res_df_sim):
    return res_df_sim.merge(
        df_raw[['eid', 'p30750_i0']].rename(columns={'p30750_i0': 'hba1c'}),
        on='eid', how='left'
    ).dropna(subset=['hba1c'])

diab_matched  = get_hba1c_matched(df,     res_df)
ifg_matched   = get_hba1c_matched(df_ifg, res_df_ifg)
hlthy_matched = get_hba1c_matched(df_h,   res_df_healthy)
all_matched   = pd.concat([diab_matched, ifg_matched, hlthy_matched], ignore_index=True)

r_all, p_all = stats.pearsonr(all_matched['End'], all_matched['hba1c'])
r_d,   _     = stats.pearsonr(diab_matched['End'],  diab_matched['hba1c'])
r_h,   _     = stats.pearsonr(hlthy_matched['End'], hlthy_matched['hba1c'])
print(f"HbA1c validation: pooled r={r_all:.3f} (n={len(all_matched)}), diabetic r={r_d:.3f}, healthy r={r_h:.3f}")