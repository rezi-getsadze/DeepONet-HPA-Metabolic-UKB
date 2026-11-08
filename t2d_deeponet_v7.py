import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp as sivp
from numpy import sin, cos, exp, sqrt, mod, heaviside
import string
import matplotlib.transforms as mtransforms
import os

# v7 - Snellius parameter sweep
# reads sweep_results_full.csv (120,000 ODE runs: G0 x S x SI_target x BMI)
# first part: setup and local test sweep
# second part: analysis of Snellius results

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
_tee = _Tee('runtime/v7/v7_output.txt')
_sys.stdout = _tee


_os.makedirs('runtime/v7', exist_ok=True)
_os.makedirs('output', exist_ok=True)
_vv7_n      = [0]
_vv7_saved  = [False]
_vv7_orig_sf  = plt.savefig
_vv7_orig_cl  = plt.close
_vv7_orig_fig = plt.figure
_vv7_orig_sp  = plt.subplots

def _vv7_sf(*a, **kw):
    _vv7_saved[0] = True
    return _vv7_orig_sf(*a, **kw)

def _vv7_cl(*a, **kw):
    if not _vv7_saved[0] and plt.get_fignums():
        _vv7_n[0] += 1
        _vv7_orig_sf(f'runtime/v7/v7_fig{_vv7_n[0]}.svg', bbox_inches='tight')
    _vv7_saved[0] = False
    return _vv7_orig_cl(*a, **kw)

def _vv7_figure(*a, **kw):
    _vv7_saved[0] = False
    return _vv7_orig_fig(*a, **kw)

def _vv7_subplots(*a, **kw):
    _vv7_saved[0] = False
    return _vv7_orig_sp(*a, **kw)

plt.savefig  = _vv7_sf
plt.close    = _vv7_cl
plt.figure   = _vv7_figure
plt.subplots = _vv7_subplots

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
pars['racor']=0.08   # was 0.54 - (1-racor*(cort-1)) goes negative when cort>2.85
pars['racor1']=0.03  # reduced proportionally
pars['maxrd']=215; pars['nhgp']=2; pars['ncrd']=2; pars['ncsi']=2; pars['tata']=2

p_base = np.array(list(pars.values()))

G_REF     = 86.5
eg0_ref   = 56.2964
ALPHA_CAL = 0.984
ALPHA_HGP = 0.3
EG0_FLOOR = 0.75

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
	gcort = 1/(1+(cort/kgr)**n)
	mcort = 1/cort
	bmi = w/height**2
	age = age_b+cage*t/365

	tsi = (kcortsi**ncsi/(kcortsi**ncsi+(cort-1)**ncsi))*target_si*(ksi_infl/(ksi_infl+infl))*(1-mffa*ffa**nsi_ffa/(ffa**nsi_ffa+ksi_ffa**nsi_ffa))

	inc_int = 0 + heav(t-it1)*heav(it2-t)*heav(it3-t)*inc_i1 + heav(t-it2)*heav(it3-t)*inc_i2 + heav(t-it3)*inc_i3
	activity_level = 1.6
	bmr = sex * (
		(age < 30) * (15.057 * w + 692.2) + ((age >= 30) & (age < 60)) * (11.472 * w + 873.1) + (age >= 60) * (11.711 * w + 587.7)
	) + (1 - sex) * (
		(age < 30) * (14.818 * w + 486.6) + ((age >= 30) & (age < 60)) * (8.126 * w + 845.6) + (age >= 60) * (9.082 * w + 658.5)
	)
	expd = activity_level * bmr
	k_w = 1/kpc
	m = mmax*g**km/(alpha_m**km + g**km)
	isr = sigma*(m)**kisr/(alpha_isr**kisr + (m)**kisr)
	p_isr = p_b + pmax*isr**kp/(alpha_p**kp + isr**kp)
	a_m = a_b + amax*m**ka/(alpha_a**ka + m**ka)
	p = p_isr; a = a_m
	fmr = sex*(-0.12 + 0.013*bmi) + (1-sex)*(0.07 + 0.01*bmi)
	ffm = w-fmass
	inc = 1+inc_int
	hepa_si = hepa_sic*si
	hgp = hgp_bas + (1+chgp*(cort-1))*hepa_max*khgp**nhgp/(khgp**nhgp+(hepa_si*i)**nhgp)
	intake = intake_i*inc
	s_glucu = sgmu*g**nsgku/(g**nsgku+sgku**nsgku)
	s_glucd = sgmd*g**nsgkd/(g**nsgkd+sgkd**nsgkd)
	s_ffa = sfm*ffa**nsfk/(ffa**nsfk+sfk**nsfk)
	s_infl = sim*infl**nsik/(infl**nsik+sik**nsik)
	s_inf = sigma_b*(s_glucu - s_glucd*s_ffa - s_infl)
	siff = si*sif
	cl0 = l0*24*60/bv; cl2 = l2*24*60/bv; al0 = 2.45*24*60
	p_vf_base = sex * 0.25 + (1 - sex) * 0.15
	p_vf = p_vf_base + p_vf_amp * (cort-1)**n_p / ((cort-1)**n_p + k_p**n_p)
	p_sf = 1.0 - p_vf

	# clip so denominator never goes negative (NaN when cort>2.85 at racor=0.54)
	racor_eff  = np.clip(racor  * (cort - 1), 0.0, 0.92)
	racor1_eff = np.clip(racor1 * (cort - 1), 0.0, 0.92)
	rd=(eg0+sci*(si*i)+(1-racor1_eff)*maxrd*(si*i)**ncrd/((1-racor_eff)*kcrd**ncrd+(si*i)**ncrd))*g

	dcrh=w1*(gcort*mcort*u-crh); dacth=w2*(cmass*gcort*stim_crh-acth)
	dcort=w3*(amass*acth-cort/cr); dcmass=wc*cmass*(stim_crh-1); damass=wa*amass*(acth-1)
	dcr=wcr*(cort-cr); dext_crh=wcrhe*(d-ext_crh)
	dg=gclamp+hgp-rd; di=iclamp+b*isr/bv-k*i
	dffa=(cl0+cl2*fmass)*(ffa_ramax*(ksif**aa)/(ksif**aa+(siff*i)**aa))-cf*w*ffa
	dsi=(tsi-si)/tau_si; db=(png+p*b-a*b-s*b**2)/tau_b; dsigma=(s_inf-sigma)/tau_sigma
	vf_term = vf**n_infl/(vf**n_infl+k_infl**n_infl)
	sf_term = sf**n_infl/(sf**n_infl+k_infl**n_infl)
	infl_drive = maxinfl_vf*vf_term + maxinfl_sf*sf_term
	dinfl = infl_b + (infl_drive - infl)/tau_infl
	i_thresh = 10.0; i_eff = max(0, i - i_thresh)
	phi_insulin = max_insulin * i_eff**n_i / (k_i**n_i + i_eff**n_i)
	cortisol_factor = 1 + max_w * (cort-1)**ncort / (kcort**ncort + (cort-1)**ncort)
	carb_frac = 0.5
	phi = carb_frac * intake * k_w * phi_insulin * cortisol_factor
	dw = (intake * k_w - expd * k_w + phi) / tau_w
	p_fmass = 0.35; dfmass = (p_fmass * (intake * k_w - expd * k_w) + phi)/tau_w
	dvf=p_vf*dfmass; dsf=p_sf*dfmass
	cort = max(y[2], 0.001); g = max(y[7], 0.001)
	dy=np.array([dcrh,dacth,dcort,dcmass,damass,dcr,dext_crh,dg,di,dffa,dsi,db,dsigma,dinfl,dw,dfmass,dvf,dsf])
	return dy

# ── sweep grid definitions
# full grid (Snellius): 50 x 30 x 20 x 4 = 120,000 simulations
# test grid (local):    20 x 10 x 5  x 2 = 2,000 simulations
# BMI capped at 30 to avoid the caloric intake artifact above BMI~32

T_END_SWEEP = 365.25 * 24 * 60  # 1 year in minutes

G0_FULL  = np.linspace(70, 320, 50); S_FULL  = np.linspace(0, 0.56, 30)
SI_FULL  = np.linspace(0.8, 2.8, 20); BMI_FULL = [22, 25, 27, 30]

G0_TEST  = np.linspace(70, 320, 20); S_TEST  = np.linspace(0, 0.56, 10)
SI_TEST  = np.linspace(0.8, 2.8, 5); BMI_TEST = [22, 27]

THRESHOLD = 5.0

total_full = len(G0_FULL)*len(S_FULL)*len(SI_FULL)*len(BMI_FULL)
total_test = len(G0_TEST)*len(S_TEST)*len(SI_TEST)*len(BMI_TEST)
print(f'Full Snellius grid: {total_full:,} simulations')
print(f'Local test grid:    {total_test:,} simulations')

# ── sweep functions
def bmi_to_weight(bmi, height=1.66):
    return bmi * height**2

def run_sweep_point(G0, S, si_target, bmi, t_end=T_END_SWEEP):
    p_sim = p_base.copy()
    p_sim[15] = 18.7 + S * 6.3
    ratio = G0 / G_REF
    eg0_i = eg0_ref * (1.0 / ratio) ** ALPHA_CAL
    if G0 > 126: eg0_i = min(eg0_i, EG0_FLOOR * eg0_ref)
    p_sim[22] = eg0_i; p_sim[63] = p_base[63] * (ratio ** ALPHA_HGP)
    p_sim[43] = si_target; weight = bmi_to_weight(bmi)
    y0 = [1,1,1,1,1,1,0,G0,9,400,0.8,900,536,0.73,weight,18.7,2.8,15.9]
    try:
        sol = sivp(odde, [0, t_end], y0, method='LSODA', args=[p_sim])
        return sol.y[7][-1] if sol.success else np.nan
    except: return np.nan

def classify(G0, G_end, threshold=THRESHOLD):
    if np.isnan(G_end): return 'crash'
    d = G_end - G0
    if d > threshold: return 'progressive'
    if d < -threshold: return 'recovering'
    return 'stable'

print('sweep functions defined')

# ── local test sweep
import time
print('running local test sweep...')
start = time.time()
test_results = []
total = len(G0_TEST)*len(S_TEST)*len(SI_TEST)*len(BMI_TEST); done = 0

for bmi in BMI_TEST:
    for si in SI_TEST:
        for s in S_TEST:
            for g0 in G0_TEST:
                g_end = run_sweep_point(g0, s, si, bmi)
                test_results.append({'G0': g0, 'S': s, 'SI_target': si, 'BMI': bmi,
                                     'G_end': g_end,
                                     'delta_G': g_end - g0 if not np.isnan(g_end) else np.nan,
                                     'class': classify(g0, g_end)})
                done += 1
                if done % 400 == 0: print(f'  {done}/{total} ({100*done/total:.0f}%)')

df_test = pd.DataFrame(test_results)
print(f'done in {time.time()-start:.0f}s')
print(df_test['class'].value_counts())

# test sweep visualization
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

cat_colors = {'recovering':'#3498db','stable':'#2ecc71','progressive':'#e74c3c','crash':'#95a5a6'}
cat_order  = ['recovering','stable','progressive','crash']
cmap_sweep = ListedColormap([cat_colors[k] for k in cat_order])
norm_sweep = BoundaryNorm([-0.5,0.5,1.5,2.5,3.5], cmap_sweep.N)
cat_map    = {c:i for i,c in enumerate(cat_order)}

n_bmi, n_si = len(BMI_TEST), len(SI_TEST)
fig, axes = plt.subplots(n_bmi, n_si, figsize=(4*n_si, 4*n_bmi))
for bi, bmi in enumerate(BMI_TEST):
    for si_i, si in enumerate(SI_TEST):
        ax = axes[bi, si_i] if n_bmi > 1 else axes[si_i]
        sub = df_test[(df_test['BMI']==bmi) & (df_test['SI_target'].round(3)==round(si,3))]
        if len(sub) == 0: continue
        grid = sub.pivot(index='S', columns='G0', values='class')
        numeric = grid.map(lambda c: cat_map.get(c, 3))
        ax.pcolormesh(G0_TEST, S_TEST, numeric.values, cmap=cmap_sweep, norm=norm_sweep, shading='nearest')
        ax.axvline(100, color='orange', ls='--', lw=1.5, alpha=0.8)
        ax.axvline(126, color='red',    ls='--', lw=1.5, alpha=0.8)
        ax.set_title(f'BMI={bmi}, SI={si:.1f}')
        ax.set_xlabel('G0 (mg/dL)'); ax.set_ylabel('Stress S')
legend_elements = [Patch(facecolor=cat_colors[c], label=c) for c in ['recovering','stable','progressive']]
fig.legend(handles=legend_elements, loc='upper right')
plt.suptitle('Test sweep: self-consistency map')
plt.tight_layout();

# ── Snellius submission files
# generates attractor/attractor_sweep.py and attractor/attractor_sweep.sh
# on snellius: copy attractor/ to ~/attractor (venv inside), sbatch attractor_sweep.sh, then merge_results.py

os.makedirs('attractor', exist_ok=True)

merge_py = (
    'import pandas as pd, glob\n'
    'files = sorted(glob.glob("sweep_task_*.csv"))\n'
    'print(f"merging {len(files)} files...")\n'
    'df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)\n'
    'df.to_csv("sweep_results_full.csv", index=False)\n'
    'print(f"saved {len(df):,} rows")\n'
    'print(df["class"].value_counts())\n'
)
with open('attractor/merge_results.py', 'w') as f:
    f.write(merge_py)

slurm = (
    '#!/bin/bash\n'
    '#SBATCH --job-name=attractor_sweep\n'
    '#SBATCH --nodes=1\n'
    '#SBATCH --ntasks=1\n'
    '#SBATCH --cpus-per-task=1\n'
    '#SBATCH --partition=genoa\n'
    '#SBATCH --time=02:00:00\n'
    '#SBATCH --array=0-79\n'
    '#SBATCH --output=logs/sweep_%A_%a.out\n'
    '#SBATCH --error=logs/sweep_%A_%a.err\n'
    'mkdir -p logs\n'
    'module load 2024\n'
    'module load Python/3.12.3-GCCcore-13.3.0\n'
    'source ~/attractor/venv/bin/activate\n'
    'python attractor_sweep.py $SLURM_ARRAY_TASK_ID\n'
)
with open('attractor/attractor_sweep.sh', 'w') as f:
    f.write(slurm)

print('wrote attractor/merge_results.py')
print('wrote attractor/attractor_sweep.sh')

# generate attractor_sweep.py from live pars + ODE function source
import inspect
odde_src = inspect.getsource(odde)
pars_lines = []
for k, v in pars.items():
    if v == np.inf: pars_lines.append(f"    '{k}': np.inf,")
    else: pars_lines.append(f"    '{k}': {v},")
pars_str = 'pars = {\n' + '\n'.join(pars_lines) + '\n}'

script_lines = [
    '#!/usr/bin/env python3',
    'import numpy as np', 'import pandas as pd', 'from scipy.integrate import solve_ivp as sivp',
    'import sys', 'from numpy import sin, cos, exp, sqrt, mod, heaviside', '',
    'inf = np.inf', '', pars_str, 'p_base = np.array(list(pars.values()))', '',
    f'G_REF     = {G_REF}', f'eg0_ref   = {eg0_ref}', f'ALPHA_CAL = {ALPHA_CAL}',
    f'ALPHA_HGP = {ALPHA_HGP}', f'EG0_FLOOR = {EG0_FLOOR}',
    'T_END     = 365.25 * 24 * 60', 'THRESHOLD = 5.0', '', odde_src, '',
    'def run_point(G0, S, si_target, bmi):',
    '    p_sim = p_base.copy()',
    '    p_sim[15] = 18.7 + S * 6.3',
    '    ratio = G0 / G_REF',
    '    eg0_i = eg0_ref * (1.0 / ratio) ** ALPHA_CAL',
    '    if G0 > 126:', '        eg0_i = min(eg0_i, EG0_FLOOR * eg0_ref)',
    '    p_sim[22] = eg0_i', '    p_sim[63] = p_base[63] * (ratio ** ALPHA_HGP)',
    '    p_sim[43] = si_target', '    weight = bmi * 1.66**2',
    '    y0 = [1,1,1,1,1,1,0,G0,9,400,0.8,900,536,0.73,weight,18.7,2.8,15.9]',
    '    try:', '        sol = sivp(odde, [0, T_END], y0, method="LSODA", args=[p_sim])',
    '        return sol.y[7][-1] if sol.success else float("nan")',
    '    except Exception as e:',
    '        print(f"ERR: G0={G0:.1f} S={S:.2f} SI={si_target:.2f} BMI={bmi}: {e}", flush=True)',
    '        return float("nan")', '',
    'G0_VALS  = np.linspace(70, 320, 50)', 'S_VALS   = np.linspace(0, 0.56, 30)',
    'SI_VALS  = np.linspace(0.8, 2.8, 20)', 'BMI_VALS = [22, 25, 27, 30]', '',
    'import os',
    'task_id = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))',
    'n_si    = len(SI_VALS)', 'bmi_idx = task_id // n_si', 'si_idx  = task_id %  n_si',
    'bmi     = BMI_VALS[bmi_idx]', 'si      = SI_VALS[si_idx]',
    'print(f"task {task_id}: BMI={bmi}, SI_target={si:.3f}")', '',
    'results = []',
    'for s in S_VALS:',
    '    for g0 in G0_VALS:',
    '        g_end = run_point(g0, s, si, bmi)',
    '        delta = g_end - g0 if not np.isnan(g_end) else float("nan")',
    '        if np.isnan(g_end): label = "crash"',
    '        elif delta > THRESHOLD: label = "progressive"',
    '        elif delta < -THRESHOLD: label = "recovering"',
    '        else: label = "stable"',
    '        results.append({"G0":g0,"S":s,"SI_target":si,"BMI":bmi,',
    '                         "G_end":g_end,"delta_G":delta,"class":label})', '',
    'df = pd.DataFrame(results)', 'out = f"sweep_task_{task_id:04d}.csv"',
    'df.to_csv(out, index=False)', 'print(f"saved {len(df)} rows to {out}")',
]
with open('attractor/attractor_sweep.py', 'w') as f:
    f.write('\n'.join(script_lines))
print('wrote attractor/attractor_sweep.py')
print('verify with: python attractor/attractor_sweep.py 0')

# ── load Snellius results
if os.path.exists('sweep_results_full.csv'):
    df_sweep = pd.read_csv('sweep_results_full.csv')
    print(f'loaded Snellius results: {len(df_sweep):,} rows')
elif os.path.exists('synthetic_sweep_results_full.csv'):
    df_sweep = pd.read_csv('synthetic_sweep_results_full.csv')
    print(f'loaded SYNTHETIC sweep (demo only, not the real runs): {len(df_sweep):,} rows')
else:
    print('no sweep csv found - falling back to local test results')
    df_sweep = df_test.copy()
    print(f'using test results: {len(df_sweep)} rows')

print(df_sweep['class'].value_counts())
print(f'crash rate: {(df_sweep["class"]=="crash").mean()*100:.1f}%')

# self-consistency map visualization (4 representative SI_target slices per BMI)
cat_colors_v = {'recovering':'#3498db','stable':'#2ecc71','progressive':'#e74c3c','crash':'#95a5a6'}
cat_order_v  = ['recovering','stable','progressive','crash']
cmap_s = ListedColormap([cat_colors_v[k] for k in cat_order_v])
norm_s = BoundaryNorm([-0.5,0.5,1.5,2.5,3.5], cmap_s.N)
cat_map_v = {c:i for i,c in enumerate(cat_order_v)}

bmi_vals = sorted(df_sweep['BMI'].unique())
si_all   = sorted(df_sweep['SI_target'].unique())
si_show  = [si_all[0], si_all[len(si_all)//3], si_all[2*len(si_all)//3], si_all[-1]]
g0_vals  = sorted(df_sweep['G0'].unique()); s_vals = sorted(df_sweep['S'].unique())

fig, axes = plt.subplots(len(bmi_vals), 4, figsize=(18, 4.5*len(bmi_vals)))
if len(bmi_vals) == 1: axes = axes.reshape(1, -1)
for bi, bmi in enumerate(bmi_vals):
    for si_i, si in enumerate(si_show):
        ax = axes[bi, si_i]
        sub = df_sweep[(df_sweep['BMI']==bmi) & (df_sweep['SI_target'].round(2)==round(si,2))]
        if len(sub) == 0: continue
        grid = sub.pivot(index='S', columns='G0', values='class')
        numeric = grid.map(lambda c: cat_map_v.get(c, 3))
        ax.pcolormesh(g0_vals, s_vals, numeric.values, cmap=cmap_s, norm=norm_s, shading='nearest')
        ax.axvline(100, color='orange', ls='--', lw=1.5, alpha=0.8)
        ax.axvline(126, color='red',    ls='--', lw=1.5, alpha=0.8)
        ax.set_xlabel('Initial glucose G0 (mg/dL)'); ax.set_ylabel('Stress S')
        ax.set_title(f'BMI={bmi}, SI_target={si:.2f}')
        ax.grid(alpha=0.15)
legend_elements = [Patch(facecolor=cat_colors_v[c], label=c) for c in ['recovering','stable','progressive']]
fig.legend(handles=legend_elements, loc='upper right')
plt.suptitle('Self-consistency map: recovery zone expands with insulin sensitivity', y=1.01)
plt.tight_layout();

# ── racor stability boundary sweep
# finds highest racor where model is still stable across full diabetic G0 range
# racor=0.08 is the chosen value; this quantifies the tradeoff with stress coupling
# stability boundary = max racor with 0 crashes on this 20-point G0 grid

RACOR_VALS   = np.linspace(0.08, 0.54, 10)
DIABETIC_G0S = np.linspace(126, 300, 20)
HEIGHT       = 1.66
RACOR_BMI = 27; RACOR_SI = 1.8; RACOR_S = 0.3; T_RACOR = 7200

def run_racor_point(G0, racor_val):
    p_sim = p_base.copy(); p_sim[15] = 18.7 + RACOR_S * 6.3
    ratio = G0 / G_REF
    eg0_i = eg0_ref * (1.0 / ratio) ** ALPHA_CAL
    if G0 > 126: eg0_i = min(eg0_i, EG0_FLOOR * eg0_ref)
    p_sim[22] = eg0_i; p_sim[63] = p_base[63] * (ratio ** ALPHA_HGP)
    p_sim[43] = RACOR_SI; p_sim[45] = RACOR_BMI
    p_sim[116] = racor_val; p_sim[117] = racor_val * 0.375  # racor1 scales proportionally
    weight = RACOR_BMI * HEIGHT**2
    y0 = [1,1,1,1,1,1,0, G0, 9,400,0.8,900,536,0.73, weight, 18.7,2.8,15.9]
    try:
        sol = sivp(odde, [0, T_RACOR], y0, method='LSODA', args=[p_sim])
        g_end = sol.y[7][-1] if sol.success else np.nan
        return g_end if (sol.success and np.isfinite(g_end) and g_end > 0) else np.nan
    except: return np.nan

crash_rates = []; mean_gends = []
print(f'racor sweep: BMI={RACOR_BMI}, SI_target={RACOR_SI}, S={RACOR_S}')
print(f'{"racor":>8}  {"crash%":>8}  {"mean G_end":>12}')
for racor_val in RACOR_VALS:
    results = np.array([run_racor_point(G0, racor_val) for G0 in DIABETIC_G0S])
    crash_pct = np.isnan(results).mean() * 100
    mean_gend = np.nanmean(results) if not np.all(np.isnan(results)) else np.nan
    crash_rates.append(crash_pct); mean_gends.append(mean_gend)
    print(f'{racor_val:>8.3f}  {crash_pct:>8.1f}  {mean_gend:>12.1f}')

crash_rates = np.array(crash_rates); mean_gends = np.array(mean_gends)
stable_mask = crash_rates == 0
racor_boundary = RACOR_VALS[stable_mask].max() if stable_mask.any() else RACOR_VALS[0]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax1 = axes[0]
ax1.plot(RACOR_VALS, crash_rates, 'ro-', lw=2, ms=6)
ax1.axvline(racor_boundary, color='steelblue', ls='--', lw=1.5, label=f'stability boundary = {racor_boundary:.3f}')
ax1.axvline(0.08, color='green', ls=':', lw=1.5, label='current racor=0.08')
ax1.axvline(0.54, color='grey',  ls=':', lw=1.5, label='original racor=0.54')
ax1.set_xlabel('racor'); ax1.set_ylabel('crash rate (%)')
ax1.set_title(f'Crash rate vs racor  BMI={RACOR_BMI}, SI={RACOR_SI}, S={RACOR_S}', loc='left')
ax1.legend(); ax1.grid(alpha=0.2)

ax2 = axes[1]; valid = ~np.isnan(mean_gends)
ax2.plot(RACOR_VALS[valid], mean_gends[valid], 'ko-', lw=2, ms=6)
ax2.axvline(racor_boundary, color='steelblue', ls='--', lw=1.5, label=f'boundary = {racor_boundary:.3f}')
ax2.axvline(0.08, color='green', ls=':', lw=1.5, label='current racor=0.08')
ax2.set_xlabel('racor'); ax2.set_ylabel('mean G_end (mg/dL)')
ax2.set_title('Attractor shift vs racor (how much stress coupling moves equilibrium)', loc='left')
ax2.legend(); ax2.grid(alpha=0.2)
plt.suptitle('Racor sensitivity: stability-coupling tradeoff')
plt.tight_layout()
plt.close()
print(f'Stability boundary: racor <= {racor_boundary:.3f} gives 0 crashes on this test grid')
print(f'Stress coupling headroom: {racor_boundary:.3f} / 0.54 = {racor_boundary/0.54:.1%} of original')

# ── SI_target dominance: sweep corroboration
si_levels = sorted(df_sweep['SI_target'].unique())
frac_progressive = []
for si in si_levels:
    sub = df_sweep[(df_sweep['SI_target'].round(3) == round(si, 3)) & (df_sweep['class'] != 'crash')]
    frac_progressive.append((sub['class'] == 'progressive').mean() if len(sub) > 0 else np.nan)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(si_levels, [f*100 for f in frac_progressive], color='#e74c3c', lw=2, marker='o', ms=5)
ax.axhline(0, color='grey', lw=0.8, ls='--')
ax.set_xlabel('SI_target (insulin sensitivity)'); ax.set_ylabel('% patients classified progressive')
ax.set_title('SI_target dominates recovery zone size (tornado rank 1)', loc='left')
ax.grid(alpha=0.25); plt.tight_layout();
print(f'SI_target={si_levels[0]:.1f}: {frac_progressive[0]*100:.0f}% progressive -> SI_target={si_levels[-1]:.1f}: {frac_progressive[-1]*100:.0f}% progressive')
for bmi in sorted(df_sweep['BMI'].unique()):
    sub = df_sweep[(df_sweep['BMI'] == bmi) & (df_sweep['class'] != 'crash')]
    print(f'BMI={bmi}: {(sub["class"] == "progressive").mean()*100:.0f}% progressive')