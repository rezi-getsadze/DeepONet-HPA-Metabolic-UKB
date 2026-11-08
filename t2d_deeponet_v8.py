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

COL_WIDTH  = 6.85   # inches, double column
HALF_WIDTH = 3.35   # inches, single column
import sys as _sys, os as _os

class _Tee:
    def __init__(self, path):
        _os.makedirs(_os.path.dirname(path), exist_ok=True)
        self._f = open(path, 'w', buffering=1)
        self._orig = _sys.stdout
    def write(self, x): self._orig.write(x); self._f.write(x)
    def flush(self): self._orig.flush(); self._f.flush()
    def close(self): self._f.close()
_tee = _Tee('runtime/v8/v8_output.txt')
_sys.stdout = _tee

_os.makedirs('runtime/v8', exist_ok=True)
_vv8_n   = [0]
_vv8_saved = [False]
_vv8_orig_sf  = plt.savefig
_vv8_orig_cl  = plt.close
_vv8_orig_fig = plt.figure

def _vv8_sf(*a, **kw):
    _vv8_saved[0] = True
    return _vv8_orig_sf(*a, **kw)

def _vv8_cl(*a, **kw):
    if not _vv8_saved[0] and plt.get_fignums():
        _vv8_n[0] += 1
        _vv8_orig_sf(f'runtime/v8/v8_fig{_vv8_n[0]}.svg', bbox_inches='tight')
    _vv8_saved[0] = False
    return _vv8_orig_cl(*a, **kw)

def _vv8_figure(*a, **kw):
    _vv8_saved[0] = False
    return _vv8_orig_fig(*a, **kw)

_vv8_orig_sp = plt.subplots
def _vv8_subplots(*a, **kw):
    _vv8_saved[0] = False
    return _vv8_orig_sp(*a, **kw)

plt.savefig  = _vv8_sf
plt.close    = _vv8_cl
plt.figure   = _vv8_figure
plt.subplots = _vv8_subplots



# v8 - full cohort N=1,798 (298 diabetic + 1,000 healthy + 500 IFG)
# training set: longitudinal patients held out for blind validation
# all key results live here. v7 does the self-consistency sweep; v10 does the Olink variant.
# run sequentially - cache files (cache/v8_*.npz, cache/v8_*.pt) skip recomputation

# ── parameters
# key calibration constants:
#   eg0_ref=56.2964   anchors healthy attractor at 86.5 mg/dL
#   ALPHA_CAL=0.984   per-patient eg0 scaling exponent
#   ALPHA_HGP=0.3     per-patient hgp_bas scaling exponent
#   EG0_FLOOR=0.75    kept from the pilot, never binds here
#   racor=0.08        was 0.54 - caused NaN when cort>2.85 before the clip

pars={}
inf = np.inf

pars['w1']=244.8
pars['w2']=50.4
pars['w3']=12.384
pars['kgr']=4
pars['n']=3
pars['whpa']=0.02083
pars['lam']=1
pars['wcrhe']=23.04
pars['wc']=0.02
pars['wa']=0.01
pars['wr']=0.023
pars['wcr']=0
pars['ub']=1
pars['uon']=0
pars['uoff']=3600
pars['uinp']=1
pars['don']=0
pars['doff']=0
pars['dinp']=20
pars['per']=30
pars['slt']=20
pars['sw']=1
pars['eg0']=24.48
pars['k']=700
pars['bv']=5
pars['mmax']=1
pars['alpha_m']=140
pars['km']=2
pars['alpha_isr']=1.2
pars['kisr']=2
pars['pmax']=4.55
pars['kp']=4
pars['alpha_p']=35
pars['p_b']=0
pars['amax']=5
pars['alpha_a']=0.37
pars['ka']=6
pars['a_b']=0.9
pars['tau_b']=1800
pars['height']=1.66
pars['age_b']=64
pars['sex']=1
pars['cage']=0
pars['target_si']=1.8
pars['tau_si']=1
pars['bmi_h']=22
pars['mffa']=0.8
pars['ksi_infl']=4.0
pars['ksi_ffa']=400
pars['nsi_ffa']=6
pars['sw11']=1
pars['inc_i1']=0
pars['inc_i2']=0
pars['inc_i3']=0
pars['it1']=0
pars['it2']=inf
pars['it3']=inf
pars['kpc']=7700
pars['tau_w']=1
pars['k_infl']=6.0
pars['n_infl']=2
pars['infl_b']=0
pars['tau_infl']=1
pars['hgp_bas']=2000
pars['hepa_max']=3000
pars['hepa_sic']=1
pars['a_hgp']=4
pars['gcg']=0.1
pars['k_gcg']=0
pars['s']=0.0002
pars['png']=350
pars['sigma_b']=600
pars['sgmu']=1.5
pars['sgmd']=1
pars['sfm']=1.2
pars['sim']=0.25
pars['sgku']=81
pars['sgkd']=137
pars['sfk']=357
pars['sik']=0.6
pars['nsgku']=6
pars['nsgkd']=6
pars['nsfk']=6
pars['nsik']=4
pars['tau_sigma']=1
pars['sci']=1
pars['gclamp']=0
pars['iclamp']=0
pars['maxinfl']=1
pars['l0']=170
pars['l2']=8.1
pars['cf']=2
pars['ksif']=11
pars['aa']=2
pars['sif']=1
pars['ffa_ramax']=1.12
pars['max_w']=0.5
pars['kcort']=1
pars['ncort']=2
pars['kcortsi'] = 1.0
pars['k_fmass']=1.2
pars['n_fmass']=2
pars['max_cort']=0.5
pars['p_vf_base'] = 0.2
pars['p_vf_amp'] = 0.2
pars['k_p'] = 1
pars['n_p'] = 2
pars['maxinfl_vf']=0.9
pars['maxinfl_sf']=0.1
pars['max_insulin']=0.1
pars['k_i']=15
pars['n_i']=2
pars['intake_i']=2380
pars['kcrd'] = 61.9
pars['khgp'] = 16.1
pars['chgp'] = 0.027
pars['racor'] = 0.08  # was 0.54 - (1-racor*(cort-1)) goes negative when cort>2.85, NaN follows
pars['racor1'] = 0.03  # reduced proportionally
pars['maxrd'] = 215
pars['nhgp'] = 2
pars['ncrd'] = 2
pars['ncsi'] = 2
pars['tata'] = 2

p_base = np.array(list(pars.values()))

# calibration constants from binary search (eg0_ref) and alpha sweep
G_REF     = 86.5
eg0_ref   = 56.2964
ALPHA_CAL = 0.984
ALPHA_HGP = 0.3
EG0_FLOOR = 0.75

# ── cache key
import hashlib as _hashlib, json as _json

def _ode_param_hash(extra=None):
    # 8-char hex from p_base + calibration constants
    # cache filenames include this so they auto-invalidate when params change
    h = _hashlib.md5(p_base.tobytes())
    calib = {'G_REF': G_REF, 'eg0_ref': eg0_ref,
             'ALPHA_CAL': ALPHA_CAL, 'ALPHA_HGP': ALPHA_HGP, 'EG0_FLOOR': EG0_FLOOR}
    h.update(_json.dumps(calib, sort_keys=True).encode())
    if extra:
        h.update(_json.dumps(extra, sort_keys=True).encode())
    return h.hexdigest()[:8]

# ── full ODE system (from hpa_fmd_diab.py)
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
	kcrd = pars_npa[113]; khgp = pars_npa[114]; chgp = pars_npa[115]
	racor = pars_npa[116]; racor1 = pars_npa[117]; maxrd = pars_npa[118]
	nhgp = pars_npa[119]; ncrd = pars_npa[120]; ncsi = pars_npa[121]; tata = pars_npa[122]

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
	w_base = 77

	activity_level = 1.6
	bmr = sex * (
		(age < 30) * (15.057 * w + 692.2) +
		((age >= 30) & (age < 60)) * (11.472 * w + 873.1) +
		(age >= 60) * (11.711 * w + 587.7)
	) + (1 - sex) * (
		(age < 30) * (14.818 * w + 486.6) +
		((age >= 30) & (age < 60)) * (8.126 * w + 845.6) +
		(age >= 60) * (9.082 * w + 658.5)
	)
	expd = activity_level * bmr

	k_w = 1/kpc
	m = mmax*g**km/(alpha_m**km + g**km)
	isr = sigma*(m)**kisr/(alpha_isr**kisr + (m)**kisr)
	p_isr = p_b + pmax*isr**kp/(alpha_p**kp + isr**kp)
	a_m = a_b + amax*m**ka/(alpha_a**ka + m**ka)
	p = p_isr
	a = a_m

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
	cl0 = l0*24*60/bv
	cl2 = l2*24*60/bv
	al0 = 2.45*24*60

	p_vf_base = sex * 0.25 + (1 - sex) * 0.15
	p_vf = p_vf_base + p_vf_amp * (cort-1)**n_p / ((cort-1)**n_p + k_p**n_p)
	p_sf = 1.0 - p_vf

	# clip racor terms so denominator never goes negative (caused NaN when cort>2.85)
	racor_eff = np.clip(racor * (cort - 1), 0.0, 0.92)
	racor1_eff = np.clip(racor1 * (cort - 1), 0.0, 0.92)
	rd=(eg0+sci*(si*i)+(1-racor1_eff)*maxrd*(si*i)**ncrd/((1-racor_eff)*kcrd**ncrd+(si*i)**ncrd))*g

	dcrh=w1*(gcort*mcort*u-crh)
	dacth=w2*(cmass*gcort*stim_crh-acth)
	dcort=w3*(amass*acth-cort/cr)
	dcmass=wc*cmass*(stim_crh-1)
	damass=wa*amass*(acth-1)
	dcr=wcr*(cort-cr)
	dext_crh=wcrhe*(d-ext_crh)

	dg=gclamp+hgp-rd
	di=iclamp+b*isr/bv-k*i
	dffa=(cl0+cl2*fmass)*(ffa_ramax*(ksif**aa)/(ksif**aa+(siff*i)**aa))-cf*w*ffa
	dsi=(tsi-si)/tau_si
	db=(png+p*b-a*b-s*b**2)/tau_b
	dsigma=(s_inf-sigma)/tau_sigma

	vf_term = vf**n_infl/(vf**n_infl+k_infl**n_infl)
	sf_term = sf**n_infl/(sf**n_infl+k_infl**n_infl)
	infl_drive = maxinfl_vf*vf_term + maxinfl_sf*sf_term
	dinfl = infl_b + (infl_drive - infl)/tau_infl

	i_thresh = 10.0
	i_eff = max(0, i - i_thresh)
	phi_insulin = max_insulin * i_eff**n_i / (k_i**n_i + i_eff**n_i)

	cortisol_factor = 1 + max_w * (cort-1)**ncort / (kcort**ncort + (cort-1)**ncort)

	carb_frac = 0.5
	phi = carb_frac * intake * k_w * phi_insulin * cortisol_factor
	dw = (intake * k_w - expd * k_w + phi) / tau_w

	p_fmass = 0.35
	dfmass = (p_fmass * (intake * k_w - expd * k_w) + phi)/tau_w

	dvf=p_vf*dfmass
	dsf=p_sf*dfmass

	cort = max(y[2], 0.001)
	g = max(y[7], 0.001)

	dy=np.array([dcrh,dacth,dcort,dcmass,damass,dcr,dext_crh,dg,di,dffa,dsi,db,dsigma,dinfl,dw,dfmass,dvf,dsf])
	return dy

# ── simulation helper
def run_twin_sim(patient, current_pars):
    y0 = [1,1,1,1,1,1,0, patient['G_mgdl'], 9,400,0.8,900,536,0.73,77,18.7,2.8,15.9]
    p_sim = current_pars.copy()
    p_sim[15] = 18.7 + patient['S_input'] * 6.3
    ratio = patient['G_mgdl'] / G_REF
    eg0_i = eg0_ref * (1.0 / ratio) ** ALPHA_CAL
    if patient['G_mgdl'] > 126:
        eg0_i = min(eg0_i, EG0_FLOOR * eg0_ref)
    p_sim[22] = eg0_i
    p_sim[63] = p_base[63] * (ratio ** ALPHA_HGP)
    try:
        sol = sivp(odde, [0, 7200], y0, method='LSODA', args=[p_sim])
        return sol.y[7][-1]
    except:
        return patient['G_mgdl']

# ── load cohorts
df   = pd.read_csv("Training Data/ukb_full_diabetics.csv")
df_h = pd.read_csv("Training Data/ukb_full_healthy.csv")

# ── diabetic ODE run
final_results = []
for idx, patient in df.iterrows():
    y0 = [1,1,1,1,1,1,0, patient['G_mgdl'], 9,400,0.8,900,536,0.73,77,18.7,2.8,15.9]
    p_fail = p_base.copy()
    p_fail[15] = 18.7 + patient['S_input'] * 6.3
    ratio = patient['G_mgdl'] / G_REF
    eg0_i = eg0_ref * (1.0 / ratio) ** ALPHA_CAL
    if patient['G_mgdl'] > 126:
        eg0_i = min(eg0_i, EG0_FLOOR * eg0_ref)
    p_fail[22] = eg0_i
    p_fail[63] = p_base[63] * (ratio ** ALPHA_HGP)
    try:
        sol = sivp(odde, [0, 7200], y0, method='LSODA', args=[p_fail])
        g_start, g_end = sol.y[7][0], sol.y[7][-1]
        status = 'Crash' if g_end > (g_start + 5) else 'Stable'
        final_results.append({'eid': patient['eid'], 'Start': g_start, 'End': g_end,
                               'Delta': g_end - g_start, 'Status': status})
    except:
        continue

res_df = pd.DataFrame(final_results)
print(res_df['Status'].value_counts())

# ── healthy ODE run
healthy_results = []
for idx, patient in df_h.iterrows():
    y0 = [1,1,1,1,1,1,0, patient['G_mgdl'], 9,400,0.8,900,536,0.73,77,18.7,2.8,15.9]
    p_fail = p_base.copy()
    p_fail[15] = 18.7 + patient['S_input'] * 6.3
    ratio = patient['G_mgdl'] / G_REF
    eg0_i = eg0_ref * (1.0 / ratio) ** ALPHA_CAL
    if patient['G_mgdl'] > 126:
        eg0_i = min(eg0_i, EG0_FLOOR * eg0_ref)
    p_fail[22] = eg0_i
    p_fail[63] = p_base[63] * (ratio ** ALPHA_HGP)
    try:
        sol = sivp(odde, [0, 7200], y0, method='LSODA', args=[p_fail])
        g_start, g_end = sol.y[7][0], sol.y[7][-1]
        status = 'Crash' if g_end > (g_start + 5) else 'Stable'
        healthy_results.append({'eid': patient['eid'], 'Start': g_start, 'End': g_end,
                                 'Delta': g_end - g_start, 'Status': status})
    except:
        continue

res_df_healthy = pd.DataFrame(healthy_results)
print(f"Healthy Verdict: {res_df_healthy['Status'].value_counts().get('Crash', 0)} Crashes out of {len(res_df_healthy)}")

# ── stress subgroup analysis (healthy cohort)
# checks whether S_input still does anything at racor=0.08
# S_input is zero-inflated so qcut uses duplicates='drop'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

stress_df = df_h[['S_input']].copy()
stress_df['G_end'] = res_df_healthy['End'].values
stress_df['G0']    = df_h['G_mgdl'].values
stress_df['delta_G'] = stress_df['G_end'] - stress_df['G0']
stress_df = stress_df[stress_df['G_end'].notna() & (stress_df['G_end'] > 0)]

_labels_pool = ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']
stress_df['S_quartile'], _bins = pd.qcut(
    stress_df['S_input'], q=4, retbins=True, duplicates='drop', labels=False
)
n_actual = int(stress_df['S_quartile'].max()) + 1
_labels_used = _labels_pool[:n_actual]
stress_df['S_quartile'] = stress_df['S_quartile'].map({i: _labels_used[i] for i in range(n_actual)})
print(f'S_input quartile bins ({n_actual} distinct): {[round(b,4) for b in _bins]}')

print('Healthy cohort: stress quartile vs G_end')
print(f'{"Quartile":15s}  {"n":>5}  {"S_input mean":>14}  {"G_end mean":>12}  {"delta_G mean":>14}')
quartile_data = {}
for q, grp in stress_df.groupby('S_quartile', observed=True):
    print(f'{str(q):15s}  {len(grp):>5}  {grp["S_input"].mean():>14.3f}  '
          f'{grp["G_end"].mean():>12.2f}  {grp["delta_G"].mean():>+14.3f}')
    quartile_data[str(q)] = grp['G_end'].values

groups = list(quartile_data.values())
h_stat, p_kruskal = stats.kruskal(*groups)
print()
print(f'Kruskal-Wallis across quartiles: H={h_stat:.3f}, p={p_kruskal:.4f}')
if p_kruskal < 0.05:
    print('  Stress quartile significantly shifts G_end')
else:
    print('  No significant difference - everyone gets the same strong stress drive, S only scales it')
    print('  glucose variance dominated by G0/eg0 scaling, not S_input')

q1_key = _labels_used[0]; q4_key = _labels_used[-1]
q1 = quartile_data[q1_key]; q4 = quartile_data[q4_key]
_, p_mw = stats.mannwhitneyu(q1, q4, alternative='two-sided')
print(f'Mann-Whitney {q1_key} vs {q4_key}: p={p_mw:.4f}')
print(f'  {q1_key} G_end mean={q1.mean():.2f}, {q4_key} G_end mean={q4.mean():.2f}')
print(f'  difference = {q4.mean()-q1.mean():+.2f} mg/dL')

colors_q = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c'][:n_actual]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax1 = axes[0]
for (q, grp), col in zip(stress_df.groupby('S_quartile', observed=True), colors_q):
    ax1.hist(grp['G_end'], bins=25, alpha=0.5, color=col, label=str(q), density=True)
ax1.set_xlabel('G_end (mg/dL)'); ax1.set_ylabel('Density')
ax1.set_title('Healthy cohort G_end by stress quartile', loc='left')
ax1.legend(); ax1.grid(alpha=0.2)

ax2 = axes[1]
q_means = [quartile_data[q].mean() for q in _labels_used]
q_sems  = [stats.sem(quartile_data[q]) for q in _labels_used]
ax2.bar(_labels_used, q_means, yerr=q_sems, color=colors_q, alpha=0.8, capsize=4)
ax2.set_xlabel('Stress quartile'); ax2.set_ylabel('Mean G_end (mg/dL)')
ax2.set_title(f'Mean steady-state glucose by stress level, Kruskal-Wallis p={p_kruskal:.4f}', loc='left')
ax2.grid(alpha=0.2, axis='y')

plt.tight_layout()


# ── EG0_FLOOR significance test
from scipy import stats

floor_results, nofloor_results = [], []
for idx, patient in df.iterrows():
    if patient['G_mgdl'] < 126:
        continue
    y0 = [1,1,1,1,1,1,0, patient['G_mgdl'], 9,400,0.8,900,536,0.73,77,18.7,2.8,15.9]
    ratio = patient['G_mgdl'] / G_REF
    p_sim = p_base.copy()
    p_sim[15] = 18.7 + patient['S_input'] * 6.3
    p_sim[63] = p_base[63] * (ratio ** ALPHA_HGP)

    eg0_floor = min(eg0_ref * (1.0 / ratio) ** ALPHA_CAL, EG0_FLOOR * eg0_ref)
    p_sim[22] = eg0_floor
    try:
        sol = sivp(odde, [0, 7200], y0, method='LSODA', args=[p_sim.copy()])
        floor_results.append(sol.y[7][-1])
    except:
        floor_results.append(np.nan)

    eg0_nofloor = eg0_ref * (1.0 / ratio) ** ALPHA_CAL
    p_sim[22] = eg0_nofloor
    try:
        sol = sivp(odde, [0, 7200], y0, method='LSODA', args=[p_sim.copy()])
        nofloor_results.append(sol.y[7][-1])
    except:
        nofloor_results.append(np.nan)

floor_arr   = np.array(floor_results)
nofloor_arr = np.array(nofloor_results)
valid = ~(np.isnan(floor_arr) | np.isnan(nofloor_arr))
floor_arr, nofloor_arr = floor_arr[valid], nofloor_arr[valid]

stat_mw, p_mw = stats.mannwhitneyu(floor_arr, nofloor_arr, alternative='two-sided')
diabetic_with    = np.sum(floor_arr >= 126)
diabetic_without = np.sum(nofloor_arr >= 126)
n = len(floor_arr)
contingency = np.array([[diabetic_with, n - diabetic_with],
                         [diabetic_without, n - diabetic_without]])
chi2, p_chi2, _, _ = stats.chi2_contingency(contingency)

print('EG0_FLOOR significance test (diabetic patients, G0 >= 126):')
print(f'  n = {n}')
print(f'  WITH floor:    mean G_end = {floor_arr.mean():.1f} +/- {floor_arr.std():.1f} mg/dL')
print(f'  WITHOUT floor: mean G_end = {nofloor_arr.mean():.1f} +/- {nofloor_arr.std():.1f} mg/dL')
print(f'  Mann-Whitney U: p = {p_mw:.4f}')
print(f'  Chi-squared: chi2 = {chi2:.2f}, p = {p_chi2:.4f}')
# p=1.0 expected for the UKB cohort - see algebraic proof in next section

# ── EG0_FLOOR algebraic analysis
# analytically showing why the floor never binds (binding threshold < 126)
floor_binding_threshold = G_REF / (EG0_FLOOR ** (1.0 / ALPHA_CAL))
print(f'Floor binds when G0 < {floor_binding_threshold:.1f} mg/dL, applied when G0 > 126')
print(f'{"Mutually exclusive" if floor_binding_threshold < 126 else "Floor CAN bind"} at these thresholds')

# ── individualized caloric intake (BMI artifact fix)
def get_weight_maintaining_intake(weight_kg, sex, age, activity_level=1.6):
    if sex == 1:
        if age < 30:   bmr = 15.057 * weight_kg + 692.2
        elif age < 60: bmr = 11.472 * weight_kg + 873.1
        else:          bmr = 11.711 * weight_kg + 587.7
    else:
        if age < 30:   bmr = 14.818 * weight_kg + 486.6
        elif age < 60: bmr = 8.126  * weight_kg + 845.6
        else:          bmr = 9.082  * weight_kg + 658.5
    return activity_level * bmr

# BMI=37 control: verify weight stays stable with individualized intake
bmi_control = 37.0; height_ctrl = 1.75; sex_ctrl = 1; age_ctrl = 55
weight_ctrl = bmi_control * height_ctrl**2
intake_ctrl = get_weight_maintaining_intake(weight_ctrl, sex_ctrl, age_ctrl)
print(f'BMI=37 control: fixed 2380 kcal vs individualized {intake_ctrl:.1f} kcal')

p_fixed = p_base.copy()
p_fixed[39] = height_ctrl; p_fixed[40] = age_ctrl; p_fixed[41] = sex_ctrl
G0_ctrl = 140.0; ratio_ctrl = G0_ctrl / G_REF
p_fixed[15] = 18.7
p_fixed[22] = min(eg0_ref * (1.0 / ratio_ctrl)**ALPHA_CAL, EG0_FLOOR * eg0_ref)
p_fixed[63] = p_base[63] * (ratio_ctrl**ALPHA_HGP)

T_90d = 90 * 24 * 60
t_eval_ctrl = np.linspace(0, T_90d, 200)
y0_ctrl = [1,1,1,1,1,1,0, G0_ctrl, 9,400,0.8,900,536,0.73, weight_ctrl, 18.7,2.8,15.9]

try:
    sol_fixed = sivp(odde, [0, T_90d], y0_ctrl, method='LSODA', args=[p_fixed], t_eval=t_eval_ctrl)
    bmi_fixed = sol_fixed.y[14] / height_ctrl**2
    print(f'Fixed intake: BMI drift = {bmi_fixed[0]:.2f} -> {bmi_fixed[-1]:.2f}')
except Exception as e:
    print(f'Fixed intake sim failed: {e}'); sol_fixed = None

p_indiv = p_fixed.copy(); p_indiv[112] = intake_ctrl
try:
    sol_indiv = sivp(odde, [0, T_90d], y0_ctrl, method='LSODA', args=[p_indiv], t_eval=t_eval_ctrl)
    bmi_indiv = sol_indiv.y[14] / height_ctrl**2
    print(f'Indiv. intake: BMI drift = {bmi_indiv[0]:.2f} -> {bmi_indiv[-1]:.2f}')
except Exception as e:
    print(f'Indiv. intake sim failed: {e}'); sol_indiv = None

# ── combined data prep (the figures below need these)
ukb_all  = np.concatenate([df['G_mgdl'], df_h['G_mgdl']])
sim_all  = np.concatenate([res_df['End'], res_df_healthy['End']])

bins   = [0, 100, 126, 1000]
labels = ['Normal', 'IFG', 'Diabetes']
ukb_sorted = np.sort(ukb_all); sim_sorted = np.sort(sim_all)
percs = np.linspace(0, 100, 100)
q_ukb = np.percentile(ukb_sorted, percs); q_sim = np.percentile(sim_sorted, percs)

plot_data = pd.DataFrame({
    'Glucose': np.concatenate([ukb_all, sim_all]),
    'Source':  ['UKB Actual'] * len(ukb_all) + ['Model Simulated'] * len(sim_all)
})
counts_ukb = pd.Series(pd.cut(ukb_all, bins=bins, labels=labels)).value_counts(normalize=True).reindex(labels).values
counts_sim = pd.Series(pd.cut(sim_all, bins=bins, labels=labels)).value_counts(normalize=True).reindex(labels).values

def add_letters(fig):
    for i, ax in enumerate(fig.get_axes()):
        ax.text(-0.1, 1.1, string.ascii_uppercase[i],
                transform=ax.transAxes, fontsize=10, weight='bold', va='top', ha='right')

# resilience comparison plot
plt.figure(figsize=(12, 7))
for status, color, label in [('Stable', '#3498db', 'T2D - Resilient'), ('Crash', '#e74c3c', 'T2D - Vulnerable')]:
    subset = res_df[res_df['Status'] == status]
    plt.scatter(subset['Start'], subset['Delta'], c=color, label=label, marker='o', s=80, edgecolors='k', alpha=0.8)
for status, color, label in [('Stable', '#85C1E9', 'Healthy - Resilient'), ('Crash', '#F1948A', 'Healthy - Vulnerable')]:
    subset = res_df_healthy[res_df_healthy['Status'] == status]
    plt.scatter(subset['Start'], subset['Delta'], c=color, label=label, marker='^', s=80, edgecolors='k', alpha=0.6)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
plt.xlabel("Baseline Blood Glucose (mg/dL)")
plt.ylabel("Glucose Change After Stress (delta G)")
plt.title("The Resilience Gap: T2D Cohort vs. Healthy Controls")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3); plt.tight_layout();

# ── sensitivity tornado
test_params = {'target_si': 43, 'hgp_bas': 63, 'eg0': 22, 'kcortsi': 99, 'racor': 116, 'k': 23}
base_mean = 178.5
tornado_results = []

for name, idx in test_params.items():
    p_up = p_base.copy(); p_down = p_base.copy()
    p_up[idx] *= 1.1; p_down[idx] *= 0.9
    mu_up   = np.mean([run_twin_sim(p, p_up)   for _, p in df.head(10).iterrows()])
    mu_down = np.mean([run_twin_sim(p, p_down) for _, p in df.head(10).iterrows()])
    tornado_results.append((name, mu_up - base_mean, mu_down - base_mean))

tornado_results.sort(key=lambda x: abs(x[1]), reverse=True)
names_raw, ups_raw, downs_raw = zip(*tornado_results)
_label_map = {'target_si': r'$SI_{target}$', 'hgp_bas': r'$HGP_{bas}$', 'eg0': r'$eg_0$',
              'kcortsi': r'$k_{cortsi}$', 'racor': r'$r_{acor}$', 'k': r'$k$'}
params_math = [_label_map[n] for n in names_raw]
ups = list(ups_raw); downs = list(downs_raw)

# ── 4-panel validation figure
import matplotlib.gridspec as gridspec
c_ukb = '#808080'; c_sim = '#0000FF'; c_ifg = '#FFD700'; c_dia = '#FF0000'

fig2 = plt.figure(figsize=(15, 12))
gs = gridspec.GridSpec(2, 2, figure=fig2)
ax_qq = fig2.add_subplot(gs[0, 0]); ax_ecdf = fig2.add_subplot(gs[0, 1])
ax_dens = fig2.add_subplot(gs[1, 0]); ax_violin = fig2.add_subplot(gs[1, 1])

ax_qq.scatter(q_ukb, q_sim, color=c_sim, alpha=0.6, s=40, edgecolors='k', zorder=3)
ax_qq.plot([0, 400], [0, 400], color='green', linestyle='--', label='Perfect Fit', lw=2)
ax_qq.set_title("A. Q-Q plot", pad=20, loc='left')
ax_qq.set_xlabel('Clinical Quantiles (mg/dL)'); ax_qq.set_ylabel('Simulated Quantiles (mg/dL)')
ax_qq.set_box_aspect(1); ax_qq.grid(True, alpha=0.3, linestyle='--')
ax_qq.set_xlim(50, 325); ax_qq.set_ylim(50, 325); ax_qq.legend()

ax_ecdf.step(np.sort(ukb_all), np.linspace(0, 1, len(ukb_all)), color=c_ukb, lw=2.5, label='UKB Clinical')
ax_ecdf.step(np.sort(sim_all), np.linspace(0, 1, len(sim_all)), color=c_sim, lw=2.5, label='Model Simulated')
ax_ecdf.axvline(x=100, color=c_ifg, linestyle='--', alpha=0.8, label='IFG (100)')
ax_ecdf.axvline(x=126, color=c_dia, linestyle='--', alpha=0.8, label='Diabetes (126)')
ax_ecdf.set_title("B. ECDF comparison", pad=20, loc='left')
ax_ecdf.set_xlabel('Glucose (mg/dL)'); ax_ecdf.set_ylabel('Cumulative Proportion')
ax_ecdf.set_box_aspect(1); ax_ecdf.grid(True, alpha=0.2); ax_ecdf.legend(loc='lower right')

sns.kdeplot(ukb_all, ax=ax_dens, fill=True, color=c_ukb, alpha=0.3, bw_adjust=1.5, label='UKB Clinical')
sns.kdeplot(sim_all, ax=ax_dens, fill=True, color=c_sim, alpha=0.5, bw_adjust=0.5, label='Model Simulated')
ax_dens.axvline(x=100, color=c_ifg, linestyle='--', linewidth=2)
ax_dens.axvline(x=126, color=c_dia, linestyle='--', linewidth=2)
ax_dens.set_title("C. Density overlay", pad=20, loc='left')
ax_dens.set_xlabel('Fasting Glucose (mg/dL)'); ax_dens.set_ylabel('Density')
ax_dens.set_ylim(0, 0.045); ax_dens.grid(True, alpha=0.2); ax_dens.legend()

sns.violinplot(ax=ax_violin, x='Source', y='Glucose', hue='Source', data=plot_data,
               inner='quartile', palette=[c_ukb, c_sim], legend=False)
ax_violin.axhline(y=126, color=c_dia, linestyle='--', alpha=0.8)
ax_violin.axhline(y=100, color=c_ifg, linestyle='--', alpha=0.8)
ax_violin.set_title("D. Quartile distribution", pad=20, loc='left')
ax_violin.set_ylabel("Glucose (mg/dL)")
plt.tight_layout();

# ── sensitivity + clinical proportions
fig3, axs3 = plt.subplots(1, 2, figsize=(14, 6))
plt.subplots_adjust(wspace=0.35)

axs3[0].barh(params_math, ups,   color=c_dia, alpha=0.6, label='+10 % change', edgecolor='k')
axs3[0].barh(params_math, downs, color=c_sim, alpha=0.6, label='-10 % change', edgecolor='k')
axs3[0].axvline(0, color='black', lw=1.5)
axs3[0].set_xlabel('Delta Glucose (mg/dL)')
axs3[0].set_title('A. Sensitivity analysis', pad=20, loc='left')
axs3[0].grid(alpha=0.3, axis='x'); axs3[0].invert_yaxis(); axs3[0].legend()

width = 0.35; x_pos = np.arange(3)
axs3[1].bar(x_pos - width/2, counts_ukb, width, label='UKB Actual',      color=c_ukb, edgecolor='black')
axs3[1].bar(x_pos + width/2, counts_sim, width, label='Model Simulated', color=c_sim, edgecolor='black', alpha=0.7)
axs3[1].set_xticks(x_pos); axs3[1].set_xticklabels(['Normal', 'IFG', 'Diabetes'])
axs3[1].set_title('B. Clinical proportions', pad=20, loc='left')
axs3[1].set_ylabel('Proportion'); axs3[1].set_ylim(0, 1.1)
axs3[1].grid(axis='y', alpha=0.3); axs3[1].legend()
plt.tight_layout();

# ── DeepONet architecture
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
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
    def forward(self, x):
        return self.net(x)

class DeepONet(nn.Module):
    """
    Branch: [S_input, BMI/40, age/80, sex, CRP/10]
    Trunk:  [t_norm, G0/200]
    tanh IC reconstruction: G(t) = G0 + tanh(k*t) * raw * 200
    """
    def __init__(self, basis_dim=64, hidden=64):
        super().__init__()
        self.branch = _MLP(5, hidden, basis_dim)
        self.trunk  = _MLP(2, hidden, basis_dim)
        self.bias   = nn.Parameter(torch.zeros(1))

    def forward(self, branch_in, trunk_in):
        b = self.branch(branch_in)
        t = self.trunk(trunk_in)
        return (b * t).sum(dim=-1, keepdim=True) + self.bias

# ── training/validation split
# IFG cohort needed early so df_ifg_train is available for training
df_ifg = pd.read_csv("Training Data/ukb_full_ifg.csv")
print(f"IFG cohort loaded: {len(df_ifg)} patients, G0 range {df_ifg['G_mgdl'].min():.1f}-{df_ifg['G_mgdl'].max():.1f} mg/dL")

# longitudinal diabetics held out for blind validation only
# healthy and IFG stay fully in the training set
has_longval_d = (df['p30740_i1'].notna() & df['p53_i1'].notna())
longval_eids = set(df.loc[has_longval_d, 'eid'].values)
print(f"Diabetic patients with longitudinal follow-up: {len(longval_eids)} (held out for blind validation)")

df_train     = df[~df['eid'].isin(longval_eids)].copy()
df_h_train   = df_h.copy()
df_ifg_train = df_ifg[~df_ifg['eid'].isin(longval_eids)].copy()
print(f"Training set: {len(df_train)} diabetic + {len(df_h_train)} healthy + {len(df_ifg_train)} IFG = {len(df_train)+len(df_h_train)+len(df_ifg_train)} total")

# ── 5-day training data builder
def build_training_dataset(n_timepoints=40, cache_path=None):
    if cache_path is None:
        cache_path = f'cache/5d_data_{_ode_param_hash({"n_tp": n_timepoints})}.npz'
    if os.path.exists(cache_path):
        print(f'loading cached dataset from {cache_path}')
        d = np.load(cache_path)
        n_pat = int(d['n_patients'][0]) if 'n_patients' in d else len(np.unique(d['trunk_in'][:,1]))
        return d['branch_in'], d['trunk_in'], d['targets'], n_pat

    all_patients = pd.concat([df_train, df_h_train, df_ifg_train], ignore_index=True)
    all_patients['p30750_i0'] = all_patients['p30750_i0'].fillna(all_patients['p30750_i0'].median())
    all_patients['p30710_i0'] = all_patients['p30710_i0'].fillna(all_patients['p30710_i0'].median())

    t_eval = np.linspace(0, 7200, n_timepoints)
    branch_rows, trunk_rows, target_rows = [], [], []
    failed = 0

    print(f'running ODE for {len(all_patients)} UKB patients x {n_timepoints} timepoints...')
    # only S and G0 change the ODE run. BMI, age, sex and CRP go into the branch net
    # but stay at model defaults in the ODE, so the surrogate never sees their effect
    for _, row in all_patients.iterrows():
        p_sim  = p_base.copy()
        p_sim[15] = 18.7 + row['S_input'] * 6.3
        ratio  = row['G_mgdl'] / G_REF
        eg0_i  = eg0_ref * (1.0 / ratio) ** ALPHA_CAL
        if row['G_mgdl'] > 126:
            eg0_i = min(eg0_i, EG0_FLOOR * eg0_ref)
        p_sim[22] = eg0_i; p_sim[63] = p_base[63] * (ratio ** ALPHA_HGP)
        y0 = [1,1,1,1,1,1,0, row['G_mgdl'], 9,400,0.8,900,536,0.73,77,18.7,2.8,15.9]
        try:
            sol = sivp(odde, [0,7200], y0, method='LSODA', args=[p_sim], t_eval=t_eval)
            if not sol.success:
                failed += 1; continue
        except:
            failed += 1; continue

        G0 = row['G_mgdl']; t_norm = t_eval / 7200.0
        branch_feat = [row['S_input'], row['p21001_i0']/40.0, row['p21022']/80.0,
                       float(row['p31']), row['p30710_i0']/10.0]
        for t, g in zip(t_norm, sol.y[7]):
            branch_rows.append(branch_feat)
            trunk_rows.append([t, G0/200.0])
            target_rows.append([(g - G0) / 200.0])

    B = np.array(branch_rows, dtype=np.float32)
    T = np.array(trunk_rows,  dtype=np.float32)
    Y = np.array(target_rows, dtype=np.float32)
    n_ok = len(B) // n_timepoints
    np.savez(cache_path, branch_in=B, trunk_in=T, targets=Y, n_patients=np.array([n_ok]))
    print(f'saved {len(B):,} rows ({failed} ODE failures) to {cache_path}')
    return B, T, Y, n_ok

# ── 5-day training loop (tanh IC)
# weighted loss tested and discarded - it fights the architecture instead of fixing it
TANH_K = 5.0

def train_deeponet_v2(epochs=300, batch_size=256, lr=3e-4, cache_path=None):
    B, T, Y, n_patients_total = build_training_dataset(cache_path=cache_path)

    t_norms = T[:, 0]
    keep = t_norms > 0  # drop t=0, IC handles it
    B, T, Y = B[keep], T[keep], Y[keep]
    tanh_ramp = np.tanh(TANH_K * T[:, 0:1]).astype(np.float32)
    Y = Y / tanh_ramp

    n_patients = n_patients_total
    n_timepoints_per = len(B) // n_patients
    patient_idx = np.arange(n_patients)
    np.random.seed(28); np.random.shuffle(patient_idx)
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
    print(f'training DeepONet - {n_params:,} params, {len(train_ds):,} train / {len(val_ds):,} val')

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for b_b, t_b, y_b, _ in train_loader:
            optimiser.zero_grad()
            loss = nn.functional.mse_loss(model(b_b, t_b), y_b)
            loss.backward(); optimiser.step()
            epoch_loss += loss.item() * len(b_b)
        train_losses.append(epoch_loss / len(train_ds))

        model.eval()
        with torch.no_grad():
            vl = sum(nn.functional.mse_loss(model(b_b, t_b), y_b).item() * len(b_b)
                     for b_b, t_b, y_b, _ in val_loader)
        val_losses.append(vl / len(val_ds))
        scheduler.step()
        if epoch % 50 == 0:
            print(f'  epoch {epoch:3d} | train {train_losses[-1]:.6f} | val {val_losses[-1]:.6f}')

    os.makedirs('cache', exist_ok=True)
    torch.save(model.state_dict(), 'cache/v8_5d_model.pt')
    return model, train_losses, val_losses, val_patients.tolist()

def predict_trajectory_tanh(model, patient, n_timepoints=100):
    G0 = patient['G_mgdl']
    t_norm = np.linspace(0, 1, n_timepoints).astype(np.float32)
    branch_feat = np.array([patient['S_input'], patient['p21001_i0']/40.0,
                             patient['p21022']/80.0, float(patient['p31']),
                             patient['p30710_i0']/10.0], dtype=np.float32)
    branch_in = np.tile(branch_feat, (n_timepoints, 1))
    trunk_in  = np.column_stack([t_norm, np.full(n_timepoints, G0/200.0)]).astype(np.float32)
    model.eval()
    with torch.no_grad():
        raw = model(torch.tensor(branch_in), torch.tensor(trunk_in)).numpy().flatten()
    ramp = np.tanh(TANH_K * t_norm)
    return t_norm * 7200.0, G0 + ramp * raw * 200.0

print('training 5-day DeepONet (delete 5d_data_*.npz to force rebuild)...')
model_tanh, tl_tanh, vl_tanh, val_idx_tanh = train_deeponet_v2()
best_model   = model_tanh
best_predict = predict_trajectory_tanh
print('done - using tanh IC model for all downstream analysis')

# ── temporal MAE on held-out patients
def compute_temporal_mae(model, predict_fn, val_indices, n_timepoints=40):
    all_patients = pd.concat([df_train.assign(_cohort='T2D'),
                              df_h_train.assign(_cohort='Healthy'),
                              df_ifg_train.assign(_cohort='IFG')], ignore_index=True)
    all_patients['p30750_i0'] = all_patients['p30750_i0'].fillna(all_patients['p30750_i0'].median())
    all_patients['p30710_i0'] = all_patients['p30710_i0'].fillna(all_patients['p30710_i0'].median())
    eval_pts = all_patients.iloc[val_indices].reset_index(drop=True)
    t_eval = np.linspace(0, 7200, n_timepoints)

    _key = _ode_param_hash({'T_END': 7200, 'n_tp': n_timepoints, 'val': sorted(map(int, val_indices))})
    traj_cache = f'cache/5d_val_ode_{_key}.npz'
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
            except:
                ode_traj = np.full(n_timepoints, np.nan)
            ode_trajs.append(ode_traj)
        np.savez(traj_cache, ode_trajs=np.array(ode_trajs))

    don_trajs = [predict_fn(model, row, n_timepoints=n_timepoints)[1] for _, row in eval_pts.iterrows()]
    ode_trajs = np.array(ode_trajs); don_trajs = np.array(don_trajs)
    mae_t = np.nanmean(np.abs(ode_trajs - don_trajs), axis=0)
    std_t = np.nanstd(np.abs(ode_trajs - don_trajs), axis=0)
    return mae_t, std_t, ode_trajs, don_trajs

n_tp = 40
t_days = np.linspace(0, 7200, n_tp) / 1440.0
mae_tanh, std_tanh, ode_t, don_t_tanh = compute_temporal_mae(model_tanh, predict_trajectory_tanh, val_idx_tanh, n_tp)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.plot(tl_tanh, label='train', color='#e74c3c', alpha=0.8)
ax.plot(vl_tanh, label='val',   color='#e74c3c', ls='--')
ax.set_xlabel('epoch'); ax.set_ylabel('MSE loss')
ax.set_title('A. Training convergence (tanh IC)', loc='left')
ax.legend(); ax.grid(alpha=0.3); ax.set_yscale('log')

ax2 = axes[1]
ax2.plot(t_days, mae_tanh, color='#e74c3c', lw=2.5, label='tanh IC')
ax2.fill_between(t_days, mae_tanh - std_tanh, mae_tanh + std_tanh, color='#e74c3c', alpha=0.15)
ax2.set_xlabel('Time (days)'); ax2.set_ylabel('MAE (mg/dL)')
ax2.set_title(f'B. Temporal MAE - held-out patients (n={len(val_idx_tanh)})', loc='left')
ax2.set_ylim(0, max(1.0, (mae_tanh + std_tanh).max() * 1.25))
ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout();

print('Temporal MAE summary (held-out patients):')
print(f'  tanh IC:')
print(f'    t=0: {mae_tanh[0]:.2f}  |  t=mid: {mae_tanh[n_tp//2]:.2f}  |  t=end: {mae_tanh[-1]:.2f} mg/dL')
print(f'    max: {mae_tanh.max():.2f}')

# ── stability analysis: approach A (5-day endpoint classifier)
def classify_endpoint(g_start, g_end, tol=5.0):
    delta = g_end - g_start
    if delta > tol: return 'progressive'
    elif delta < -tol: return 'recovering'
    else: return 'stable'

def endpoint_stability_sweep(model, predict_fn, n_G0=80, n_S=60, bmi_values=[22, 27, 32]):
    all_pts = pd.concat([df_train, df_h_train, df_ifg_train], ignore_index=True)
    age_med = all_pts['p21022'].median()
    sex_med = float(all_pts['p31'].mode()[0])
    crp_med = all_pts['p30710_i0'].median()
    G0_range = np.linspace(65, 400, n_G0); S_range = np.linspace(0.0, 1.0, n_S)
    results = {}
    for bmi in bmi_values:
        landscape = np.full((n_S, n_G0), np.nan)
        classes   = np.empty((n_S, n_G0), dtype='U15')
        for i, S in enumerate(S_range):
            for j, G0 in enumerate(G0_range):
                fake_patient = {'S_input': S, 'p21001_i0': bmi, 'p21022': age_med,
                                'p31': sex_med, 'p30710_i0': crp_med, 'G_mgdl': G0}
                _, g_traj = predict_fn(model, fake_patient, n_timepoints=50)
                landscape[i, j] = g_traj[-1]
                classes[i, j] = classify_endpoint(G0, g_traj[-1])
        results[bmi] = {'landscape': landscape, 'classes': classes, 'G0_range': G0_range, 'S_range': S_range}
        print(f'  BMI={bmi}: done')
    return results

sweep_A = endpoint_stability_sweep(best_model, best_predict)

# approach A figures
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
bmi_labels = {22: 'Healthy BMI (22)', 27: 'Overweight (27)', 32: 'Obese (32)'}
cat_colors_A = {'recovering': '#3498db', 'stable': '#2ecc71', 'progressive': '#e74c3c'}
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

for ax, bmi in zip(axes, [22, 27, 32]):
    res = sweep_A[bmi]
    cat_map = {'recovering': 0, 'stable': 1, 'progressive': 2}
    numeric = np.vectorize(lambda c: cat_map.get(c, -1))(res['classes'])
    cmap = ListedColormap([cat_colors_A[k] for k in ['recovering', 'stable', 'progressive']])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.pcolormesh(res['G0_range'], res['S_range'], numeric, cmap=cmap, norm=norm, shading='nearest')
    ax.axvline(100, color='orange', ls='--', lw=1.5); ax.axvline(126, color='red', ls='--', lw=1.5)
    all_patients = pd.concat([df.assign(_cohort='T2D'), df_h.assign(_cohort='Healthy'), df_ifg.assign(_cohort='IFG')], ignore_index=True)
    for cohort, color, marker in [('T2D', 'white', 'o'), ('Healthy', 'black', '^')]:
        mask = (all_patients['_cohort'] == cohort) & (all_patients['p21001_i0'].between(bmi - 5, bmi + 5))
        ax.scatter(all_patients.loc[mask, 'G_mgdl'], all_patients.loc[mask, 'S_input'],
                   color=color, marker=marker, s=30, edgecolors='k', lw=0.5, zorder=5)
    ax.set_xlabel('Initial glucose G0 (mg/dL)'); ax.set_ylabel('Stress score S')
    ax.set_title(f'{bmi_labels[bmi]}')
plt.tight_layout();

# ── approach B: 90-day retrained DeepONet
T_END_90 = 129600  # 90 days in minutes

def build_training_dataset_90d(n_timepoints=60, cache_path=None):
    if cache_path is None:
        cache_path = f'cache/90d_data_{_ode_param_hash({"T_END": T_END_90, "n_tp": n_timepoints})}.npz'
    if os.path.exists(cache_path):
        print(f'loading cached 90-day dataset from {cache_path}')
        d = np.load(cache_path)
        n_pat = int(d['n_patients'][0]) if 'n_patients' in d else len(np.unique(d['trunk_in'][:,1]))
        return d['branch_in'], d['trunk_in'], d['targets'], n_pat

    all_patients = pd.concat([df_train, df_h_train, df_ifg_train], ignore_index=True)
    all_patients['p30750_i0'] = all_patients['p30750_i0'].fillna(all_patients['p30750_i0'].median())
    all_patients['p30710_i0'] = all_patients['p30710_i0'].fillna(all_patients['p30710_i0'].median())
    t_eval = np.linspace(0, T_END_90, n_timepoints)
    branch_rows, trunk_rows, target_rows = [], [], []
    failed = 0
    print(f'running ODE for {len(all_patients)} patients x {n_timepoints} timepoints at 90 days...')
    for _, row in all_patients.iterrows():
        p_sim = p_base.copy(); p_sim[15] = 18.7 + row['S_input'] * 6.3
        ratio = row['G_mgdl'] / G_REF
        eg0_i = eg0_ref * (1.0 / ratio) ** ALPHA_CAL
        if row['G_mgdl'] > 126: eg0_i = min(eg0_i, EG0_FLOOR * eg0_ref)
        p_sim[22] = eg0_i; p_sim[63] = p_base[63] * (ratio ** ALPHA_HGP)
        y0 = [1,1,1,1,1,1,0, row['G_mgdl'], 9,400,0.8,900,536,0.73,77,18.7,2.8,15.9]
        try:
            sol = sivp(odde, [0, T_END_90], y0, method='LSODA', args=[p_sim], t_eval=t_eval)
            if not sol.success:
                failed += 1; continue
        except:
            failed += 1; continue
        G0 = row['G_mgdl']; t_norm = t_eval / T_END_90
        branch_feat = [row['S_input'], row['p21001_i0']/40.0, row['p21022']/80.0,
                       float(row['p31']), row['p30710_i0']/10.0]
        for t, g in zip(t_norm, sol.y[7]):
            branch_rows.append(branch_feat); trunk_rows.append([t, G0/200.0])
            target_rows.append([(g - G0) / 200.0])
    B = np.array(branch_rows, dtype=np.float32); T = np.array(trunk_rows, dtype=np.float32)
    Y = np.array(target_rows, dtype=np.float32); n_ok = len(B) // n_timepoints
    np.savez(cache_path, branch_in=B, trunk_in=T, targets=Y, n_patients=np.array([n_ok]))
    print(f'saved {len(B):,} rows ({failed} ODE failures) to {cache_path}')
    return B, T, Y, n_ok

TANH_K_90 = 5.0

def train_deeponet_90d(epochs=400, batch_size=256, lr=3e-4, cache_path=None):
    B, T, Y, n_patients_total = build_training_dataset_90d(cache_path=cache_path)
    t_norms = T[:, 0]; keep = t_norms > 0
    B, T, Y = B[keep], T[keep], Y[keep]
    tanh_ramp = np.tanh(TANH_K_90 * T[:, 0:1]).astype(np.float32); Y = Y / tanh_ramp
    n_patients = n_patients_total; n_tp = len(B) // n_patients
    patient_idx = np.arange(n_patients)
    np.random.seed(28); np.random.shuffle(patient_idx)
    split = int(0.8 * n_patients); train_p = patient_idx[:split]; val_p = patient_idx[split:]
    train_rows = np.concatenate([np.arange(i*n_tp, (i+1)*n_tp) for i in train_p])
    val_rows   = np.concatenate([np.arange(i*n_tp, (i+1)*n_tp) for i in val_p])
    Bt, Tt, Yt = torch.tensor(B), torch.tensor(T), torch.tensor(Y)
    ones = torch.ones(len(B))
    train_ds = TensorDataset(Bt[train_rows], Tt[train_rows], Yt[train_rows], ones[train_rows])
    val_ds   = TensorDataset(Bt[val_rows],   Tt[val_rows],   Yt[val_rows],   ones[val_rows])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    model_90 = DeepONet()
    optimiser = torch.optim.Adam(model_90.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=100, gamma=0.4)
    loss_fn = nn.MSELoss()
    train_losses, val_losses = [], []
    print(f'training 90-day DeepONet - {sum(p.numel() for p in model_90.parameters()):,} params')
    for epoch in range(1, epochs + 1):
        model_90.train()
        epoch_loss = 0.0
        for b_b, t_b, y_b, _ in train_loader:
            optimiser.zero_grad(); loss = loss_fn(model_90(b_b, t_b), y_b)
            loss.backward(); optimiser.step(); epoch_loss += loss.item() * len(b_b)
        train_losses.append(epoch_loss / len(train_ds))
        model_90.eval()
        with torch.no_grad():
            vl = sum(loss_fn(model_90(b, t), y).item() * len(b) for b, t, y, _ in val_loader) / len(val_ds)
        val_losses.append(vl); scheduler.step()
        if epoch % 50 == 0:
            print(f'  epoch {epoch:3d} | train {train_losses[-1]:.6f} | val {val_losses[-1]:.6f}')
    torch.save(model_90.state_dict(), 'cache/v8_90d_model.pt')
    return model_90, train_losses, val_losses, val_p.tolist()

print('training 90-day model...')
model_90d, tl_90, vl_90, val_idx_90 = train_deeponet_90d()

def predict_trajectory_90d(model, patient, n_timepoints=100):
    G0 = patient['G_mgdl']
    t_norm = np.linspace(0, 1, n_timepoints).astype(np.float32)
    branch_feat = np.array([patient['S_input'], patient['p21001_i0']/40.0,
                             patient['p21022']/80.0, float(patient['p31']),
                             patient['p30710_i0']/10.0], dtype=np.float32)
    branch_in = np.tile(branch_feat, (n_timepoints, 1))
    trunk_in  = np.column_stack([t_norm, np.full(n_timepoints, G0/200.0)]).astype(np.float32)
    model.eval()
    with torch.no_grad():
        raw = model(torch.tensor(branch_in), torch.tensor(trunk_in)).numpy().flatten()
    ramp = np.tanh(TANH_K_90 * t_norm)
    return t_norm * T_END_90, G0 + ramp * raw * 200.0

# ── 90-day temporal validation
def temporal_validation_90d(model, val_indices, n_timepoints=60):
    all_patients = pd.concat([df.assign(_cohort='T2D'), df_h.assign(_cohort='Healthy'),
                              df_ifg.assign(_cohort='IFG')], ignore_index=True)
    all_patients['p30750_i0'] = all_patients['p30750_i0'].fillna(all_patients['p30750_i0'].median())
    all_patients['p30710_i0'] = all_patients['p30710_i0'].fillna(all_patients['p30710_i0'].median())
    eval_pts = all_patients.iloc[val_indices].reset_index(drop=True)
    t_eval = np.linspace(0, T_END_90, n_timepoints); t_days = t_eval / 1440.0
    _key = _ode_param_hash({'T_END': T_END_90, 'n_tp': n_timepoints, 'val': sorted(map(int, val_indices))})
    traj_cache = f'cache/90d_val_ode_{_key}.npz'
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
                sol = sivp(odde, [0, T_END_90], y0, method='LSODA', args=[p_sim], t_eval=t_eval)
                ode_traj = sol.y[7] if sol.success else np.full(n_timepoints, np.nan)
            except:
                ode_traj = np.full(n_timepoints, np.nan)
            ode_trajs.append(ode_traj)
        np.savez(traj_cache, ode_trajs=np.array(ode_trajs))
    don_trajs = [predict_trajectory_90d(model, row, n_timepoints=n_timepoints)[1] for _, row in eval_pts.iterrows()]
    ode_trajs = np.array(ode_trajs); don_trajs = np.array(don_trajs)
    mae_t = np.nanmean(np.abs(ode_trajs - don_trajs), axis=0)
    std_t = np.nanstd(np.abs(ode_trajs - don_trajs), axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(t_days, mae_t, color='#e74c3c', lw=2, label='MAE')
    ax.fill_between(t_days, mae_t - std_t, mae_t + std_t, color='#e74c3c', alpha=0.2, label='+-1 SD')
    ax.axhline(10, color='gray', ls='--', lw=1, label='10 mg/dL target')
    ax.set_xlabel('Time (days)'); ax.set_ylabel('MAE (mg/dL)')
    ax.set_title('A. 90-day DeepONet temporal error', loc='left')
    ax.legend(); ax.grid(alpha=0.3)
    ax2 = axes[1]
    checkpoints = [0, n_timepoints//2, -1]
    cp_labels = ['t=0', f't={t_days[n_timepoints//2]:.0f}d (mid)', f't={t_days[-1]:.0f}d (end)']
    percs = np.linspace(5, 95, 50)
    for cp, cp_label, color in zip(checkpoints, cp_labels, ['#3498db', '#e67e22', '#e74c3c']):
        q_ode = np.nanpercentile(ode_trajs[:, cp], percs)
        q_don = np.nanpercentile(don_trajs[:, cp], percs)
        ax2.scatter(q_ode, q_don, label=cp_label, color=color, alpha=0.7, s=30, edgecolors='k', lw=0.5)
    lo = min(np.nanmin(ode_trajs), np.nanmin(don_trajs)) - 5
    hi = max(np.nanmax(ode_trajs), np.nanmax(don_trajs)) + 5
    ax2.plot([lo, hi], [lo, hi], 'k--', lw=1.5, label='Perfect fit')
    ax2.set_xlabel('ODE quantiles (mg/dL)'); ax2.set_ylabel('DeepONet quantiles (mg/dL)')
    ax2.set_title('B. QQ plot (90-day model)', loc='left')
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout();
    print(f'90-day temporal MAE (held-out, n={len(eval_pts)}):')
    print(f'  t=0: {mae_t[0]:.2f}  |  t=mid ({t_days[n_timepoints//2]:.0f}d): {mae_t[n_timepoints//2]:.2f}  |  t=end ({t_days[-1]:.0f}d): {mae_t[-1]:.2f} mg/dL')
    return mae_t, ode_trajs, don_trajs, t_days

mae_90, ode_90, don_90, tdays_90 = temporal_validation_90d(model_90d, val_idx_90)

# ── 90-day trajectory classifier
def classify_trajectory_90d(g_traj, tol=5.0, skip_frac=0.15):
    n_skip = max(1, int(len(g_traj) * skip_frac))
    g_settled = g_traj[n_skip:]
    kernel = min(5, len(g_settled) // 3)
    g_smooth = np.convolve(g_settled, np.ones(kernel)/kernel, mode='valid') if kernel > 1 else g_settled
    delta = g_smooth[-1] - g_smooth[0]
    diffs = np.diff(g_smooth)
    pos_frac = np.mean(diffs > 0.01); neg_frac = np.mean(diffs < -0.01)
    if abs(delta) < tol: return 'stable'
    elif delta < -tol and neg_frac > 0.6: return 'recovering'
    elif delta > tol and pos_frac > 0.6: return 'progressive'
    else: return 'oscillatory'

def trajectory_stability_sweep_90d(model, n_G0=80, n_S=60, bmi_values=[22, 27, 32]):
    all_pts = pd.concat([df_train, df_h_train, df_ifg_train], ignore_index=True)
    age_med = all_pts['p21022'].median(); sex_med = float(all_pts['p31'].mode()[0]); crp_med = all_pts['p30710_i0'].median()
    G0_range = np.linspace(65, 400, n_G0); S_range = np.linspace(0.0, 1.0, n_S)
    results = {}
    for bmi in bmi_values:
        landscape = np.full((n_S, n_G0), np.nan); classes = np.empty((n_S, n_G0), dtype='U15')
        for i, S in enumerate(S_range):
            for j, G0 in enumerate(G0_range):
                fake_patient = {'S_input': S, 'p21001_i0': bmi, 'p21022': age_med,
                                'p31': sex_med, 'p30710_i0': crp_med, 'G_mgdl': G0}
                _, g_traj = predict_trajectory_90d(model, fake_patient, n_timepoints=60)
                landscape[i, j] = g_traj[-1]; classes[i, j] = classify_trajectory_90d(g_traj)
        results[bmi] = {'landscape': landscape, 'classes': classes, 'G0_range': G0_range, 'S_range': S_range}
        print(f'  BMI={bmi}: done')
    return results

sweep_B = trajectory_stability_sweep_90d(model_90d)

# A vs B agreement (divergence = timescale separation, not measurement error)
res_a = sweep_A[27]; res_b = sweep_B[27]
b_mapped = np.where(res_b['classes'].flatten() == 'oscillatory', 'progressive', res_b['classes'].flatten())
agree = np.mean(res_a['classes'].flatten() == b_mapped)
print(f'Agreement A vs B (BMI=27): {100*agree:.1f}%')

# ── IFG cohort ODE run
# IFG expected to drift downward - EG0_FLOOR doesn't activate below 126
# the model captures fast dynamics but not the slow structural decay that would eventually push them into the irreversible diabetic level
df_ifg = pd.read_csv("Training Data/ukb_full_ifg.csv")
ifg_results = []
for idx, patient in df_ifg.iterrows():
    y0 = [1,1,1,1,1,1,0, patient['G_mgdl'], 9,400,0.8,900,536,0.73,77,18.7,2.8,15.9]
    p_sim = p_base.copy(); p_sim[15] = 18.7 + patient['S_input'] * 6.3
    ratio = patient['G_mgdl'] / G_REF
    eg0_i = eg0_ref * (1.0 / ratio) ** ALPHA_CAL
    if patient['G_mgdl'] > 126: eg0_i = min(eg0_i, EG0_FLOOR * eg0_ref)
    p_sim[22] = eg0_i; p_sim[63] = p_base[63] * (ratio ** ALPHA_HGP)
    try:
        sol = sivp(odde, [0, 7200], y0, method='LSODA', args=[p_sim])
        g_start, g_end = sol.y[7][0], sol.y[7][-1]
        status = 'Crash' if g_end > (g_start + 5) else 'Stable'
        ifg_results.append({'eid': patient['eid'], 'Start': g_start, 'End': g_end, 'Delta': g_end - g_start, 'Status': status})
    except:
        continue
res_df_ifg = pd.DataFrame(ifg_results)
print(f"IFG ODE complete: {len(res_df_ifg)} patients, mean delta G: {res_df_ifg['Delta'].mean():.2f} mg/dL")

# ── HbA1c validation (independent check)
from scipy import stats

def get_hba1c_matched(df_raw, res_df_sim):
    merged = res_df_sim.merge(
        df_raw[['eid', 'p30750_i0']].rename(columns={'p30750_i0': 'hba1c'}),
        on='eid', how='left'
    ).dropna(subset=['hba1c'])
    return merged

diab_matched  = get_hba1c_matched(df,     res_df)
ifg_matched   = get_hba1c_matched(df_ifg, res_df_ifg)
hlthy_matched = get_hba1c_matched(df_h,   res_df_healthy)
all_matched   = pd.concat([diab_matched, ifg_matched, hlthy_matched], ignore_index=True)

r_all, p_all = stats.pearsonr(all_matched['End'], all_matched['hba1c'])
r_d,   p_d   = stats.pearsonr(diab_matched['End'],  diab_matched['hba1c'])
r_i,   p_i   = stats.pearsonr(ifg_matched['End'],   ifg_matched['hba1c'])
r_h,   p_h   = stats.pearsonr(hlthy_matched['End'], hlthy_matched['hba1c'])

print("Pearson r: simulated G_end vs observed HbA1c:")
print(f"  All cohorts pooled:  r = {r_all:.3f}, p = {p_all:.4f}  (n={len(all_matched)})")
print(f"  Diabetic only:       r = {r_d:.3f},  p = {p_d:.4f}  (n={len(diab_matched)})")
print(f"  IFG only:            r = {r_i:.3f},  p = {p_i:.4f}  (n={len(ifg_matched)})")
print(f"  Healthy only:        r = {r_h:.3f},  p = {p_h:.4f}  (n={len(hlthy_matched)})")

# ── same concordance in glucose units, not just r
# r shows the trend but not how many mg/dL i'm actually off by. regressing my sim g_end on
# observed hba1c and taking the residual rmse gives the scatter around the fit line in mg/dL,
# which is the number a clinician reads. (same fit as the fig 2c concordance line.)
m_fit, b_fit, *_ = stats.linregress(all_matched['hba1c'], all_matched['End'])
resid = all_matched['End'] - (m_fit * all_matched['hba1c'] + b_fit)
print(f"  concordance scatter: residual RMSE = {(resid**2).mean()**0.5:.1f}, MAE = {resid.abs().mean():.1f} mg/dL  (n={len(all_matched)})")

# ── G0 vs HbA1c: independence check
# G_end is ~a function of G0 (eg0 and hgp scale off it, stress barely matters), so r(G_end, hba1c)
# can just inherit r(G0, hba1c). print the G0 baseline next to it - if G_end doesn't beat it, the
# "independent" wording is overstated and i soften it in the text.
print()
print("r(G0, HbA1c) baseline vs r(G_end, HbA1c):")
for _name, _d in [('pooled', all_matched), ('diabetic', diab_matched), ('IFG', ifg_matched), ('healthy', hlthy_matched)]:
    _rg, _ = stats.pearsonr(_d['Start'], _d['hba1c'])
    _re, _ = stats.pearsonr(_d['End'], _d['hba1c'])
    print(f"  {_name:9s} r(G0)={_rg:+.3f}  r(G_end)={_re:+.3f}  gain={_re - _rg:+.3f}  (n={len(_d)})")

# ── longitudinal validation
df_long = df.copy()
df_long['date_i0'] = pd.to_datetime(df_long['p53_i0'], errors='coerce')
df_long['date_i1'] = pd.to_datetime(df_long['p53_i1'], errors='coerce')
has_followup = df_long['date_i0'].notna() & df_long['date_i1'].notna() & df_long['p30740_i1'].notna()
df_followup = df_long[has_followup].copy()
df_followup['years_gap'] = (df_followup['date_i1'] - df_followup['date_i0']).dt.days / 365.25
print(f"Patients with longitudinal glucose: {len(df_followup)}")

longval_results = []
for _, patient in df_followup.iterrows():
    t_end_min = patient['years_gap'] * 365.25 * 24 * 60
    p_sim = p_base.copy(); p_sim[15] = 18.7 + patient['S_input'] * 6.3
    ratio = patient['G_mgdl'] / G_REF
    eg0_i = eg0_ref * (1.0 / ratio) ** ALPHA_CAL
    if patient['G_mgdl'] > 126: eg0_i = min(eg0_i, EG0_FLOOR * eg0_ref)
    p_sim[22] = eg0_i; p_sim[63] = p_base[63] * (ratio ** ALPHA_HGP)
    y0 = [1,1,1,1,1,1,0, patient['G_mgdl'], 9,400,0.8,900,536,0.73,77,18.7,2.8,15.9]
    try:
        sol = sivp(odde, [0, t_end_min], y0, method='LSODA', args=[p_sim])
        g_predicted = sol.y[7][-1]; g_observed = patient['G_i1_mgdl']
        in_training = patient['eid'] in pd.concat([df_train, df_h_train, df_ifg_train])['eid'].values
        longval_results.append({'eid': patient['eid'], 'G_i0': patient['G_mgdl'],
                                 'G_predicted': g_predicted, 'G_observed': g_observed,
                                 'years_gap': patient['years_gap'], 'in_training': in_training,
                                 'error': abs(g_predicted - g_observed)})
    except:
        continue

df_lv = pd.DataFrame(longval_results)
if len(df_lv) > 0:
    blind = df_lv[~df_lv['in_training']]
    print(f"Total longitudinal predictions: {len(df_lv)}, blind: {len(blind)}")
    print(f"All - MAE: {df_lv['error'].mean():.1f} mg/dL")
    # high MAE is expected - natural history vs managed patients
    if len(blind) > 0:
        r, p = stats.pearsonr(blind['G_predicted'], blind['G_observed'])
        print(f"Blind - MAE: {blind['error'].mean():.1f} mg/dL, r={r:.3f}, p={p:.4f}, n={len(blind)}")

# ── medication codes in longitudinal patients
if len(df_lv) > 0:
    long_eids = df_lv['eid'].tolist()
    df_full = pd.read_csv('Training Data/ukb_full_diabetics.csv')
    df_long_patients = df_full[df_full['eid'].isin(long_eids)].copy()
    med_cols_i0 = [f'p20003_i0_a{a}' for a in range(8)]
    med_cols_i1 = [f'p20003_i1_a{a}' for a in range(8)]
    all_med_cols = [c for c in med_cols_i0 + med_cols_i1 if c in df_long_patients.columns]
    codes = (df_long_patients[all_med_cols].stack().dropna().astype(int).value_counts())
    fill_values = {0, 99999}

    # must match treated_codes_by_class in t2d_preprocess.py exactly
    excluded_codes = {
        1140884600: 'metformin', 1140874686: 'glucophage (metformin brand)', 1140874690: 'orabet (metformin brand)',
        1140874744: 'gliclazide', 1140874746: 'diamicron (gliclazide brand)',
        1140874718: 'glibenclamide', 1140874724: 'daonil (glibenclamide brand)',
        1140874726: 'semi-daonil', 1140874728: 'euglucon', 1140874732: 'malix',
        1140874736: 'diabetamide', 1140874740: 'calabren',
        1140874646: 'glipizide', 1140874650: 'glibenese (glipizide brand)', 1140874652: 'minodiab (glipizide brand)',
        1140874658: 'gliquidone', 1140874660: 'glurenorm (gliquidone brand)',
        1140874664: 'tolazamide', 1140874666: 'tolanase (tolazamide brand)',
        1140874674: 'tolbutamide', 1140874678: 'glyconon (tolbutamide brand)',
        1140874680: 'rastinon (tolbutamide brand)', 1140857506: 'pramidex (tolbutamide brand)',
        1140874706: 'chlorpropamide', 1140874712: 'diabinese (chlorpropamide brand)',
        1140874716: 'glymese (chlorpropamide brand)',
        1140857584: 'acetohexamide', 1140857586: 'dimelor (acetohexamide brand)',
        1140857494: 'glibornuride', 1140857496: 'glutril (glibornuride brand)',
        1140857500: 'glymidine', 1140857502: 'gondafon (glymidine brand)',
        1141152590: 'glimepiride', 1141156846: 'amaryl (glimepiride brand)',
        1141171646: 'pioglitazone', 1141171652: 'actos (pioglitazone brand)',
        1141177600: 'rosiglitazone', 1141177606: 'avandia (rosiglitazone brand)',
        1141189090: 'rosiglitazone/metformin combo', 1141189094: 'avandamet combo',
        1141168660: 'repaglinide', 1141168668: 'novonorm (repaglinide brand)',
        1141173882: 'nateglinide', 1141173786: 'starlix (nateglinide brand)',
        1140868902: 'acarbose', 1140868908: 'glucobay (acarbose brand)',
        1140883066: 'insulin product',
    }

    codes = codes[~codes.index.isin(fill_values)]
    print(f"Medication codes in the {len(df_long_patients)} longitudinal patients:")
    for code, count in codes.items():
        label = excluded_codes.get(code, 'other')
        flag = '  <-- excluded by filter' if code in excluded_codes else '  (non-T2D medication)'
        print(f"  code {code}: n={count}  [{label}]{flag}")
    leaked = [c for c in codes.index if c not in excluded_codes and c not in fill_values]
    print(f"Non-T2D codes present: {len(leaked)}")

# ── Fig 3: surrogate validation (1x3: MAE, ECDF, HbA1c)
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.patches as mpatches
from scipy import stats
os.makedirs('output', exist_ok=True)

# ── build merged Figure 2: surrogate validation + parameter response (5 rows x 3 cols)
# this single matplotlib figure replaces 3 separate stacked SVGs (surrogate_validation,
# ode_confirm, parameter_response). it stays alive across the script - row 0 filled here,
# rows 2-4 at line ~2120, row 1 at line ~2541, single savefig at the very end of v8.
# layout='constrained' lets matplotlib handle spacing automatically so panel labels and
# axis titles don't clip even with 15 panels packed in.
# fixed gridspec instead of layout='constrained'. constrained let each row's left edge
# drift by a few points with tick-label width, so the first-column letters (A,D,E,H,K) never
# truly lined up and the titled rows (D, K-M) got extra vertical gap. a manual grid pins every
# column-0 axis to the same left edge and gives every row the same height, so the letters align
# and the spacing is even. left margin is sized for the long row-3 ylabel; savefig bbox='tight'
# still captures the right-side legends.
_FIG2 = plt.figure(figsize=(COL_WIDTH, 11.0))
_GS2 = _FIG2.add_gridspec(5, 3, left=0.105, right=0.985, top=0.965, bottom=0.038,
                          hspace=0.52, wspace=0.34)
_AXES2 = np.array([[_FIG2.add_subplot(_GS2[r, c]) for c in range(3)] for r in range(5)])
axes = _AXES2[0]
ax1, ax2, ax3 = axes

# one place for every panel letter so they stop drifting. i anchor each letter a fixed
# number of POINTS from its axes' top-left corner instead of a fraction of the axes box.
# fixed points means first-column panels (which share a left edge) line up on the same x
# no matter how wide their y-tick labels are, and same-row panels share a y. this is what
# replaced the old hand-typed -0.14/1.10, -0.10/1.08, -0.10/1.22 offsets that never matched.
def _panel_label(ax, letter):
    # offset in POINTS from each axes' top-left corner. dy lifts the letter clear of any
    # title on that axes; bump _PANEL_DY if a title still touches a letter.
    ax.annotate(letter, xy=(0.0, 1.0), xycoords='axes fraction',
                xytext=(-24, 13), textcoords='offset points',
                fontsize=11, fontweight='bold', va='bottom', ha='left',
                annotation_clip=False)

ax1.plot(t_days, mae_tanh, color=THESIS_PALETTE['slate'], lw=1.5, label='DeepONet vs ODE')
ax1.fill_between(t_days, np.maximum(mae_tanh - std_tanh, 0), mae_tanh + std_tanh, color=THESIS_PALETTE['slate'], alpha=0.18, label=r'$\pm 1\sigma$')
ax1.axhline(10.0, color=THESIS_PALETTE['aubergine'], ls='--', lw=0.8, alpha=0.7, label='10 mg/dL threshold')
ax1.set_ylim(0, max(1.2, (mae_tanh + std_tanh).max() * 1.35))
ax1.set_xlabel('Time (days)'); ax1.set_ylabel('MAE (mg/dL)')
ax1.legend(loc='upper left'); ax1.grid(alpha=0.18)
_panel_label(ax1, 'A')

sim_vals  = pd.concat([res_df['End'], res_df_healthy['End'], res_df_ifg['End']]).dropna()
sim_vals  = sim_vals[sim_vals > 0].values
clin_vals = pd.concat([df['G_mgdl'], df_h['G_mgdl'], df_ifg['G_mgdl']]).dropna().values

def ecdf(x):
    xs = np.sort(x)
    return xs, np.arange(1, len(xs)+1) / len(xs)

xc, yc = ecdf(clin_vals); xs, ys = ecdf(sim_vals)
ax2.fill_between(xc, 0, yc, color=THESIS_PALETTE['grey'], alpha=0.20, label='Clinical (UKB)')
ax2.step(xc, yc, color=THESIS_PALETTE['grey'],  lw=1.2, where='post')
ax2.step(xs, ys, color=THESIS_PALETTE['slate'], lw=1.5, where='post', label='Simulated (ODE)')
ax2.axvline(100, color=THESIS_PALETTE['ifg'],       ls='--', lw=0.8, alpha=0.85, label='IFG threshold')
ax2.axvline(126, color=THESIS_PALETTE['aubergine'], ls='--', lw=0.8, alpha=0.85, label='T2D threshold')
ax2.set_xlabel('Glucose (mg/dL)'); ax2.set_ylabel('Cumulative probability')
ax2.legend(loc='lower right'); ax2.grid(alpha=0.18); ax2.set_xlim(left=50)
_panel_label(ax2, 'B')

for mdf, col, label in [(hlthy_matched, THESIS_PALETTE['healthy'], 'Healthy'),
                         (ifg_matched,   THESIS_PALETTE['ifg'],     'IFG'),
                         (diab_matched,  THESIS_PALETTE['t2d'],     'Diabetic')]:
    if len(mdf) == 0: continue
    ax3.scatter(mdf['hba1c'], mdf['End'], color=col, alpha=0.4, s=14, edgecolors='none', label=label)
all_m = pd.concat([hlthy_matched, ifg_matched, diab_matched]).dropna(subset=['hba1c','End'])
m, b, *_ = stats.linregress(all_m['hba1c'], all_m['End'])
xr = np.linspace(all_m['hba1c'].min(), all_m['hba1c'].max(), 100)
ax3.plot(xr, m*xr + b, color=THESIS_PALETTE['aubergine'], lw=0.8, ls='--', alpha=0.8)
legend_c = [mpatches.Patch(color=THESIS_PALETTE['healthy'], label='Healthy  r=' + str(round(r_h, 3))),
            mpatches.Patch(color=THESIS_PALETTE['ifg'],     label='IFG      r=' + str(round(r_i, 3))),
            mpatches.Patch(color=THESIS_PALETTE['t2d'],     label='Diabetic r=' + str(round(r_d, 3)))]
ax3.legend(handles=legend_c, loc='upper left')
ax3.text(0.97, 0.04, 'pooled r = ' + str(round(r_all, 3)), transform=ax3.transAxes,
         ha='right', va='bottom', fontsize=10, fontstyle='italic')
ax3.set_xlabel('Observed HbA1c (mmol/mol)')
ax3.set_ylabel(r'Simulated $G_{\mathrm{end}}$ (mg/dL)')
ax3.grid(alpha=0.18)
_panel_label(ax3, 'C')
# row 0 of _FIG2 done. row 1 (ode_confirm) and rows 2-4 (parameter response) come later.
# don't tight_layout / savefig / close here - we want _FIG2 alive until everything's drawn.

# ── Fig 4: parameter response (3x3)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp as sivp

diab_med_idx = (df['G_mgdl'] - df['G_mgdl'].median()).abs().idxmin()
patient_diab = df.loc[diab_med_idx]

# healthy representative: same median selection as diabetic
hlth_med_idx = (df_h['G_mgdl'] - df_h['G_mgdl'].median()).abs().idxmin()
patient_hlth = df_h.loc[hlth_med_idx]

SWEEP = {'BMI': (np.linspace(20, 30, 15), 'BMI (kg/m$^2$)'),
          'S_input': (np.linspace(0.0, 0.9, 15), 'Stress score $S$'),
          'SI_target': (np.linspace(0.8, 3.5, 15), '$S_{I,\\mathrm{target}}$')}
HEIGHT = 1.66; T_PHASE = 7200

def run_with_param(patient, param_name, param_value):
    p_sim = p_base.copy(); p_sim[15] = 18.7 + patient['S_input'] * 6.3
    ratio = patient['G_mgdl'] / G_REF
    eg0_i = eg0_ref * (1.0 / ratio) ** ALPHA_CAL
    if patient['G_mgdl'] > 126: eg0_i = min(eg0_i, EG0_FLOOR * eg0_ref)
    p_sim[22] = eg0_i; p_sim[63] = p_base[63] * (ratio ** ALPHA_HGP)
    bmi = patient['p21001_i0']; s = patient['S_input']
    if param_name == 'BMI': bmi = param_value
    elif param_name == 'S_input': s = param_value; p_sim[15] = 18.7 + s * 6.3
    elif param_name == 'SI_target': p_sim[43] = param_value
    p_sim[45] = bmi; weight = bmi * HEIGHT**2
    y0 = [1,1,1,1,1,1,0, patient['G_mgdl'], 9,400,0.8,900,536,0.73, weight, 18.7,2.8,15.9]
    try:
        sol = sivp(odde, [0, T_PHASE], y0, method='LSODA', args=[p_sim])
        return sol.y[7][-1] if sol.success else np.nan
    except: return np.nan

all_patients_4 = pd.concat([df.assign(_cohort='Diabetic'), df_h.assign(_cohort='Healthy'),
                              df_ifg.assign(_cohort='IFG')], ignore_index=True)
all_patients_4['p30710_i0'] = all_patients_4['p30710_i0'].fillna(all_patients_4['p30710_i0'].median())

def predict_with_param_override(patient, param_name, param_value, model=best_model):
    G0 = patient['G_mgdl']
    branch_feat = np.array([
        patient['S_input'] if param_name != 'S_input' else param_value,
        (patient['p21001_i0'] if param_name != 'BMI' else param_value) / 40.0,
        patient['p21022'] / 80.0, float(patient['p31']), patient['p30710_i0'] / 10.0,
    ], dtype=np.float32)
    if param_name == 'SI_target': return None
    branch_in = np.tile(branch_feat, (1, 1)); trunk_in = np.array([[1.0, G0/200.0]], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        raw = model(torch.tensor(branch_in), torch.tensor(trunk_in)).numpy().flatten()[0]
    return G0 + np.tanh(TANH_K * 1.0) * raw * 200.0

def ode_endpoint_with_param(patient, param_name, param_value):
    p_sim = p_base.copy(); p_sim[15] = 18.7 + (patient['S_input'] if param_name != 'S_input' else param_value) * 6.3
    ratio = patient['G_mgdl'] / G_REF
    eg0_i = eg0_ref * (1.0 / ratio) ** ALPHA_CAL
    if patient['G_mgdl'] > 126: eg0_i = min(eg0_i, EG0_FLOOR * eg0_ref)
    p_sim[22] = eg0_i; p_sim[63] = p_base[63] * (ratio ** ALPHA_HGP)
    bmi = patient['p21001_i0'] if param_name != 'BMI' else param_value
    p_sim[45] = bmi
    if param_name == 'SI_target': p_sim[43] = param_value
    weight = bmi * 1.66**2
    y0 = [1,1,1,1,1,1,0, patient['G_mgdl'], 9,400,0.8,900,536,0.73, weight, 18.7,2.8,15.9]
    try:
        sol = sivp(odde, [0, 7200], y0, method='LSODA', args=[p_sim])
        return sol.y[7][-1] if sol.success else np.nan
    except: return np.nan

SWEEP_POP = {'BMI': np.linspace(20, 30, 12), 'S_input': np.linspace(0.0, 0.9, 12), 'SI_target': np.linspace(0.8, 3.5, 12)}

# the SI_target path calls ode_endpoint_with_param for every patient - cache it
# without cache: 12 values x 1,798 patients x LSODA = hour+ runtime
_rpop_cache = f'cache/results_pop_{_ode_param_hash({"n_pop": 12})}.npz'
if os.path.exists(_rpop_cache):
    print(f'loading cached population sweep from {_rpop_cache}')
    _d = np.load(_rpop_cache, allow_pickle=True)
    results_pop = _d['results_pop'].item()
else:
    results_pop = {}
    for param, vals in SWEEP_POP.items():
        pct_diabetic, pct_ifg, pct_healthy = [], [], []
        for v in vals:
            gs = []
            for _, p in all_patients_4.iterrows():
                g = ode_endpoint_with_param(p, param, v) if param == 'SI_target' else predict_with_param_override(p, param, v)
                gs.append(g)
            gs = np.array(gs); gs = gs[~np.isnan(gs)]; n = len(gs)
            pct_diabetic.append(100 * (gs >= 126).sum() / n)
            pct_ifg.append(100 * ((gs >= 100) & (gs < 126)).sum() / n)
            pct_healthy.append(100 * (gs < 100).sum() / n)
        results_pop[param] = {'vals': vals, 'pct_diabetic': pct_diabetic, 'pct_ifg': pct_ifg, 'pct_healthy': pct_healthy}
        print(f'  pop sweep {param}: done')
    np.savez(_rpop_cache, results_pop=np.array(results_pop, dtype=object))
    print(f'saved population sweep to {_rpop_cache}')

xlabels_pop = {'BMI': 'BMI (kg/m$^2$)', 'S_input': 'Stress score $S$', 'SI_target': '$S_{I,\\mathrm{target}}$'}

# stability landscape - custom diverging colormap from palette
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
_cmap_severity = LinearSegmentedColormap.from_list(
    'severity', [THESIS_PALETTE['healthy'], THESIS_PALETTE['grid'], THESIS_PALETTE['t2d']], N=3)
_cat_map_4    = {'recovering': 0, 'stable': 1, 'progressive': 2}
_norm_4       = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)
_bmi_labels_4 = {22: 'BMI 22 (healthy weight)', 27: 'BMI 27 (overweight)', 32: 'BMI 32 (obese)'}

axes = _AXES2[2:]   # rows 2,3,4 of the merged Fig 2 (parameter-response family)

# ── row A-C: single patient parameter response (T2D + healthy)
for col_idx, (param, (vals, xlabel)) in enumerate(SWEEP.items()):
    ax = axes[0, col_idx]
    g_ends_d = [run_with_param(patient_diab, param, v) for v in vals]
    g_ends_h = [run_with_param(patient_hlth, param, v) for v in vals]
    ax.plot(vals, g_ends_d, '-o', color=THESIS_PALETTE['t2d'],     ms=3, lw=1.5, label=f'T2D  $G_0$={patient_diab["G_mgdl"]:.0f}')
    ax.plot(vals, g_ends_h, '-s', color=THESIS_PALETTE['healthy'], ms=3, lw=1.5, label=f'Healthy  $G_0$={patient_hlth["G_mgdl"]:.0f}')
    ax.axhline(126, color=THESIS_PALETTE['aubergine'], ls='--', lw=0.8, alpha=0.6, label='T2D threshold')
    ax.axhline(100, color=THESIS_PALETTE['aubergine'], ls=':',  lw=0.8, alpha=0.6, label='IFG threshold')
    ax.set_xlabel(xlabel)
    if col_idx == 0: ax.set_ylabel('Representative patient\n$G_{\\mathrm{end}}$ (mg/dL)')
    if col_idx == 2: ax.legend(fontsize=7.5, loc='best')
    ax.grid(alpha=0.2)

# ── row D-F: population fraction sweep
for col_idx, param in enumerate(SWEEP_POP.keys()):
    ax = axes[1, col_idx]; res = results_pop[param]
    ax.stackplot(res['vals'], res['pct_healthy'], res['pct_ifg'], res['pct_diabetic'],
                 labels=['Healthy (<100)', 'IFG (100-126)', 'Diabetic ($\\geq$126)'],
                 colors=[THESIS_PALETTE['healthy'], THESIS_PALETTE['ifg'], THESIS_PALETTE['t2d']], alpha=0.85)
    ax.set_xlabel(xlabels_pop[param])
    if col_idx == 0: ax.set_ylabel('Cohort fraction (%)')
    if col_idx == 2: ax.legend(loc='lower right')
    ax.set_ylim(0, 100); ax.set_xlim(res['vals'][0], res['vals'][-1]); ax.grid(alpha=0.2)

# row G-I: stability landscape (approach A, DeepONet)
# uses multi-seed cache when available (alpha = agreement across seeds), single-seed otherwise
_multiseed_path = 'cache/multiseed_sweep.npz'
_multiseed_km = np.load(_multiseed_path) if os.path.exists(_multiseed_path) else None
if _multiseed_km is not None:
    print(f'K-M panels: multi-seed view ({len(_multiseed_km["seeds"])} seeds, alpha = agreement)')

from matplotlib.colors import to_rgba as _to_rgba

for col_idx, bmi in enumerate([22, 27, 32]):
    ax = axes[2, col_idx]
    if _multiseed_km is not None:
        modal = _multiseed_km[f'modal_{bmi}']
        agree = _multiseed_km[f'agreement_{bmi}']
        G0r   = _multiseed_km[f'G0_{bmi}']
        Sr    = _multiseed_km[f'S_{bmi}']
        img = np.zeros((modal.shape[0], modal.shape[1], 4))
        for _cls, _col in (('recovering',  THESIS_PALETTE['healthy']),
                           ('stable',      THESIS_PALETTE['grid']),
                           ('progressive', THESIS_PALETTE['t2d'])):
            img[modal == _cls] = np.array(_to_rgba(_col))
        img[..., 3] = agree
        ax.imshow(img, extent=[G0r[0], G0r[-1], Sr[0], Sr[-1]],
                  origin='lower', aspect='auto', interpolation='nearest')
    else:
        res = sweep_A[bmi]
        numeric = np.vectorize(lambda c: _cat_map_4.get(c, -1))(res['classes'])
        ax.pcolormesh(res['G0_range'], res['S_range'], numeric, cmap=_cmap_severity, norm=_norm_4, shading='nearest')
    ax.axvline(100, color=THESIS_PALETTE['ifg'],       ls='--', lw=0.8, alpha=0.8)
    ax.axvline(126, color=THESIS_PALETTE['aubergine'],  ls='--', lw=0.8, alpha=0.8)
    ax.set_xlabel('Baseline glucose $G_0$ (mg/dL)')
    if col_idx == 0:
        ax.set_ylabel('Stress score $S$')
    if col_idx == 2:
        import matplotlib.patches as _mp4
        ax.legend(handles=[_mp4.Patch(color=THESIS_PALETTE['healthy'], label='Recovering'),
                            _mp4.Patch(color=THESIS_PALETTE['grid'],   label='Stable'),
                            _mp4.Patch(color=THESIS_PALETTE['t2d'],    label='Progressive')],
                  fontsize=7.5, loc='lower right')
    ax.set_title(_bmi_labels_4[bmi])

# panel letters - continuous from D (ODE confirm in row 1). E,F,G per-patient, H,I,J cohort, K,L,M landscape.
# panel letters via the shared helper so E-M line up with A-D (was: hand-tuned 1.08/1.22
# offsets to dodge the stacked-area edges and the BMI subtitles; the fixed-points helper
# clears both automatically).
for col_idx, letter in enumerate(['E','F','G']):
    _panel_label(axes[0, col_idx], letter)
for col_idx, letter in enumerate(['H','I','J']):
    _panel_label(axes[1, col_idx], letter)
for col_idx, letter in enumerate(['K','L','M']):
    _panel_label(axes[2, col_idx], letter)

# rows 2-4 of _FIG2 done. row 1 (ode_confirm) follows at line ~2541. final savefig at end of v8.

# ── statistical tests

import os, numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
os.makedirs('cache', exist_ok=True)

# ── 1. McNemar: 5-day (A) vs 90-day (B) classification agreement
# tests whether systematic disagreement exists beyond chance
# oscillatory folded into progressive to get a shared 3-class label space
bmi_show = 27 if 27 in sweep_A else list(sweep_A.keys())[0]
a_cls = sweep_A[bmi_show]['classes'].flatten()
b_cls_raw = sweep_B[bmi_show]['classes'].flatten()
b_cls = np.where(b_cls_raw == 'oscillatory', 'progressive', b_cls_raw)

# build 2x2 table: "agrees" vs "disagrees" per direction
# McNemar is strictly 2x2 so we use (A=progressive, B=not) vs (A=not, B=progressive)
# this captures the dominant asymmetry direction
a_prog = (a_cls == 'progressive'); b_prog = (b_cls == 'progressive')
n_agree     = np.sum(a_prog == b_prog)
n_a_not_b   = np.sum(a_prog & ~b_prog)   # A=progressive, B=not
n_b_not_a   = np.sum(~a_prog & b_prog)   # B=progressive, A=not
n_both_not  = np.sum(~a_prog & ~b_prog)
table_mc = np.array([[np.sum(a_prog & b_prog), n_a_not_b],
                     [n_b_not_a,                n_both_not]])
result_mc = mcnemar(table_mc, exact=(n_a_not_b + n_b_not_a < 25))
agree_pct = 100 * n_agree / len(a_cls)
print(f'\nMcNemar test: binary progressive/not  (BMI={bmi_show}, distinct from 3-class agreement above)')
print(f'  binary agreement: {agree_pct:.1f}%')
print(f'  A=prog, B=not: {n_a_not_b}   |   B=prog, A=not: {n_b_not_a}')
print(f'  McNemar statistic={result_mc.statistic:.3f}, p={result_mc.pvalue:.4f}')
print(f'  dominant direction: {"A classifies more as progressive (fast drift > slow settle)" if n_a_not_b > n_b_not_a else "B classifies more as progressive"}')

# ── 2. Fisher z: pairwise HbA1c subgroup r comparisons
# r and n already computed upstream in the HbA1c validation
n_d = len(diab_matched.dropna(subset=['hba1c','End']))
n_h = len(hlthy_matched.dropna(subset=['hba1c','End']))
n_i = len(ifg_matched.dropna(subset=['hba1c','End']))

def fisher_z_compare(r1, n1, r2, n2):
    z1 = np.arctanh(np.clip(r1, -0.9999, 0.9999))
    z2 = np.arctanh(np.clip(r2, -0.9999, 0.9999))
    se = np.sqrt(1/(n1 - 3) + 1/(n2 - 3))
    z_stat = (z1 - z2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return z_stat, p

alpha_bonf = 0.05 / 3
pairs = [('diabetic vs healthy', r_d, n_d, r_h, n_h),
         ('diabetic vs IFG',     r_d, n_d, r_i, n_i),
         ('healthy vs IFG',      r_h, n_h, r_i, n_i)]
print(f'\nFisher z: HbA1c subgroup r comparisons (Bonferroni alpha={alpha_bonf:.4f})')
fisher_results = {}
for label, r1, n1, r2, n2 in pairs:
    z, p = fisher_z_compare(r1, n1, r2, n2)
    sig = '**' if p < alpha_bonf else 'ns'
    print(f'  {label:25s}: z={z:+.3f}  p={p:.4f}  {sig}')
    fisher_results[label] = {'z': z, 'p': p}

# save all numerical results to cache so tex can cite them without rerunning
import json
_test_results = {
    'mcnemar': {'agree_pct': round(agree_pct, 1), 'statistic': round(float(result_mc.statistic), 3),
                'p': round(float(result_mc.pvalue), 4), 'n_a_not_b': int(n_a_not_b), 'n_b_not_a': int(n_b_not_a)},
    'fisher_z': {k: {'z': round(v['z'], 3), 'p': round(v['p'], 4)} for k, v in fisher_results.items()},
}
with open('cache/hypothesis_test_results.json', 'w') as _f:
    json.dump(_test_results, _f, indent=2)
print('\nSaved: cache/hypothesis_test_results.json')

# ── ODE spot checks on stability landscape
import numpy as np, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp as sivp
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import matplotlib.patches as _mpa

# reuse the sweep_A results and the same grid points as the landscape
_bmi_confirm = [22, 27, 32]

# pick 5 evenly-spaced grid points in G0 (fixed S=0.0 for clarity)
def _ode_endpoint_confirm(G0, bmi, S=0.0):
    p_sim = p_base.copy()
    p_sim[15] = 18.7 + S * 6.3
    ratio = G0 / G_REF
    eg0_i = eg0_ref * (1.0 / ratio) ** ALPHA_CAL
    if G0 > 126: eg0_i = min(eg0_i, EG0_FLOOR * eg0_ref)
    p_sim[22] = eg0_i; p_sim[63] = p_base[63] * (ratio ** ALPHA_HGP)
    p_sim[45] = bmi
    y0 = [1,1,1,1,1,1,0, G0, 9,400,0.8,900,536,0.73, bmi*1.66**2, 18.7,2.8,15.9]
    try:
        sol = sivp(odde, [0, 7200], y0, method='LSODA', args=[p_sim])
        return sol.y[7][-1] if sol.success else np.nan
    except:
        return np.nan

_cat_map_c  = {'recovering': 0, 'stable': 1, 'progressive': 2}
_cmap_sev_c = LinearSegmentedColormap.from_list(
    'sev', [THESIS_PALETTE['healthy'], THESIS_PALETTE['grid'], THESIS_PALETTE['t2d']], N=3)
_norm_c = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)

axes = _AXES2[1]   # row 1 of merged Fig 2 (the ODE-vs-surrogate confirmation D panel)
total_pts = 0; concordant = 0

for ax, bmi in zip(axes, _bmi_confirm):
    res = sweep_A[bmi]
    numeric = np.vectorize(lambda c: _cat_map_c.get(c, -1))(res['classes'])
    ax.pcolormesh(res['G0_range'], res['S_range'], numeric,
                  cmap=_cmap_sev_c, norm=_norm_c, shading='nearest')
    ax.axvline(100, color=THESIS_PALETTE['ifg'],       ls='--', lw=0.8, alpha=0.8)
    ax.axvline(126, color=THESIS_PALETTE['aubergine'],  ls='--', lw=0.8, alpha=0.8)
    ax.set_xlabel('Baseline glucose $G_0$ (mg/dL)')
    if bmi == 22: ax.set_ylabel('Stress score $S$')
    ax.set_title(f'BMI {bmi}')

    # 5 confirmation points at S=0, evenly spaced across G0 range
    G0_pts = np.linspace(80, 320, 5)
    S_pt = 0.0
    # find closest S index in grid for S=0
    s_idx = int(np.argmin(np.abs(res['S_range'] - S_pt)))
    for G0_pt in G0_pts:
        g_idx = int(np.argmin(np.abs(res['G0_range'] - G0_pt)))
        surr_cls = res['classes'][s_idx, g_idx]
        g_end_ode = _ode_endpoint_confirm(G0_pt, bmi, S=S_pt)
        if np.isnan(g_end_ode):
            continue
        ode_cls = 'progressive' if (g_end_ode - G0_pt) > 5 else ('recovering' if (g_end_ode - G0_pt) < -5 else 'stable')
        agrees = (surr_cls == ode_cls)
        total_pts += 1
        if agrees: concordant += 1
        fc = THESIS_PALETTE['aubergine'] if agrees else 'white'
        ax.scatter([G0_pt], [S_pt + 0.04], s=35, marker='D', facecolors=fc,
                   edgecolors=THESIS_PALETTE['aubergine'], linewidths=0.8, zorder=8)

concord_rate = 100 * concordant / total_pts if total_pts > 0 else 0
print(f'ODE vs surrogate: {concordant}/{total_pts} agree ({concord_rate:.0f}%)')

axes[2].legend(handles=[
    _mpa.Patch(color=THESIS_PALETTE['healthy'], label='Recovering'),
    _mpa.Patch(color=THESIS_PALETTE['grid'],   label='Stable'),
    _mpa.Patch(color=THESIS_PALETTE['t2d'],    label='Progressive'),
    plt.Line2D([0],[0], marker='D', color=THESIS_PALETTE['aubergine'], lw=0, ms=5, label='ODE match'),
    plt.Line2D([0],[0], marker='D', color='white', markeredgecolor=THESIS_PALETTE['aubergine'],
               lw=0, ms=5, label='ODE mismatch'),
], fontsize=7.5, loc='upper right')

# panel D label on leftmost panel (the whole row is panel D in the paper caption)
_panel_label(axes[0], 'D')

# _FIG2 is now fully populated: row 0 (A B C validation), row 1 (D ODE-confirm at 3 BMIs),
# rows 2-4 (parameter response E-M continuing from D). single savefig for the entire merged Figure 2,
# saved as panel2.svg so it doesn't collide with the individual SVG files now redirected to runtime/.
plt.figure(_FIG2.number)
plt.savefig('output/panel2.svg', bbox_inches='tight')

# preserve the individual sub-figures as runtime artifacts using bbox cropping on _FIG2.
# this gives separate surrogate_validation.svg, ode_confirm.svg, parameter_response.svg
# in runtime/v8/ for debugging without double-rendering. constrained_layout requires a draw()
# call before tightbbox is accurate, so force render first.
import matplotlib.transforms as _mtrans_v8
os.makedirs('runtime/v8', exist_ok=True)
_FIG2.canvas.draw()
_renderer_v8 = _FIG2.canvas.get_renderer()

def _save_axes_subset(fig, axes_iter, filename, pad=0.1):
    """save a cropped view of `fig` containing only `axes_iter` by computing the
    union tightbbox (which includes axis labels, titles, and panel letters)."""
    bboxes = [ax.get_tightbbox(_renderer_v8) for ax in axes_iter]
    union  = _mtrans_v8.Bbox.union(bboxes).transformed(fig.dpi_scale_trans.inverted()).padded(pad)
    fig.savefig(filename, bbox_inches=union)

_save_axes_subset(_FIG2, _AXES2[0],            'runtime/v8/surrogate_validation.svg')   # row 0: A B C
_save_axes_subset(_FIG2, _AXES2[1],            'runtime/v8/ode_confirm.svg')            # row 1: D
_save_axes_subset(_FIG2, _AXES2[2:].flatten(), 'runtime/v8/parameter_response.svg')     # rows 2-4: E-M

plt.close(_FIG2)
print(f'saved output/panel2.svg  ({concord_rate:.0f}% ODE match in row 1)')
print('saved runtime/v8/{surrogate_validation,ode_confirm,parameter_response}.svg  (cropped views of panel2)')