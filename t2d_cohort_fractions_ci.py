import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob, os, sys

# small post-process that adds wilson 95% CIs to the cohort-fraction panels (H-J of fig 2).
# loads v8's existing cache (results_pop_*.npz), computes wilson bounds for each proportion
# at each parameter value, and renders an updated stackplot with thin error bands.
# no ODE runs - this is pure post-processing on the cached counts.
#
# wilson is preferred over normal-approximation CI because it stays in [0, 1] and
# behaves well at the extremes (0% or 100%), which we get when the cohort sits
# fully inside one category at the edges of the SI_target sweep.
#
# usage: run after v8 finishes once. produces runtime/ci/cohort_fractions.svg

THESIS_PALETTE = {
    'healthy': '#7FB069', 'ifg': '#E6AA68', 't2d': '#D8657D',
    'slate': '#5B8FA8', 'grey': '#7C7C7C', 'aubergine': '#3D2645',
    'text': '#2B2B2B', 'grid': '#EAEAEA', 'bg': '#FFFFFF',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Inter', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': THESIS_PALETTE['text'], 'axes.linewidth': 0.8,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'svg.fonttype': 'none',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

# wilson interval for a binomial proportion. p in [0,1], n is the sample size.
# returns (lo, hi) half-bounds. z=1.96 for 95%.
def wilson(p, n, z=1.96):
    if n == 0: return p, p
    denom = 1.0 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half   = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return max(centre - half, 0.0), min(centre + half, 1.0)

def find_cache():
    matches = sorted(glob.glob('cache/results_pop_*.npz'))
    if not matches:
        print('ERROR: no cache/results_pop_*.npz - run v8 first to build the cohort sweep cache.')
        sys.exit(1)
    return matches[-1]

def load_results_pop(path):
    d = np.load(path, allow_pickle=True)
    return d['results_pop'].item()

def add_wilson(results_pop, n_assumed=1798):
    # the existing cache stores percentages without the underlying counts.
    # since the population sweep iterates over all_patients_4 (n ~ 1,798 minus
    # rare NaN drops), we assume n=1798 per cell. small NaN rates make this
    # tight to within fractions of a percent for the CI half-widths.
    out = {}
    for param, res in results_pop.items():
        vals = np.asarray(res['vals'])
        pd_  = np.asarray(res['pct_diabetic']) / 100.0
        pi_  = np.asarray(res['pct_ifg'])      / 100.0
        ph_  = np.asarray(res['pct_healthy'])  / 100.0

        ci_d = np.array([wilson(p, n_assumed) for p in pd_])
        ci_i = np.array([wilson(p, n_assumed) for p in pi_])
        ci_h = np.array([wilson(p, n_assumed) for p in ph_])

        out[param] = {
            'vals': vals,
            'pct_healthy':  ph_ * 100, 'lo_healthy':  ci_h[:, 0] * 100, 'hi_healthy':  ci_h[:, 1] * 100,
            'pct_ifg':      pi_ * 100, 'lo_ifg':      ci_i[:, 0] * 100, 'hi_ifg':      ci_i[:, 1] * 100,
            'pct_diabetic': pd_ * 100, 'lo_diabetic': ci_d[:, 0] * 100, 'hi_diabetic': ci_d[:, 1] * 100,
        }
    return out

def render(results_with_ci, save_prefix='runtime/ci/cohort_fractions'):
    # 1 x 3 panels, one per swept parameter. each panel: line + filled band per category.
    # this version drops the stackplot in favour of three lines + bands - lets CI bands actually breathe.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    xlabels = {'BMI': r'BMI (kg/m$^2$)', 'S_input': r'Stress score $S$',
               'SI_target': r'$S_{I,\mathrm{target}}$'}
    titles  = {'BMI': 'BMI sweep', 'S_input': 'Stress sweep', 'SI_target': 'Insulin-sensitivity sweep'}
    cats    = [('healthy', 'Healthy ($<$100)', THESIS_PALETTE['healthy']),
               ('ifg',     'IFG (100-126)',     THESIS_PALETTE['ifg']),
               ('diabetic', r'Diabetic ($\geq$126)', THESIS_PALETTE['t2d'])]

    for col_idx, param in enumerate(['BMI', 'S_input', 'SI_target']):
        if param not in results_with_ci:
            axes[col_idx].set_visible(False); continue
        ax  = axes[col_idx]
        res = results_with_ci[param]
        v   = res['vals']
        for key, label, col in cats:
            mid = res[f'pct_{key}']; lo = res[f'lo_{key}']; hi = res[f'hi_{key}']
            ax.fill_between(v, lo, hi, color=col, alpha=0.22, lw=0)
            ax.plot(v, mid, '-', color=col, lw=1.6, label=label)

        ax.set_xlabel(xlabels[param]); ax.set_title(titles[param])
        if col_idx == 0: ax.set_ylabel('Cohort fraction (%)')
        ax.set_ylim(0, 100); ax.set_xlim(v[0], v[-1])
        ax.grid(alpha=0.2)
        if col_idx == 2: ax.legend(fontsize=8, loc='center right')

    fig.text(0.5, -0.04, r'shaded bands: Wilson 95\% CI on the per-value proportions ($N \approx 1{,}798$).',
             ha='center', va='top', style='italic', color=THESIS_PALETTE['grey'])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
    plt.savefig(save_prefix + '.svg', bbox_inches='tight')
    plt.savefig(save_prefix + '.png', dpi=300, bbox_inches='tight')
    print(f'  saved {save_prefix}.png and .svg')
    plt.close(fig)


def report_widths(results_with_ci):
    # quick numeric sanity check - print the widest CI per category per parameter
    print('\nwidest 95% CI half-widths (percentage points):')
    for param, res in results_with_ci.items():
        for key in ('healthy', 'ifg', 'diabetic'):
            widths = np.asarray(res[f'hi_{key}']) - np.asarray(res[f'lo_{key}'])
            print(f'  {param:>10s} / {key:>9s}: max half-width = {widths.max()/2:.2f}%')


if __name__ == '__main__':
    cache = find_cache()
    print(f'loading {cache}')
    results_pop = load_results_pop(cache)

    print(f'computing wilson 95% CIs (assuming n=1,798)')
    with_ci = add_wilson(results_pop, n_assumed=1798)
    report_widths(with_ci)

    print('rendering figure')
    render(with_ci)

    # save the with-CI table as a CSV so the paper can cite specific values
    rows = []
    for param, res in with_ci.items():
        for k, v in enumerate(res['vals']):
            rows.append({
                'param': param, 'value': v,
                'pct_healthy':  res['pct_healthy'][k],  'lo_healthy':  res['lo_healthy'][k],  'hi_healthy':  res['hi_healthy'][k],
                'pct_ifg':      res['pct_ifg'][k],      'lo_ifg':      res['lo_ifg'][k],      'hi_ifg':      res['hi_ifg'][k],
                'pct_diabetic': res['pct_diabetic'][k], 'lo_diabetic': res['lo_diabetic'][k], 'hi_diabetic': res['hi_diabetic'][k],
            })
    import pandas as pd
    pd.DataFrame(rows).to_csv('runtime/ci/cohort_fractions.csv', index=False)
    print('saved runtime/ci/cohort_fractions.csv')