import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os, glob, sys

# multi-seed extension to v8. retrains the 5-day DeepONet K times with different seeds,
# runs the BMI 22/27/32 stability sweep for each, then renders panels where colour = modal
# class and alpha = agreement fraction across seeds.
# threshold cells where seeds disagree show up washed out, which is the visual point.

# usage:
#   1. run v8 first - it writes cache/5d_data_*.npz which i pick up below
#   2. run this. trains 5 models (caches each), then sweeps + aggregates + plots
#   the trained models cache to cache/multiseed_5d_seed{n}.pt so reruns are cheap

SEEDS = [28, 42, 137, 1729, 31415]   # the original v8 seed + 4 more
N_G0  = 80
N_S   = 60
BMIS  = (22, 27, 32)
TANH_K = 5.0

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
    'axes.labelsize':     9,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.edgecolor':     THESIS_PALETTE['text'],
    'axes.linewidth':     0.8,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'svg.fonttype':       'none',
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
})


# DeepONet (must match v8 exactly, otherwise the cached features mean nothing)
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
    def __init__(self, basis_dim=64, hidden=64):
        super().__init__()
        self.branch = _MLP(5, hidden, basis_dim)
        self.trunk  = _MLP(2, hidden, basis_dim)
        self.bias   = nn.Parameter(torch.zeros(1))
    def forward(self, b_in, t_in):
        b = self.branch(b_in); t = self.trunk(t_in)
        return (b * t).sum(dim=-1, keepdim=True) + self.bias


def find_training_cache():
    matches = sorted(glob.glob('cache/5d_data_*.npz'))
    if not matches:
        print('ERROR: no cache/5d_data_*.npz found. run v8 first to populate the training cache.')
        sys.exit(1)
    return matches[-1]  # most recent


def load_training_data(npz_path):
    d = np.load(npz_path)
    B = d['branch_in']; T = d['trunk_in']; Y = d['targets']
    n_pat = int(d['n_patients'][0])
    keep = T[:, 0] > 0
    B, T, Y = B[keep], T[keep], Y[keep]
    ramp = np.tanh(TANH_K * T[:, 0:1]).astype(np.float32)
    Y = Y / ramp
    return B, T, Y, n_pat


def train_one_seed(seed, B, T, Y, n_pat, epochs=300, batch_size=256, lr=3e-4):
    cache = f'cache/multiseed_5d_seed{seed}.pt'
    if os.path.exists(cache):
        m = DeepONet(); m.load_state_dict(torch.load(cache, weights_only=True)); m.eval()
        print(f'  seed {seed:>5}: loaded cached model')
        return m

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_tp_per = len(B) // n_pat
    pidx = np.arange(n_pat); np.random.shuffle(pidx)
    split = int(0.8 * n_pat)
    tr_p, va_p = pidx[:split], pidx[split:]
    tr_rows = np.concatenate([np.arange(i*n_tp_per, (i+1)*n_tp_per) for i in tr_p])
    va_rows = np.concatenate([np.arange(i*n_tp_per, (i+1)*n_tp_per) for i in va_p])

    Bt, Tt, Yt = torch.from_numpy(B), torch.from_numpy(T), torch.from_numpy(Y)
    train_ds = TensorDataset(Bt[tr_rows], Tt[tr_rows], Yt[tr_rows])
    val_ds   = TensorDataset(Bt[va_rows], Tt[va_rows], Yt[va_rows])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)

    model = DeepONet()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=80, gamma=0.4)

    for epoch in range(1, epochs+1):
        model.train()
        for b, t, y in train_loader:
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(b, t), y)
            loss.backward(); opt.step()
        sched.step()
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                vl = sum(nn.functional.mse_loss(model(b, t), y).item() * len(b)
                         for b, t, y in val_loader)
            print(f'  seed {seed:>5}: epoch {epoch:>3}, val mse {vl/len(val_ds):.6f}')

    os.makedirs('cache', exist_ok=True)
    torch.save(model.state_dict(), cache)
    return model


def predict_endpoint(model, S, bmi, age, sex, crp, G0, n_tp=50):
    t_norm = np.linspace(0, 1, n_tp).astype(np.float32)
    bf = np.array([S, bmi/40.0, age/80.0, float(sex), crp/10.0], dtype=np.float32)
    branch_in = torch.from_numpy(np.tile(bf, (n_tp, 1)))
    G0_col = np.full((n_tp, 1), G0/200.0, dtype=np.float32)
    trunk_in = torch.from_numpy(np.hstack([t_norm[:, None], G0_col]))
    with torch.no_grad():
        raw = model(branch_in, trunk_in).numpy().flatten()
    ramp = np.tanh(TANH_K * t_norm)
    return (G0 + ramp * raw * 200)[-1]


def classify(g0, g_end, tol=5.0):
    d = g_end - g0
    if d > tol:  return 'progressive'
    if d < -tol: return 'recovering'
    return 'stable'


def sweep_model(model, age, sex, crp, n_G0=N_G0, n_S=N_S, bmis=BMIS):
    G0_range = np.linspace(65, 400, n_G0)
    S_range  = np.linspace(0.0, 1.0, n_S)
    out = {}
    for bmi in bmis:
        cls = np.empty((n_S, n_G0), dtype='U15')
        for i, S in enumerate(S_range):
            for j, G0 in enumerate(G0_range):
                g_end = predict_endpoint(model, S, bmi, age, sex, crp, G0)
                cls[i, j] = classify(G0, g_end)
        out[bmi] = {'classes': cls, 'G0': G0_range, 'S': S_range}
        print(f'    bmi={bmi}: swept')
    return out


def aggregate(per_seed, bmi):
    stack = np.stack([s[bmi]['classes'] for s in per_seed])
    K, n_S, n_G0 = stack.shape
    modal     = np.empty((n_S, n_G0), dtype='U15')
    agreement = np.zeros((n_S, n_G0))
    for i in range(n_S):
        for j in range(n_G0):
            uniq, counts = np.unique(stack[:, i, j], return_counts=True)
            modal[i, j] = uniq[np.argmax(counts)]
            agreement[i, j] = counts.max() / K
    return modal, agreement, per_seed[0][bmi]['G0'], per_seed[0][bmi]['S']


def render_panels(per_seed, save_prefix='runtime/multiseed/stability_landscape'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cat_colors = {
        'recovering':  THESIS_PALETTE['healthy'],
        'stable':      THESIS_PALETTE['slate'],
        'progressive': THESIS_PALETTE['t2d'],
    }
    titles = {22: 'BMI 22 (healthy)', 27: 'BMI 27 (overweight)', 32: 'BMI 32 (obese)'}

    for ax, bmi in zip(axes, BMIS):
        modal, agree, G0r, Sr = aggregate(per_seed, bmi)
        H, W = modal.shape
        img = np.zeros((H, W, 4))
        for cls, hex_col in cat_colors.items():
            mask = modal == cls
            img[mask] = np.array(to_rgba(hex_col))
        img[..., 3] = agree  # alpha = agreement
        ax.imshow(img, extent=[G0r[0], G0r[-1], Sr[0], Sr[-1]],
                  origin='lower', aspect='auto', interpolation='nearest')
        ax.axvline(100, color=THESIS_PALETTE['ifg'], ls='--', lw=0.8, alpha=0.7)
        ax.axvline(126, color=THESIS_PALETTE['t2d'], ls='--', lw=0.8, alpha=0.7)
        ax.set_xlabel('Initial glucose $G_0$ (mg/dL)')
        ax.set_ylabel('Stress score $S$')
        ax.set_title(titles[bmi])

    # legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=cat_colors[c]) for c in ['recovering', 'stable', 'progressive']]
    fig.legend(handles, ['recovering', 'stable', 'progressive'],
               loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.text(0.5, -0.12, f'colour = modal class across $K = {len(SEEDS)}$ seeds; alpha = agreement fraction',
             ha='center', va='top', style='italic', color=THESIS_PALETTE['grey'])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
    plt.savefig(save_prefix + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_prefix + '.svg', bbox_inches='tight')
    print(f'  saved {save_prefix}.png and .svg')
    plt.close(fig)


def render_disagreement_overlay(per_seed, save_prefix='runtime/multiseed/disagreement'):
    # second figure: shows where the seeds disagreed regardless of modal class
    # cell = 1 - agreement; high values = where seeds split
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = {22: 'BMI 22', 27: 'BMI 27', 32: 'BMI 32'}

    for ax, bmi in zip(axes, BMIS):
        _, agree, G0r, Sr = aggregate(per_seed, bmi)
        disagree = 1.0 - agree
        im = ax.imshow(disagree, extent=[G0r[0], G0r[-1], Sr[0], Sr[-1]],
                       origin='lower', aspect='auto', cmap='Purples', vmin=0, vmax=1)
        ax.axvline(100, color=THESIS_PALETTE['ifg'], ls='--', lw=0.8, alpha=0.7)
        ax.axvline(126, color=THESIS_PALETTE['t2d'], ls='--', lw=0.8, alpha=0.7)
        ax.set_xlabel('Initial glucose $G_0$ (mg/dL)')
        ax.set_ylabel('Stress score $S$')
        ax.set_title(titles[bmi])
    fig.colorbar(im, ax=axes, shrink=0.8, label='disagreement fraction')

    plt.savefig(save_prefix + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_prefix + '.svg', bbox_inches='tight')
    print(f'  saved {save_prefix}.png and .svg')
    plt.close(fig)


def summary_table(per_seed):
    # cohort-level summary: what fraction of cells in each class, mean +/- std across seeds
    rows = []
    for bmi in BMIS:
        for cls in ('recovering', 'stable', 'progressive'):
            fracs = [(s[bmi]['classes'] == cls).mean() for s in per_seed]
            rows.append({
                'bmi': bmi, 'class': cls,
                'frac_mean': float(np.mean(fracs)),
                'frac_std':  float(np.std(fracs, ddof=1)),
                'frac_min':  float(np.min(fracs)),
                'frac_max':  float(np.max(fracs)),
            })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    print(f'multi-seed sweep with {len(SEEDS)} seeds: {SEEDS}')

    cache_npz = find_training_cache()
    print(f'using training cache: {cache_npz}')
    B, T, Y, n_pat = load_training_data(cache_npz)
    print(f'  {len(B):,} rows, {n_pat:,} patients')

    print('training models')
    models = [train_one_seed(s, B, T, Y, n_pat) for s in SEEDS]

    # cohort medians for the sweep. v8's branch feature layout is
    # [S_input, BMI/40, age/80, sex, CRP/10] - so we pull medians directly from B
    # rather than re-loading the raw CSVs. unique patients first since B has
    # n_timepoints rows per patient (same branch features each).
    n_tp_per = len(B) // n_pat
    B_unique = B[::n_tp_per]                       # one row per patient
    age_med = float(np.median(B_unique[:, 2]) * 80.0)
    sex_med = float(np.round(np.mean(B_unique[:, 3])))   # binary 0/1, take majority
    crp_med = float(np.median(B_unique[:, 4]) * 10.0)
    print(f'cohort medians (from training cache): age={age_med:.1f}, sex={sex_med}, crp={crp_med:.2f}')

    print('running per-seed stability sweeps')
    per_seed = []
    for k, m in enumerate(models):
        print(f'  seed {SEEDS[k]:>5}')
        per_seed.append(sweep_model(m, age_med, sex_med, crp_med))

    # save raw and aggregated results for downstream reuse
    os.makedirs('cache', exist_ok=True)
    save_dict = {'seeds': np.array(SEEDS)}
    for bmi in BMIS:
        save_dict[f'G0_{bmi}'] = per_seed[0][bmi]['G0']
        save_dict[f'S_{bmi}']  = per_seed[0][bmi]['S']
        save_dict[f'all_classes_{bmi}'] = np.stack([s[bmi]['classes'] for s in per_seed])
        modal, agree, _, _ = aggregate(per_seed, bmi)
        save_dict[f'modal_{bmi}']     = modal
        save_dict[f'agreement_{bmi}'] = agree
    np.savez('cache/multiseed_sweep.npz', **save_dict)
    print('saved cache/multiseed_sweep.npz')

    print('rendering figures')
    render_panels(per_seed)
    render_disagreement_overlay(per_seed)

    summary = summary_table(per_seed)
    summary.to_csv('runtime/multiseed/class_fractions.csv', index=False)
    print('\nclass fraction summary across seeds:')
    print(summary.to_string(index=False))