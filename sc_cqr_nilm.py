import os, time, json, glob, shutil, random
import numpy as np, pandas as pd
import torch, torch.nn as nn
import matplotlib.pyplot as plt, matplotlib as mpl
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from scipy.stats import spearmanr

T0 = time.time()
TIME_BUDGET_H = 7.5
def hours_left(): return TIME_BUDGET_H - (time.time() - T0) / 3600

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True

DATA_DIR = '/kaggle/input/datasets/rabianaz22/refit-dataset'
FILE_TPL = 'CLEAN_House{h}.csv'
OUT = '/kaggle/working'

TRAIN_HOUSES, VAL_HOUSE, TEST_HOUSE = [2, 3], 9, 20
EXTRA_HOUSES = [5, 6, 13]

APP_COLS = {
    'kettle':          {2:'Appliance8', 3:'Appliance9', 9:'Appliance7', 20:'Appliance9'},
    'microwave':       {2:'Appliance5', 3:'Appliance8', 9:'Appliance6', 20:'Appliance8'},
    'fridge':          {2:'Appliance1', 3:'Appliance2', 9:'Appliance1', 20:'Appliance1'},
    'dishwasher':      {2:'Appliance3', 3:'Appliance5', 9:'Appliance4', 20:'Appliance5'},
    'washing_machine': {2:'Appliance2', 3:'Appliance6', 9:'Appliance3', 20:'Appliance4'},
}
MANUAL_OVERRIDE = {
    'kettle':          {5:'Appliance8', 6:'Appliance7'},
    'microwave':       {5:'Appliance7', 6:'Appliance6'},
    'fridge':          {5:'Appliance1', 6:'Appliance1'},
    'dishwasher':      {5:'Appliance4', 6:'Appliance3'},
    'washing_machine': {5:'Appliance3', 6:'Appliance2'},
}
APP_PARAMS = {'kettle': (2000, 3100), 'microwave': (200, 3000), 'fridge': (50, 400), 'dishwasher': (10, 2500), 'washing_machine': (20, 2500)}
NAME = {'kettle': 'Kettle', 'microwave': 'Microwave', 'fridge': 'Fridge', 'dishwasher': 'Dishwasher', 'washing_machine': 'Washing machine'}
APPS = list(APP_PARAMS)

WINDOW, SAMPLE_SECS = 599, 8
MAX_TRAIN_WINDOWS, MAX_EPOCHS, PATIENCE = 250_000, 10, 2
SEEDS, ARCHS = [1, 2, 3], ['cnn', 'bilstm']
ALPHA, MC_T, CAL_FRAC, EVAL_STRIDE = 0.10, 30, 0.10, 6
MIN_ON, MIN_GROUP = 50, 20
FACTORS = [0.50, 0.75, 1.00, 1.25, 1.50]
STEP_SECS = SAMPLE_SECS * EVAL_STRIDE

for f in glob.glob(f'{OUT}/bilstm_*.pt') + glob.glob(f'{OUT}/cnn_*.pt'): os.remove(f)
hits = glob.glob('/kaggle/input/**/cnn_kettle_s1.pt', recursive=True)
if hits:
    PREV = os.path.dirname(hits[0])
    for f in (glob.glob(f'{PREV}/cnn_*.pt') + glob.glob(f'{PREV}/bilstm_*.pt') + glob.glob(f'{PREV}/house_*.parquet')):
        shutil.copy(f, f'{OUT}/{os.path.basename(f)}')

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_house(h, cache_it=True):
    cache = f'{OUT}/house_{h}.parquet'
    if cache_it and os.path.exists(cache): return pd.read_parquet(cache)
    df = pd.read_csv(os.path.join(DATA_DIR, FILE_TPL.format(h=h)))
    idx = pd.to_datetime(df['Unix'], unit='s') if 'Unix' in df.columns else pd.to_datetime(df[df.columns[0]], errors='coerce')
    df = df.select_dtypes('number').drop(columns=['Unix'], errors='ignore')
    df.index = idx
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated()].sort_index()
    df = df.resample(f'{SAMPLE_SECS}s').mean().ffill(limit=int(180 / SAMPLE_SECS)).astype('float32')
    if cache_it: df.to_parquet(cache)
    return df

for f in glob.glob(f'{OUT}/house_*.parquet'):
    if os.path.getsize(f) < 20_000_000: os.remove(f)

raw = {h: load_house(h) for h in sorted(set(TRAIN_HOUSES + [VAL_HOUSE, TEST_HOUSE]))}
agg = pd.concat([raw[h]['Aggregate'] for h in TRAIN_HOUSES]).dropna()
AGG_MEAN, AGG_STD = float(agg.mean()), float(agg.std())
meta = {'AGG_MEAN': AGG_MEAN, 'AGG_STD': AGG_STD, 'WINDOW': WINDOW, 'SAMPLE_SECS': SAMPLE_SECS, 'APP_COLS': APP_COLS, 'APP_PARAMS': APP_PARAMS}
meta.update({'TRAIN_HOUSES': TRAIN_HOUSES, 'VAL_HOUSE': VAL_HOUSE, 'TEST_HOUSE': TEST_HOUSE, 'SEEDS': SEEDS, 'ARCHS': ARCHS, 'VERSION': 'v3.1'})
json.dump(meta, open(f'{OUT}/metadata.json', 'w'))

def build_xy(d, col, app):
    d = d[['Aggregate', col]].dropna()
    mx = APP_PARAMS[app][1]
    x = ((d['Aggregate'].values - AGG_MEAN) / AGG_STD).astype(np.float32)
    y = (np.clip(d[col].values, 0, mx) / mx).astype(np.float32)
    return x, y

def house_xy(h, app): return build_xy(raw[h], APP_COLS[app][h], app)

class S2P(Dataset):
    def __init__(self, x, y, idxs): self.x, self.y, self.idxs = x, y, idxs
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        s = self.idxs[i]
        return self.x[s:s + WINDOW][None, :], self.y[s + WINDOW // 2]

def make_idxs(x, y, app, n_max=None, balance=None, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x) - WINDOW + 1
    if n <= 0: return np.array([], dtype=np.int64)
    if balance is None: return np.arange(n, dtype=np.int64)
    thr = APP_PARAMS[app][0] / APP_PARAMS[app][1]
    mid = y[WINDOW // 2: WINDOW // 2 + n]
    on_i, off_i = np.where(mid > thr)[0], np.where(mid <= thr)[0]
    n_tot = min(n_max, n)
    n_on = min(len(on_i), int(n_tot * balance))
    n_off = min(len(off_i), n_tot - n_on)
    idx = np.concatenate([rng.choice(on_i, n_on, replace=len(on_i) < n_on), rng.choice(off_i, n_off, replace=False)])
    rng.shuffle(idx)
    return idx.astype(np.int64)

class Seq2PointCNN(nn.Module):
    def __init__(self, p=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 30, 10, padding='same'), nn.ReLU(), nn.Dropout(p),
            nn.Conv1d(30, 30, 8, padding='same'), nn.ReLU(), nn.Dropout(p),
            nn.Conv1d(30, 40, 6, padding='same'), nn.ReLU(), nn.Dropout(p),
            nn.Conv1d(40, 50, 5, padding='same'), nn.ReLU(), nn.Dropout(p),
            nn.Conv1d(50, 50, 5, padding='same'), nn.ReLU(), nn.Dropout(p),
            nn.Flatten(), nn.Linear(50 * WINDOW, 1024), nn.ReLU(), nn.Dropout(p), nn.Linear(1024, 3))
    def forward(self, x): return self.net(x)

class BiLSTM(nn.Module):
    def __init__(self, p=0.25):
        super().__init__()
        self.front = nn.Sequential(nn.Conv1d(1, 16, 8, stride=4, padding=2), nn.ReLU())
        self.l1 = nn.LSTM(16, 64, bidirectional=True, batch_first=True)
        self.l2 = nn.LSTM(128, 128, bidirectional=True, batch_first=True)
        self.drop = nn.Dropout(p)
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Dropout(p), nn.Linear(128, 3))
    def forward(self, x):
        h = self.front(x).transpose(1, 2)
        h, _ = self.l1(h); h = self.drop(h)
        h, _ = self.l2(h)
        return self.fc(h[:, h.size(1) // 2, :])

Q = torch.tensor([0.05, 0.50, 0.95])
def pinball(pred, y):
    d = y[:, None] - pred
    q = Q.to(pred.device)[None, :]
    return torch.maximum(q * d, (q - 1) * d).mean()

def train_one(arch, app, seed):
    ck = f'{OUT}/{arch}_{app}_s{seed}.pt'
    if os.path.exists(ck): return
    set_seed(seed)
    model = (Seq2PointCNN() if arch == 'cnn' else BiLSTM()).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler()
    xs, ys = zip(*[house_xy(h, app) for h in TRAIN_HOUSES])
    xtr, ytr = np.concatenate(xs), np.concatenate(ys)
    tr = DataLoader(S2P(xtr, ytr, make_idxs(xtr, ytr, app, MAX_TRAIN_WINDOWS, 0.3, seed)), batch_size=512, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    xv, yv = house_xy(VAL_HOUSE, app)
    vidx = make_idxs(xv, yv, app)
    vidx = vidx[::max(1, len(vidx) // 50_000)]
    va = DataLoader(S2P(xv, yv, vidx), batch_size=1024, num_workers=2)
    best, best_state, wait = 1e9, None, 0
    for ep in range(MAX_EPOCHS):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast('cuda'): loss = pinball(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        model.eval(); errs = []
        with torch.no_grad(), autocast('cuda'):
            for xb, yb in va: errs.append((model(xb.to(DEVICE))[:, 1].float().cpu() - yb).abs())
        vmae = torch.cat(errs).mean().item() * APP_PARAMS[app][1]
        if vmae < best - 0.1: best, best_state, wait = vmae, model.state_dict(), 0
        else:
            wait += 1
            if wait >= PATIENCE: break
        if hours_left() < 0.3: break
    torch.save(best_state, ck)

for arch in ARCHS:
    for app in APPS:
        for seed in SEEDS:
            if hours_left() >= 0.6: train_one(arch, app, seed)

def load_model(arch, app, seed):
    ck = f'{OUT}/{arch}_{app}_s{seed}.pt'
    if not os.path.exists(ck): return None
    m = (Seq2PointCNN() if arch == 'cnn' else BiLSTM()).to(DEVICE)
    m.load_state_dict(torch.load(ck, map_location=DEVICE))
    return m

@torch.no_grad()
def predict(model, loader):
    model.eval(); P, Ys = [], []
    for xb, yb in loader: P.append(model(xb.to(DEVICE)).cpu()); Ys.append(yb)
    return {'q': torch.cat(P).numpy(), 'y': torch.cat(Ys).numpy()}

@torch.no_grad()
def mc_sigma(model, loader):
    model.eval()
    for mod in model.modules():
        if isinstance(mod, nn.Dropout): mod.train()
    out = []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        out.append(torch.stack([model(xb)[:, 1].cpu() for _ in range(MC_T)]).std(0))
    return torch.cat(out).numpy()

def scores(q, y): return np.maximum(q[:, 0] - y, y - q[:, 2])

def qhat(s, a=ALPHA):
    n = len(s)
    return np.quantile(s, min(np.ceil((n + 1) * (1 - a)) / n, 1.0), method='higher')

def cqr(qc, yc, qt, alpha=ALPHA):
    qh = qhat(scores(qc, yc), alpha)
    return qt[:, 0] - qh, qt[:, 2] + qh

def mondrian_cqr(qc, yc, qt, thr, alpha=ALPHA):
    gc, gt = qc[:, 1] > thr, qt[:, 1] > thr
    lo, hi = np.empty(len(qt)), np.empty(len(qt))
    for g in [True, False]:
        mc, mt = gc == g, gt == g
        if mt.sum() == 0: continue
        cc = mc if mc.sum() >= MIN_GROUP else np.ones_like(mc, bool)
        qh = qhat(scores(qc[cc], yc[cc]), alpha)
        lo[mt], hi[mt] = qt[mt, 0] - qh, qt[mt, 2] + qh
    return lo, hi

def cov(y, lo, hi, m=None):
    if m is None: return float(((y >= lo) & (y <= hi)).mean())
    return float(((y[m] >= lo[m]) & (y[m] <= hi[m])).mean()) if m.sum() > MIN_ON else np.nan

def f1_score(p, y, thr):
    tp = ((p > thr) & (y > thr)).sum(); fp = ((p > thr) & (y <= thr)).sum(); fn = ((p <= thr) & (y > thr)).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))

rows = []
for arch in ARCHS:
    for app in APPS:
        models = {s: load_model(arch, app, s) for s in SEEDS}
        if any(m is None for m in models.values()): continue
        mx, thr = APP_PARAMS[app][1], APP_PARAMS[app][0] / APP_PARAMS[app][1]
        xt, yt = house_xy(TEST_HOUSE, app)
        idxs = np.arange(len(xt) - WINDOW + 1, dtype=np.int64)[::EVAL_STRIDE]
        n_cal = int(len(idxs) * CAL_FRAC)
        dl_cal = DataLoader(S2P(xt, yt, idxs[:n_cal]), batch_size=1024, num_workers=2)
        dl_test = DataLoader(S2P(xt, yt, idxs[n_cal:]), batch_size=1024, num_workers=2)
        xv, yv = house_xy(VAL_HOUSE, app)
        vidx = np.arange(len(xv) - WINDOW + 1, dtype=np.int64)[::EVAL_STRIDE * 3]
        dl_src = DataLoader(S2P(xv, yv, vidx), batch_size=1024, num_workers=2)
        t0 = time.time()
        preds = {s: predict(models[s], dl_test) for s in SEEDS}
        t_ens = time.time() - t0
        t0 = time.time()
        sig_mc = mc_sigma(models[SEEDS[0]], dl_test)
        t_mc = time.time() - t0
        cal_t = predict(models[SEEDS[0]], dl_cal)
        cal_s = predict(models[SEEDS[0]], dl_src)
        y, q1 = preds[SEEDS[0]]['y'], preds[SEEDS[0]]['q']
        p = np.mean([preds[s]['q'][:, 1] for s in SEEDS], axis=0)
        sig_ens = np.std([preds[s]['q'][:, 1] for s in SEEDS], axis=0)
        sig_qw = (q1[:, 2] - q1[:, 0]) / 2
        err = np.abs(y - p)
        seed_maes = [float(np.abs(y - preds[s]['q'][:, 1]).mean() * mx) for s in SEEDS]
        on = y > thr
        lo_t, hi_t = cqr(cal_t['q'], cal_t['y'], q1)
        lo_s, hi_s = cqr(cal_s['q'], cal_s['y'], q1)
        lo_m, hi_m = mondrian_cqr(cal_t['q'], cal_t['y'], q1, thr)
        order = np.argsort(-sig_ens)
        rc = {k: float(np.abs(y[order[int(len(order) * k / 100):]] - p[order[int(len(order) * k / 100):]]).mean() * mx) for k in [0, 5, 10, 20, 30]}
        np.savez_compressed(f'{OUT}/preds_{arch}_{app}.npz', y=y, q1=q1, p=p, sig_ens=sig_ens, sig_mc=sig_mc, q_cal_t=cal_t['q'], y_cal_t=cal_t['y'], q_cal_s=cal_s['q'], y_cal_s=cal_s['y'])
        r = dict(arch=arch, app=app, MAE=round(float(err.mean() * mx), 1), MAE_std=round(float(np.std(seed_maes)), 1))
        r.update(MAE_on=round(float(np.abs(y[on] - p[on]).mean() * mx) if on.sum() > MIN_ON else np.nan, 1))
        r.update(SAE=round(float(abs(p.sum() - y.sum()) / max(y.sum(), 1e-8)), 3), F1=round(f1_score(p, y, thr), 3))
        r.update(raw_cov=round(cov(y, q1[:, 0], q1[:, 2]), 3), raw_cov_on=round(cov(y, q1[:, 0], q1[:, 2], on), 3))
        r.update(CQR_cov_t=round(cov(y, lo_t, hi_t), 3), CQR_cov_t_on=round(cov(y, lo_t, hi_t, on), 3), CQR_w_t=round(float((hi_t - lo_t).mean() * mx), 1))
        r.update(CQR_cov_s=round(cov(y, lo_s, hi_s), 3), CQR_cov_s_on=round(cov(y, lo_s, hi_s, on), 3), CQR_w_s=round(float((hi_s - lo_s).mean() * mx), 1))
        r.update(M_cov=round(cov(y, lo_m, hi_m), 3), M_cov_on=round(cov(y, lo_m, hi_m, on), 3), M_w=round(float((hi_m - lo_m).mean() * mx), 1))
        r.update(rho_ens=round(float(spearmanr(sig_ens, err).statistic), 3), rho_mc=round(float(spearmanr(sig_mc, err).statistic), 3), rho_qw=round(float(spearmanr(sig_qw, err).statistic), 3))
        r.update(t_ens_s=round(t_ens, 1), t_mc_s=round(t_mc, 1))
        r.update({f'MAE_rej{k}': round(v, 1) for k, v in rc.items()})
        rows.append(r)
        del preds, cal_t, cal_s, models
        torch.cuda.empty_cache()

df_main = pd.DataFrame(rows)
df_main.to_csv(f'{OUT}/results_main.csv', index=False)

def load_npz(arch, app):
    z = np.load(f'{OUT}/preds_{arch}_{app}.npz')
    return z['q_cal_t'], z['y_cal_t'], z['q1'], z['y'], z['p']

HRS = SAMPLE_SECS * EVAL_STRIDE / 3600
rowsA, rowsB, rowsC, roll = [], [], [], {}
for app in APPS:
    qc, yc, qt, yt, _ = load_npz('cnn', app)
    mx, thr = APP_PARAMS[app][1], APP_PARAMS[app][0] / APP_PARAMS[app][1]
    on = yt > thr
    for frac in [0.05, 0.10, 0.25, 0.50, 1.00]:
        n = max(20, int(len(yc) * frac))
        lo, hi = cqr(qc[:n], yc[:n], qt)
        rowsA.append(dict(app=app, frac=frac, n_cal=n, days=round(n * HRS / 24, 2), cov=round(cov(yt, lo, hi), 3), cov_on=round(cov(yt, lo, hi, on), 3), width_W=round(float((hi - lo).mean() * mx), 1)))
    for a in [0.20, 0.10, 0.05]:
        lo, hi = cqr(qc, yc, qt, a)
        rowsB.append(dict(app=app, nominal=1 - a, cov=round(cov(yt, lo, hi), 3)))
    s_all = np.concatenate([scores(qc, yc), scores(qt, yt)])
    n0, STEP = len(yc), 200
    lo, hi = np.empty(len(yt)), np.empty(len(yt))
    for a in range(0, len(yt), STEP):
        b = min(a + STEP, len(yt))
        qh = qhat(s_all[max(0, n0 + a - n0):n0 + a])
        lo[a:b], hi[a:b] = qt[a:b, 0] - qh, qt[a:b, 2] + qh
    lo_st, hi_st = cqr(qc, yc, qt)
    hit_on = ((yt >= lo) & (yt <= hi)).astype(float)
    hit_st = ((yt >= lo_st) & (yt <= hi_st)).astype(float)
    rowsC.append(dict(app=app, static_cov=round(float(hit_st.mean()), 3), online_cov=round(float(hit_on.mean()), 3), static_w=round(float((hi_st - lo_st).mean() * mx), 1), online_w=round(float((hi - lo).mean() * mx), 1)))
    k = max(1000, len(yt) // 30)
    roll[app] = (pd.Series(hit_st).rolling(k).mean(), pd.Series(hit_on).rolling(k).mean())

pd.DataFrame(rowsA).to_csv(f'{OUT}/ablation_calsize.csv', index=False)
pd.DataFrame(rowsB).to_csv(f'{OUT}/coverage_levels.csv', index=False)
pd.DataFrame(rowsC).to_csv(f'{OUT}/online_recal.csv', index=False)
np.save(f'{OUT}/roll_wm_static.npy', roll['washing_machine'][0].values)
np.save(f'{OUT}/roll_wm_online.npy', roll['washing_machine'][1].values)

def pick_columns(h):
    df = raw_extra[h]
    cols = [c for c in df.columns if c.lower().startswith('appliance')]
    st = {c: dict(mx=float(df[c].dropna().max()), p99=float(df[c].dropna().quantile(.99)), duty=float((df[c].dropna() > 50).mean()), d2k=float((df[c].dropna() > 2000).mean())) for c in cols}
    checks = {'kettle': lambda s: s['mx'] >= 2500 and s['duty'] < 0.03 and s['d2k'] > 0, 'microwave': lambda s: 800 <= s['mx'] <= 3200 and s['duty'] < 0.03, 'fridge': lambda s: s['duty'] >= 0.20 and s['p99'] <= 500, 'dishwasher': lambda s: s['mx'] >= 1800 and 0.005 <= s['duty'] <= 0.15, 'washing_machine': lambda s: s['mx'] >= 1800 and 0.005 <= s['duty'] <= 0.15}
    prefer = {'kettle': lambda s: s['d2k'], 'microwave': lambda s: -s['duty'], 'fridge': lambda s: s['duty'], 'dishwasher': lambda s: s['p99'], 'washing_machine': lambda s: -s['p99']}
    used = set()
    for app in ['kettle', 'fridge', 'dishwasher', 'washing_machine', 'microwave']:
        col = MANUAL_OVERRIDE.get(app, {}).get(h)
        if not col:
            cands = [(c, st[c]) for c in cols if c not in used and checks[app](st[c])]
            col = max(cands, key=lambda cs: prefer[app](cs[1]))[0] if cands else None
        extra_cols[app][h] = col
        if col: used.add(col)

raw_extra = {h: load_house(h, cache_it=False) for h in EXTRA_HOUSES}
extra_cols = {a: {} for a in APPS}
for h in EXTRA_HOUSES: pick_columns(h)

rowsM = []
for h in EXTRA_HOUSES:
    for app in APPS:
        col = extra_cols[app][h]
        if col is None: continue
        mx, thr = APP_PARAMS[app][1], APP_PARAMS[app][0] / APP_PARAMS[app][1]
        x, yv = build_xy(raw_extra[h], col, app)
        idxs = np.arange(len(x) - WINDOW + 1, dtype=np.int64)[::EVAL_STRIDE]
        if len(idxs) < 5000: continue
        nc = int(len(idxs) * CAL_FRAC)
        dlc = DataLoader(S2P(x, yv, idxs[:nc]), batch_size=1024, num_workers=2)
        dlt = DataLoader(S2P(x, yv, idxs[nc:]), batch_size=1024, num_workers=2)
        ms = {s: load_model('cnn', app, s) for s in SEEDS}
        preds = {s: predict(ms[s], dlt) for s in SEEDS}
        cal = predict(ms[SEEDS[0]], dlc)
        q1, y = preds[SEEDS[0]]['q'], preds[SEEDS[0]]['y']
        p = np.mean([preds[s]['q'][:, 1] for s in SEEDS], axis=0)
        on = y > thr
        lo, hi = cqr(cal['q'], cal['y'], q1)
        lom, him = mondrian_cqr(cal['q'], cal['y'], q1, thr)
        r = dict(house=h, app=app, col=col, MAE=round(float(np.abs(y - p).mean() * mx), 1), F1=round(f1_score(p, y, thr), 3))
        r.update(raw_on=round(cov(y, q1[:, 0], q1[:, 2], on), 3), CQR_cov=round(cov(y, lo, hi), 3), CQR_on=round(cov(y, lo, hi, on), 3))
        r.update(M_on=round(cov(y, lom, him, on), 3), M_w=round(float((him - lom).mean() * mx), 1))
        rowsM.append(r)
        del preds, ms
        torch.cuda.empty_cache()

pd.DataFrame(rowsM).to_csv(f'{OUT}/results_multi_house.csv', index=False)

rowsS = []
for app in APPS:
    qc, yc, qt, yt, p = load_npz('cnn', app)
    mx, thr0 = APP_PARAMS[app][1], APP_PARAMS[app][0] / APP_PARAMS[app][1]
    on0 = yt > thr0
    lo_c, hi_c = cqr(qc, yc, qt)
    for f in FACTORS:
        thr = thr0 * f
        on = yt > thr
        tp = ((p > thr) & on).sum(); fp = ((p > thr) & ~on).sum(); fn = ((p <= thr) & on).sum()
        lo_m, hi_m = mondrian_cqr(qc, yc, qt, thr)
        r = dict(app=app, factor=f, thr_W=round(thr * mx, 1), n_on=int(on.sum()), on_frac=round(float(on.mean()), 4), F1=round(float(2 * tp / max(2 * tp + fp + fn, 1)), 3))
        r.update(CQR_on=round(cov(yt, lo_c, hi_c, on), 3), SC_on=round(cov(yt, lo_m, hi_m, on), 3), SC_on_fixedmask=round(cov(yt, lo_m, hi_m, on0), 3))
        r.update(SC_marg=round(cov(yt, lo_m, hi_m), 3), SC_width_W=round(float((hi_m - lo_m).mean() * mx), 1))
        rowsS.append(r)

dS = pd.DataFrame(rowsS)
dS.to_csv(f'{OUT}/threshold_sensitivity.csv', index=False)

ref = df_main[df_main.arch == 'cnn'].set_index('app').loc[APPS]
base = dS[dS.factor == 1.00].set_index('app').loc[APPS]
chk = pd.DataFrame({'F1_paper': ref.F1, 'F1_here': base.F1, 'CQRon_paper': ref.CQR_cov_t_on, 'CQRon_here': base.CQR_on, 'SCon_paper': ref.M_cov_on, 'SCon_here': base.SC_on, 'SCmarg_paper': ref.M_cov, 'SCmarg_here': base.SC_marg, 'SCw_paper': ref.M_w, 'SCw_here': base.SC_width_W})
mism = [(a, c) for a in APPS for c in ['F1', 'CQRon', 'SCon', 'SCmarg', 'SCw'] if not np.isclose(chk.loc[a, f'{c}_paper'], chk.loc[a, f'{c}_here'], atol=1e-3)]
if mism: print('WARNING: mismatch', mism)

r_st = roll['washing_machine'][0].values
r_on = roll['washing_machine'][1].values
ok = ~np.isnan(r_st) & ~np.isnan(r_on)
n_test = len(r_st)
k_roll = max(1000, n_test // 30)
to_days = lambda n: n * STEP_SECS / 3600 / 24
n_cal_wm = int(len(np.load(f'{OUT}/preds_cnn_washing_machine.npz')['y_cal_t']))
stat_rows = [('test_days', to_days(n_test), to_days(n_test)), ('rolling_window_days', to_days(k_roll), to_days(k_roll)), ('cal_days', to_days(n_cal_wm), to_days(n_cal_wm))]
stat_rows += [('mean_cov', float(np.mean(r_st[ok])), float(np.mean(r_on[ok]))), ('mean_abs_dev', float(np.mean(np.abs(r_st[ok] - 0.90))), float(np.mean(np.abs(r_on[ok] - 0.90))))]
stat_rows += [('frac_below_0.90', float(np.mean(r_st[ok] < 0.90)), float(np.mean(r_on[ok] < 0.90))), ('frac_below_0.85', float(np.mean(r_st[ok] < 0.85)), float(np.mean(r_on[ok] < 0.85)))]
stat_rows += [('minimum', float(np.min(r_st[ok])), float(np.min(r_on[ok]))), ('std', float(np.std(r_st[ok])), float(np.std(r_on[ok])))]
pd.DataFrame(stat_rows, columns=['stat', 'static', 'online']).round(3).to_csv(f'{OUT}/rolling_stats.csv', index=False)

mpl.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'figure.dpi': 300, 'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'pdf.fonttype': 42, 'ps.fonttype': 42})

MXv = {a: APP_PARAMS[a][1] for a in APPS}
ks = np.arange(0, 41)
fig, axes = plt.subplots(2, 3, figsize=(3.5, 2.7), sharex=True, sharey=True)
axf = axes.flatten()
for ax, app in zip(axf[:5], APPS):
    z = np.load(f'{OUT}/preds_cnn_{app}.npz')
    err = np.abs(z['y'] - z['p']) * MXv[app]
    order = np.argsort(-z['sig_ens'])
    mae = np.array([err[order[int(len(order) * k / 100):]].mean() for k in ks])
    ax.axhline(100, color='0.75', ls=(0, (4, 3)), lw=0.7)
    ax.plot(ks, 100 * mae / mae[0], color='#1d3557', lw=1.3, solid_capstyle='round')
    ax.set_title(NAME[app], fontsize=7.5, pad=2)
    ax.set_xlim(0, 40); ax.set_xticks([0, 20, 40]); ax.set_ylim(0, 115); ax.set_yticks([0, 50, 100])
    ax.tick_params(length=2, labelsize=7, labelbottom=True)
axf[5].axis('off')
fig.supylabel('Remaining MAE (%)', fontsize=9, x=0.01)
fig.supxlabel('Windows rejected (%)', fontsize=9, y=0.0)
fig.subplots_adjust(wspace=0.15, hspace=0.85)
fig.savefig(f'{OUT}/fig_risk_coverage.pdf'); plt.show()

t = np.arange(n_test) * STEP_SECS / 3600 / 24
fig, ax = plt.subplots(figsize=(3.5, 2.2))
ax.axhline(0.90, color='0.1', ls=(0, (4, 3)), lw=0.9)
ax.plot(t, r_st, color='#457b9d', lw=1.1, label='Static CQR')
ax.plot(t, r_on, color='#e07a5f', lw=1.1, label='Online recalibration')
ax.set_xlabel('Days into test period'); ax.set_ylabel('Rolling coverage')
ax.set_ylim(0.65, 1.02); ax.set_xlim(0, t[-1])
ax.xaxis.grid(True, ls=':', lw=0.5, color='0.85'); ax.set_axisbelow(True)
ax.tick_params(length=2.5)
ax.legend(frameon=False, fontsize=8, loc='lower left', handlelength=1.4, handletextpad=0.4)
d = 0.015
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, linewidth=1)
ax.plot((-d, d), (-d, d), **kwargs)
ax.plot((-d, d), (2 * d, 4 * d), **kwargs)
fig.savefig(f'{OUT}/fig_online_recal.pdf'); plt.show()