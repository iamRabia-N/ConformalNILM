
import os, time, json, glob, shutil, random, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
T0 = time.time()
TIME_BUDGET_H = 7.5
def hours_left(): return TIME_BUDGET_H - (time.time()-T0)/3600
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True
print('Device:', DEVICE)
DATA_DIR = '/kaggle/input/datasets/rabianaz22/refit-dataset'
FILE_TPL = 'CLEAN_House{h}.csv'
TRAIN_HOUSES, VAL_HOUSE, TEST_HOUSE = [2, 3], 9, 20
QUICK_TEST = False
APP_COLS = {
 'kettle':          {2:'Appliance8', 3:'Appliance9', 9:'Appliance7', 20:'Appliance9'},
 'microwave':       {2:'Appliance5', 3:'Appliance8', 9:'Appliance6', 20:'Appliance8'},
 'fridge':          {2:'Appliance1', 3:'Appliance2', 9:'Appliance1', 20:'Appliance1'},
 'dishwasher':      {2:'Appliance3', 3:'Appliance5', 9:'Appliance4', 20:'Appliance5'},
 'washing_machine': {2:'Appliance2', 3:'Appliance6', 9:'Appliance3', 20:'Appliance4'},
}
APP_PARAMS = {'kettle':(2000,3100),'microwave':(200,3000),'fridge':(50,400), 'dishwasher':(10,2500),'washing_machine':(20,2500)}
WINDOW, SAMPLE_SECS = 599, 8
MAX_TRAIN_WINDOWS, MAX_EPOCHS, PATIENCE = 250_000, 10, 2
SEEDS, ARCHS = [1,2,3], ['cnn','bilstm']
APPS = list(APP_PARAMS)
OUT = '/kaggle/working'
ALPHA, MC_T, CAL_FRAC, EVAL_STRIDE = 0.10, 30, 0.10, 6
for f in glob.glob(f'{OUT}/bilstm_*.pt'): os.remove(f)
for f in glob.glob(f'{OUT}/cnn_*.pt'): os.remove(f)
hits = glob.glob('/kaggle/input/**/cnn_kettle_s1.pt', recursive=True)
if hits:
    PREV = os.path.dirname(hits[0])
    print('Reusing artifacts from:', PREV)
    for f in glob.glob(f'{PREV}/cnn_*.pt') + glob.glob(f'{PREV}/house_*.parquet'):
        shutil.copy(f, f'{OUT}/{os.path.basename(f)}')
else:
    print('No previous version found - training everything fresh.')
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def load_house(h):
    cache = f'{OUT}/house_{h}.parquet'
    if os.path.exists(cache): return pd.read_parquet(cache)
    df = pd.read_csv(os.path.join(DATA_DIR, FILE_TPL.format(h=h)))
    idx = pd.to_datetime(df['Unix'], unit='s') if 'Unix' in df.columns else \
          pd.to_datetime(df[df.columns[0]], errors='coerce')
    df = df.select_dtypes('number').drop(columns=['Unix'], errors='ignore')
    df.index = idx
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated()].sort_index()
    df = df.resample(f'{SAMPLE_SECS}s').mean()
    df = df.ffill(limit=int(180/SAMPLE_SECS)).astype('float32')
    df.to_parquet(cache)
    return df
for f in glob.glob(f'{OUT}/house_*.parquet'):
    if os.path.getsize(f) < 20_000_000: os.remove(f)
HOUSES = sorted(set(TRAIN_HOUSES + [VAL_HOUSE, TEST_HOUSE]))
raw = {h: load_house(h) for h in HOUSES}
print(f'Data loaded @ {(time.time()-T0)/60:.1f} min')
agg = pd.concat([raw[h]['Aggregate'] for h in TRAIN_HOUSES]).dropna()
AGG_MEAN, AGG_STD = float(agg.mean()), float(agg.std())
json.dump({'AGG_MEAN':AGG_MEAN,'AGG_STD':AGG_STD,'WINDOW':WINDOW,
           'SAMPLE_SECS':SAMPLE_SECS,'APP_COLS':APP_COLS,'APP_PARAMS':APP_PARAMS,
           'TRAIN_HOUSES':TRAIN_HOUSES,'VAL_HOUSE':VAL_HOUSE,'TEST_HOUSE':TEST_HOUSE,
           'SEEDS':SEEDS,'ARCHS':ARCHS,'VERSION':'v3.1'},
          open(f'{OUT}/metadata.json','w'))
print(f'AGG_MEAN={AGG_MEAN:.1f} AGG_STD={AGG_STD:.1f}')
def build_xy(h, app):
    col = APP_COLS[app][h]
    d = raw[h][['Aggregate', col]].dropna()
    x = ((d['Aggregate'].values - AGG_MEAN)/AGG_STD).astype(np.float32)
    mx = APP_PARAMS[app][1]
    y = (np.clip(d[col].values, 0, mx)/mx).astype(np.float32)
    return x, y
class S2P(Dataset):
    def __init__(self, x, y, idxs): self.x, self.y, self.idxs = x, y, idxs
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        s = self.idxs[i]
        return self.x[s:s+WINDOW][None,:], self.y[s+WINDOW//2]
def make_idxs(x, y, app, n_max=None, balance=None, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x) - WINDOW + 1
    if n <= 0: return np.array([], dtype=np.int64)
    if balance is None: return np.arange(n, dtype=np.int64)
    thr = APP_PARAMS[app][0]/APP_PARAMS[app][1]
    mid = y[WINDOW//2 : WINDOW//2 + n]
    on_i, off_i = np.where(mid > thr)[0], np.where(mid <= thr)[0]
    n_tot = min(n_max, n)
    n_on  = min(len(on_i), int(n_tot*balance))
    n_off = min(len(off_i), n_tot - n_on)
    idx = np.concatenate([rng.choice(on_i, n_on, replace=len(on_i)<n_on), rng.choice(off_i, n_off, replace=False)])
    rng.shuffle(idx); return idx.astype(np.int64)
class Seq2PointCNN(nn.Module):
    def __init__(self, p=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,30,10,padding='same'), nn.ReLU(), nn.Dropout(p),
            nn.Conv1d(30,30,8,padding='same'), nn.ReLU(), nn.Dropout(p),
            nn.Conv1d(30,40,6,padding='same'), nn.ReLU(), nn.Dropout(p),
            nn.Conv1d(40,50,5,padding='same'), nn.ReLU(), nn.Dropout(p),
            nn.Conv1d(50,50,5,padding='same'), nn.ReLU(), nn.Dropout(p),
            nn.Flatten(), nn.Linear(50*WINDOW,1024), nn.ReLU(), nn.Dropout(p),
            nn.Linear(1024,3))
    def forward(self,x): return self.net(x)
class BiLSTM(nn.Module):
    def __init__(self, p=0.25):
        super().__init__()
        self.front = nn.Sequential(nn.Conv1d(1,16,8,stride=4,padding=2), nn.ReLU())
        self.l1 = nn.LSTM(16,64, bidirectional=True, batch_first=True)
        self.l2 = nn.LSTM(128,128, bidirectional=True, batch_first=True)
        self.drop = nn.Dropout(p)
        self.fc = nn.Sequential(nn.Linear(256,128), nn.Tanh(), nn.Dropout(p), nn.Linear(128,3))
    def forward(self,x):
        h = self.front(x).transpose(1,2)
        h,_ = self.l1(h); h = self.drop(h)
        h,_ = self.l2(h)
        return self.fc(h[:, h.size(1)//2, :])
Q = torch.tensor([0.05,0.50,0.95])
def pinball(pred, y):
    d = y[:,None] - pred
    q = Q.to(pred.device)[None,:]
    return torch.maximum(q*d, (q-1)*d).mean()
def train_one(arch, app, seed):
    ck = f'{OUT}/{arch}_{app}_s{seed}.pt'
    if os.path.exists(ck): print('SKIP:', ck); return
    set_seed(seed)
    model = (Seq2PointCNN() if arch=='cnn' else BiLSTM()).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler()
    xs, ys = zip(*[build_xy(h, app) for h in TRAIN_HOUSES])
    xtr, ytr = np.concatenate(xs), np.concatenate(ys)
    tr = DataLoader(S2P(xtr,ytr, make_idxs(xtr,ytr,app,MAX_TRAIN_WINDOWS,0.3,seed)), batch_size=512, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    xv, yv = build_xy(VAL_HOUSE, app)
    vidx = make_idxs(xv,yv,app)
    vidx = vidx[::max(1, len(vidx)//50_000)]
    va = DataLoader(S2P(xv,yv,vidx), batch_size=1024, num_workers=2)
    best, best_state, wait = 1e9, None, 0
    for ep in range(MAX_EPOCHS):
        model.train(); t_ep = time.time()
        for xb, yb in tr:
            xb, yb = xb.to(DEVICE,non_blocking=True), yb.to(DEVICE,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast('cuda'):
                loss = pinball(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        model.eval(); errs = []
        with torch.no_grad(), autocast('cuda'):
            for xb, yb in va:
                errs.append((model(xb.to(DEVICE))[:,1].float().cpu()-yb).abs())
        vmae = torch.cat(errs).mean().item()*APP_PARAMS[app][1]
        print(f'{arch}/{app}/s{seed} ep{ep}: valMAE={vmae:6.1f}W '
              f'({(time.time()-t_ep)/60:.1f} min/ep, {hours_left():.2f} h left)')
        if vmae < best - 0.1: best, best_state, wait = vmae, model.state_dict(), 0
        else:
            wait += 1
            if wait >= PATIENCE: break
        if hours_left() < 0.3: break
    torch.save(best_state, ck)
for arch in ARCHS:
    for app in APPS:
        for seed in SEEDS:
            if hours_left() < 0.6: print('BUDGET skip', arch, app, seed); continue
            train_one(arch, app, seed)
print(f'TRAINING DONE @ {(time.time()-T0)/3600:.2f} h')
def load_model(arch, app, seed):
    ck = f'{OUT}/{arch}_{app}_s{seed}.pt'
    if not os.path.exists(ck): return None
    m = (Seq2PointCNN() if arch=='cnn' else BiLSTM()).to(DEVICE)
    m.load_state_dict(torch.load(ck, map_location=DEVICE)); return m
@torch.no_grad()
def predict(model, loader):
    model.eval(); P, Ys = [], []
    for xb, yb in loader:
        P.append(model(xb.to(DEVICE)).cpu()); Ys.append(yb)
    return {'q': torch.cat(P).numpy(), 'y': torch.cat(Ys).numpy()}
@torch.no_grad()
def mc_sigma(model, loader):
    model.eval()
    for mod in model.modules():
        if isinstance(mod, nn.Dropout): mod.train()
    MCs = []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        mm = torch.stack([model(xb)[:,1].cpu() for _ in range(MC_T)])
        MCs.append(mm.std(0))
    return torch.cat(MCs).numpy()
def cqr(q_cal, y_cal, q_test, alpha=ALPHA):
    s = np.maximum(q_cal[:,0]-y_cal, y_cal-q_cal[:,2]); n = len(s)
    qh = np.quantile(s, min(np.ceil((n+1)*(1-alpha))/n, 1.0), method='higher')
    return q_test[:,0]-qh, q_test[:,2]+qh
def mondrian_cqr(q_cal, y_cal, q_test, thr, alpha=ALPHA):
    g_cal, g_test = q_cal[:,1] > thr, q_test[:,1] > thr
    lo, hi = np.empty(len(q_test)), np.empty(len(q_test))
    for g in [True, False]:
        mc, mt = g_cal==g, g_test==g
        if mt.sum() == 0: continue
        cc = mc if mc.sum() >= 20 else np.ones_like(mc, bool)
        s = np.maximum(q_cal[cc,0]-y_cal[cc], y_cal[cc]-q_cal[cc,2]); n = len(s)
        qh = np.quantile(s, min(np.ceil((n+1)*(1-alpha))/n, 1.0), method='higher')
        lo[mt], hi[mt] = q_test[mt,0]-qh, q_test[mt,2]+qh
    return lo, hi
rows, rc_curves, missing = [], {}, []
for arch in ARCHS:
    for app in APPS:
        models = {s: load_model(arch, app, s) for s in SEEDS}
        if any(m is None for m in models.values()):
            missing.append((arch, app)); continue
        mx, thr = APP_PARAMS[app][1], APP_PARAMS[app][0]/APP_PARAMS[app][1]
        xt, yt = build_xy(TEST_HOUSE, app)
        idxs = np.arange(len(xt)-WINDOW+1, dtype=np.int64)[::EVAL_STRIDE]
        n_cal = int(len(idxs)*CAL_FRAC)
        dl_cal  = DataLoader(S2P(xt,yt,idxs[:n_cal]),  batch_size=1024, num_workers=2)
        dl_test = DataLoader(S2P(xt,yt,idxs[n_cal:]), batch_size=1024, num_workers=2)
        xv, yv = build_xy(VAL_HOUSE, app)
        vidx = np.arange(len(xv)-WINDOW+1, dtype=np.int64)[::EVAL_STRIDE*3]
        dl_src = DataLoader(S2P(xv,yv,vidx), batch_size=1024, num_workers=2)
        t0 = time.time()
        preds = {s: predict(models[s], dl_test) for s in SEEDS}
        t_ens = time.time()-t0
        t0 = time.time()
        sig_mc = mc_sigma(models[SEEDS[0]], dl_test)
        t_mc = time.time()-t0
        cal_t = predict(models[SEEDS[0]], dl_cal)
        cal_s = predict(models[SEEDS[0]], dl_src)
        y = preds[SEEDS[0]]['y']; q1 = preds[SEEDS[0]]['q']
        p       = np.mean([preds[s]['q'][:,1] for s in SEEDS], axis=0)
        sig_ens = np.std ([preds[s]['q'][:,1] for s in SEEDS], axis=0)
        sig_qw  = (q1[:,2]-q1[:,0])/2
        err = np.abs(y-p)
        seed_maes = [float(np.abs(y-preds[s]['q'][:,1]).mean()*mx) for s in SEEDS]
        mae = float(err.mean()*mx)
        sae = float(abs(p.sum()-y.sum())/max(y.sum(),1e-8))
        tp=((p>thr)&(y>thr)).sum(); fp=((p>thr)&(y<=thr)).sum(); fn=((p<=thr)&(y>thr)).sum()
        f1 = float(2*tp/max(2*tp+fp+fn,1))
        raw_cov = float(((y>=q1[:,0])&(y<=q1[:,2])).mean())
        on = y > thr; has_on = on.sum() > 50
        raw_cov_on = float(((y[on]>=q1[on,0])&(y[on]<=q1[on,2])).mean()) if has_on else np.nan
        mae_on     = float(np.abs(y[on]-p[on]).mean()*mx) if has_on else np.nan
        lo_t,hi_t = cqr(cal_t['q'], cal_t['y'], q1)
        lo_s,hi_s = cqr(cal_s['q'], cal_s['y'], q1)
        lo_m,hi_m = mondrian_cqr(cal_t['q'], cal_t['y'], q1, thr)
        cov_t = float(((y>=lo_t)&(y<=hi_t)).mean()); w_t = float((hi_t-lo_t).mean()*mx)
        cov_s = float(((y>=lo_s)&(y<=hi_s)).mean()); w_s = float((hi_s-lo_s).mean()*mx)
        mcov  = float(((y>=lo_m)&(y<=hi_m)).mean()); m_w = float((hi_m-lo_m).mean()*mx)
        cov_t_on = float(((y[on]>=lo_t[on])&(y[on]<=hi_t[on])).mean()) if has_on else np.nan
        cov_s_on = float(((y[on]>=lo_s[on])&(y[on]<=hi_s[on])).mean()) if has_on else np.nan
        mcov_on  = float(((y[on]>=lo_m[on])&(y[on]<=hi_m[on])).mean()) if has_on else np.nan
        order = np.argsort(-sig_ens)
        rc = {k: float(np.abs(y[o]-p[o]).mean()*mx)
              for k in [0,5,10,20,30]
              for o in [order[int(len(order)*k/100):]]}
        rc_curves[(arch,app)] = rc
        np.savez_compressed(f'{OUT}/preds_{arch}_{app}.npz', y=y, q1=q1, p=p, sig_ens=sig_ens, sig_mc=sig_mc, q_cal_t=cal_t['q'], y_cal_t=cal_t['y'], q_cal_s=cal_s['q'], y_cal_s=cal_s['y'])
        rows.append(dict(arch=arch, app=app,
            MAE=round(mae,1), MAE_std=round(float(np.std(seed_maes)),1),
            MAE_on=round(mae_on,1), SAE=round(sae,3), F1=round(f1,3),
            raw_cov=round(raw_cov,3), raw_cov_on=round(raw_cov_on,3),
            CQR_cov_t=round(cov_t,3), CQR_cov_t_on=round(cov_t_on,3), CQR_w_t=round(w_t,1),
            CQR_cov_s=round(cov_s,3), CQR_cov_s_on=round(cov_s_on,3), CQR_w_s=round(w_s,1),
            M_cov=round(mcov,3), M_cov_on=round(mcov_on,3), M_w=round(m_w,1),
            rho_ens=round(float(spearmanr(sig_ens,err).statistic),3),
            rho_mc=round(float(spearmanr(sig_mc,err).statistic),3),
            rho_qw=round(float(spearmanr(sig_qw,err).statistic),3),
            t_ens_s=round(t_ens,1), t_mc_s=round(t_mc,1),
            **{f'MAE_rej{k}': round(v,1) for k,v in rc.items()}))
        print(rows[-1], f'| {hours_left():.2f} h left')
        del preds, cal_t, cal_s, models
        torch.cuda.empty_cache()
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/results_main.csv', index=False)
print(df.to_string(index=False))
if missing: print('MISSING:', missing)
print(f'MAIN RUN DONE | total {(time.time()-T0)/3600:.2f} h')

import numpy as np, pandas as pd, os, json
import matplotlib.pyplot as plt
SRC = '/kaggle/working'; print('Artifacts:', SRC)
md = json.load(open(f'{SRC}/metadata.json'))
APP_PARAMS = {k: tuple(v) for k, v in md['APP_PARAMS'].items()}
APPS, ALPHA = list(APP_PARAMS), 0.10
HRS = md['SAMPLE_SECS'] * 6 / 3600
def scores(q, y): return np.maximum(q[:,0]-y, y-q[:,2])
def qhat(s, a=ALPHA):
    n = len(s)
    return np.quantile(s, min(np.ceil((n+1)*(1-a))/n, 1.0), method='higher')
def load(app):
    z = np.load(f'{SRC}/preds_cnn_{app}.npz')
    return z['q_cal_t'], z['y_cal_t'], z['q1'], z['y'], z['p']
rowsA = []
for app in APPS:
    qc, yc, qt, yt, _ = load(app)
    mx, thr = APP_PARAMS[app][1], APP_PARAMS[app][0]/APP_PARAMS[app][1]
    on = yt > thr
    for frac in [0.05, 0.10, 0.25, 0.50, 1.00]:
        n = max(20, int(len(yc)*frac))
        qh = qhat(scores(qc[:n], yc[:n]))
        lo, hi = qt[:,0]-qh, qt[:,2]+qh
        rowsA.append(dict(app=app, frac=frac, n_cal=n, days=round(n*HRS/24,2), cov=round(float(((yt>=lo)&(yt<=hi)).mean()),3), cov_on=round(float(((yt[on]>=lo[on])&(yt[on]<=hi[on])).mean()),3)
                   if on.sum()>50 else np.nan, width_W=round(float((hi-lo).mean()*mx),1)))
dA = pd.DataFrame(rowsA); dA.to_csv(f'{SRC}/ablation_calsize.csv', index=False)
for v in ['cov','cov_on','width_W']:
    print(dA.pivot(index='app', columns='frac', values=v).to_string())
rowsB = []
for app in APPS:
    qc, yc, qt, yt, _ = load(app)
    s = scores(qc, yc)
    for a in [0.20, 0.10, 0.05]:
        qh = qhat(s, a)
        rowsB.append(dict(app=app, nominal=1-a, cov=round(float(((yt>=qt[:,0]-qh)&(yt<=qt[:,2]+qh)).mean()),3)))
dB = pd.DataFrame(rowsB); dB.to_csv(f'{SRC}/coverage_levels.csv', index=False)
print(dB.pivot(index='app', columns='nominal', values='cov').to_string())
rowsC, roll = [], {}
for app in APPS:
    qc, yc, qt, yt, _ = load(app)
    mx = APP_PARAMS[app][1]
    s_all = np.concatenate([scores(qc, yc), scores(qt, yt)])
    n0 = len(yc); N = n0; STEP = 200
    lo = np.empty(len(yt)); hi = np.empty(len(yt))
    for a in range(0, len(yt), STEP):
        b = min(a+STEP, len(yt))
        qh = qhat(s_all[max(0, n0+a-N):n0+a])
        lo[a:b], hi[a:b] = qt[a:b,0]-qh, qt[a:b,2]+qh
    hit_on = ((yt>=lo)&(yt<=hi)).astype(float)
    qh_s = qhat(scores(qc, yc))
    hit_st = ((yt>=qt[:,0]-qh_s)&(yt<=qt[:,2]+qh_s)).astype(float)
    rowsC.append(dict(app=app,
        static_cov=round(float(hit_st.mean()),3),
        online_cov=round(float(hit_on.mean()),3),
        static_w=round(float(((qt[:,2]+qh_s)-(qt[:,0]-qh_s)).mean()*mx),1),
        online_w=round(float((hi-lo).mean()*mx),1)))
    k = max(1000, len(yt)//30)
    roll[app] = (pd.Series(hit_st).rolling(k).mean(), pd.Series(hit_on).rolling(k).mean())
dC = pd.DataFrame(rowsC); dC.to_csv(f'{SRC}/online_recal.csv', index=False)
print(dC.to_string(index=False))
np.save(f'{SRC}/roll_wm_static.npy', roll['washing_machine'][0].values)
np.save(f'{SRC}/roll_wm_online.npy', roll['washing_machine'][1].values)

import numpy as np, pandas as pd, torch, torch.nn as nn, glob, os, json, time
from torch.utils.data import Dataset, DataLoader
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ART = '/kaggle/working'
DATA_DIR = os.path.dirname(glob.glob('/kaggle/input/**/CLEAN_House2.csv', recursive=True)[0])
md = json.load(open(f'{ART}/metadata.json'))
AGG_MEAN, AGG_STD, WINDOW = md['AGG_MEAN'], md['AGG_STD'], md['WINDOW']
APP_PARAMS = {k: tuple(v) for k, v in md['APP_PARAMS'].items()}
APPS, SEEDS, ALPHA, CAL_FRAC, STRIDE = list(APP_PARAMS), [1,2,3], 0.10, 0.10, 6
EXTRA_HOUSES = [5, 6, 13]
print('Device:', DEVICE)
MANUAL_OVERRIDE = {
 'kettle':          {5:'Appliance8', 6:'Appliance7'},
 'microwave':       {5:'Appliance7', 6:'Appliance6'},
 'fridge':          {5:'Appliance1', 6:'Appliance1'},
 'dishwasher':      {5:'Appliance4', 6:'Appliance3'},
 'washing_machine': {5:'Appliance3', 6:'Appliance2'},
}
def load_house(h):
    df = pd.read_csv(f'{DATA_DIR}/CLEAN_House{h}.csv')
    idx = pd.to_datetime(df['Unix'], unit='s')
    df = df.select_dtypes('number').drop(columns=['Unix'], errors='ignore')
    df.index = idx
    df = df[~df.index.duplicated()].sort_index()
    return df.resample('8s').mean().ffill(limit=22).astype('float32')
raw = {h: load_house(h) for h in EXTRA_HOUSES}
def stats(s):
    return dict(mx=float(s.max()), p99=float(s.quantile(.99)), duty=float((s>50).mean()), d2k=float((s>2000).mean()))
CHECKS = {
 'kettle':          lambda st: st['mx']>=2500 and st['duty']<0.03 and st['d2k']>0,
 'microwave':       lambda st: 800<=st['mx']<=3200 and st['duty']<0.03,
 'fridge':          lambda st: st['duty']>=0.20 and st['p99']<=500,
 'dishwasher':      lambda st: st['mx']>=1800 and 0.005<=st['duty']<=0.15,
 'washing_machine': lambda st: st['mx']>=1800 and 0.005<=st['duty']<=0.15,
}
PICK = {'kettle': lambda st: st['d2k'], 'microwave': lambda st: -st['duty'], 'fridge': lambda st: st['duty'], 'dishwasher': lambda st: st['p99'], 'washing_machine': lambda st: -st['p99']}
APP_COLS = {a: {} for a in APPS}
for h in EXTRA_HOUSES:
    cols = [c for c in raw[h].columns if c.lower().startswith('appliance')]
    st = {c: stats(raw[h][c].dropna()) for c in cols}
    used = set()
    for app in ['kettle','fridge','dishwasher','washing_machine','microwave']:
        ov = MANUAL_OVERRIDE.get(app, {}).get(h)
        if ov: col, src = ov, 'MANUAL'
        else:
            cands = [(c, st[c]) for c in cols if c not in used and CHECKS[app](st[c])]
            col = max(cands, key=lambda cs: PICK[app](cs[1]))[0] if cands else None
            src = 'auto'
        APP_COLS[app][h] = col
        if col: used.add(col)
        print(f'house {h} {app:16s} -> {col} ({src})')
class S2P(Dataset):
    def __init__(s_, x, y, i): s_.x, s_.y, s_.i = x, y, i
    def __len__(s_): return len(s_.i)
    def __getitem__(s_, k):
        j = s_.i[k]; return s_.x[j:j+WINDOW][None,:], s_.y[j+WINDOW//2]
class Seq2PointCNN(nn.Module):
    def __init__(self, p=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,30,10,padding='same'), nn.ReLU(), nn.Dropout(p),
            nn.Conv1d(30,30,8,padding='same'), nn.ReLU(), nn.Dropout(p),
            nn.Conv1d(30,40,6,padding='same'), nn.ReLU(), nn.Dropout(p),
            nn.Conv1d(40,50,5,padding='same'), nn.ReLU(), nn.Dropout(p),
            nn.Conv1d(50,50,5,padding='same'), nn.ReLU(), nn.Dropout(p),
            nn.Flatten(), nn.Linear(50*WINDOW,1024), nn.ReLU(), nn.Dropout(p),
            nn.Linear(1024,3))
    def forward(self,x): return self.net(x)
@torch.no_grad()
def predict(m, dl):
    m.eval(); P, Y = [], []
    for xb, yb in dl: P.append(m(xb.to(DEVICE)).cpu()); Y.append(yb)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()
def qhat(s, a=ALPHA):
    n = len(s)
    return np.quantile(s, min(np.ceil((n+1)*(1-a))/n, 1.0), method='higher')
rows, t0 = [], time.time()
for h in EXTRA_HOUSES:
    for app in APPS:
        col = APP_COLS[app][h]
        if col is None: print(f'SKIP house {h} {app}'); continue
        d = raw[h][['Aggregate', col]].dropna()
        mx, thr = APP_PARAMS[app][1], APP_PARAMS[app][0]/APP_PARAMS[app][1]
        x  = ((d['Aggregate'].values-AGG_MEAN)/AGG_STD).astype(np.float32)
        yv = (np.clip(d[col].values,0,mx)/mx).astype(np.float32)
        idxs = np.arange(len(x)-WINDOW+1, dtype=np.int64)[::STRIDE]
        if len(idxs) < 5000: print(f'SKIP house {h} {app}: too short'); continue
        nc = int(len(idxs)*CAL_FRAC)
        dlc = DataLoader(S2P(x,yv,idxs[:nc]),  batch_size=1024, num_workers=2)
        dlt = DataLoader(S2P(x,yv,idxs[nc:]), batch_size=1024, num_workers=2)
        ms = {}
        for s in SEEDS:
            m = Seq2PointCNN().to(DEVICE)
            m.load_state_dict(torch.load(f'{ART}/cnn_{app}_s{s}.pt', map_location=DEVICE))
            ms[s] = m
        preds = {s: predict(ms[s], dlt) for s in SEEDS}
        qcal, ycal = predict(ms[SEEDS[0]], dlc)
        q1, y = preds[SEEDS[0]]
        p = np.mean([preds[s][0][:,1] for s in SEEDS], axis=0)
        on = y > thr; has_on = bool(on.sum() > 50)
        qh = qhat(np.maximum(qcal[:,0]-ycal, ycal-qcal[:,2]))
        lo, hi = q1[:,0]-qh, q1[:,2]+qh
        gc, gt = qcal[:,1] > thr, q1[:,1] > thr
        lom, him = lo.copy(), hi.copy()
        for g in [True, False]:
            msk = gc==g
            if msk.sum() >= 20:
                qg = qhat(np.maximum(qcal[msk,0]-ycal[msk], ycal[msk]-qcal[msk,2]))
                lom[gt==g], him[gt==g] = q1[gt==g,0]-qg, q1[gt==g,2]+qg
        tp=((p>thr)&on).sum(); fp=((p>thr)&~on).sum(); fn=((p<=thr)&on).sum()
        rows.append(dict(house=h, app=app, col=col,
            MAE=round(float(np.abs(y-p).mean()*mx),1),
            F1=round(float(2*tp/max(2*tp+fp+fn,1)),3),
            raw_on=round(float(((y[on]>=q1[on,0])&(y[on]<=q1[on,2])).mean()),3) if has_on else np.nan,
            CQR_cov=round(float(((y>=lo)&(y<=hi)).mean()),3),
            CQR_on=round(float(((y[on]>=lo[on])&(y[on]<=hi[on])).mean()),3) if has_on else np.nan,
            M_on=round(float(((y[on]>=lom[on])&(y[on]<=him[on])).mean()),3) if has_on else np.nan,
            M_w=round(float((him-lom).mean()*mx),1)))
        print(rows[-1], f'| {(time.time()-t0)/60:.1f} min')
        del preds, ms
        if DEVICE=='cuda': torch.cuda.empty_cache()
dm = pd.DataFrame(rows); dm.to_csv(f'{ART}/results_multi_house.csv', index=False)
print(dm.to_string(index=False))

import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
SRC = '/kaggle/working'; print('Using:', SRC)
mpl.rcParams.update({
 'font.family':'serif','font.size':9,'axes.labelsize':9,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.6,
 'xtick.major.width':0.6,'ytick.major.width':0.6,
 'figure.dpi':300,'savefig.dpi':600,'savefig.bbox':'tight',
 'savefig.pad_inches':0.02})
APPS = ['kettle','microwave','fridge','dishwasher','washing_machine']
NAME = ['Kettle','Microwave','Fridge','Dishwasher','Washing machine']
MXv  = {'kettle':3100,'microwave':3000,'fridge':400, 'dishwasher':2500,'washing_machine':2500}
fig, axes = plt.subplots(2, 3, figsize=(3.5, 2.7), sharex=True, sharey=True)
ks = np.arange(0, 41); axf = axes.flatten()
for ax, app, nm in zip(axf[:5], APPS, NAME):
    z = np.load(f'{SRC}/preds_cnn_{app}.npz')
    err = np.abs(z['y'] - z['p']) * MXv[app]
    order = np.argsort(-z['sig_ens'])
    mae = np.array([err[order[int(len(order)*k/100):]].mean() for k in ks])
    ax.axhline(100, color='0.75', ls=(0,(4,3)), lw=0.7)
    ax.plot(ks, 100*mae/mae[0], color='#1d3557', lw=1.3, solid_capstyle='round')
    ax.set_title(nm, fontsize=7.5, pad=2)
    ax.set_xlim(0, 40); ax.set_xticks([0, 20, 40])
    ax.set_ylim(0, 115); ax.set_yticks([0, 50, 100])
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(length=2, labelsize=7, labelbottom=True)
axf[5].axis('off')
fig.supylabel('Remaining MAE (%)', fontsize=9, x=0.01)
fig.supxlabel('Windows rejected (%)', fontsize=9, y=0.0)
fig.subplots_adjust(wspace=0.15, hspace=0.85)
fig.savefig(f'{SRC}/fig_risk_coverage.pdf'); plt.show()
r_st = np.load(f'{SRC}/roll_wm_static.npy')
r_on = np.load(f'{SRC}/roll_wm_online.npy')
t = np.arange(len(r_st)) * 48/3600/24
fig, ax = plt.subplots(figsize=(3.5, 2.0))
ax.axhline(0.9, color='0.1', ls=(0,(4,3)), lw=0.9)
ax.plot(t, r_st, color='#457b9d', lw=1.1, label='Static CQR')
ax.plot(t, r_on, color='#e07a5f', lw=1.1, label='Online recalibration')
ax.set_xlabel('Days into test period'); ax.set_ylabel('Rolling coverage')
ax.set_ylim(0.62, 1.01); ax.set_xlim(0, t[-1])
ax.xaxis.grid(True, ls=':', lw=0.5, color='0.85'); ax.set_axisbelow(True)
ax.tick_params(length=2.5)
ax.legend(frameon=False, fontsize=8, loc='lower left', handlelength=1.4, handletextpad=0.4)
fig.savefig(f'{SRC}/fig_online_recal.pdf'); plt.show()
