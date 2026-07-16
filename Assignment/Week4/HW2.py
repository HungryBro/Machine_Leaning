# Assignment 2: Performance Estimation — resubstitution / holdout / k-fold CV
# target f(x)=sin(pi*x), x~U(-1,1), y=f(x)+noise ; models: Constant, Linear
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

np.random.seed(42) 
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')  #สร้างโฟลเดอร์ไว้เก็บรูปกราฟ
os.makedirs(OUT, exist_ok=True)
WEEK3_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Week3'))
sys.path.insert(0, WEEK3_DIR)
from bias_variance_lab_compact import fit_predict, expected_eout

f = lambda x: np.sin(np.pi * x)    #กำหนดฟังก์ชันจริงของข้อมูล
MODELS = ['Constant', 'Linear']
XG = np.linspace(-1, 1, 4000)   #สร้างจุด 4000 จุดไว้คำนวณ True E_out ให้แม่นยำ       

def gen_data(n, sigma):
    X = np.random.uniform(-1, 1, n)
    return X, f(X) + np.random.normal(0, sigma, n)

def true_eout(model, X, y, sigma): # คำนวณ Out-of-sample Error ที่แท้จริงใช้ 4000 จุดบน XG
    # ใช้นิยามเดียวกับ Week3: signal error + Noise^2
    return expected_eout(model, X, y, f, XG, sigma)

def resub(model, X, y): 
    return np.mean((fit_predict(model, X, y, X) - y) ** 2) #ใช้ข้อมูลเดิมทั้งTrainและTest

def holdout(model, X, y, frac=0.7):
    n = len(X); idx = np.random.permutation(n) #สลับลำดับข้อมูลแบบสุ่ม
    nt = min(max(int(round(frac * n)), 1), n - 1) #คำนวณจำนวนข้อสอบที่จะให้ฝึก
    tr, te = idx[:nt], idx[nt:]  #แบ่งกองข้อมูล
    return np.mean((fit_predict(model, X[tr], y[tr], X[te]) - y[te]) ** 2)

def kfold(model, X, y, k=5):  #แบงข้อมูลเป็น k ส่วนแล้วเฉลี่ย Error ของแต่ละ Fold
    n = len(X); k = max(2, min(k, n))
    folds = np.array_split(np.random.permutation(n), k)
    errs = []
    for i in range(k):
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        prediction = fit_predict(model, X[train_idx], y[train_idx], X[folds[i]])
        errs.append(np.mean((prediction - y[folds[i]]) ** 2))
    return np.mean(errs)

def run(model, n, sigma, frac=0.7, k=5, reps=2000): # ทดลองหลายๆครั้ง
    rows = {name: [] for name in ['true', 'Resub', 'Holdout', 'KFold']}
    for _ in range(reps):
        X, y = gen_data(n, sigma)
        rows['true'].append(true_eout(model, X, y, sigma))
        rows['Resub'].append(resub(model, X, y))
        rows['Holdout'].append(holdout(model, X, y, frac))
        rows['KFold'].append(kfold(model, X, y, k))
    return {name: np.asarray(values, dtype=float) for name, values in rows.items()}

def bias_var(df): #คำนวณ Bias, Variance และ MSE 
    stats = {}
    for method in ['Resub', 'Holdout', 'KFold']:
        diff = df[method] - df['true'] # หาความต่างระหว่างค่าประมาณกับค่าจริง
        stats[method] = {
            'bias': float(diff.mean()),
            'var': float(diff.var(ddof=1)),
            'mse': float(np.mean(diff ** 2)),
        }
    return stats

# ---------- 1) single dataset ----------
print('### 1) Single dataset: estimate vs true E_out ###')
for m in MODELS:
    X, y = gen_data(20, 0.3) #สร้างข้อมูล 20 จุดแล้วเปรียบเทียบ True Resub Holdout KFold เพื่อดูความแตกต่างของแต่ละวิธี
    print(f"{m:9s} true={true_eout(m,X,y,0.3):.3f}  resub={resub(m,X,y):.3f}  "
          f"holdout={holdout(m,X,y):.3f}  kfold={kfold(m,X,y):.3f}")

# ---------- 2) bias / variance / mse ----------
print('\n### 2) Bias / Variance / MSE over many datasets ###') 
dfs = {m: run(m, 20, 0.3, reps=2000) for m in MODELS} # สร้างข้อมูล 2000 datasets สำหรับคำนวณ Bias Variance MSE ของแต่ละวิธี
for m in MODELS:
    print(f'\n{m}')
    for method, values in bias_var(dfs[m]).items():
        print(f"  {method:8s} bias={values['bias']:.4f}  var={values['var']:.4f}  mse={values['mse']:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, m in zip(axes, MODELS):
    methods = ['Resub', 'Holdout', 'KFold']
    err = np.column_stack([dfs[m][method] - dfs[m]['true'] for method in methods])
    ax.boxplot(err, tick_labels=methods, showmeans=True)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_title(m)
    ax.set_xlabel('Estimator')
    ax.set_ylabel('estimate − true Eout')
    ax.grid(alpha=0.3)
    ax.legend(handles=[
        Patch(facecolor='white', edgecolor='black', label='Box = middle 50% (IQR)'),
        Line2D([0], [0], color='#E67E22', lw=2, label='Orange = median'),
        Line2D([0], [0], marker='^', color='none', markerfacecolor='#2CA02C',
               markeredgecolor='#2CA02C', markersize=7, label='Green triangle = mean'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='none',
               markeredgecolor='black', markersize=5, label='Circle = outlier'),
    ], loc='upper right', fontsize=7, framealpha=0.92)
plt.tight_layout(); plt.savefig(f'{OUT}/part2.png', dpi=150); plt.close()

# ---------- 3) effect of holdout split ratio & k ----------
print('\n### 3) Effect of split ratio (holdout) and k (k-fold) ###')
fracs, ks = [0.1, 0.3, 0.5, 0.7, 0.9], [2, 5, 10, 20] #ทดลองแบ่งข้อมูล Holdout เป็น 10%,30%,50%,70%,90% ดูว่า Bias Variance เปลี่ยนอย่างไร 


def sweep(model, kind, values, n=20, sigma=0.3, reps=1000): #สร้างข้อมูลหลายๆชุดแล้วคำนวณ Bias Variance เก็บไว้วาดกราฟ
    data = [gen_data(n, sigma) for _ in range(reps)]
    truth = np.array([true_eout(model, X, y, sigma) for X, y in data])
    out = []
    for v in values:
        est = np.array([holdout(model, X, y, v) if kind == 'frac' else kfold(model, X, y, v)
                         for X, y in data])
        out.append(((est - truth).mean(), est.var(ddof=1)))
    return np.array(out)


fig, axes = plt.subplots(2, 4, figsize=(16, 7))  #การสร้างกราฟ Bias Variance ของ Holdout และ KFold สำหรับแต่ละโมเดล
for row, m in enumerate(MODELS):
    hb, hv = sweep(m, 'frac', fracs).T
    kb, kv = sweep(m, 'k', ks).T
    print(f'\n{m} holdout (frac,bias,var):', list(zip(fracs, hb.round(3).tolist(), hv.round(3).tolist())))
    print(f'{m} kfold  (k,bias,var):   ', list(zip(ks, kb.round(3).tolist(), kv.round(3).tolist())))
    for ax, x, y, title, logy in [
        (axes[row, 0], fracs, hb, f'{m}: Holdout bias vs frac', False),
        (axes[row, 1], fracs, hv, f'{m}: Holdout var vs frac', True),
        (axes[row, 2], ks, kb, f'{m}: K-fold bias vs k', False),
        (axes[row, 3], ks, kv, f'{m}: K-fold var vs k', False)]:
        metric_label = 'Bias (estimate − true Eout)' if 'bias' in title else 'Variance across runs'
        ax.plot(x, y, 'o-', label=metric_label)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Holdout train fraction' if 'Holdout' in title else 'k (number of folds)')
        ax.set_ylabel('Bias (estimate − true)' if 'bias' in title else 'Variance')
        ax.grid(alpha=0.3)
        ax.legend(loc='best', fontsize=7, framealpha=0.92)
        if logy: ax.set_yscale('log')
        if not logy: ax.axhline(0, color='gray', lw=0.6)
plt.tight_layout(); plt.savefig(f'{OUT}/part3.png', dpi=150); plt.close()

# ---------- 4) effect of n and sigma ----------
print('\n### 4) Effect of n and sigma on bias/variance ###')
n_list, sigma_list = [5, 10, 20, 50, 100], [0.0, 0.2, 0.4, 0.6, 0.8] # กำหนดตัวแปรต้นที่ต้องการศึกษา 

fig, axes = plt.subplots(2, 4, figsize=(18, 7))
for row, m in enumerate(MODELS):
    bv_n = [bias_var(run(m, n, 0.3, reps=800)) for n in n_list] # รันการทดลอง (Simulation แบบ List Comprehension)
    bv_s = [bias_var(run(m, 20, s, reps=800)) for s in sigma_list]
    print(f'\n{m} vs n (sigma=0.3):')
    print('  Bias by method:')
    for method in ['Resub', 'Holdout', 'KFold']:
        print(f"    {method:8s}", [round(bv[method]['bias'], 4) for bv in bv_n])
    print(f'\n{m} vs sigma (n=20):')
    print('  Bias by method:')
    for method in ['Resub', 'Holdout', 'KFold']:
        print(f"    {method:8s}", [round(bv[method]['bias'], 4) for bv in bv_s])
    for ax, x, key, ylab, title in [
        (axes[row, 0], n_list, ('bias', bv_n), 'bias', f'{m}: Bias vs n'),
        (axes[row, 1], n_list, ('var', bv_n), 'var', f'{m}: Variance vs n'),
        (axes[row, 2], sigma_list, ('bias', bv_s), 'bias', f'{m}: Bias vs sigma'),
        (axes[row, 3], sigma_list, ('var', bv_s), 'var', f'{m}: Variance vs sigma')]:
        col, bvs = key
        for method in ['Resub', 'Holdout', 'KFold']:
            ax.plot(x, [bv[method][col] for bv in bvs], 'o-', label=method)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Training set size n' if x == n_list else 'Noise level σ')
        ax.set_ylabel('Bias' if col == 'bias' else 'Variance')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        if col == 'bias':
            ax.axhline(0, color='gray', lw=0.6)
plt.tight_layout(); plt.savefig(f'{OUT}/part4.png', dpi=150); plt.close()

print('\nDone! plots saved in plots/')
