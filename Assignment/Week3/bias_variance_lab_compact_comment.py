import os
import numpy as np
import matplotlib
# ตั้งค่า matplotlib ให้บันทึกไฟล์ภาพโดยไม่ต้องมี GUI/หน้าต่างแสดงผล
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ล็อคการสุ่มเพื่อให้ผลลัพธ์การสุ่มคงที่ (สร้างผลลัพธ์ซ้ำได้)
np.random.seed(42)

# จัดการเกี่ยวกับเส้นทางจัดเก็บไฟล์
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# กำหนดฟังก์ชันเป้าหมาย f(x) และชนิดของโมเดล
TARGETS = {'sin(pi*x)': lambda x: np.sin(np.pi * x), 'x^2': lambda x: x ** 2}
MODELS = ['Constant', 'Linear', 'Linear through origin']
NOISE = [0.0, 0.3]

# กำหนดสีสำหรับแต่ละระดับเสียงรบกวน (Noise level) และลิสต์ขนาดข้อมูล N สำหรับทำ Learning Curve
COLORS = {0.0: '#0072B2', 0.3: '#D55E00'}
N_LIST = [2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100]
YMAX = {'x^2': 0.6, 'sin(pi*x)': 1.0}


# ฟังก์ชันสำหรับเทรนและทำนายผลการพยากรณ์ของแต่ละโมเดล
def fit_predict(model, X, y, xq):
    X, y, xq = map(np.asarray, (X, y, xq)) # แปลงข้อมูลทั้งหมดเป็น Numpy Array
    
    if model == 'Constant':
        # โมเดลค่าคงที่: คาดเดาเป็นค่าเฉลี่ยของ y เสมอ
        return np.full_like(xq, np.mean(y))
        
    if model == 'Linear':
        # โมเดลเชิงเส้นตรง (h(x) = w0 + w1*x): ฟิตหาจุดตัดและน้ำหนักด้วย Least Squares
        w = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)[0]
        return w[0] + w[1] * xq
        
    # โมเดลเชิงเส้นผ่านจุดกำเนิด (h(x) = w*x)
    return np.linalg.lstsq(X.reshape(-1, 1), y, rcond=None)[0] * xq


# ฟังก์ชันรัน Simulation เพื่อประมาณค่า Bias^2 และ Variance ด้วยวิธี Monte Carlo
def simulate(f, model, n_datasets=50000, n_test=300):
    x_test = np.linspace(-1, 1, n_test)
    preds = np.zeros((n_datasets, n_test))
    
    # วนลูปสุ่มชุดข้อมูลเพื่อมาเทรนและเก็บผลลัพธ์ทำนาย 50,000 รอบ
    for i in range(n_datasets):
        X = np.random.uniform(-1, 1, 2) # สุ่มข้อมูลขนาด N = 2 จุด
        preds[i] = fit_predict(model, X, f(X), x_test)
        
    # หาโมเดลทำนายเฉลี่ย (g_bar) และส่วนเบี่ยงเบนมาตรฐาน (std) ณ แต่ละจุด x_test
    g_bar, std = preds.mean(axis=0), preds.std(axis=0)
    
    # แยกคำนวณ Bias^2 และ Variance
    bias2 = np.mean((g_bar - f(x_test)) ** 2)
    variance = np.mean(np.var(preds, axis=0))
    
    return {'bias2': float(bias2), 'variance': float(variance), 'eout': float(bias2 + variance),
            'g_bar': g_bar.tolist(), 'std': std.tolist(), 'x_test': x_test.tolist()}


# ฟังก์ชันจำลองเพื่อคำนวณ Ein และ Eout สำหรับสร้างกราฟ Learning Curve
def learning_curve(f, model, n_list, sigma=0.0, n_datasets=3000, n_test=1000):
    x_test = np.linspace(-1, 1, n_test)
    Ein, Eout = [], []
    
    for n in n_list:
        ein_sum = eout_sum = 0.0
        # รันจำลองรอบละ 3,000 ชุดข้อมูล เพื่อหาค่าเฉลี่ยข้อผิดพลาดในและนอกโมเดล
        for _ in range(n_datasets):
            X = np.random.uniform(-1, 1, n)
            y = f(X) + np.random.normal(0, sigma, n) # บวก Noise (sigma) ลงในคำตอบของเทรนเซ็ต
            
            # คำนวณความคลาดเคลื่อนกำลังสองเฉลี่ยบนเซ็ตเทรน (Ein)
            ein_sum += np.mean((fit_predict(model, X, y, X) - y) ** 2)
            # คำนวณความคลาดเคลื่อนบนเซ็ตเทสใหม่ที่มี Noise (Eout)
            eout_sum += np.mean((fit_predict(model, X, y, x_test) - f(x_test) + np.random.normal(0, sigma, n_test)) ** 2)
            
        Ein.append(ein_sum / n_datasets)
        Eout.append(eout_sum / n_datasets)
        
    return Ein, Eout


# --- เริ่มต้นประมวลผลหลักและพิมพ์ผลลัพธ์ ---
print('=' * 80)
print('Summary Table')
print('=' * 80)
print(f"{'Target':<12} {'Model':<22} {'bias^2':<10} {'variance':<10} {'Eout':<10}")
print('-' * 80)

x_plot = np.linspace(-1, 1, 500)
fig_avg, axes_avg = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

# รันแยก Bias-Variance และสร้างกราฟ Average Fit (2x3 Subplots)
for row, (target_name, f) in enumerate(TARGETS.items()):
    for col, model in enumerate(MODELS):
        sim = simulate(f, model)
        print(f"{target_name:<12} {model:<22} {sim['bias2']:<10.4f} {sim['variance']:<10.4f} {sim['eout']:<10.4f}")

        # วาดพล็อตเปรียบเทียบ f(x), g_bar, ช่วง std, และตัวอย่างโมเดล 20 เส้นสีดำบาง
        ax = axes_avg[row, col]
        ax.plot(x_plot, f(x_plot), 'g-', linewidth=2, label='Target f(x)')
        g_bar, std = np.array(sim['g_bar']), np.array(sim['std'])
        ax.plot(sim['x_test'], g_bar, 'r--', linewidth=2, label='g_bar(x)')
        ax.fill_between(sim['x_test'], g_bar - std, g_bar + std,
                        color='red', alpha=0.3, label='±1 std')
        for _ in range(20):
            X = np.random.uniform(-1, 1, 2)
            ax.plot(x_plot, fit_predict(model, X, f(X), x_plot), 'k-', alpha=0.2, linewidth=0.5)
        ax.set_title(f"{target_name} | {model}")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-4, 4)
        ax.legend(fontsize=9, loc='upper center')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'average_fit.png'), dpi=150)
print('\nSaved: plots/average_fit.png')

# รันและวาดกราฟสร้างเส้นพล็อต Learning Curve (2x3 Subplots)
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey='row')
for row, (target_name, f) in enumerate(TARGETS.items()):
    ymax = YMAX[target_name]
    for col, model in enumerate(MODELS):
        ax = axes2[row, col]
        for sigma in NOISE:
            Ein, Eout = learning_curve(f, model, N_LIST, sigma=sigma)
            ax.plot(N_LIST, np.clip(Ein, 0, ymax), '--', color=COLORS[sigma], label=f'Ein σ={sigma}', alpha=0.7)
            ax.plot(N_LIST, np.clip(Eout, 0, ymax), '-', color=COLORS[sigma], label=f'Eout σ={sigma}', alpha=0.7)
        if row == 1:
            ax.set_xlabel('n')
        if col == 0:
            ax.set_ylabel('Expected Error')
            ax.tick_params(labelleft=True)
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(f'{target_name} | {model}')
        ax.set_xscale('log')
        ax.set_ylim(0, ymax)
        ax.legend(fontsize=9, loc='upper center')
        ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'learning_curve.png'), dpi=150)
print('Saved: plots/learning_curve.png')

print('\nDone!')