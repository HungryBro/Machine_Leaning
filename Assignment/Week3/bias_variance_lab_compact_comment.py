# ==============================================================================
# โค้ดแล็บวิเคราะห์ Bias-Variance Decomposition และ Learning Curves (ฉบับอธิบายละเอียดรายบรรทัด)
# ==============================================================================

# นำเข้าไลบรารีระบบปฏิบัติการ (os) เพื่อจัดการตำแหน่งโฟลเดอร์เก็บไฟล์และจัดการเส้นทาง (Path) ต่าง ๆ
import os

# นำเข้าไลบรารีคำนวณทางคณิตศาสตร์และการจัดการอาเรย์ (numpy) โดยตั้งชื่อย่อว่า np
import numpy as np

# นำเข้าไลบรารี matplotlib สำหรับการกำหนดคุณลักษณะของกราฟ
import matplotlib

# สั่งให้ Matplotlib ทำงานแบบไม่แสดงหน้าต่าง (Non-interactive/Agg Backend) เพื่อบันทึกรูปกราฟเป็นไฟล์ได้โดยตรงโดยไม่ต้องเปิดหน้าต่างใหม่
matplotlib.use('Agg')

# นำเข้าโมดูล pyplot จาก matplotlib เพื่อใช้วาดกราฟ ตกแต่งกราฟ และบันทึกไฟล์รูปภาพ โดยใช้ชื่อย่อว่า plt
import matplotlib.pyplot as plt

# ล็อค Seed ของระบบสุ่ม NumPy ให้เป็นเลข 42 เพื่อให้เวลาสุ่มข้อมูลใหม่ ผลลัพธ์จะออกมาเหมือนกันทุกรอบ (สร้างความถูกต้องซ้ำได้)
np.random.seed(42)

# ดึงเอาตำแหน่งโฟลเดอร์ของไฟล์ Python ปัจจุบันนี้เก็บไว้ในตัวแปร BASE_DIR
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# สร้างตำแหน่งโฟลเดอร์เป้าหมายสำหรับเก็บไฟล์รูปภาพที่จะวาด โดยนำโฟลเดอร์ปัจจุบันมารวมกับโฟลเดอร์ชื่อ 'plots'
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

# สร้างโฟลเดอร์เก็บกราฟ (plots) ขึ้นมาจริง ๆ หากโฟลเดอร์นี้ยังไม่มีอยู่ในเครื่องคอมพิวเตอร์ (ไม่สร้างซ้ำหากมีอยู่แล้วเพราะ exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# สร้าง Dictionary เก็บฟังก์ชันเป้าหมาย f(x) สองตัว โดยใช้ Lambda Function (ฟังก์ชันย่อบรรทัดเดียว)
# 'sin(pi*x)' เป็นคีย์เก็บฟังก์ชันหาค่าไซน์ และ 'x^2' เป็นคีย์เก็บฟังก์ชันยกกำลังสอง
TARGETS = {'sin(pi*x)': lambda x: np.sin(np.pi * x), 'x^2': lambda x: x ** 2}

# ลิสต์เก็บชื่อประเภทโมเดลจำลอง 3 รูปแบบที่เราต้องการนำมาเปรียบเทียบหา Bias-Variance และวาดกราฟ
MODELS = ['Constant', 'Linear', 'Linear through origin']

# ลิสต์กำหนดระดับสเปกตรัมของสัญญาณรบกวน (Noise Standard Deviation หรือ σ) ที่ระดับ 0.0 (ไม่มีเลย) และ 0.3
NOISE = [0.0, 0.3]

# กำหนดเฉดสีตามทฤษฎีสีที่เหมาะสำหรับคนตาบอดสี (Okabe-Ito Palette): สีน้ำเงินแทน σ=0.0 และสีส้มแทน σ=0.3
COLORS = {0.0: '#0072B2', 0.3: '#D55E00'}

# ลิสต์เก็บขนาดของชุดข้อมูลฝึกสอน (N) ต่าง ๆ ที่เราต้องการนำมาใช้สร้างและดูทิศทางของกราฟ Learning Curve
N_LIST = [2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100]

# กำหนดขอบเขตสูงสุดในแนวตั้ง (y-axis limit) สำหรับการแสดงผลการประเมินความผิดพลาดของแต่ละฟังก์ชันเป้าหมาย f(x)
YMAX = {'x^2': 0.6, 'sin(pi*x)': 1.0}


# ฟังก์ชันสำหรับเทรน (Fit) และทำนายผล (Predict) ค่าจากโมเดล
# รับค่าชนิดโมเดล (model), ข้อมูลเทรน X, เฉลย y, และตำแหน่งที่ต้องการให้โมเดลทำนายค่า xq (Query Points)
def fit_predict(model, X, y, xq):
    # แปลงตัวแปร X, y, xq ให้เป็น Numpy Array เสมอ เพื่อความสะดวกและรวดเร็วในการคำนวณทางคณิตศาสตร์
    X, y, xq = map(np.asarray, (X, y, xq))
    
    # กรณีโมเดลเป็นโมเดลค่าคงที่ (Constant model: h(x) = b)
    if model == 'Constant':
        # หาค่าเฉลี่ยของ y ทั้งหมดแล้วนำไปสร้างอาร์เรย์ที่มีขนาดและมิติเท่ากับ xq โดยค่าทุกตัวเป็นค่าเฉลี่ยนี้
        return np.full_like(xq, np.mean(y))
        
    # กรณีโมเดลเป็นโมเดลเชิงเส้นปกติที่มีจุดตัดแกน (Linear model: h(x) = w0 + w1*x)
    if model == 'Linear':
        # สร้างเมทริกซ์การออกแบบ (Design Matrix) โดยรวมคอลัมน์เลข 1 (แทนพจน์ w0) และคอลัมน์ X (แทนพจน์ w1*X)
        # จากนั้นหาค่าน้ำหนัก w ด้วยวิธี Normal Equation (กำลังสองน้อยที่สุด) ผ่านคำสั่ง np.linalg.lstsq
        w = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)[0]
        # ส่งค่าทำนายกลับออกไปตามสมการพหุนามอันดับ 1: w[0] (จุดตัดแกน) + w[1] (ความชัน)คูณ xq
        return w[0] + w[1] * xq
        
    # กรณีสุดท้ายคือโมเดลเชิงเส้นผ่านจุดกำเนิด (Linear through origin: h(x) = w*x)
    # หาค่าความชัน w ด้วยการคำนวณกำลังสองน้อยที่สุดโดยจัดรูป X ให้เป็น 2D เมทริกซ์คอลัมน์เดียวแล้วคูณด้วยตำแหน่ง xq
    return np.linalg.lstsq(X.reshape(-1, 1), y, rcond=None)[0] * xq


# ฟังก์ชันจำลองกระบวนการเพื่อแยกส่วนประกอบความผิดพลาดออกมาเป็น Bias² และ Variance
# รับฟังก์ชันเป้าหมาย f, ชนิดโมเดล, จำนวนชุดข้อมูลสุ่ม (n_datasets), และจำนวนจุดทดสอบ (n_test)
def simulate(f, model, n_datasets=50000, n_test=300):
    # สร้างจุดทดสอบเรียงกันอย่างเท่ากันจำนวน 300 จุด ในช่วงตั้งแต่ -1 ถึง 1 เพื่อใช้วัดความแม่นยำของโมเดล
    x_test = np.linspace(-1, 1, n_test)
    
    # สร้างเมทริกซ์เริ่มต้นที่เป็นเลขศูนย์ทั้งหมด ขนาด (50000 แถว, 300 คอลัมน์) เพื่อเตรียมเก็บผลลัพธ์ทำนาย
    preds = np.zeros((n_datasets, n_test))
    
    # วนลูปทดสอบสุ่มข้อมูลทีละชุดจำนวน 50,000 รอบ
    for i in range(n_datasets):
        # สุ่มค่า X จำนวน 2 จุดอย่างสม่ำเสมอในช่วง [-1, 1]
        X = np.random.uniform(-1, 1, 2)
        # คำนวณ Fit โมเดลจากจุดสุ่ม X และค่าเฉลยที่แท้จริง f(X) แล้วพยากรณ์ค่าที่จุด x_test จากนั้นบันทึกในแถวที่ i
        preds[i] = fit_predict(model, X, f(X), x_test)
        
    # หาโมเดลทำนายเฉลี่ย (g_bar) ในแนวตั้ง (หาค่าเฉลี่ยคอลัมน์) และคำนวณส่วนเบี่ยงเบนมาตรฐาน (std) ของผลพยากรณ์
    g_bar, std = preds.mean(axis=0), preds.std(axis=0)
    
    # คำนวณค่า Bias² โดยนำโมเดลทำนายเฉลี่ยมาลบด้วยค่าเป้าหมายจริง ยกกำลังสอง แล้วหาค่าเฉลี่ยทั่วแกน x_test
    bias2 = np.mean((g_bar - f(x_test)) ** 2)
    
    # คำนวณความแปรปรวน (Variance) โดยหาความแตกต่างกำลังสองของโมเดลแต่ละตัวจากโมเดลเฉลี่ยของมัน
    variance = np.mean(np.var(preds, axis=0))
    
    # ส่งค่าความเอนเอียง ความแปรปรวน ความคลาดเคลื่อนภายนอกทั้งหมด (Eout = Bias² + Variance) และอาร์เรย์ผลลัพธ์กลับในรูป Dictionary
    return {'bias2': float(bias2), 'variance': float(variance), 'eout': float(bias2 + variance),
            'g_bar': g_bar.tolist(), 'std': std.tolist(), 'x_test': x_test.tolist()}


# ฟังก์ชันสร้างเส้นข้อมูล Learning Curve (กราฟการเรียนรู้)
# รับเป้าหมาย f, ชนิดโมเดล, รายชื่อจำนวนข้อมูล n_list, ระดับสัญญาณรบกวน sigma, จำนวนชุดข้อมูล, และจำนวนจุดทดสอบ
def learning_curve(f, model, n_list, sigma=0.0, n_datasets=3000, n_test=1000):
    # สร้างพิกัดจุดทดสอบจำนวน 1000 จุด อย่างละเอียดในช่วง [-1, 1]
    x_test = np.linspace(-1, 1, n_test)
    
    # กำหนดลิสต์เปล่าสำหรับบันทึกประวัติข้อผิดพลาดเฉลี่ยฝั่งฝึกสอน (Ein) และนอกขอบเขตฝึกสอน (Eout)
    Ein, Eout = [], []
    
    # วนลูปตามจำนวนขนาดชุดข้อมูลเทรน (n) เช่น จาก 2 ไปถึง 100
    for n in n_list:
        # กำหนดตัวแปรสะสมผลรวมข้อผิดพลาด Ein และ Eout เริ่มต้นเป็น 0.0
        ein_sum = eout_sum = 0.0
        
        # วนลูปสุ่มทดสอบสร้างชุดข้อมูลจำลองจำนวน 3,000 รอบ
        for _ in range(n_datasets):
            # สุ่มข้อมูล X_train จำนวน n จุดจากช่วง [-1, 1]
            X = np.random.uniform(-1, 1, n)
            # หาค่าผลลัพธ์จริง f(X) แล้วทำการบวกเสียงรบกวนแบบสุ่มตามค่าส่วนเบี่ยงเบนมาตรฐาน sigma
            y = f(X) + np.random.normal(0, sigma, n)
            
            # คำนวณ Ein (Error ในชุดข้อมูลฝึก): นำผลทำนายบนตัวเทรนเทียบกับ y แล้วเฉลี่ยกำลังสองความคลาดเคลื่อนสะสมไว้
            ein_sum += np.mean((fit_predict(model, X, y, X) - y) ** 2)
            
            # คำนวณ Eout (Error นอกชุดฝึก): ทำนายผลบน x_test เทียบกับข้อมูลเป้าหมายที่มีการเติม Noise ลงไปด้วย
            eout_sum += np.mean((fit_predict(model, X, y, x_test) - f(x_test) + np.random.normal(0, sigma, n_test)) ** 2)
            
        # หาค่าเฉลี่ยข้อผิดพลาดที่คำนวณได้จาก 3,000 รอบนั้น บันทึกเพิ่มลงในลิสต์ผลรวม Ein และ Eout
        Ein.append(ein_sum / n_datasets)
        Eout.append(eout_sum / n_datasets)
        
    # ส่งลิสต์ Ein และ Eout กลับไปตามลำดับขนาดข้อมูล N
    return Ein, Eout


# พิมพ์ตัวอักษรตกแต่งหัวข้อรายงานผลลัพธ์จำลองสรุปรวมทางสถิติ
print('=' * 80)
print('Summary Table')
print('=' * 80)
# จัดหัวข้อของคอลัมน์ตารางสรุปให้ตรงแนวและชัดเจน
print(f"{'Target':<12} {'Model':<22} {'bias^2':<10} {'variance':<10} {'Eout':<10}")
print('-' * 80)

# สร้างแกนสำหรับการทำนายผลเพื่อไปวาดกราฟเฉลี่ย (500 จุด)
x_plot = np.linspace(-1, 1, 500)

# สร้างพื้นที่สำหรับรูปภาพเฉลี่ยรวมย่อย (2 แถว, 3 คอลัมน์) ขนาดกว้าง 15 นิ้ว สูง 8 นิ้ว
# sharex=True และ sharey=True เพื่อใช้แกน x และ y ร่วมกันสำหรับทุกกราฟ
fig_avg, axes_avg = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

# วนลูปสองชั้น: ชั้นนอกตามฟังก์ชันเป้าหมายจริง (row ย่อยของรูปภาพ)
for row, (target_name, f) in enumerate(TARGETS.items()):
    # ชั้นในตามรูปแบบจำลองชนิดต่าง ๆ (column ย่อยของรูปภาพ)
    for col, model in enumerate(MODELS):
        # รันฟังก์ชันจำลองเพื่อดึงค่าเฉลี่ยของ bias, variance และข้อมูลวาดกราฟ
        sim = simulate(f, model)
        # พิมพ์ตัวเลขผลคำนวณที่จำลองได้ลงในตารางรายงานสรุปทางจอภาพ
        print(f"{target_name:<12} {model:<22} {sim['bias2']:<10.4f} {sim['variance']:<10.4f} {sim['eout']:<10.4f}")

        # อ้างอิงพิกัดกราฟในแถวและคอลัมน์นั้น ๆ เพื่อเตรียมแต่งเติม
        ax = axes_avg[row, col]
        
        # วาดเส้นทึบสีเขียว ความหนา 2 พิกเซล แสดงเป้าหมายจริง f(x)
        ax.plot(x_plot, f(x_plot), 'g-', linewidth=2, label='Target f(x)')
        
        # แปลงข้อมูลโมเดลเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานที่เก็บในรูปลิสต์ให้อยู่ในรูป Numpy Array
        g_bar, std = np.array(sim['g_bar']), np.array(sim['std'])
        
        # วาดเส้นประสีแดง แสดงโมเดลทำนายเฉลี่ย (g_bar)
        ax.plot(sim['x_test'], g_bar, 'r--', linewidth=2, label='g_bar(x)')
        
        # ระบายแถบสีแดงโปร่งแสง (ความเข้มสี 0.3) ตั้งแต่ระดับ (g_bar - 1 std) ถึง (g_bar + 1 std)
        ax.fill_between(sim['x_test'], g_bar - std, g_bar + std,
                        color='red', alpha=0.3, label='±1 std')
        
        # วาดตัวอย่างโมเดลจากการรันตัวแทน 20 ชุดสุ่มข้อมูลเป็นเส้นดำบาง ๆ (alpha=0.2, ความหนา 0.5)
        for _ in range(20):
            X = np.random.uniform(-1, 1, 2)
            ax.plot(x_plot, fit_predict(model, X, f(X), x_plot), 'k-', alpha=0.2, linewidth=0.5)
            
        # ตั้งชื่อหัวข้อกราฟย่อยแต่ละกราฟ เช่น "sin(pi*x) | Linear"
        ax.set_title(f"{target_name} | {model}")
        # ล็อคขอบเขตการดูแกน x ตั้งแต่ -1 ถึง 1
        ax.set_xlim(-1, 1)
        # ล็อคขอบเขตการดูแกน y ตั้งแต่ -4 ถึง 4
        ax.set_ylim(-4, 4)
        # แสดงป้ายคำอธิบายสัญลักษณ์ (Legend) ขนาดอักษร 9 พิกเซล ที่ด้านบนตรงกลางของกราฟ
        ax.legend(fontsize=9, loc='upper center')
        # แสดงเส้นตาราง (Grid) แบบจาง ๆ
        ax.grid(True, alpha=0.3)

# ปรับระยะขอบของภาพและกราฟย่อยต่าง ๆ ให้มีระยะห่างที่พอดี ไม่ซ้อนทับกัน
plt.tight_layout()
# บันทึกรูปกราฟเฉลี่ยทั้งหมดลงในโฟลเดอร์ plots ชื่อรูป 'average_fit.png' ความละเอียด 150 dpi
plt.savefig(os.path.join(PLOTS_DIR, 'average_fit.png'), dpi=150)
print('\nSaved: plots/average_fit.png')

# ==============================================================================
# วนลูปวาดกราฟที่ 2: กราฟเส้นการเรียนรู้ Learning Curves (เปรียบเทียบ Ein และ Eout)
# ==============================================================================

# สร้างรูปภาพสำหรับการเปรียบเทียบการเรียนรู้ (2 แถว, 3 คอลัมน์) ขนาด 15x8 นิ้ว
# sharex=True ใช้แกน x ร่วมกัน และ sharey='row' ล็อคแกน y ร่วมกันเฉพาะในแต่ละแถวเท่านั้น
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey='row')

# วนลูปสลับฟังก์ชันเป้าหมายแถวบนและแถวล่าง
for row, (target_name, f) in enumerate(TARGETS.items()):
    # กำหนดเพดานค่าข้อผิดพลาดแนวตั้งสูงสุดเฉพาะแต่ละฟังก์ชันตาม Dictionary YMAX
    ymax = YMAX[target_name]
    
    # วนลูปตามคอลัมน์โมเดลจำลอง
    for col, model in enumerate(MODELS):
        # เข้าถึงตารางย่อยพิกัด [row, col]
        ax = axes2[row, col]
        
        # วนลูปสร้างเส้นตามระดับสัญญาณรบกวน (0.0 และ 0.3)
        for sigma in NOISE:
            # คำนวณความคลาดเคลื่อนในและนอกแบบจำลองสำหรับขนาดข้อมูล N ตั้งแต่ 2 ถึง 100
            Ein, Eout = learning_curve(f, model, N_LIST, sigma=sigma)
            
            # วาดกราฟข้อผิดพลาดภายใน Ein (เส้นประสีตามระดับเสียงรบกวน) คลุมค่าไม่ให้เกินขอบเขต ymax
            ax.plot(N_LIST, np.clip(Ein, 0, ymax), '--', color=COLORS[sigma], label=f'Ein σ={sigma}', alpha=0.7)
            # วาดกราฟข้อผิดพลาดนอกแบบจำลอง Eout (เส้นทึบสีตามระดับเสียงรบกวน) คลุมค่าไม่ให้เกินขอบเขต ymax
            ax.plot(N_LIST, np.clip(Eout, 0, ymax), '-', color=COLORS[sigma], label=f'Eout σ={sigma}', alpha=0.7)
            
        # ถ้าเป็นแถวสุดท้าย ให้แสดงชื่อฉลากแกน x เป็นตัวอักษร 'n' (จำนวนตัวอย่าง)
        if row == 1:
            ax.set_xlabel('n')
            
        # ถ้าเป็นคอลัมน์ซ้ายสุด ให้แสดงป้ายชื่อแกน y เป็น 'Expected Error'
        if col == 0:
            ax.set_ylabel('Expected Error')
            ax.tick_params(labelleft=True) # แสดงตัวเลขฉลากแกน y ฝั่งซ้าย
        else:
            ax.tick_params(labelleft=False) # ซ่อนตัวเลขแนวตั้งสำหรับกราฟย่อยด้านในเพื่อไม่ให้ซ้อนทับกัน
            
        # แสดงชื่อประเภทกราฟย่อยแสดงฟังก์ชันและรูปแบบ เช่น "x^2 | Constant"
        ax.set_title(f'{target_name} | {model}')
        # กำหนดให้สเกลแกน x เป็นการแสดงผลแบบทวีคูณ (Logarithmic Scale) เพื่อให้ดูแนวโน้มตอน N มีค่าน้อยได้ง่ายขึ้น
        ax.set_xscale('log')
        # ล็อคขอบเขตการดูแกน y ในกรอบตั้งแต่ 0 ถึงค่าสูงสุดขอบที่เก็บไว้ใน ymax
        ax.set_ylim(0, ymax)
        # แสดงป้ายชื่อระบุสัญลักษณ์ของเส้นกราฟที่กึ่งกลางตอนบน
        ax.legend(fontsize=9, loc='upper center')
        # วาดตารางประกอบพื้นหลังจาง ๆ
        ax.grid(True, alpha=0.3)

# ปรับความเหมาะสมของที่ว่างในขอบเขตภาพให้สมดุล
plt.tight_layout()
# บันทึกภาพลงไฟล์ชื่อ 'learning_curve.png' ที่ความละเอียด 150 dpi
plt.savefig(os.path.join(PLOTS_DIR, 'learning_curve.png'), dpi=150)
print('Saved: plots/learning_curve.png')

# พิมพ์ข้อความว่าการประมวลผลทั้งหมดเสร็จสิ้นเรียบร้อย
print('\nDone!')
