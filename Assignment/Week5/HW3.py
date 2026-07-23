"""Week 5: polynomial degree selection and overfitting."""

# แผนที่เชื่อมโยงกับการคำนวณมือและภาพใน visual_guides
# -----------------------------------------------------------------------------
# 1) visual_guides/Dregree 3 และ Degree 8:
#       สร้าง X = [1, x, x^2, ..., x^d]
#       คำนวณ X^T X และ X^T y
#       แก้สมการปกติ (normal equation) เพื่อหา w
#       ได้โมเดล y_hat = w0 + w1*x + ... + wd*x^d
#
# 2) visual_guides/Training,CV:
#       Training = resubstitution MSE หรือ E_in
#       CV = ค่าเฉลี่ย MSE จาก 10-fold cross-validation หรือ E_cv
#       ตารางในภาพใช้ degree = 1, 3, 8 และ n = 10, 20, 40, 80
#
# 3) HW3.py ทำต่อจากสูตรมือให้ครบ degree d = 0, 1, ..., D โดย default D=8
#       และคำนวณ true E_out, simulation หลายชุดข้อมูล, ผลของ n/sigma
#       รวมถึง max|w| เพื่อดูสัญญาณของ overfitting


import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
# เพิ่ม path ของ Week3 และ Week4 เพื่อ reuse ฟังก์ชันเดิม
sys.path[:0] = [str(WEEK / "Week3"), str(WEEK / "Week4")]
# Week3: sin_target, EOUT_GRID และ reference_data ใช้ target/grid/dataset เดียวกัน
# Week4: gen_data ใช้วิธีสุ่ม x~Uniform(-1,1) และ noise~N(0,sigma^2) เหมือนโจทย์
from bias_variance_lab_compact import EOUT_GRID, SEED, reference_data, sin_target  # pyright: ignore
from HW2 import gen_data  # pyright: ignore

DATA, PLOTS, RESULTS = HERE / "sin experiment", HERE / "plots", HERE / "results"
NS, WEKA_D, SIGMAS = [10, 20, 40, 80], [1, 3, 8], [0.0, 0.3, 0.6]


def design(x, d):
    """สร้าง Design Matrix X=[1,x,...,x^d] ตามภาพคำนวณมือ."""
    # ถ้า x มี n จุด จะได้ X ขนาด n x (d+1)
    # แถวที่ i คือ [1, x_i, x_i^2, ..., x_i^d]
    # เช่น d=3: [1, x_i, x_i^2, x_i^3]
    # เช่น d=8: [1, x_i, x_i^2, ..., x_i^8]
    return np.vander(np.asarray(x), d + 1, increasing=True)


def fit(x, y, d):
    """หา w จาก normal equation เดียวกับขั้นตอนคำนวณมือ."""
    # สูตรมือเขียนเป็น:
    #       w = (X^T X)^(-1) X^T y
    # แต่ในโค้ดใช้ solve(A,b) แทนการสร้าง A^(-1) โดยตรง ซึ่งให้คำตอบเดียวกัน
    X = design(x, d)
    with np.errstate(all="ignore"):
        # X^T X คือเมทริกซ์สมการปกติ
        # ถ้าใช้เลขยกกำลังแทน ภาพคำนวณมือจะได้ว่า:
        #       (X^T X)_(p,q) = sum_i x_i^(p+q) = S_(p+q)
        # ดังนั้น degree 3 ใช้ S0 ถึง S6 และ degree 8 ใช้ S0 ถึง S16
        XT_X, XT_y = X.T @ X, X.T @ y
        # X^T y คือเวกเตอร์ด้านขวาของระบบสมการ
        #       (X^T y)_p = sum_i x_i^p*y_i = T_p
    # โดย p=0,...,d และลำดับสมาชิกคือ [T0,T1,...,Td]
        # จุดตรวจเทียบภาพ Degree 3, noiseless, n=10:
        #   w0=0.02540358, w1=2.67394270, w2=-0.13231034, w3=-2.86976200
        # ดังนั้นค่า w1 ที่คำนวณมือเขียนเป็น 2.67394270 จะตรงกับโค้ด
        # สำหรับ Degree 8 หลักการเหมือนกันทุกประการ แต่ w มี 9 ตัวคือ w0 ถึง w8
        try:
            # แก้ (X^T X)w = X^T y ได้ w0,w1,...,wd
            w = np.linalg.solve(XT_X, XT_y)
        except np.linalg.LinAlgError:
            # ใช้ pseudo-inverse เฉพาะกรณีเมทริกซ์เกือบ singular
            # ซึ่งอาจเกิดได้เมื่อ degree สูงและข้อมูลมีน้อย/รูปแบบ x ไม่เหมาะสม
            w = np.linalg.pinv(XT_X) @ XT_y
    # ป้องกันค่า NaN/Inf จากกรณี numerical instability ไม่ให้ทำให้การจำลองหยุด
    return np.nan_to_num(w, nan=0.0, posinf=1e150, neginf=-1e150)


def predict(x, w):
    # คำนวณ y_hat = Xw โดย X มี degree = len(w)-1
    # ตรงกับการแทนค่า x ลงใน polynomial ที่ได้จากการแก้ระบบสมการ
    with np.errstate(all="ignore"):
        y = design(x, len(w) - 1) @ w
    # clip เป็น safety guard กรณี polynomial degree สูงแกว่งจนเกิดค่ามากผิดปกติ
    return np.clip(np.nan_to_num(y, nan=1e150, posinf=1e150, neginf=-1e150), -1e150, 1e150)


def mse(y, yhat):
    # MSE = (1/n) * sum_i (y_i - y_hat_i)^2
    # ใช้ทั้ง training error และ test error ในแต่ละ fold
    with np.errstate(all="ignore"):
        return float(np.mean(np.square(np.clip(yhat - y, -1e150, 1e150))))


class JavaRandom:
    """Java/Weka Random(seed) สำหรับทำลำดับ fold ให้ใกล้กับ Weka."""
    # Weka ใช้ Java Random ไม่ใช่ NumPy RNG โดยตรง
    # จึงเขียนตัวสร้างเลขสุ่มนี้ไว้สำหรับ supplied CSV/Weka comparison เท่านั้น
    MASK, MULT, ADD = (1 << 48) - 1, 0x5DEECE66D, 0xB

    def __init__(self, seed=1):
        self.seed = (seed ^ self.MULT) & self.MASK

    def next(self, bits):
        self.seed = (self.seed * self.MULT + self.ADD) & self.MASK
        return self.seed >> (48 - bits)

    def next_int(self, bound):
        if bound & (bound - 1) == 0:
            return (bound * self.next(31)) >> 31
        while True:
            bits = self.next(31)
            value = bits % bound
            if bits - value + bound - 1 < 1 << 31:
                return value


def make_folds(n, k, seed=SEED, weka=False):
    # สุ่มลำดับ index ก่อน แล้วแบ่งเป็น k กลุ่ม
    order = list(range(n))
    if weka:
        rng = JavaRandom(seed)
        for i in range(n, 1, -1):
            j = rng.next_int(i)
            order[i - 1], order[j] = order[j], order[i - 1]
    else:
        order = np.random.default_rng(seed).permutation(n)
    # สำหรับ 10-fold:
    #   n=10 -> test fold ละ 1 จุด, train 9 จุด
    #   n=20 -> test fold ละ 2 จุด, train 18 จุด
    #   n=80 -> test fold ละ 8 จุด, train 72 จุด
    # np.array_split ทำให้แต่ละข้อมูลถูกใช้เป็น test ครบหนึ่งครั้ง
    return np.array_split(np.asarray(order), min(max(2, k), n))


def cv_mse(x, y, d, folds):
    # ในแต่ละ fold:
    #   1. ใช้ข้อมูล train ที่เหลือ fit polynomial degree d
    #   2. ทำนายเฉพาะข้อมูล test ของ fold นั้น
    #   3. เก็บ squared error ของ test ทุกจุด
    # เมื่อจบทุก fold จึงเฉลี่ยเป็น E_cv
    errors = []
    all_idx = np.arange(len(x))
    for test in folds:
        train = np.setdiff1d(all_idx, test)
        errors.extend(np.square(predict(x[test], fit(x[train], y[train], d)) - y[test]))
    # ค่าเฉลี่ยนี้เทียบเท่าการรวม squared error ของ test ทุก fold แล้วหารด้วย n
    return float(np.mean(errors))


def true_eout(w, sigma):
    """คำนวณ true E_out โดยประมาณบน grid เดียวกับ Week3."""
    # ตามสูตรมือ/Week3:
    #   E_out = E_x[(g(x)-f(x))^2] + sigma^2
    # โดย f(x)=sin(pi*x) และ noise variance = sigma^2
    # EOUT_GRID เป็นจุด x จำนวนมากใน [-1,1] จึงใช้ค่าเฉลี่ยแทน integral
    # ถ้า noiseless sigma=0 จะเหลือเฉพาะ signal approximation error
    return mse(sin_target(EOUT_GRID), predict(EOUT_GRID, w)) + sigma ** 2


def evaluate(x, y, D, k, sigma, seed=SEED, weka=False):
    # สร้าง fold หนึ่งครั้ง แล้วใช้ fold ชุดเดียวกันกับทุก degree
    # ทำให้การเปรียบเทียบ degree ยุติธรรม เพราะต่างกันเฉพาะความซับซ้อนของโมเดล
    folds = make_folds(len(x), k, 1 if weka else seed, weka)
    rows, weights = [], []
    for d in range(D + 1):
        # fit ด้วยข้อมูลทั้งหมดเพื่อหา training error และ true E_out ของ degree นี้
        w = fit(x, y, d)
        weights.append(w)
        # rows[d] = [E_in, E_cv, true E_out, max|w|]
        rows.append([mse(y, predict(x, w)), cv_mse(x, y, d, folds),
                     true_eout(w, sigma), np.max(np.abs(w))])
    return np.asarray(rows), weights


def simulate(n, sigma, D, k, reps, seed):
    # ทำซ้ำหลายชุดข้อมูลตามโจทย์ เพื่อหา mean E_in, E_cv, E_out และ max|w|
    # gen_data มาจาก Week4: x~Uniform(-1,1), y=sin(pi*x)+N(0,sigma^2)
    np.random.seed(seed)
    out = []
    for r in range(reps):
        x, y = gen_data(n, sigma)
        # เปลี่ยน seed ของการแบ่ง fold เล็กน้อยในแต่ละรอบ
        out.append(evaluate(x, y, D, k, sigma, seed + r)[0])
    return np.asarray(out)


def save_csv(name, header, rows):
    # บันทึกผลลัพธ์ไว้ตรวจสอบ/นำไปสร้างตารางรายงานภายหลัง
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / name, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def supplied_data(D, k):
    # ส่วนนี้ใช้ข้อมูล CSV ที่เตรียมไว้เพื่อเทียบกับตาราง Weka ในภาพ
    # ไม่ได้สุ่มข้อมูลใหม่ และไม่ได้อ่านภาพ Weka เข้ามาในโปรแกรม
    metrics, coefficients = [], []
    print("\nSUPPLIED CSV DATA (same formula/folds as Weka; values shown as RMSE)")
    print(f"{'Noise':<10} {'n':>3} {'d':>3} {'Training':>11} {'CV':>11}")
    print("-" * 43)
    for noise in ("noiseless", "noisy"):
        for n in NS:
            # CSV แต่ละไฟล์มี x และ y ที่ตรงกับกรณี noiseless/noisy และ sample size นั้น
            a = np.genfromtxt(DATA / f"sin_{noise}_{n}sample.csv", delimiter=",",
                              names=True, encoding="utf-8-sig")
            x, y = a[a.dtype.names[0]], a[a.dtype.names[-1]]
            for d in (value for value in WEKA_D if value <= D):
                # weka=True ทำให้การสุ่ม fold ใช้ JavaRandom(seed=1) แบบ Weka
                table, weights = evaluate(x, y, d, k, 0.0 if noise == "noiseless" else 0.3,
                                          weka=True)
                ein, ecv = table[d, :2]
                # ในไฟล์ผลเก็บทั้ง MSE และ RMSE แต่ terminal ตาราง Weka เดิมแสดง RMSE
                metrics.append([noise, n, d, ein, np.sqrt(ein), ecv, np.sqrt(ecv)])
                print(f"{noise:<10} {n:3d} {d:3d} {np.sqrt(ein):11.4f} {np.sqrt(ecv):11.4f}")
                if n in (10, 80) and d in (3, 8):
                    # เก็บ coefficient ของกรณีที่มีภาพคำนวณมือ Degree 3 และ Degree 8
                    w = list(weights[d]) + [""] * (D + 1 - len(weights[d]))
                    coefficients.append([noise, n, d, *w[:D + 1], np.max(np.abs(weights[d]))])
    save_csv("provided_data_metrics.csv",
             ["noise", "n", "degree", "train_mse", "train_rmse", "cv_mse", "cv_rmse"], metrics)
    save_csv("provided_data_coefficients.csv",
             ["noise", "n", "degree", *[f"w{i}" for i in range(D + 1)], "max_abs_w"],
             coefficients)


def print_table(title, table):
    # ตารางหลักของ HW3: ทุกแถวคือ degree d=0,...,D
    # Ein  = training/resubstitution MSE
    # Ecv  = 10-fold cross-validation MSE
    # Eout = true test error ของ target function + noise variance
    # max|w| = ค่าสัมประสิทธิ์ที่มี absolute value มากที่สุด
    print(f"\n{title}")
    print(f"{'d':>3} {'Ein':>12} {'Ecv':>12} {'Eout':>12} {'max|w|':>12}")
    print("-" * 55)
    for d, row in enumerate(table):
        print(f"{d:3d}" + "".join(f"{v:12.6g}" for v in row))


def plot_lines(name, title, table, labels, ylabel="MSE", log=False):
    # กราฟนี้ใช้ดูแนวโน้ม error เมื่อ degree เพิ่ม:
    #   Training มักลดลงเรื่อย ๆ เพราะโมเดลยืดหยุ่นขึ้น
    #   CV/true Eout มักลดก่อนแล้วเพิ่มเมื่อเริ่ม overfit
    #   จุดต่ำสุดของ CV คือ degree ที่เลือกด้วย cross-validation
    fig, ax = plt.subplots(figsize=(8, 5))
    d = np.arange(len(table))
    for values, label in zip(np.asarray(table).T, labels):
        ax.plot(d, np.maximum(values, 1e-15) if log else values, "o-", label=label)
    ax.set(title=title, xlabel="Polynomial degree d", ylabel=ylabel, xticks=d)
    if log:
        ax.set_yscale("log")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / name, dpi=150)
    plt.close(fig)


def main():
    # ค่าเริ่มต้นตามการทดลองหลัก:
    # n=20, sigma=0.3, ทดลอง degree 0..8 และใช้ 10-fold CV
    # ปรับค่าได้จาก terminal เช่น --n 80 --sigma 0.3 --D 8 --k 10
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--sigma", type=float, default=.3)
    p.add_argument("--D", type=int, default=8)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--reps", type=int, default=300)
    p.add_argument("--sensitivity-reps", type=int, default=50)
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args()
    PLOTS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    # sigma คือ standard deviation ของ noise
    # ดังนั้น noise variance ที่ถูกบวกใน true Eout คือ sigma^2
    print("=" * 72)
    print("WEEK 5: POLYNOMIAL DEGREE SELECTION")
    print(f"target=sin(pi*x), n={a.n}, sigma={a.sigma}, d=0..{a.D}, {a.k}-fold CV")
    print("All errors below are MSE; true Eout = signal error + sigma^2")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # ส่วนที่ 1: ข้อมูลจาก CSV ที่เตรียมไว้เพื่อเทียบกับตาราง Weka
    # -------------------------------------------------------------------------
    # ตารางนี้วนครบ noiseless/noisy และ n=10,20,40,80
    # แต่แสดงเฉพาะ degree 1,3,8 เหมือนภาพ Training,CV
    supplied_data(a.D, a.k)

    # -------------------------------------------------------------------------
    # ส่วนที่ 2: ทดลองกับข้อมูลชุดเดียว
    # -------------------------------------------------------------------------
    # reference_data มาจาก Week3 และใช้ RNG/seed เดียวกัน:
    #   x_i ~ Uniform(-1,1)
    #   y_i = sin(pi*x_i) + epsilon_i
    x, y = reference_data(a.n, a.sigma, a.seed)
    single, _ = evaluate(x, y, a.D, a.k, a.sigma, a.seed)
    print_table("ONE DATASET", single)
    # argmin หา degree ที่ให้ error ต่ำสุดของแต่ละวิธี
    # training เลือกจาก E_in, CV เลือกจาก E_cv, true เลือกจาก E_out
    print(f"Selected degree: training={single[:, 0].argmin()}, CV={single[:, 1].argmin()}, "
          f"true Eout={single[:, 2].argmin()}")
    # ใช้ log scale เพราะค่า error/coefficients อาจต่างกันหลายลำดับขนาด
    plot_lines("single_dataset.png", "One dataset", single[:, :3],
               ["Training (Ein)", "k-fold CV (Ecv)", "True Eout"], log=True)

    # -------------------------------------------------------------------------
    # ส่วนที่ 3: simulation หลายชุดข้อมูล
    # -------------------------------------------------------------------------
    # ทำซ้ำ reps ครั้ง แล้วเฉลี่ยแต่ละ column ของ:
    #   [E_in, E_cv, E_out, max|w|]
    # การเฉลี่ยช่วยให้เห็นแนวโน้ม bias/variance ของ degree ชัดกว่าข้อมูลชุดเดียว
    runs = simulate(a.n, a.sigma, a.D, a.k, a.reps, a.seed)
    mean = runs.mean(0)
    print_table(f"MEAN OF {a.reps} DATASETS", mean)
    # เปรียบเทียบ degree ที่ดีที่สุดจากค่าเฉลี่ยของแต่ละ metric
    print(f"Selected from mean: training={mean[:, 0].argmin()}, CV={mean[:, 1].argmin()}, "
          f"true Eout={mean[:, 2].argmin()}")
    # นอกจากดู degree จากค่าเฉลี่ยแล้ว นับความถี่ว่าแต่ละรอบเลือก degree ใด
    for col, name in enumerate(("training", "CV", "true Eout")):
        freq = np.bincount(runs[:, :, col].argmin(1), minlength=a.D + 1) / a.reps
        print(f"{name:>10} selection frequency: " +
              ", ".join(f"d{d}={v:.1%}" for d, v in enumerate(freq) if v))
    save_csv("mean_random_simulation.csv",
             ["degree", "mean_Ein", "mean_Ecv", "mean_Eout", "median_max_abs_w"],
             [[d, *mean[d, :3], np.median(runs[:, d, 3])] for d in range(a.D + 1)])
    plot_lines("mean_errors.png", f"Mean of {a.reps} datasets", mean[:, :3],
               ["Mean Ein", "Mean Ecv", "Mean true Eout"], log=True)
    coefficient = np.column_stack([runs[:, :, 3].mean(0), np.median(runs[:, :, 3], 0)])
    plot_lines("coefficient_size.png", "Coefficient size vs degree", coefficient,
               ["Mean max|w|", "Median max|w|"], "max |coefficient|", True)

    # -------------------------------------------------------------------------
    # ส่วนที่ 4: sensitivity ต่อจำนวนข้อมูล n และ noise sigma
    # -------------------------------------------------------------------------
    # ทดลอง n=10,20,40,80 และ sigma=0,0.3,0.6 ตามโจทย์/ภาพ
    # gap@D = Eout ของ degree สูงสุด - Ein ของ degree สูงสุด
    # gap ใหญ่หมายถึง fit ข้อมูล train ได้ดี แต่ generalize ไปยังข้อมูลใหม่ไม่ดี
    sensitivity = []
    print("\nEFFECT OF n AND sigma")
    print(f"{'n':>4} {'sigma':>7} {'best Ein':>9} {'best CV':>8} {'best Eout':>10} "
          f"{'gap@D':>12} {'median max|w|@D':>17}")
    print("-" * 76)
    for sigma in SIGMAS:
        for n in NS:
            # ใช้หลายชุดข้อมูลในแต่ละคู่ (n,sigma) เพื่อให้การเปรียบเทียบเสถียรขึ้น
            cube = simulate(n, sigma, a.D, a.k, a.sensitivity_reps,
                            a.seed + n + round(100 * sigma))
            m = cube.mean(0)
            # row เก็บ degree ที่ดีที่สุดจาก training, CV, true Eout และตัวชี้วัด overfit
            row = [n, sigma, m[:, 0].argmin(), m[:, 1].argmin(), m[:, 2].argmin(),
                   m[-1, 2] - m[-1, 0], np.median(cube[:, -1, 3])]
            sensitivity.append(row)
            print(f"{n:4d} {sigma:7.2f} {row[2]:9d} {row[3]:8d} {row[4]:10d} "
                  f"{row[5]:12.4g} {row[6]:17.4g}")
    save_csv("n_sigma_summary.csv",
             ["n", "sigma", "best_train_degree", "best_cv_degree", "best_eout_degree",
              "overfit_gap_at_D", "median_max_abs_w_at_D"], sensitivity)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for sigma in SIGMAS:
        rows = np.asarray([r for r in sensitivity if r[1] == sigma])
        # กราฟซ้าย: เมื่อ n เพิ่ม degree ที่ CV เลือกเปลี่ยนอย่างไร
        # กราฟขวา: เมื่อ n เพิ่ม ช่องว่าง Eout-Ein ของ degree 8 ลดลงหรือไม่
        axes[0].plot(rows[:, 0], rows[:, 3], "o-", label=f"sigma={sigma}")
        axes[1].plot(rows[:, 0], np.maximum(rows[:, 5], 1e-15), "o-", label=f"sigma={sigma}")
    axes[0].set(xlabel="n", ylabel="degree selected by CV", xticks=NS)
    axes[1].set(xlabel="n", ylabel="Eout(D)-Ein(D)", yscale="log", xticks=NS)
    for ax in axes:
        ax.grid(alpha=.3)
        ax.legend()
    fig.suptitle("Effect of n and noise on overfitting")
    fig.tight_layout()
    fig.savefig(PLOTS / "n_sigma_summary.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved plots to {PLOTS}\nSaved tables to {RESULTS}")


if __name__ == "__main__":
    main()
