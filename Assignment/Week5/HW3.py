"""Week 5: polynomial degree selection and overfitting."""
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
sys.path[:0] = [str(WEEK / "Week3"), str(WEEK / "Week4")]
from bias_variance_lab_compact import EOUT_GRID, SEED, reference_data, sin_target
from HW2 import gen_data

DATA, PLOTS, RESULTS = HERE / "sin experiment", HERE / "plots", HERE / "results"
NS, WEKA_D, SIGMAS = [10, 20, 40, 80], [1, 3, 8], [0.0, 0.3, 0.6]


def design(x, d):
    """X=[1,x,...,x^d], exactly as in the hand calculation."""
    return np.vander(np.asarray(x), d + 1, increasing=True)


def fit(x, y, d):
    """w=(X^T X)^(-1)X^T y; solve avoids explicitly forming the inverse."""
    X = design(x, d)
    with np.errstate(all="ignore"):
        XT_X, XT_y = X.T @ X, X.T @ y
        try:
            w = np.linalg.solve(XT_X, XT_y)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(XT_X) @ XT_y
    return np.nan_to_num(w, nan=0.0, posinf=1e150, neginf=-1e150)


def predict(x, w):
    with np.errstate(all="ignore"):
        y = design(x, len(w) - 1) @ w
    return np.clip(np.nan_to_num(y, nan=1e150, posinf=1e150, neginf=-1e150), -1e150, 1e150)


def mse(y, yhat):
    with np.errstate(all="ignore"):
        return float(np.mean(np.square(np.clip(yhat - y, -1e150, 1e150))))


class JavaRandom:
    """Java/Weka Random(seed), used only to reproduce Weka's folds."""
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
    order = list(range(n))
    if weka:
        rng = JavaRandom(seed)
        for i in range(n, 1, -1):
            j = rng.next_int(i)
            order[i - 1], order[j] = order[j], order[i - 1]
    else:
        order = np.random.default_rng(seed).permutation(n)
    return np.array_split(np.asarray(order), min(max(2, k), n))


def cv_mse(x, y, d, folds):
    errors = []
    all_idx = np.arange(len(x))
    for test in folds:
        train = np.setdiff1d(all_idx, test)
        errors.extend(np.square(predict(x[test], fit(x[train], y[train], d)) - y[test]))
    return float(np.mean(errors))


def true_eout(w, sigma):
    """Eout = E[(g(x)-sin(pi*x))^2] + sigma^2."""
    return mse(sin_target(EOUT_GRID), predict(EOUT_GRID, w)) + sigma ** 2


def evaluate(x, y, D, k, sigma, seed=SEED, weka=False):
    folds = make_folds(len(x), k, 1 if weka else seed, weka)
    rows, weights = [], []
    for d in range(D + 1):
        w = fit(x, y, d)
        weights.append(w)
        rows.append([mse(y, predict(x, w)), cv_mse(x, y, d, folds),
                     true_eout(w, sigma), np.max(np.abs(w))])
    return np.asarray(rows), weights


def simulate(n, sigma, D, k, reps, seed):
    np.random.seed(seed)
    out = []
    for r in range(reps):
        x, y = gen_data(n, sigma)
        out.append(evaluate(x, y, D, k, sigma, seed + r)[0])
    return np.asarray(out)


def save_csv(name, header, rows):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / name, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def supplied_data(D, k):
    metrics, coefficients = [], []
    print("\nSUPPLIED CSV DATA (same formula/folds as Weka; values shown as RMSE)")
    print(f"{'Noise':<10} {'n':>3} {'d':>3} {'Training':>11} {'CV':>11}")
    print("-" * 43)
    for noise in ("noiseless", "noisy"):
        for n in NS:
            a = np.genfromtxt(DATA / f"sin_{noise}_{n}sample.csv", delimiter=",",
                              names=True, encoding="utf-8-sig")
            x, y = a[a.dtype.names[0]], a[a.dtype.names[-1]]
            for d in (value for value in WEKA_D if value <= D):
                table, weights = evaluate(x, y, d, k, 0.0 if noise == "noiseless" else 0.3,
                                          weka=True)
                ein, ecv = table[d, :2]
                metrics.append([noise, n, d, ein, np.sqrt(ein), ecv, np.sqrt(ecv)])
                print(f"{noise:<10} {n:3d} {d:3d} {np.sqrt(ein):11.4f} {np.sqrt(ecv):11.4f}")
                if n in (10, 80) and d in (3, 8):
                    w = list(weights[d]) + [""] * (D + 1 - len(weights[d]))
                    coefficients.append([noise, n, d, *w[:D + 1], np.max(np.abs(weights[d]))])
    save_csv("provided_data_metrics.csv",
             ["noise", "n", "degree", "train_mse", "train_rmse", "cv_mse", "cv_rmse"], metrics)
    save_csv("provided_data_coefficients.csv",
             ["noise", "n", "degree", *[f"w{i}" for i in range(D + 1)], "max_abs_w"],
             coefficients)


def print_table(title, table):
    print(f"\n{title}")
    print(f"{'d':>3} {'Ein':>12} {'Ecv':>12} {'Eout':>12} {'max|w|':>12}")
    print("-" * 55)
    for d, row in enumerate(table):
        print(f"{d:3d}" + "".join(f"{v:12.6g}" for v in row))


def plot_lines(name, title, table, labels, ylabel="MSE", log=False):
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

    print("=" * 72)
    print("WEEK 5: POLYNOMIAL DEGREE SELECTION")
    print(f"target=sin(pi*x), n={a.n}, sigma={a.sigma}, d=0..{a.D}, {a.k}-fold CV")
    print("All errors below are MSE; true Eout = signal error + sigma^2")
    print("=" * 72)
    supplied_data(a.D, a.k)

    x, y = reference_data(a.n, a.sigma, a.seed)
    single, _ = evaluate(x, y, a.D, a.k, a.sigma, a.seed)
    print_table("ONE DATASET", single)
    print(f"Selected degree: training={single[:, 0].argmin()}, CV={single[:, 1].argmin()}, "
          f"true Eout={single[:, 2].argmin()}")
    plot_lines("single_dataset.png", "One dataset", single[:, :3],
               ["Training (Ein)", "k-fold CV (Ecv)", "True Eout"], log=True)

    runs = simulate(a.n, a.sigma, a.D, a.k, a.reps, a.seed)
    mean = runs.mean(0)
    print_table(f"MEAN OF {a.reps} DATASETS", mean)
    print(f"Selected from mean: training={mean[:, 0].argmin()}, CV={mean[:, 1].argmin()}, "
          f"true Eout={mean[:, 2].argmin()}")
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

    sensitivity = []
    print("\nEFFECT OF n AND sigma")
    print(f"{'n':>4} {'sigma':>7} {'best Ein':>9} {'best CV':>8} {'best Eout':>10} "
          f"{'gap@D':>12} {'median max|w|@D':>17}")
    print("-" * 76)
    for sigma in SIGMAS:
        for n in NS:
            cube = simulate(n, sigma, a.D, a.k, a.sensitivity_reps,
                            a.seed + n + round(100 * sigma))
            m = cube.mean(0)
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

