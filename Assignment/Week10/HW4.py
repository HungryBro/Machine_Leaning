"""Assignment 4: Bayes Decision Theory.

แก้ค่าที่ SETTINGS ด้านบน แล้วรันเพียง:  python3 HW4.py
โปรแกรมทำครบ 4 ข้อ พร้อมกราฟ likelihood, posterior และ decision boundary.
"""

from pathlib import Path
import os

# เก็บ Matplotlib cache นอกโฟลเดอร์งาน
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-hw4-cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


# =============================================================================
# SETTINGS: ปรับค่าตรงนี้ แล้วรันเพียง python3 HW4.py
# =============================================================================
# "all", "equal_1d", "unequal_1d", "qda_2d", หรือ "lda_2d"
SCENARIO_TO_RUN = "all"

# "manual" = ใช้ค่าด้านล่าง, "estimate" = สุ่มข้อมูลแล้วประมาณค่าด้วย MLE
PARAMETER_MODE = "manual"
N_SAMPLES = 120
RANDOM_SEED = 42
PRIOR_C0 = 0.50                 # P(C1) = 1 - PRIOR_C0
RUN_SENSITIVITY_DEMO = True     # สร้างกราฟทดลอง n, μ, σ, prior เพิ่ม

# 1D: mu = [mu_C0, mu_C1], sigma = [sigma_C0, sigma_C1]
# equal_1d ต้องมี sigma สองค่าที่เท่ากัน
ONE_D_PARAMETERS = {
    "equal_1d":   {"mu": [-1.5, 1.5], "sigma": [1.0, 1.0]},
    "unequal_1d": {"mu": [-1.0, 1.0], "sigma": [0.55, 1.80]},
}

# 2D: mu มีสองแถว (C0, C1), cov มี covariance ของ C0 และ C1
# LDA ใช้ covariance เดียวกันทั้งสองคลาส
TWO_D_PARAMETERS = {
    "qda_2d": {
        "mu": [[-1.2, -0.8], [1.1, 0.9]],
        "cov": [[[1.25, 0.55], [0.55, 0.80]],
                [[0.65, -0.35], [-0.35, 1.40]]],
    },
    "lda_2d": {
        "mu": [[-1.2, -0.8], [1.1, 0.9]],
        "cov": [[[1.10, 0.45], [0.45, 0.85]],
                [[1.10, 0.45], [0.45, 0.85]]],
    },
}

HERE = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = HERE / "output"
COLORS = ("#20639B", "#ED553B")
REGION_COLORS = ListedColormap(["#dcecf7", "#fde2dc"])
CASES = ("equal_1d", "unequal_1d", "qda_2d", "lda_2d")
LABELS = {
    "equal_1d": "1) 1D Gaussian - equal variance (single threshold)",
    "unequal_1d": "2) 1D Gaussian - unequal variance (quadratic / double threshold)",
    "qda_2d": "3) 2D Gaussian - QDA (different covariance)",
    "lda_2d": "4) 2D Gaussian - LDA (common covariance)",
}


def make_model(case, prior, changed=None):
    """สร้าง model = {mu, cov, prior}; changed ใช้เฉพาะชุดทดลอง demo."""
    if case in ONE_D_PARAMETERS:
        p = {**ONE_D_PARAMETERS[case], **(changed or {})}
        mu = np.asarray(p["mu"], dtype=float)[:, None]
        sigma = np.asarray(p["sigma"], dtype=float)
        cov = np.array([[[s ** 2]] for s in sigma])
    else:
        p = {**TWO_D_PARAMETERS[case], **(changed or {})}
        mu, cov = np.asarray(p["mu"], dtype=float), np.asarray(p["cov"], dtype=float)

    # โจทย์ข้อ 1 และ 4 บังคับใช้ covariance เดียวกัน
    if case in ("equal_1d", "lda_2d"):
        cov[1] = cov[0]
    if not 0 < prior < 1 or np.min(np.linalg.eigvalsh(cov)) <= 0:
        raise ValueError("prior ต้องอยู่ระหว่าง 0 กับ 1 และ covariance ต้อง positive definite")
    return {"mu": mu, "cov": cov, "prior": np.array([prior, 1 - prior])}


def sample(model, n, rng):
    """สุ่ม n จุด โดยสัดส่วนคลาสตาม prior."""
    n0 = min(max(round(n * model["prior"][0]), 2), n - 2)
    X0 = rng.multivariate_normal(model["mu"][0], model["cov"][0], n0)
    X1 = rng.multivariate_normal(model["mu"][1], model["cov"][1], n - n0)
    X = np.vstack((X0, X1))
    y = np.r_[np.zeros(n0, int), np.ones(n - n0, int)]
    order = rng.permutation(n)
    return X[order], y[order]


def estimate(X, y, common_covariance):
    """Maximum Likelihood Estimation ของ mean, covariance และ prior."""
    groups = [X[y == k] for k in (0, 1)]
    mu = np.array([g.mean(axis=0) for g in groups])
    cov = np.array([(g - m).T @ (g - m) / len(g) for g, m in zip(groups, mu)])
    if common_covariance:
        centered = np.vstack([g - m for g, m in zip(groups, mu)])
        cov[:] = centered.T @ centered / len(X)
    cov += np.eye(X.shape[1])[None] * 1e-6  # ป้องกัน matrix เกือบ singular
    return {"mu": mu, "cov": cov, "prior": np.array([len(g) / len(X) for g in groups])}


def posterior(X, model):
    """คืน likelihood, posterior และ class ที่ตัดสิน โดยคำนวณใน log scale."""
    log_like = []
    for mu, cov in zip(model["mu"], model["cov"]):
        d = X - mu
        log_det = np.linalg.slogdet(cov)[1]
        mahal = np.einsum("...i,ij,...j->...", d, np.linalg.inv(cov), d)
        log_like.append(-.5 * (len(mu) * np.log(2 * np.pi) + log_det + mahal))
    log_like = np.column_stack(log_like)
    log_joint = log_like + np.log(model["prior"])
    z = np.exp(log_joint - log_joint.max(axis=1, keepdims=True))
    return np.exp(log_like), z / z.sum(axis=1, keepdims=True), log_joint.argmax(axis=1)


def boundaries_1d(model):
    """แก้ g0(x)-g1(x)=0; ได้ 0, 1 หรือ 2 thresholds."""
    m0, m1 = model["mu"][:, 0]
    v0, v1 = model["cov"][:, 0, 0]
    p0, p1 = model["prior"]
    a = -.5 * (1 / v0 - 1 / v1)
    b = m0 / v0 - m1 / v1
    c = np.log(p0 / p1) - .5 * np.log(v0 / v1) - .5 * (m0 ** 2 / v0 - m1 ** 2 / v1)
    if np.isclose(a, 0):
        return np.array([] if np.isclose(b, 0) else [-c / b])
    roots = np.roots([a, b, c])
    return np.sort(roots[np.isreal(roots)].real)


def limits(model):
    std = np.array([np.sqrt(np.diag(c)) for c in model["cov"]])
    lo = np.min(model["mu"] - 3.5 * std, axis=0)
    hi = np.max(model["mu"] + 3.5 * std, axis=0)
    return lo - .08 * (hi - lo), hi + .08 * (hi - lo)


def plot_1d(model, X, y, title, filename):
    lo, hi = limits(model)
    grid = np.linspace(lo[0], hi[0], 1000)[:, None]
    likelihood, post, decision = posterior(grid, model)
    roots = boundaries_1d(model)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.7))
    for k in (0, 1):
        ax[0].plot(grid[:, 0], likelihood[:, k], color=COLORS[k], lw=2, label=f"p(x | C{k})")
        ax[1].plot(grid[:, 0], post[:, k], color=COLORS[k], lw=2, label=f"P(C{k} | x)")
        ax[0].scatter(X[y == k, 0], np.full((y == k).sum(), -.03 + .02 * k),
                      color=COLORS[k], s=20, alpha=.65, label=f"train C{k}")
    ax[0].set(title="Likelihood", xlabel="x", ylabel="density")
    ax[1].set(title="Posterior probability", xlabel="x", ylim=(-.04, 1.04))
    for k in (0, 1):
        ax[2].fill_between(grid[:, 0], 0, 1, where=decision == k, color=COLORS[k],
                           alpha=.25, label=f"decide C{k}")
    ax[2].plot(grid[:, 0], post[:, 1], color="#555555", lw=1.6, label="P(C1 | x)")
    ax[2].set(title="Decision region / threshold(s)", xlabel="x", ylim=(-.04, 1.04))
    for root in roots:
        for a in ax:
            a.axvline(root, color="#1D3557", ls="--", lw=1.3)
    for a in ax:
        a.grid(alpha=.25)
        a.legend(fontsize=8)
    root_text = ", ".join(f"{r:.3f}" for r in roots) or "none"
    fig.suptitle(f"{title} | threshold(s): {root_text}", y=1.02)
    fig.tight_layout()
    fig.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return roots


def scatter_2d(axis, X, y):
    for k in (0, 1):
        points = X[y == k]
        axis.scatter(points[:, 0], points[:, 1], s=23, color=COLORS[k], edgecolor="white",
                     linewidth=.35, alpha=.78, label=f"train C{k}")


def plot_2d(model, X, y, title, filename):
    lo, hi = limits(model)
    x0, x1 = np.meshgrid(np.linspace(lo[0], hi[0], 250), np.linspace(lo[1], hi[1], 250))
    grid = np.c_[x0.ravel(), x1.ravel()]
    likelihood, post, decision = posterior(grid, model)
    likelihood = likelihood.reshape(x0.shape + (2,))
    post1, decision = post[:, 1].reshape(x0.shape), decision.reshape(x0.shape)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.9), sharex=True, sharey=True)
    for k in (0, 1):
        z = likelihood[:, :, k]
        ax[0].contour(x0, x1, z, levels=np.linspace(z.max()*.1, z.max()*.9, 5),
                      colors=COLORS[k], linewidths=1.4)
    scatter_2d(ax[0], X, y)
    ax[0].set(title="Likelihood contours", xlabel="x1", ylabel="x2")

    image = ax[1].contourf(x0, x1, post1, levels=np.linspace(0, 1, 21), cmap="coolwarm")
    ax[1].contour(x0, x1, post1, levels=[.5], colors="#1D3557", linewidths=2)
    scatter_2d(ax[1], X, y)
    ax[1].set(title="Posterior P(C1 | x)", xlabel="x1")
    fig.colorbar(image, ax=ax[1], fraction=.046, pad=.04)

    ax[2].contourf(x0, x1, decision, levels=[-.5, .5, 1.5], cmap=REGION_COLORS)
    ax[2].contour(x0, x1, post1, levels=[.5], colors="#1D3557", linewidths=2)
    scatter_2d(ax[2], X, y)
    ax[2].set(title="Decision region and boundary", xlabel="x1")
    for a in ax:
        a.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
        a.grid(alpha=.18)
        a.legend(fontsize=8, loc="upper left")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(case, mode, n, prior, tag="", changed=None):
    true_model = make_model(case, prior, changed)
    rng = np.random.default_rng(RANDOM_SEED + sum(map(ord, case + tag)))
    X, y = sample(true_model, n, rng)
    common = case in ("equal_1d", "lda_2d")
    model = true_model if mode == "manual" else estimate(X, y, common)
    filename = OUTPUT_DIRECTORY / f"{case}_{mode}{tag}.png"
    title = f"{LABELS[case]} | mode={mode}, n={n}"

    if model["mu"].shape[1] == 1:
        roots = plot_1d(model, X, y, title, filename)
        boundary = ", ".join(f"{r:.4f}" for r in roots) or "no threshold"
    else:
        plot_2d(model, X, y, title, filename)
        boundary = "curve (QDA)" if case == "qda_2d" else "line (LDA)"
    print(f"{case}: {boundary} -> {filename.name}")


def run_demo():
    """สร้างหลักฐานการเปลี่ยนแปลงของ n, μ, σ และ prior ตามโจทย์."""
    demos = [
        ("equal_1d", "estimate", 20,  .50, "_n20", None),
        ("equal_1d", "estimate", 400, .50, "_n400", None),
        ("equal_1d", "manual",   80,  .50, "_mu_close", {"mu": [-.5, .5], "sigma": [1, 1]}),
        ("equal_1d", "manual",   80,  .50, "_mu_far", {"mu": [-2, 2], "sigma": [1, 1]}),
        ("unequal_1d", "manual", 80,  .50, "_sigma_unequal", {"mu": [-1, 1], "sigma": [.55, 1.8]}),
        ("lda_2d", "manual",    120, .80, "_prior_c0_080", None),
    ]
    print("\nDEMO: n, μ, σ and prior")
    for case, mode, n, prior, tag, changed in demos:
        run(case, mode, n, prior, tag, changed)


def main():
    if SCENARIO_TO_RUN not in ("all", *CASES) or PARAMETER_MODE not in ("manual", "estimate"):
        raise SystemExit("ตรวจ SCENARIO_TO_RUN และ PARAMETER_MODE ใน SETTINGS")
    if N_SAMPLES < 4:
        raise SystemExit("N_SAMPLES ต้องไม่น้อยกว่า 4")
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    for case in CASES if SCENARIO_TO_RUN == "all" else (SCENARIO_TO_RUN,):
        run(case, PARAMETER_MODE, N_SAMPLES, PRIOR_C0)
    if RUN_SENSITIVITY_DEMO:
        run_demo()


if __name__ == "__main__":
    main()
