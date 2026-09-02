"""Assignment 4 - Bayes Decision Theory.

สร้างตัวจำแนกแบบ Bayes สองคลาส 4 กรณีจากสไลด์ Week 10:
  1) 1D Normal, equal variance  -> single (linear) threshold
  2) 1D Normal, unequal variance -> quadratic / possibly double threshold
  3) 2D Normal, unequal covariance -> QDA
  4) 2D Normal, common covariance -> LDA

แก้ไขค่าที่ส่วน "SETTINGS: ปรับค่าตรงนี้" ด้านบนไฟล์ แล้วรันเพียง
``python3 HW4.py`` โดยไม่ต้องสร้าง web playground. ทุกกรณีบันทึกกราฟ
likelihood, posterior และ decision boundary/region ลงโฟลเดอร์ output.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, Optional, Union

# เก็บ font cache ชั่วคราวนอกโฟลเดอร์งาน จึงไม่ใช่ไฟล์ที่ต้องส่ง
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-hw4-cache")

import matplotlib

# ทำให้รันบน terminal หรือเครื่องที่ไม่มีหน้าต่าง GUI ได้
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


HERE = Path(__file__).resolve().parent
EPS = 1e-6
CLASS_COLORS = ("#20639B", "#ED553B")
REGION_COLORS = ListedColormap(["#dcecf7", "#fde2dc"])


# =============================================================================
# SETTINGS: ปรับค่าตรงนี้ แล้วรันเพียง `python3 HW4.py`
# =============================================================================
# เลือก "all" เพื่อรันครบ 4 ข้อ หรือเลือกเพียงข้อเดียว:
# "equal_1d", "unequal_1d", "qda_2d", "lda_2d"
SCENARIO_TO_RUN = "all"

# "manual"   = ใช้พารามิเตอร์ที่กำหนดเอง
# "estimate" = สุ่ม N_SAMPLES จุด แล้วประมาณ mean/covariance/prior จากข้อมูล
PARAMETER_MODE = "manual"
N_SAMPLES = 120
RANDOM_SEED = 42
PRIOR_C0 = 0.50                 # P(C0); P(C1) = 1 - PRIOR_C0

# ถ้าเป็น None โปรแกรมใช้ค่าตั้งต้นที่เหมาะกับแต่ละข้อ
# หากรันเฉพาะ 1D: เช่น MU_C0 = "-1.5", MU_C1 = "1.5", SIGMA_C0 = 1.0
# หากรันเฉพาะ 2D: เช่น MU_C0 = "-1.2,-0.8", MU_C1 = "1.1,0.9"
MU_C0 = None
MU_C1 = None
SIGMA_C0 = None                # ใช้เฉพาะ 1D; ต้องมากกว่า 0
SIGMA_C1 = None                # equal_1d ต้องให้เท่ากับ SIGMA_C0

# covariance 1D: "1.0"; covariance 2D: "a,b,c,d" = [[a,b],[c,d]]
# สำหรับ LDA ให้ระบุ COV_C0 ค่าเดียว เพราะทั้งสองคลาสใช้ covariance เดียวกัน
COV_C0 = None
COV_C1 = None

# True จะสร้างกราฟทดลอง n, mean, sigma และ prior เพิ่มเติมให้ครบงาน
RUN_SENSITIVITY_DEMO = True
OUTPUT_DIRECTORY = HERE / "output"


@dataclass
class GaussianModel:
    """พารามิเตอร์ Gaussian ของสองคลาส C0 และ C1."""

    means: np.ndarray       # shape = (2, d)
    covariances: np.ndarray # shape = (2, d, d)
    priors: np.ndarray      # shape = (2,)

    @property
    def dimension(self) -> int:
        return int(self.means.shape[1])


@dataclass
class RunSettings:
    """ค่าที่นำมาจาก SETTINGS ด้านบน เพื่อส่งให้ฟังก์ชันต่าง ๆ ใช้งาน."""

    scenario: str
    mode: str
    n: int
    seed: int
    prior0: float
    mu0: Optional[str]
    mu1: Optional[str]
    sigma0: Optional[float]
    sigma1: Optional[float]
    cov0: Optional[str]
    cov1: Optional[str]
    demo: bool


def settings_from_top_of_file() -> RunSettings:
    """รวมค่าที่ผู้ใช้แก้ด้านบนให้เป็น object เดียวสำหรับการรันโปรแกรม."""
    return RunSettings(
        scenario=SCENARIO_TO_RUN,
        mode=PARAMETER_MODE,
        n=N_SAMPLES,
        seed=RANDOM_SEED,
        prior0=PRIOR_C0,
        mu0=MU_C0,
        mu1=MU_C1,
        sigma0=SIGMA_C0,
        sigma1=SIGMA_C1,
        cov0=COV_C0,
        cov1=COV_C1,
        demo=RUN_SENSITIVITY_DEMO,
    )


def parse_vector(text: str, dimension: int, name: str) -> np.ndarray:
    """อ่าน '1.0,-0.5' เป็น vector; 1D ก็รับค่าเดี่ยวได้."""
    try:
        values = np.array([float(item.strip()) for item in text.split(",")], dtype=float)
    except ValueError as error:
        raise ValueError(f"{name} ต้องเป็นตัวเลขคั่นด้วย comma") from error
    if values.size != dimension:
        raise ValueError(f"{name} ต้องมี {dimension} ค่า เช่น "
                         f"{'0.0' if dimension == 1 else '0.0,1.0'}")
    return values


def parse_covariance(text: str, dimension: int, name: str) -> np.ndarray:
    """อ่าน covariance แบบ 1D (variance) หรือ 2D 'a,b,c,d'."""
    values = parse_vector(text, dimension * dimension, name)
    covariance = values.reshape(dimension, dimension)
    if not np.allclose(covariance, covariance.T):
        raise ValueError(f"{name} ต้องเป็นเมทริกซ์สมมาตร")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if np.min(eigenvalues) <= 0:
        raise ValueError(f"{name} ต้อง positive definite")
    return covariance


def normalise_priors(prior0: float) -> np.ndarray:
    return np.array([prior0, 1.0 - prior0], dtype=float)


def default_model(scenario: str, prior0: float) -> GaussianModel:
    """ค่าตั้งต้นที่ทำให้เห็น boundary ชัดเจนในแต่ละโจทย์."""
    priors = normalise_priors(prior0)
    if scenario == "equal_1d":
        return GaussianModel(
            means=np.array([[-1.5], [1.5]]),
            covariances=np.array([[[1.0]], [[1.0]]]),
            priors=priors,
        )
    if scenario == "unequal_1d":
        # sigma ต่างกัน จึงมีโอกาสเกิด two thresholds
        return GaussianModel(
            means=np.array([[-1.0], [1.0]]),
            covariances=np.array([[[0.30]], [[2.25]]]),
            priors=priors,
        )
    if scenario == "qda_2d":
        return GaussianModel(
            means=np.array([[-1.2, -0.8], [1.1, 0.9]]),
            covariances=np.array([
                [[1.25, 0.55], [0.55, 0.80]],
                [[0.65, -0.35], [-0.35, 1.40]],
            ]),
            priors=priors,
        )
    if scenario == "lda_2d":
        common_covariance = np.array([[1.10, 0.45], [0.45, 0.85]])
        return GaussianModel(
            means=np.array([[-1.2, -0.8], [1.1, 0.9]]),
            covariances=np.array([common_covariance, common_covariance]),
            priors=priors,
        )
    raise ValueError(f"ไม่รู้จัก scenario: {scenario}")


def override_model(model: GaussianModel, args: RunSettings) -> GaussianModel:
    """แทนค่าตั้งต้นด้วยค่าจาก SETTINGS; ใช้ได้ทั้ง 1D และ 2D."""
    d = model.dimension
    means = model.means.copy()
    covariances = model.covariances.copy()
    if args.mu0 is not None:
        means[0] = parse_vector(args.mu0, d, "mu0")
    if args.mu1 is not None:
        means[1] = parse_vector(args.mu1, d, "mu1")

    # sigma เป็นทางลัดที่อ่านง่ายสำหรับกรณี 1D
    if d == 1:
        if args.sigma0 is not None:
            covariances[0, 0, 0] = args.sigma0 ** 2
        if args.sigma1 is not None:
            covariances[1, 0, 0] = args.sigma1 ** 2
    if args.cov0 is not None:
        covariances[0] = parse_covariance(args.cov0, d, "cov0")
    if args.cov1 is not None:
        covariances[1] = parse_covariance(args.cov1, d, "cov1")

    # ข้อ 1 และ 4 ต้องใช้ covariance เดียวกันเสมอ
    if args.scenario in ("equal_1d", "lda_2d"):
        if args.sigma0 is not None and args.sigma1 is not None and not np.isclose(
            args.sigma0, args.sigma1
        ):
            raise ValueError("equal_1d ต้องให้ sigma0 และ sigma1 เท่ากัน")
        if args.cov0 is not None and args.cov1 is not None and not np.allclose(
            covariances[0], covariances[1]
        ):
            raise ValueError(f"{args.scenario} ต้องให้ cov0 และ cov1 เท่ากัน")
        if args.cov1 is not None and args.cov0 is None:
            covariances[0] = covariances[1]
        covariances[1] = covariances[0]

    return GaussianModel(means=means, covariances=covariances,
                         priors=normalise_priors(args.prior0))


def sample_from_model(model: GaussianModel, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """สุ่มข้อมูล train โดย n คือจำนวนตัวอย่างรวม และ class ratio ตาม prior."""
    n0 = int(round(n * model.priors[0]))
    n0 = min(max(n0, 2), n - 2)  # แต่ละคลาสต้องมีอย่างน้อย 2 จุดเพื่อ estimate covariance
    counts = (n0, n - n0)
    samples = [rng.multivariate_normal(model.means[k], model.covariances[k], counts[k])
               for k in range(2)]
    X = np.vstack(samples)
    y = np.concatenate([np.full(counts[k], k, dtype=int) for k in range(2)])
    order = rng.permutation(n)
    return X[order], y[order]


def estimate_model(X: np.ndarray, y: np.ndarray, common_covariance: bool) -> GaussianModel:
    """MLE: estimate mu, covariance (หาร n) และ prior จากข้อมูล train."""
    d = X.shape[1]
    means, covariances, priors = [], [], []
    centered_by_class = []
    for k in range(2):
        group = X[y == k]
        mean = group.mean(axis=0)
        centered = group - mean
        means.append(mean)
        centered_by_class.append(centered)
        # MLE covariance = 1/n sum (x-mu)(x-mu)^T
        covariances.append(centered.T @ centered / len(group))
        priors.append(len(group) / len(X))
    if common_covariance:
        centered_all = np.vstack(centered_by_class)
        common = centered_all.T @ centered_all / len(X)
        covariances = [common, common.copy()]
    # regularisation เล็กน้อยเพื่อป้องกัน inverse matrix ไม่ได้เมื่อ n น้อย
    covariances = np.asarray(covariances, dtype=float)
    covariances += np.eye(d)[None, :, :] * EPS
    return GaussianModel(np.asarray(means), covariances, np.asarray(priors))


def log_likelihood(X: np.ndarray, model: GaussianModel) -> np.ndarray:
    """log p(x|Ck) ของทุก x และทุกคลาส; ผลลัพธ์ shape = (m, 2)."""
    values = []
    for k in range(2):
        covariance = model.covariances[k]
        difference = X - model.means[k]
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise ValueError("covariance ต้อง positive definite")
        mahalanobis = np.einsum("...i,ij,...j->...", difference,
                                np.linalg.inv(covariance), difference)
        values.append(-0.5 * (model.dimension * np.log(2 * np.pi) + logdet + mahalanobis))
    return np.column_stack(values)


def posterior(X: np.ndarray, model: GaussianModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """คืน likelihood, posterior และ predicted class โดยคำนวณบน log scale."""
    log_like = log_likelihood(X, model)
    log_joint = log_like + np.log(model.priors)
    maximum = log_joint.max(axis=1, keepdims=True)
    joint_shifted = np.exp(log_joint - maximum)
    post = joint_shifted / joint_shifted.sum(axis=1, keepdims=True)
    return np.exp(log_like), post, np.argmax(log_joint, axis=1)


def one_dimensional_boundaries(model: GaussianModel) -> np.ndarray:
    """หา g0(x)-g1(x)=0 แบบ analytic; อาจได้ 0, 1 หรือ 2 thresholds."""
    mu0, mu1 = model.means[:, 0]
    variance0, variance1 = model.covariances[:, 0, 0]
    a = -0.5 * (1.0 / variance0 - 1.0 / variance1)
    b = mu0 / variance0 - mu1 / variance1
    c = (np.log(model.priors[0] / model.priors[1])
         - 0.5 * np.log(variance0 / variance1)
         - 0.5 * (mu0 ** 2 / variance0 - mu1 ** 2 / variance1))
    if np.isclose(a, 0.0):
        roots = [] if np.isclose(b, 0.0) else [-c / b]
    else:
        roots = np.roots([a, b, c])
        roots = [root.real for root in roots if np.isclose(root.imag, 0.0)]
    return np.asarray(sorted(roots), dtype=float)


def axis_limit(model: GaussianModel, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    """ช่วงแกนที่ครอบคลุม mean และ 3.5 sigma ของทั้งสองคลาส."""
    stds = np.array([np.sqrt(np.diag(covariance)) for covariance in model.covariances])
    lower = np.min(model.means - 3.5 * stds, axis=0)
    upper = np.max(model.means + 3.5 * stds, axis=0)
    margin = 0.08 * (upper - lower)
    return lower - margin, upper + margin


def annotate_boundaries_1d(axes: Iterable[plt.Axes], boundaries: np.ndarray) -> None:
    for boundary in boundaries:
        for axis in axes:
            axis.axvline(boundary, color="#1D3557", ls="--", lw=1.3)


def scatter_1d(axis: plt.Axes, X: np.ndarray, y: np.ndarray) -> None:
    jitter = np.where(y == 0, -0.025, 0.025)
    for k in range(2):
        points = X[y == k, 0]
        axis.scatter(points, np.full(len(points), -0.028) + jitter[y == k],
                     s=22, alpha=.65, color=CLASS_COLORS[k], edgecolors="white",
                     linewidth=.35, label=f"train C{k}")


def plot_1d(model: GaussianModel, X: np.ndarray, y: np.ndarray, title: str,
            output: Path) -> np.ndarray:
    lower, upper = axis_limit(model, 1)
    grid = np.linspace(lower[0], upper[0], 1000)[:, None]
    likelihood, post, prediction = posterior(grid, model)
    boundaries = one_dimensional_boundaries(model)

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    for k in range(2):
        axes[0].plot(grid[:, 0], likelihood[:, k], color=CLASS_COLORS[k],
                     lw=2.2, label=f"p(x | C{k})")
        axes[1].plot(grid[:, 0], post[:, k], color=CLASS_COLORS[k],
                     lw=2.2, label=f"P(C{k} | x)")
    scatter_1d(axes[0], X, y)
    axes[0].set(title="Likelihood", xlabel="x", ylabel="density")
    axes[1].set(title="Posterior probability", xlabel="x", ylabel="probability", ylim=(-.04, 1.04))

    axes[2].fill_between(grid[:, 0], 0, 1, where=prediction == 0,
                         color=CLASS_COLORS[0], alpha=.25, label="decide C0")
    axes[2].fill_between(grid[:, 0], 0, 1, where=prediction == 1,
                         color=CLASS_COLORS[1], alpha=.25, label="decide C1")
    axes[2].plot(grid[:, 0], post[:, 1], color="#5B5B5B", lw=1.6,
                 label="P(C1 | x)")
    axes[2].set(title="Decision regions / threshold(s)", xlabel="x", ylabel="class / posterior",
                ylim=(-.04, 1.04))
    annotate_boundaries_1d(axes, boundaries)
    for axis in axes:
        axis.grid(alpha=.23)
        axis.legend(fontsize=8, loc="best")
    boundary_text = ", ".join(f"{value:.3f}" for value in boundaries) or "none"
    figure.suptitle(f"{title} | threshold(s): {boundary_text}", fontsize=13, y=1.02)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return boundaries


def scatter_2d(axis: plt.Axes, X: np.ndarray, y: np.ndarray) -> None:
    for k in range(2):
        points = X[y == k]
        axis.scatter(points[:, 0], points[:, 1], s=23, color=CLASS_COLORS[k],
                     edgecolor="white", linewidth=.35, alpha=.78, label=f"train C{k}")


def plot_2d(model: GaussianModel, X: np.ndarray, y: np.ndarray, title: str,
            output: Path) -> None:
    lower, upper = axis_limit(model, 2)
    x0, x1 = np.meshgrid(np.linspace(lower[0], upper[0], 260),
                         np.linspace(lower[1], upper[1], 260))
    grid = np.column_stack([x0.ravel(), x1.ravel()])
    likelihood, post, prediction = posterior(grid, model)
    likelihood = likelihood.reshape(x0.shape + (2,))
    posterior_c1 = post[:, 1].reshape(x0.shape)
    decision = prediction.reshape(x0.shape)

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.9), sharex=True, sharey=True)
    # 1) likelihood: contour ของ p(x|C0) และ p(x|C1)
    for k in range(2):
        positive = likelihood[:, :, k]
        levels = np.linspace(positive.max() * .10, positive.max() * .90, 5)
        axes[0].contour(x0, x1, positive, levels=levels, colors=CLASS_COLORS[k], linewidths=1.4)
    scatter_2d(axes[0], X, y)
    axes[0].set(title="Likelihood contours", xlabel="x1", ylabel="x2")

    # 2) posterior p(C1|x) พร้อมเส้น equal posterior เป็น boundary
    filled = axes[1].contourf(x0, x1, posterior_c1, levels=np.linspace(0, 1, 21), cmap="coolwarm")
    axes[1].contour(x0, x1, posterior_c1, levels=[0.5], colors="#1D3557", linewidths=2)
    scatter_2d(axes[1], X, y)
    axes[1].set(title="Posterior P(C1 | x)", xlabel="x1")
    figure.colorbar(filled, ax=axes[1], fraction=.046, pad=.04)

    # 3) decision regions; QDA จะเป็นเส้นโค้ง ส่วน LDA จะเป็นเส้นตรง
    axes[2].contourf(x0, x1, decision, levels=[-.5, .5, 1.5], cmap=REGION_COLORS, alpha=.9)
    axes[2].contour(x0, x1, posterior_c1, levels=[0.5], colors="#1D3557", linewidths=2)
    scatter_2d(axes[2], X, y)
    axes[2].set(title="Decision region and boundary", xlabel="x1")
    for axis in axes:
        axis.set_xlim(lower[0], upper[0])
        axis.set_ylim(lower[1], upper[1])
        axis.grid(alpha=.17)
        axis.legend(fontsize=8, loc="upper left")
    figure.suptitle(title, fontsize=13, y=1.02)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def test_accuracy(fitted: GaussianModel, true_model: GaussianModel,
                  rng: np.random.Generator, n_test: int = 10_000) -> float:
    X_test, y_test = sample_from_model(true_model, n_test, rng)
    _, _, prediction = posterior(X_test, fitted)
    return float(np.mean(prediction == y_test))


def format_model(model: GaussianModel) -> str:
    lines = []
    for k in range(2):
        mu = np.array2string(model.means[k], precision=3, suppress_small=True)
        covariance = np.array2string(model.covariances[k], precision=3, suppress_small=True)
        lines.append(f"  C{k}: prior={model.priors[k]:.3f}, mu={mu}, covariance={covariance}")
    return "\n".join(lines)


def run_scenario(scenario: str, args: RunSettings, output_dir: Path,
                 tag: str = "") -> dict[str, object]:
    """ทำหนึ่งข้อ: manual หรือ estimate, สร้างกราฟ และคืนค่าไว้สรุป."""
    true_model = override_model(default_model(scenario, args.prior0), args)
    rng = np.random.default_rng(args.seed + sum(ord(ch) for ch in scenario + tag))
    X, y = sample_from_model(true_model, args.n, rng)
    common = scenario in ("equal_1d", "lda_2d")
    fitted = true_model if args.mode == "manual" else estimate_model(X, y, common)
    labels = {
        "equal_1d": "1) 1D Gaussian - equal variance (single threshold)",
        "unequal_1d": "2) 1D Gaussian - unequal variance (quadratic / double threshold)",
        "qda_2d": "3) 2D Gaussian - QDA (different covariance)",
        "lda_2d": "4) 2D Gaussian - LDA (common covariance)",
    }
    name = f"{scenario}_{args.mode}{tag}.png"
    output = output_dir / name
    title = labels[scenario] + f" | mode={args.mode}, n={args.n}"
    if fitted.dimension == 1:
        boundaries: Union[np.ndarray, str] = one_dimensional_boundaries(fitted)
        plot_1d(fitted, X, y, title, output)
    else:
        boundaries = "curve (QDA)" if scenario == "qda_2d" else "line (LDA)"
        plot_2d(fitted, X, y, title, output)
    accuracy = test_accuracy(fitted, true_model, rng)
    return {
        "scenario": scenario,
        "fitted": fitted,
        "true": true_model,
        "boundaries": boundaries,
        "accuracy": accuracy,
        "image": output,
    }


def print_result(result: dict[str, object], mode: str) -> None:
    print(f"\n[{result['scenario']}] mode={mode}")
    print("Parameters used by classifier:")
    print(format_model(result["fitted"]))  # type: ignore[arg-type]
    boundaries = result["boundaries"]
    if isinstance(boundaries, np.ndarray):
        text = ", ".join(f"{value:.4f}" for value in boundaries) or "no crossing"
        print(f"Decision threshold(s): {text}")
    else:
        print(f"Decision boundary: {boundaries}")
    print(f"Test accuracy (10,000 new points from specified distribution): {result['accuracy']:.3%}")
    print(f"Saved plot: {result['image']}")


def run_demo(args: RunSettings, output_dir: Path) -> None:
    """ทดลองค่าหลัก 4 แบบเพื่อใช้ตอบส่วน "ดูการเปลี่ยนแปลงของตัวแปร"."""
    print("\nDEMO: sensitivity to n, mean separation, sigma, and prior")
    demos = [
        # n เล็ก/ใหญ่: estimate parameter จากตัวอย่าง
        ("equal_1d", "estimate", 20, None, None, None, None, .50, "_n20"),
        ("equal_1d", "estimate", 400, None, None, None, None, .50, "_n400"),
        # mean ใกล้/ไกล: overlap และ accuracy เปลี่ยน
        ("equal_1d", "manual", 80, "-0.5", "0.5", "1.0", "1.0", .50, "_mu_close"),
        ("equal_1d", "manual", 80, "-2.0", "2.0", "1.0", "1.0", .50, "_mu_far"),
        # unequal sigma: แสดง possible double threshold
        ("unequal_1d", "manual", 80, "-1.0", "1.0", "0.55", "1.8", .50, "_sigma_unequal"),
        # prior ทำให้ LDA decision line ขยับ
        ("lda_2d", "manual", 120, None, None, None, None, .80, "_prior_c0_080"),
    ]
    original = vars(args).copy()
    for scenario, mode, n, mu0, mu1, sigma0, sigma1, prior0, tag in demos:
        args.scenario = scenario
        args.mode = mode
        args.n = n
        args.mu0, args.mu1 = mu0, mu1
        args.sigma0, args.sigma1 = (float(sigma0) if sigma0 else None,
                                     float(sigma1) if sigma1 else None)
        args.cov0 = args.cov1 = None
        args.prior0 = prior0
        print_result(run_scenario(scenario, args, output_dir, tag), mode)
    for key, value in original.items():
        setattr(args, key, value)


def validate_settings(settings: RunSettings) -> None:
    """ตรวจค่าที่ตั้งไว้ด้านบนก่อนเริ่มคำนวณ เพื่อให้ error อ่านเข้าใจง่าย."""
    valid_scenarios = {"all", "equal_1d", "unequal_1d", "qda_2d", "lda_2d"}
    if settings.scenario not in valid_scenarios:
        raise ValueError("SCENARIO_TO_RUN ต้องเป็น all, equal_1d, unequal_1d, qda_2d หรือ lda_2d")
    if settings.mode not in {"manual", "estimate"}:
        raise ValueError('PARAMETER_MODE ต้องเป็น "manual" หรือ "estimate"')
    if settings.n < 4:
        raise ValueError("N_SAMPLES ต้องไม่น้อยกว่า 4")
    if not 0 < settings.prior0 < 1:
        raise ValueError("PRIOR_C0 ต้องอยู่ระหว่าง 0 และ 1")
    for name, sigma in (("SIGMA_C0", settings.sigma0), ("SIGMA_C1", settings.sigma1)):
        if sigma is not None and sigma <= 0:
            raise ValueError(f"{name} ต้องมากกว่า 0")


def main() -> None:
    settings = settings_from_top_of_file()
    try:
        validate_settings(settings)
        output_dir = Path(OUTPUT_DIRECTORY).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        scenarios = (["equal_1d", "unequal_1d", "qda_2d", "lda_2d"]
                     if settings.scenario == "all" else [settings.scenario])
        for scenario in scenarios:
            settings.scenario = scenario
            print_result(run_scenario(scenario, settings, output_dir), settings.mode)
        if settings.demo:
            run_demo(settings, output_dir)
    except ValueError as error:
        raise SystemExit(f"SETTINGS ERROR: {error}") from error


if __name__ == "__main__":
    main()
