"""
Gaussian Process Regression pipeline for the Fe-Cr-Ni-Al-Cu manuscript.

"Interfacial sealing and noble reinforcement shift the strength-corrosion
trade-off in high-entropy alloys"

Reproduces Figure 1 (parity plots and model response curves) and
Supplementary Figure S7 (augmentation sensitivity, learning curves, and an
ablation over literature sources), together with the supplementary tables.

Dataset
-------
data/TableS1_dataset.csv -- the verified dataset of Supplementary Table S1:
36 literature entries plus the 6 Fe60Cr17Ni(15-x)Al8Cux alloys synthesised in
this study. Corrosion entries are restricted to measurements made in
3.5 wt.% NaCl. Feature vector: [Cr, Ni, Al, Cu, AnnTemp] in at.% and degC.

Validation
----------
Leave-one-out over the 6 synthesised alloys. The literature entries form the
training background and are never held out; the reported coefficients of
determination are therefore computed over six predictions.

Augmentation
------------
Experimental entries are replicated with 1% multiplicative feature-space
jitter while the target value is held fixed ("target-pinned" replication).
Because this reduces the apparent observation noise, every metric is reported
at replication factors of 0x, 5x, 10x and 30x rather than at a single value.

Known limitation, retained for comparability with the submitted figures:
entries with no post-fabrication heat treatment are encoded as AnnTemp = 25 degC,
which places room temperature on the same kernel axis as 850-1300 degC.

Usage
-----
    pip install -r requirements.txt
    python3 gpr_pipeline.py

Outputs: figures/*.png (300 dpi), tables/*.csv
Runtime: roughly 10-20 minutes, dominated by the learning curves.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (ConstantKernel, Matern, RBF,
                                              DotProduct, WhiteKernel)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

SEED = 42
AUG_LEVELS = [0, 5, 10, 30]
FEATURES = ["Cr", "Ni", "Al", "Cu", "AnnTemp"]

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "TableS1_dataset.csv")
FIGDIR = os.path.join(HERE, "figures")
TABDIR = os.path.join(HERE, "tables")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(TABDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 11.5, "axes.titlesize": 11.5,
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA)
    df["Used_in"] = df["Used_in"].fillna("")
    lit = df[df.Ref != "Present"]
    exp = df[df.Ref == "Present"]

    def block(frame, target, keyword):
        f = frame[frame.Used_in.str.contains(keyword)]
        f = f.dropna(subset=[target])
        X = f[FEATURES].to_numpy(float)
        y = f[target].to_numpy(float)
        return X, y, f

    Xl_c, yl_c, fl_c = block(lit, "Icorr_A_cm2", "corrosion")
    Xe_c, ye_c, _ = block(exp, "Icorr_A_cm2", "corrosion")
    Xl_s, yl_s, _ = block(lit, "YS_MPa", "strength")
    Xe_s, ye_s, _ = block(exp, "YS_MPa", "strength")

    return dict(
        Icorr=dict(Xl=Xl_c, yl=np.log10(yl_c), Xe=Xe_c, ye=np.log10(ye_c),
                   lit_ref=fl_c["Ref"].to_numpy()),
        YS=dict(Xl=Xl_s, yl=yl_s, Xe=Xe_s, ye=ye_s, lit_ref=None),
        exp_names=exp["Alloy"].to_numpy())


D = load_data()

KERNEL_CORR = (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5, length_scale=1.0)
               + WhiteKernel(noise_level=1e-3))
KERNEL_YS = (ConstantKernel(1.0) * RBF(length_scale=np.ones(5))
             + DotProduct(sigma_0=1.0) + WhiteKernel(noise_level=0.1))

MODELS = {
    "Icorr": dict(kernel=KERNEL_CORR, alpha=1e-10, aug_lit=True, pub_aug=30,
                  label=r"log($I_{corr}$) (A/cm$^2$)", name=r"log($I_{corr}$)",
                  plain="log(Icorr)", fmt="{:.3f}"),
    "YS":    dict(kernel=KERNEL_YS, alpha=0.5, aug_lit=False, pub_aug=10,
                  label="Yield strength (MPa)", name="Yield strength",
                  plain="Yield strength", fmt="{:.1f}"),
}
for k, cfg in MODELS.items():
    cfg.update(D[k])


# ---------------------------------------------------------------------------
# CORE
# ---------------------------------------------------------------------------
def build_training_set(Xl, yl, Xe, ye, aug, rng, aug_lit):
    """Literature background plus target-pinned replicates of the experiments."""
    X, y = [], []
    for row, t in zip(Xl, yl):
        X.append(row); y.append(t)
        if aug_lit and aug > 0:                       # one noisy literature copy
            X.append(row * (1 + rng.normal(0, 0.01, 5)))
            y.append(t + rng.normal(0, 0.01))
    for row, t in zip(Xe, ye):
        X.append(row); y.append(t)
        for _ in range(aug):                          # feature jitter, y pinned
            X.append(row * (1 + rng.normal(0, 0.01, 5)))
            y.append(t)
    return np.array(X), np.array(y)


def loocv(cfg, aug, Xl=None, yl=None, n_restarts=20, seed=SEED):
    """Leave-one-out over the 6 synthesised alloys -> (predictions, sigmas)."""
    Xl = cfg["Xl"] if Xl is None else Xl
    yl = cfg["yl"] if yl is None else yl
    Xe, ye = cfg["Xe"], cfg["ye"]
    rng = np.random.default_rng(seed)
    preds, sigmas = [], []
    for i in range(len(Xe)):
        Xtr, ytr = build_training_set(Xl, yl,
                                      np.delete(Xe, i, axis=0),
                                      np.delete(ye, i),
                                      aug, rng, cfg["aug_lit"])
        sc = StandardScaler().fit(Xtr)
        gpr = GaussianProcessRegressor(kernel=cfg["kernel"], alpha=cfg["alpha"],
                                       n_restarts_optimizer=n_restarts,
                                       normalize_y=True, random_state=SEED)
        gpr.fit(sc.transform(Xtr), ytr)
        mu, sd = gpr.predict(sc.transform(Xe[i:i + 1]), return_std=True)
        preds.append(mu[0]); sigmas.append(sd[0])
    return np.array(preds), np.array(sigmas)


def fit_full(cfg, aug, seed=SEED):
    """Fit on all data (used for the response curves of Figure 1c,d)."""
    rng = np.random.default_rng(seed)
    Xtr, ytr = build_training_set(cfg["Xl"], cfg["yl"], cfg["Xe"], cfg["ye"],
                                  aug, rng, cfg["aug_lit"])
    sc = StandardScaler().fit(Xtr)
    gpr = GaussianProcessRegressor(kernel=cfg["kernel"], alpha=cfg["alpha"],
                                   n_restarts_optimizer=20, normalize_y=True,
                                   random_state=SEED)
    gpr.fit(sc.transform(Xtr), ytr)
    return gpr, sc


def metrics(y, p):
    return dict(R2=r2_score(y, p), MAE=mean_absolute_error(y, p),
                RMSE=np.sqrt(mean_squared_error(y, p)))


# ---------------------------------------------------------------------------
# 1. AUGMENTATION SENSITIVITY  ->  Figure S7(a), Table S2
# ---------------------------------------------------------------------------
def sensitivity():
    rows, store = [], {}
    for key, cfg in MODELS.items():
        y = cfg["ye"]
        for aug in AUG_LEVELS:
            p, s = loocv(cfg, aug)
            mt = metrics(y, p)
            rows.append(dict(Model=cfg["plain"], Replication=f"{aug}x",
                             R2=round(mt["R2"], 3), MAE=round(mt["MAE"], 4),
                             RMSE=round(mt["RMSE"], 4),
                             Mean_posterior_sigma=round(s.mean(), 4),
                             Reported_in_main_text=(aug == cfg["pub_aug"])))
            store[(key, aug)] = (p, s)
            print(f"   {cfg['plain']:>14s} {aug:>2d}x -> R2={mt['R2']:+.3f}  "
                  f"MAE={mt['MAE']:.4f}  sigma={s.mean():.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABDIR, "TableS2_augmentation_sensitivity.csv"),
              index=False)
    return df, store


# ---------------------------------------------------------------------------
# 2. LEARNING CURVES  ->  Figure S7(b,c), Table S3
# ---------------------------------------------------------------------------
def learning_curve(key, aug, sizes, n_rep=6):
    cfg = MODELS[key]
    y = cfg["ye"]
    mean, std = [], []
    for n in sizes:
        scores = []
        for r in range(n_rep):
            rng = np.random.default_rng(1000 + r)
            idx = rng.choice(len(cfg["Xl"]), size=n, replace=False)
            p, _ = loocv(cfg, aug, Xl=cfg["Xl"][idx], yl=cfg["yl"][idx],
                         n_restarts=5, seed=1000 + r)
            scores.append(r2_score(y, p))
        mean.append(np.mean(scores)); std.append(np.std(scores))
        print(f"     n_lit={n:>2d}   R2 = {np.mean(scores):+.3f} "
              f"+/- {np.std(scores):.3f}")
    return np.array(mean), np.array(std)


# ---------------------------------------------------------------------------
# 3. ABLATION OVER LITERATURE SOURCES  ->  Figure S7(d), Table S4
# ---------------------------------------------------------------------------
def ablation(key="Icorr"):
    cfg = MODELS[key]
    refs = cfg["lit_ref"]
    uniq = sorted(set(refs))
    subsets = [("none", np.zeros(len(refs), bool))]
    subsets += [(f"{r} only", refs == r) for r in uniq]
    subsets += [("all four", np.ones(len(refs), bool))]

    rows = {}
    for aug in (0, cfg["pub_aug"]):
        for label, mask in subsets:
            p, _ = loocv(cfg, aug, Xl=cfg["Xl"][mask], yl=cfg["yl"][mask],
                         n_restarts=10)
            r2 = r2_score(cfg["ye"], p)
            rows[(label, aug)] = (r2, int(mask.sum()))
            print(f"     {label:>12s} (n={mask.sum():>2d})  {aug:>2d}x -> "
                  f"R2 = {r2:+.3f}")
    df = pd.DataFrame([dict(Subset=l, n_literature=rows[(l, a)][1],
                            Replication=f"{a}x", R2=round(rows[(l, a)][0], 3))
                       for l, _ in subsets for a in (0, cfg["pub_aug"])])
    df.to_csv(os.path.join(TABDIR, "TableS4_source_ablation.csv"), index=False)
    return [l for l, _ in subsets], rows, cfg["pub_aug"]


# ---------------------------------------------------------------------------
# 4. MODEL RESPONSE CURVES  ->  Figure 1(c,d), Table S5
# ---------------------------------------------------------------------------
def response_curves(aug=30):
    cfg = MODELS["Icorr"]
    gpr, sc = fit_full(cfg, aug)

    def predict(cr, ni, al, cu, T):
        x = np.array([[cr, ni, al, cu, T]])
        mu, sd = gpr.predict(sc.transform(x), return_std=True)
        return mu[0], sd[0]

    cu_grid = np.linspace(0.0, 1.5, 61)
    rows_cu = [dict(AnnTemp=T, Cu=cu,
                    log_Icorr=predict(17.0, 15.0 - cu, 8.0, cu, T)[0],
                    sigma=predict(17.0, 15.0 - cu, 8.0, cu, T)[1])
               for T in (900, 1000) for cu in cu_grid]

    T_grid = np.linspace(850, 1050, 61)
    rows_T = [dict(Cu=cu, AnnTemp=T,
                   log_Icorr=predict(17.0, 15.0 - cu, 8.0, cu, T)[0],
                   sigma=predict(17.0, 15.0 - cu, 8.0, cu, T)[1])
              for cu in (0.0, 0.5, 1.0) for T in T_grid]

    dfc, dfT = pd.DataFrame(rows_cu), pd.DataFrame(rows_T)
    dfc.to_csv(os.path.join(TABDIR, "TableS5_Cu_response.csv"), index=False)
    dfT.to_csv(os.path.join(TABDIR, "TableS5_temperature_response.csv"),
               index=False)
    return dfc, dfT


# ---------------------------------------------------------------------------
# 5. FEATURE VARIANCE AUDIT  ->  Table S6
# ---------------------------------------------------------------------------
def feature_audit():
    Xe = MODELS["Icorr"]["Xe"]
    rows = [dict(Feature=f, Minimum=Xe[:, i].min(), Maximum=Xe[:, i].max(),
                 Std_dev=round(Xe[:, i].std(), 4),
                 Constant=bool(Xe[:, i].std() == 0))
            for i, f in enumerate(FEATURES)]
    r = np.corrcoef(Xe[:, 1], Xe[:, 3])[0, 1]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABDIR, "TableS6_feature_audit.csv"), index=False)
    print(df.to_string(index=False))
    print(f"   Pearson r(Ni, Cu) over the six evaluated alloys = {r:+.3f}")
    return df, r


# ---------------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------------
BLUE, RED, GREY = "#1f77b4", "#d62728", "#777777"


def _parity(ax, key, aug, store, panel):
    cfg = MODELS[key]
    y = cfg["ye"]
    p, s = store[(key, aug)]
    mt = metrics(y, p)
    lo = min(y.min(), (p - s).min()); hi = max(y.max(), (p + s).max())
    pad = 0.08 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.45", lw=1.2)
    ax.errorbar(y, p, yerr=s, fmt="o", ms=7, color=BLUE, mec="0.25", mew=0.7,
                ecolor="0.35", elinewidth=1.1, capsize=3, ls="none")
    box = (f"$R^2$ = {mt['R2']:.3f}\nMAE = {cfg['fmt'].format(mt['MAE'])}\n"
           f"mean $\\sigma$ = {cfg['fmt'].format(s.mean())}\nn = {len(y)}")
    ax.text(0.04, 0.96, box, transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, bbox=dict(fc="white", ec="0.7", lw=0.7, pad=4))
    ax.set_xlabel(f"Measured {cfg['label']}")
    ax.set_ylabel(f"GPR predicted {cfg['label']}")
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(panel, loc="left", fontweight="bold")
    ax.grid(alpha=0.25, ls="--")


def figure1(store, dfc, dfT):
    fig, ax = plt.subplots(2, 2, figsize=(9.6, 7.6))
    _parity(ax[0, 0], "Icorr", MODELS["Icorr"]["pub_aug"], store, "(a)")
    _parity(ax[0, 1], "YS", MODELS["YS"]["pub_aug"], store, "(b)")

    cfg = MODELS["Icorr"]
    a = ax[1, 0]
    for T, c in ((900, BLUE), (1000, RED)):
        s = dfc[dfc.AnnTemp == T]
        a.plot(s.Cu, s.log_Icorr, color=c, lw=2.0, label=f"{T} °C")
        a.fill_between(s.Cu, s.log_Icorr - s.sigma, s.log_Icorr + s.sigma,
                       color=c, alpha=0.15)
        msk = cfg["Xe"][:, 4] == T
        a.plot(cfg["Xe"][msk, 3], cfg["ye"][msk], "o", color=c, mec="0.3",
               mew=0.9, ms=8, ls="none", zorder=5)
    a.plot([], [], "o", color="0.75", mec="0.3", label="measured")
    a.set_xlabel("Cu content (at.%)"); a.set_ylabel(cfg["label"])
    a.set_title("(c)", loc="left", fontweight="bold")
    a.legend(frameon=False, fontsize=9); a.grid(alpha=0.25, ls="--")

    b = ax[1, 1]
    for cu, c in ((0.0, GREY), (0.5, BLUE), (1.0, RED)):
        s = dfT[dfT.Cu == cu]
        b.plot(s.AnnTemp, s.log_Icorr, color=c, lw=2.0, label=f"{cu:.1f} at.% Cu")
        b.fill_between(s.AnnTemp, s.log_Icorr - s.sigma, s.log_Icorr + s.sigma,
                       color=c, alpha=0.15)
        msk = cfg["Xe"][:, 3] == cu
        b.plot(cfg["Xe"][msk, 4], cfg["ye"][msk], "o", color=c, mec="0.3",
               mew=0.9, ms=8, ls="none", zorder=5)
    b.set_xlabel("Annealing temperature (°C)"); b.set_ylabel(cfg["label"])
    b.set_title("(d)", loc="left", fontweight="bold")
    b.legend(frameon=False, fontsize=9); b.grid(alpha=0.25, ls="--")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "Figure1_GPR.png"))
    fig.savefig(os.path.join(FIGDIR, "Figure1_GPR.pdf"))
    plt.close(fig)


def figureS7(df_sens, lc, sizes, abl):
    labels, rows, pub = abl
    fig, ax = plt.subplots(2, 2, figsize=(10.2, 8.0))

    a = ax[0, 0]
    for plain, mk, c in (("log(Icorr)", "o", BLUE),
                         ("Yield strength", "s", RED)):
        s = df_sens[df_sens.Model == plain]
        x = range(len(AUG_LEVELS))
        a.plot(x, s.R2, marker=mk, ms=7, lw=1.8, color=c,
               label=MODELS["Icorr" if plain == "log(Icorr)" else "YS"]["name"])
        rep = s[s.Reported_in_main_text]
        if len(rep):
            i = AUG_LEVELS.index(int(rep.Replication.iloc[0].rstrip("x")))
            a.plot([i], rep.R2, marker=mk, ms=14, mfc="none", mec=c, mew=1.6)
    a.set_xticks(range(len(AUG_LEVELS)))
    a.set_xticklabels([f"{v}×" for v in AUG_LEVELS])
    a.axhline(0, color="0.6", ls=":", lw=1)
    a.set_xlabel("Replication factor")
    a.set_ylabel("LOOCV $R^2$ ($n$ = 6)")
    a.text(0.02, 0.05, "open rings: value reported in main text",
           transform=a.transAxes, fontsize=8, color="0.45")
    a.set_title("(a)", loc="left", fontweight="bold")
    a.legend(frameon=False, fontsize=9, loc="center right")
    a.grid(alpha=0.25, ls="--")

    for axis, key, panel in ((ax[0, 1], "Icorr", "(b)"), (ax[1, 0], "YS", "(c)")):
        for aug, (mu, sd) in lc[key].items():
            c = GREY if aug == 0 else BLUE
            axis.plot(sizes[key], mu, marker="o", ms=6, lw=1.8, color=c,
                      label=f"{aug}×")
            axis.fill_between(sizes[key], mu - sd, mu + sd, color=c, alpha=0.18)
        axis.axhline(0, color="0.6", ls=":", lw=1)
        axis.set_xlabel("Number of literature training points")
        axis.set_ylabel("LOOCV $R^2$")
        axis.set_title(panel, loc="left", fontweight="bold")
        axis.text(0.5, 1.01, MODELS[key]["name"], transform=axis.transAxes,
                  ha="center", fontsize=10)
        axis.set_ylim(-1.6, 1.15)
        axis.legend(frameon=False, fontsize=9, loc="lower right")
        axis.grid(alpha=0.25, ls="--")

    d = ax[1, 1]
    x = np.arange(len(labels)); w = 0.38
    for off, aug, c in ((-w / 2, 0, GREY), (w / 2, pub, BLUE)):
        vals = [rows[(l, aug)][0] for l in labels]
        d.bar(x + off, vals, w, color=c, label=f"{aug}×")
    d.axhline(0, color="0.35", lw=1)
    d.set_xticks(x)
    d.set_xticklabels([f"{l}\n($n$={rows[(l, 0)][1]})" for l in labels],
                      fontsize=8.5)
    d.set_ylabel("LOOCV $R^2$")
    d.set_title("(d)", loc="left", fontweight="bold")
    d.text(0.5, 1.01, MODELS["Icorr"]["name"], transform=d.transAxes,
           ha="center", fontsize=10)
    d.legend(frameon=False, fontsize=9)
    d.grid(alpha=0.25, ls="--", axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "FigureS7_validation.png"))
    fig.savefig(os.path.join(FIGDIR, "FigureS7_validation.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Dataset: {DATA}")
    print(f"  corrosion model: {len(MODELS['Icorr']['Xl'])} literature + "
          f"{len(MODELS['Icorr']['Xe'])} synthesised entries")
    print(f"  strength  model: {len(MODELS['YS']['Xl'])} literature + "
          f"{len(MODELS['YS']['Xe'])} synthesised entries")

    print("\n[1/5] Augmentation sensitivity (LOOCV over the 6 synthesised alloys)")
    df_sens, store = sensitivity()

    print("\n[2/5] Learning curves")
    sizes = {"Icorr": [0, 5, 10, 15, 20, 23], "YS": [0, 3, 6, 9, 12, 15]}
    lc = {}
    for key in ("Icorr", "YS"):
        lc[key] = {}
        for aug in (0, MODELS[key]["pub_aug"]):
            print(f"   {MODELS[key]['plain']}  {aug}x:")
            lc[key][aug] = learning_curve(key, aug, sizes[key])
    pd.DataFrame([dict(Model=MODELS[k]['plain'], Replication=f"{a}x",
                       n_literature=n, R2_mean=round(mu, 3), R2_sd=round(sd, 3))
                  for k in lc for a, (m_, s_) in lc[k].items()
                  for n, mu, sd in zip(sizes[k], m_, s_)]
                 ).to_csv(os.path.join(TABDIR, "TableS3_learning_curves.csv"),
                          index=False)

    print("\n[3/5] Ablation over literature sources")
    abl = ablation("Icorr")

    print("\n[4/5] Feature variance audit across the evaluated alloys")
    feature_audit()

    print("\n[5/5] Model response curves")
    dfc, dfT = response_curves(MODELS["Icorr"]["pub_aug"])

    figure1(store, dfc, dfT)
    figureS7(df_sens, lc, sizes, abl)
    print(f"\nDone.\n  figures -> {FIGDIR}\n  tables  -> {TABDIR}")
