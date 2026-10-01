# HEA-GPR — Fe–Cr–Ni–Al–Cu high-entropy alloys

Machine-learning analysis supporting *"Interfacial sealing and noble
reinforcement shift the strength–corrosion trade-off in high-entropy alloys"*.

Running `gpr_pipeline.py` regenerates **Figure 1** and **Supplementary Figure S7**
of the manuscript, together with Supplementary Tables S2–S6, from the dataset of
Supplementary Table S1.

## Contents

| Path | Purpose |
|---|---|
| `data/TableS1_dataset.csv` | The verified dataset: 36 literature entries + the 6 alloys synthesised in this study |
| `gpr_pipeline.py` | Full analysis and all figures |
| `requirements.txt` | Pinned package versions |
| `figures/` | Generated figures (PNG, 300 dpi, and PDF) |
| `tables/` | Generated CSV tables |

## Reproducing

```bash
pip install -r requirements.txt
python3 gpr_pipeline.py
```

Runtime is roughly 10–20 minutes, dominated by the learning curves.

Results depend on the scikit-learn version: the Gaussian Process
hyperparameter optimiser and its restart behaviour have changed between
releases, so a run under a different version will give coefficients of
determination that differ in the third decimal place. Install the pinned
versions in `requirements.txt` to reproduce the published values.

## Method

**Features.** `[Cr, Ni, Al, Cu, AnnTemp]`, in at.% and °C.

**Targets.** `log10(Icorr)` in A cm⁻², and yield strength in MPa. The corrosion
entries are restricted to measurements made in 3.5 wt.% NaCl, since the
corrosion potential and current density are properties of the
material–electrolyte system and cannot be pooled across electrolytes.

**Models.** Gaussian Process Regression. The corrosion model uses a
Matérn (ν = 5/2) kernel with a white-noise term; the strength model uses an
RBF kernel with anisotropic length scales plus a dot-product and white-noise
term. Features are standardised, targets are normalised, and the optimiser is
run from 20 restarts.

**Validation.** Leave-one-out over the 6 synthesised alloys. The literature
entries form the training background and are never held out, so the reported
coefficients of determination are computed over **six predictions**. This is
stated explicitly rather than presented as general predictive accuracy.

**Replication.** Experimental entries are replicated with 1 % multiplicative
feature-space jitter while the target is held fixed. Because this reduces the
apparent observation noise, every metric is reported at replication factors of
0×, 5×, 10× and 30× (Figure S7a, Table S2) rather than at a single value. The
corrosion model is reported at 30× and the strength model at 10× in the main
text; both are marked with open rings in Figure S7(a).

**Scope.** The analysis is a surrogate model of the measured
composition–processing–property surface, used for interpolation within the
range examined and for quantifying the relative influence of Cu content and
annealing temperature. It is not a prospective discovery framework: the six
synthesised alloys are inside the training set. The Cu axis spans 0–1 at.% and
the model cannot be extrapolated beyond it.

## Known limitations, stated rather than hidden

- Entries with no post-fabrication heat treatment are encoded as
  `AnnTemp = 25 °C`, placing room temperature on the same kernel axis as
  850–1300 °C.
- Across the six evaluated alloys, Cr (17.0 at.%) and Al (8.0 at.%) are
  constant and Ni is perfectly anticorrelated with Cu by construction
  (Ni = 15 − Cu, Pearson r = −1.000). A feature-importance ranking over these
  variables is therefore not interpretable, which is why direct model-response
  curves are reported instead of SHAP values. See `tables/TableS6_feature_audit.csv`.
- The literature entries originate from sintered powder-metallurgy, additively
  deposited and as-cast material, and several contain elements outside the
  feature space (Co, Mn, Mo). The ablation in Figure S7(d) shows that they
  define the compositional domain without contributing transferable predictive
  information: equivalent performance is obtained from the six synthesised
  alloys alone.

## Citation

Cite the manuscript and this repository's archived release DOI.
