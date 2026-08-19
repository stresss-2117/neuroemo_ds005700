# NeuroEmo QPP Analysis — OpenNeuro ds005700

Quasi-Periodic Pattern (QPP) analysis of emotion processing during an emotion
viewing task ("fe") versus resting state ("rest"), in 40 subjects from the
OpenNeuro dataset [ds005700](https://openneuro.org/datasets/ds005700).

QPPs are recurring, quasi-periodic spatiotemporal BOLD patterns that capture
slow (~0.02–0.05 Hz) global brain dynamics. Instead of looking at *where* the
brain activates on average, QPP analysis finds a recurring template pattern in
each subject's fMRI time series, then measures how strongly and how often that
pattern re-occurs. This project extracts QPP templates per subject and task,
normalizes them to MNI space, summarizes them per atlas region and Yeo7
network, and finally asks **which regions/networks differ between emotion and
rest** and **which of those look like domain-general "cognitive" areas**
(cross-modality agreement).

---

## Repository layout

```
neuroemo_ds005700/
├── scripts/            # Your analysis pipeline (numbered 01→09)
├── reference/          # External/reference code (NOT part of the pipeline)
│   ├── jatin_code/     #   Jatin sachan ReHo / fALFF / GLM / visualize scripts
│   └── notebooks/      #   Dr. Puneet Sir's brain-plot notebooks (PK, Winson)
├── data/
│   ├── raw/            # OpenNeuro ds005700 downloads (BIDS)
│   └── processed/      # cleaned BOLD + brain masks per subject/task
├── results/            # all outputs (see "Outputs" section)
├── logs/               # pipeline.log
├── venv/               # virtual environment
└── requirements.txt    # pinned dependencies
```

`reference/` is kept for provenance only — the scripts there came from other
people (Jatin sir's ReHo/fALFF/GLM code and your prof's brain-plot notebooks)
and their hard-coded paths do not match this pipeline.

---

## Pipeline overview

```
download_01  →  preprocess_02  →  extract_qpp_03  →  group_analysis_04
                                                          │
                    (orchestrated by 05_run_pipeline.py)   │
                                                          ▼
                    ┌─────── normalize to MNI ────────┐
                    │  coreg_normalize_ants.py  (ANTs,     │
                    │  preferred)  or  normalize_to_mni.py │  (fallback)
                    └────────────────────────────────┘
                                                          ▼
              extract_roi_multiatlas  →  plot_roi_brain / plot_qpp_brain
                                                          ▼
              08_roi_stats_significant.py   (ROI-level fe vs rest stats)
                                                          ▼
              07_cognition_emotion.py  |  07_qpp_cognition_overlap.py
              09_overlap.py            |  (cognitive / overlap analyses)
```

| Step | Script | What it does |
|---|---|---|
| 1 | `download_01.py` | Downloads ds005700 (BOLD + JSON for `rest` and `fe` tasks) into `data/raw/`. Skips subjects already downloaded. |
| 2 | `preprocess_02.py` | Per subject/task: slice-timing correction → brain extraction (`compute_epi_mask`) → 6 mm smoothing → bandpass (0.01–0.10 Hz) → CSF/WM/global-signal confound regression → per-voxel z-scoring. Output: clean BOLD + mask in `data/processed/`. |
| 3 | `extract_qpp_03.py` | QPP extraction per subject/task: sliding-window Pearson correlation of a candidate template, iterative template refinement, peak detection, best-of-3 random starts. Outputs: 4D template, peak times, correlation time course, info dict in `results/{task}/`. |
| 4 | `group_analysis_04.py` | Aggregates `qpp_info.npy` across subjects; plots per-subject QPP strength / periodicity summary and writes group CSVs. |
| 5 | `05_run_pipeline.py` | Master pipeline: runs steps 1–4 for a subject range, crash-resilient, logs to `logs/pipeline.log`. |
| 6 | `coreg_normalize_ants.py` | Rigid coregistration of mean BOLD → T1w, then SyN nonlinear T1w → MNI152 (2 mm); composes transforms and warps each QPP template. Output: `*_qpp_template_MNI_ANTs.nii.gz`. |
| 6b | `normalize_to_mni.py` | Fallback normalization (no T1w needed, ds005700 T1w files are broken on OpenNeuro): center-of-mass alignment of mean BOLD to MNI152 + resampling. Output: `*_qpp_template_MNI.nii.gz`. |
| 7 | `extract_roi_multiatlas.py` | Averages the middle QPP frame across subjects per task, extracts mean QPP per ROI for AAL / Schaefer-400 / Yeo7, writes tables to `results/roi_tables/`. |
| 8 | `plot_roi_brain.py` / `plot_qpp_brain.py` / `make_brain_plots.py` | Brain figures: atlas-ROI QPP maps (volume + surface), raw group-average QPP maps, and per-subject brain plots. |
| 9 | `08_roi_stats_significant.py` | Per-ROI paired t-test (fe vs rest) on full-cycle mean \|QPP\| amplitude, BH-FDR across ROIs; paints only significant ROIs (volume + surface figures). |
| 10 | `07_cognition_emotion.py`, `07_qpp_cognition_overlap.py`, `09_overlap.py` | Cognitive analyses: modality overlap and Yeo7 network statistics (see "Cognitive analysis"). |

> Note: the reho/falff/glm scripts in `reference/jatin_code/` are **not** run by
> this pipeline. Scripts `07_cognition_emotion.py` and `09_overlap.py` consume
> their group-level outputs (`reho_analysis/`, `activation_contrast/`,
> `glm_analysis/`), which must be produced separately if you want the
> multimodal overlap analysis.

---

## Parameters

All configuration is done by editing module-level constants inside the
scripts (there is no CLI/config file).

### 05_run_pipeline.py — master pipeline settings

| Constant | Default | Meaning |
|---|---|---|
| `START_SUBJECT` | `1` | First subject to process |
| `END_SUBJECT` | `40` | Last subject (use 5 to test) |
| `TR_REST` | `2.0` | TR of resting-state scan (s) |
| `TR_EMOTION` | `2.0` | TR of emotion-task scan (s) |
| `SKIP_DOWNLOAD` | `False` | Skip downloading (already downloaded) |
| `SKIP_PREPROCESS` | `False` | Skip preprocessing (already done) |
| `SKIP_QPP` | `False` | Skip QPP extraction |

### download_01.py

| Parameter | Value |
|---|---|
| `dataset` | `ds005700` |
| `target_dir` | `data/raw` |
| subject ID format | `sub-{NN}` (zero-padded) |
| files per subject | `task-rest_bold.{nii.gz,json}`, `task-fe_bold.{nii.gz,json}` |
| standalone `__main__` range | sub-01 … sub-05 (the master pipeline downloads all 40) |

### preprocess_02.py

| Parameter | Value |
|---|---|
| TR auto-detect | from header `zooms[3]`; fallback `2.0` if invalid |
| Slice timing | per-slice `np.linspace(0, TR·(1−1/n_slices))`, reference `TR/2`, linear interpolation |
| Brain extraction | nilearn `compute_epi_mask` |
| Spatial smoothing | FWHM `6` mm |
| Bandpass | `0.01` – `0.10` Hz, Butterworth order 2 (`filtfilt`) |
| CSF threshold | 95th percentile of mean volume |
| WM threshold | 70th percentile of mean volume |
| Confound regressors | global signal, CSF, WM, linear trend, constant |
| Normalization | per-voxel z-score within brain mask |

### extract_qpp_03.py — QPP algorithm

| Parameter | Default | Meaning |
|---|---|---|
| `window_sec` | `20` | Template duration (s) |
| `tr` | `2.0` | Repetition time (s) |
| `window_len` | `int(window_sec/tr)` = `10` | Template size (frames) |
| `n_iter` | `10` | Max template-refinement iterations |
| `peak_thresh` | `0.2` | Initial peak correlation threshold (×0.85 each retry, floor `0.05`) |
| `n_starts` | `3` | Random template starts (best score kept) |
| random seed | `42` | Fixed seed for reproducibility |
| min peaks | `3` | Iteration requires ≥3 peaks above threshold |
| peak distance | `window_len // 2` | Minimum frames between peaks |
| convergence | similarity > `0.999` | Stop refining when template stabilizes |
| best-score | `n_peaks × mean(peak corr)` | Start with the highest score wins |

### Normalization scripts

| Parameter | coreg_normalize_ants.py | normalize_to_mni.py |
|---|---|---|
| Tasks | `["rest", "fe"]` | `["rest", "fe"]` |
| Subjects | sub-01 … sub-40 | sub-01 … sub-40 |
| MNI template | 2 mm MNI152 | 2 mm MNI152 |
| Registration | `Rigid` (BOLD→T1w) + `SyN` (T1w→MNI) | center-of-mass translation only |
| Interpolation | linear | continuous |
| Output | `*_qpp_template_MNI_ANTs.nii.gz` | `*_qpp_template_MNI.nii.gz` |

### ROI extraction & plotting

| Parameter | Value |
|---|---|
| Atlases | `aal` (116 ROIs), `schaefer` (400 ROIs, 7 networks, 2 mm), `yeo7` (7 networks, thick) |
| Frame used | middle frame (`n_frames // 2`) of the QPP template |
| Axial slices (`plot_roi_brain`) | 8 cuts `np.linspace(-62, 72)` |
| Color scaling | 98th percentile max, 40th percentile threshold, `cold_hot` cmap |
| Group-average inputs | `results/{task}/sub-XX_qpp_template_MNI_ANTs.nii.gz` |

### Statistical analysis scripts

| Parameter | Value |
|---|---|
| `FDR_ALPHA` | `0.05` (BH-FDR in every stats script) |
| paired test | fe vs rest, per ROI (AAL/Schaefer/Yeo7) or per voxel |
| QPP value convention | full-cycle mean \|QPP amplitude\| per ROI (`07_qpp_*`, `08`) or middle-frame value (`07_cognition_emotion`, `09`) |
| `NEAR_ZERO_ABS_THRESHOLD` | `1e-4` — near-zero amplitude flag |
| `LOW_COVERAGE_FRACTION_THRESHOLD` | `0.5` — network flagged unreliable if >50% subjects near-zero |
| `ROI_COVERAGE_THRESHOLD` (`09`) | `0.10` — ROI "significant" if ≥10% of its voxels are active |
| `NEAR_UNANIMOUS_THRESHOLD` (`09`) | `3` — overlap in ≥3 of 4 modalities ⇒ cognitive candidate |
| Classification heuristic | >10% of voxels/ROIs with near-unanimous agreement ⇒ "cognitive/domain-general" |
| Yeo7 networks | Visual, Somatomotor, DorsalAttention, VentralAttention, Limbic, Frontoparietal, Default |

---

## Outputs

```
results/
├── fe/  rest/  emotion/      per-subject QPP outputs per task
│     sub-XX_qpp_template.nii.gz       4D QPP template (native space)
│     sub-XX_qpp_template_MNI.nii.gz   center-of-mass MNI version
│     sub-XX_qpp_template_MNI_ANTs.nii.gz  ANTs MNI version (used for stats)
│     sub-XX_qpp_peaks.npy             peak frame indices
│     sub-XX_qpp_corr.npy              template correlation time course
│     sub-XX_qpp_info.npy              dict: n_peaks, mean_corr, avg_interval,
│                                      peak_times_s, window_sec, tr, ...
│     sub-XX_qpp_brainplot.png         per-subject brain plot (make_brain_plots)
├── group/
│     group_{rest,emotion}_results.csv  per-subject QPP summary
│     group_qpp_summary.png             3-panel group summary figure
│     group_average_{fe,rest}_MNI_ANTs.nii.gz  group-average QPP map
├── roi_tables/
│     group_{fe,rest,all_tasks}_multiatlas.csv  ROI × Weights per atlas
├── figures/
│     qpp_group/        group_average_*_qpp.png, group_diff_emotion_minus_rest.png,
│                       sub-01_*_qpp.png   (raw voxelwise QPP maps)
│     roi_plots/        group_{task}_{atlas}.png + _surface.png (atlas ROI maps)
│     roi_plots_significant/  *_roi_stats.csv, *_sig_diff.nii.gz,
│                       group_fe_vs_rest_*_SIG(_surface).png
│     yeo7_overlap/     subject_network_qpp_values*.csv, yeo7_network_stats*.csv,
│                       yeo7_classification_*_fullcycle.png
│     multimodal_overlap/  (09_overlap.py outputs when run)
└── emotion/            (reserved; pipeline uses tasks `rest` and `fe`)
```

---

## What the results show

### Emotion task (fe) vs rest — activated regions

The group-average QPP maps (`figures/qpp_group/`) show:

- **Emotion task:** strongest QPP activity in **visual/occipital cortex** and
  **thalamus** (AAL: `Thalamus` (PuA/PuL/Re), `Calcarine_R`, `Occipital_Sup_R`;
  Schaefer: top-5 parcels all in the **Visual** network; Yeo7: **Visual**
  network is the only one with positive group-average weight).
- **Rest:** activity shifts to **cerebellum/vermis** and **default-mode /
  precuneus** (AAL: `Vermis_4_5`, `Vermis_1_2`, `Cerebellum_3_L`; Schaefer:
  `SalVentAttn_Med_2`, `Default_pCunPCC_*`; Yeo7: **Default** network positive).

The emotion-task QPP cycle therefore recruits occipital (visual) circuitry,
while the rest QPP cycle is anchored in cerebellum and default-mode cortex.

### Statistically significant regions (fe vs rest, paired t-test, FDR q<0.05, N=40)

| Atlas | Significant regions |
|---|---|
| **AAL** (6/116) | `Thalamus PuL_L`, `Thalamus PuM_L/R`, `Cingulate_Post_R`, `Temporal_Mid_R`, `Temporal_Sup_R` |
| **Schaefer** (8/400) | `SalVentAttn_TempOccPar_1` (top, t=4.96), 4 somatomotor parcels, `Vis_15`, `DorsAttn_Post_2`, `SalVentAttn_Med_2` |
| **Yeo7** (4/7) | **Default** (t=2.49), **Somatomotor** (t=2.33), **VentralAttention** (t=2.32), **Frontoparietal** (t=2.31); Visual = trend (p=0.059) |

Interpretation: emotion viewing significantly increases QPP engagement
(relative to rest) in **sensorimotor, ventral-attention, temporo-parietal,
thalamic, and default/precuneus regions** — i.e. the emotion paradigm recruits
attention–sensorimotor–DMN circuitry on top of the visual drive.

### What the generic brain plots show

- `qpp_group/group_average_{task}_qpp.png` — voxelwise group mean of the QPP
  template (middle frame) per task; red = positive BOLD phase, blue = negative.
- `qpp_group/group_diff_emotion_minus_rest.png` — direct voxelwise
  fe−rest difference of the group-average QPP.
- `roi_plots/group_{task}_{atlas}.png` — the same QPP weights but parcellated:
  each atlas ROI is filled with its mean QPP weight (cold_hot, 98th-pct clip);
  `_surface` variants map the same values onto the fsaverage cortical surface.
- `roi_plots_significant/group_fe_vs_rest_{atlas}_SIG*.png` — **only
  FDR-significant ROIs** are painted (with their mean fe−rest difference);
  everything else is left as plain anatomy.

---

## Cognitive analysis

The final scripts ask: *which brain systems are engaged during emotional
"thinking" (the emotion task), and which of those are domain-general cognitive
areas rather than emotion-only ones?*

### What the code does

1. **`07_cognition_emotion.py`** — multi-modality overlap at the voxel level.
   It combines group rest-vs-emotion contrasts from **4 modalities — ReHo,
   fALFF, GLM and QPP** (the QPP contrast is computed fresh: paired t-test on
   middle-frame QPP values, FDR-corrected; ReHo/fALFF/GLM maps come from the
   reference pipeline). Each modality's significant map is binarized and summed
   into an **overlap-count map (0–4 modalities agree)**. The map is summarized
   per Yeo7 network, and networks where >10% of voxels show near-unanimous
   agreement (≥ n_modalities − 1) are classified as **cognitive/domain-general**
   candidates; the rest as emotion/modality-specific.

2. **`07_qpp_cognition_overlap.py`** — QPP-only, self-contained version. For
   each subject, task and Yeo7 network it computes a **full-cycle engagement**
   value: the mean absolute QPP amplitude across all template frames within the
   network's voxels (abs is used because a QPP cycle swings high→low and signed
   averaging would cancel out). Then per network: one-sample t-test vs 0 for
   rest and fe, paired fe-vs-rest t-test, BH-FDR across the 7 networks, and a
   4-tier classification (emotion-modulated / trend / no evidence /
   low-coverage). Networks with >50% of subjects at near-zero amplitude are
   flagged unreliable instead of being reported as null results.

3. **`09_overlap.py`** — the same multimodal-overlap idea but at the
   **Schaefer-400 ROI level**: each ROI is "significant" for a modality if ≥10%
   of its voxels are active; the per-ROI overlap count (0–4) is rolled up to
   Yeo7 networks. Networks with >10% of ROIs in near-unanimous agreement
   (overlap ≥ 3) are classified **cognitive (domain-general)**.

### What the results show

- **Full-cycle QPP engagement** is highest in the **VentralAttention**
  network for both tasks (fe mean 0.181, rest 0.161), followed by
  Somatomotor and Visual; Limbic is essentially absent (flagged
  **low-coverage/unreliable** — the Yeo7 limbic atlas has few voxels in these
  scans, so its null result should not be interpreted).
- **Emotion-modulated networks (FDR-corrected fe vs rest):** **Default,
  Somatomotor, VentralAttention, Frontoparietal** — QPP engagement increases in
  all four during the emotion task. Visual and DorsalAttention show
  uncorrected trends.
- **Which areas are activated in cognitive (thinking) processing:** the
  candidate domain-general "cognitive" systems are the **frontoparietal
  (executive/control) network**, the **default-mode network**, and the
  **ventral-attention (salience) network** — i.e., exactly the networks you
  would expect when subjects are cognitively engaged while watching emotional
  stimuli: attention re-orienting (ventral attention), executive control
  (frontoparietal), and self-referential/mnemonic processing (default mode).
  The somatomotor and visual components reflect the perceptual and response
  aspects of the task rather than "thinking" per se.

### What the overlap maps show

- `yeo7_overlap/yeo7_classification_bar_fullcycle.png` and
  `yeo7_classification_brain_fullcycle.png` — per-network fe-vs-rest t-values
  colored by classification (red = emotion-modulated, orange = trend, grey = no
  evidence, black = low coverage).
- `multimodal_overlap/overlap_count_axial.png` (when `09_overlap.py` has been
  run) — brain slices colored by how many of the 4 modalities (ReHo/fALFF/GLM/
  QPP) agree that a voxel is significant; yellow (count 3–4) = near-unanimous
  multimodal agreement, which is the strongest evidence of a genuinely
  domain-general cognitive region. The accompanying
  `yeo7_network_overlap_summary.csv` + `overlap_summary.json` list the networks
  classified as cognitive vs emotion-specific.
- `schaefer400_overlap_table.csv` — per-Schaefer-parcel overlap counts with
  per-modality significance flags.

> The multimodal overlap (`09_overlap.py`, and the ReHo/fALFF/GLM inputs of
> `07_cognition_emotion.py`) depends on group maps from the reference pipeline
> (`reho_analysis/`, `activation_contrast/`, `glm_analysis/`), which are not
> shipped in this repo — run those reference scripts first if you need the
> 4-modality overlap results.

---

## How to run

```powershell
# 1. Setup (once)
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt

# 2. Full pipeline (download → preprocess → QPP → group summary)
.\venv\Scripts\python scripts/05_run_pipeline.py

# 3. Normalize QPP templates to MNI (ANTs preferred)
.\venv\Scripts\python scripts/coreg_normalize_ants.py

# 4. ROI tables + brain plots
.\venv\Scripts\python scripts/extract_roi_multiatlas.py
.\venv\Scripts\python scripts/plot_roi_brain.py
.\venv\Scripts\python scripts/plot_qpp_brain.py

# 5. Statistics + cognitive analyses
.\venv\Scripts\python scripts/08_roi_stats_significant.py
.\venv\Scripts\python scripts/07_qpp_cognition_overlap.py
```

All scripts skip subjects/outputs that already exist, so re-runs are safe and
incremental.

---

## Known quirks

- `group_analysis_04.py` overwrites `results/group/group_{rest,emotion}_results.csv`
  per subject — the file currently contains only the last processed subject.
  The complete per-subject data is always available in the per-subject
  `sub-XX_qpp_info.npy` files.
- There are two MNI normalization outputs: `*_MNI_ANTs.nii.gz` (SyN, used by
  all statistics/ROI scripts) and `*_MNI.nii.gz` (center-of-mass fallback,
  used by the raw plotting scripts). Results figures should be read from the
  ANTs version.
- The single-middle-frame QPP metric is noisy (signed values cancel across the
  cycle); the **full-cycle mean |amplitude|** metric (scripts `07_qpp_*` and
  `08`) is the reliable one.