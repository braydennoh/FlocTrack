# FlocTrack

**Constraining the Distribution of 3D Fractal Structures in Mud Flocs**

Brayden Noh, Justin A. Nghiem, Kimberly Litwin Miller, Dongchen Wang, Michael P. Lamb

---

This repository contains the video tracking algorithm, experimental data, analysis code, and manuscript for our study on the fractal structure of cohesive sediment flocs.

<p align="center">
  <img src="https://github.com/braydennoh/FlocTrack/blob/main/gif/floc1.gif" width="400" height="250">
  <img src="https://github.com/braydennoh/FlocTrack/blob/main/gif/floc3.gif" width="400" height="250">
</p>
<p align="center">
  <img src="https://github.com/braydennoh/FlocTrack/blob/main/gif/floc4.gif" width="400" height="250">
  <img src="https://github.com/braydennoh/FlocTrack/blob/main/gif/floc5.gif" width="400" height="250">
</p>

## Repository Structure

```
FlocTrack/
├── FlocTrack.ipynb            # Main tracking notebook
├── FlocAnalysis.py            # Floc analysis pipeline
├── ParticleProcess.py         # Particle detection and processing
├── ParticleTracking.py        # Frame-to-frame particle tracking
├── ParticlePIVProcessing.py   # PIV-based velocity processing
├── AnalyzeFrames.py           # Frame extraction and preprocessing
│
├── ExperimentalData/          # Raw experimental measurements
│   ├── HIGHSHEAR.csv
│   ├── MIDSHEAR.csv
│   ├── LOWSHEAR.csv
│   └── OM1.csv, OM2.csv, OM3.csv
│
├── Analaysis/                 # Post-processing and figure generation
│   ├── basictest/             # Measured values and model comparisons
│   ├── densitytest/           # Density and fractal dimension analysis
│   ├── bulktest/              # Bulk property analysis
│   ├── 2dto3d/                # 2D-to-3D fractal reconstruction
│   ├── dpdata/                # Primary particle size analysis
│   ├── highshear/             # High shear experiment results
│   ├── midshear/              # Mid shear experiment results
│   ├── lowshear/              # Low shear experiment results
│   ├── om1/, om2/, om3/       # Organic matter experiment results
│   └── parameter_sweep.py     # Parameter sweep across experiments
│
├── InputSedData/              # Sediment input parameters
├── Figures/                   # Publication-ready figures
├── Paper/                     # Manuscript (.tex, .bib, compiled .pdf)
│   └── reviewerfiles/         # Reviewer response and tracked changes
└── gif/                       # Demo animations
```

## FlocTrack Algorithm

[**FlocTrack.ipynb**](https://github.com/braydennoh/FlocTrack/blob/main/FlocTrack.ipynb) extracts frames from floc tank experiment videos, processes images, and tracks individual floc particles to measure their **diameter**, **velocity**, **perimeter fractal dimension**, and **concavity** over time.

| Module | Description |
|--------|-------------|
| `AnalyzeFrames.py` | Frame extraction from video over a specified time range |
| `ParticleProcess.py` | Image segmentation, particle detection, and morphometric measurement |
| `ParticleTracking.py` | Frame-to-frame particle matching and trajectory construction |
| `ParticlePIVProcessing.py` | PIV-based bulk velocity field estimation |
| `FlocAnalysis.py` | Statistical analysis pipeline across experiments |

## Experimental Data

Six flume tank experiments at varying shear rates and organic matter content:

| Experiment | Description |
|------------|-------------|
| `HIGHSHEAR` | High turbulent shear rate |
| `MIDSHEAR` | Intermediate shear rate |
| `LOWSHEAR` | Low shear rate |
| `OM1` | Organic matter condition 1 |
| `OM2` | Organic matter condition 2 |
| `OM3` | Organic matter condition 3 |

Each CSV contains per-floc measurements: concave diameter, convex diameter, settling velocity, perimeter fractal dimension, and Laplacian sharpness.

## Usage

```python
jupyter notebook FlocTrack.ipynb
```
