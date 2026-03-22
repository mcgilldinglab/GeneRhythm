# Tutorial Notebooks

The notebooks below are rendered for reading only. Read the Docs will display their contents without executing any code cells during the build.

## Inference and Analysis of Gene Expression Rhythmicity using GeneRhythm

### Notebook

- [scRNA-seq_mouse_embryo_blood.ipynb](tutorial/scRNA-seq_mouse_embryo_blood)

### Contents

- Part 1: Data loading, time and frequency information acquisition
- Obtain time information with monocle3
- Frequency information generation
- Part 2: Model preparation and training
- GCN graph preparation
- Modle training
- Part 3: Showing result
- Part 4: Music sonification
- Convert the .musicxml to .wav music sound.
- Convert the .musicxml to .pdf sheet music.

## Inference and Analysis of Spatial data Rhythmicity using GeneRhythm

### Notebook

- [Spatial_LIBD_human_dorsolateral_prefrontal_cortex.ipynb](tutorial/Spatial_LIBD_human_dorsolateral_prefrontal_cortex)

### Contents

- Part 1: Data loading and frequency information acquisition
- Frequency information generation
- Part 2: Model preparation and training
- Modle training
- Part 3: Showing result
- Part 4: Differential frequency peaks

## Inference and Analysis of scATAC-seq Rhythmicity using GeneRhythm

### Notebook

- [scATAC-seq_mouse_atherosclerotic_plaque_immune_cells.ipynb](tutorial/scATAC-seq_mouse_atherosclerotic_plaque_immune_cells)

### Contents

- Part 1: Data loading, time and frequency information acquisition
- Obtain time information with monocle3
- Frequency information generation
- Part 2: Model preparation and training
- Modle training
- Part 3: Showing result
- Part 4: Differential frequency peaks

## Rhythmicity Perturbation using GeneRhythm

### Notebook

- [scRNA-seq_PDAC_perturbation.ipynb](tutorial/scRNA-seq_PDAC_perturbation)

### Contents

- Part 1: Data loading, time and frequency information acquisition
- Obtain time information with monocle3
- Frequency information generation
- Part 2: Model preparation and training
- GCN graph preparation
- Modle training
- Part 3: Perturbation
- 1. Gene perturbation
- 2. Drug-target perturbation
- 3. Pathway-level perturbation
- Part 4: Survival Analysis

```{toctree}
:hidden:
:maxdepth: 1

tutorial/scRNA-seq_mouse_embryo_blood
tutorial/Spatial_LIBD_human_dorsolateral_prefrontal_cortex
tutorial/scATAC-seq_mouse_atherosclerotic_plaque_immune_cells
tutorial/scRNA-seq_PDAC_perturbation
```
