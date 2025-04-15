# GeneRhythm
## Overview
<img title="Model Overview" alt="Alt text" src="/figures/main.png">
Gene expression dynamics are central to biological processes such as cell differentiation, disease progression, and tissue remodeling. Most analyses focus on changes in the time domain, overlooking frequency associated patterns that may reveal distinct regulatory mechanisms. We introduce GeneRhythm, a deep learning framework that defines and extracts gene rhythm, a novel feature that quantifies the frequency of gene expression changes via wavelet transformation. By modeling gene expression in time and frequency domains, GeneRhythm captures transcriptional programs missed by conventional methods. We show that gene rhythm improves gene clustering by grouping rhythmically similar genes into functionally coherent modules. It also identifies rhythm differential genes, which have similar expression levels but distinct rhythmic patterns that reveal complementary biological insights. Applied to spatial transcriptomics, GeneRhythm reveals spatially organized rhythmic patterns reflecting tissue structure. GeneRhythm provides a generalizable and frequency aware framework for uncovering dynamic regulatory programs in single cell and spatial omics data.

## Key Capabilities

1. Utilize wavelet transformation to obatin frequency information of gene expression.

2. Acurately identify gene clusters with frequency information and deep generative model.

3. Acurately identify gene markers with differential analysis based on frequency information.

4. Expand the frequency information analysis to Spatial data.

5. Expand the frequency information analysis to Multi-omics data and get frequency primed genes.

6. Explore the bio-insight of the genes identified with frequency inforamtion.







## Installation


### Prerequisites

* Python >= 3.7
* R >= 4.1.2
* R dependencies
    * monocle3 >= 1.3.1


You may install GeneRhythm by the following command:

```
git clone https://github.com/mcgilldinglab/GeneRhythm.git
cd GeneRhythm
pip3 install -r requirements.txt 
```
## Tutorials

Please check the tutorial directory of the repository.

* [Tutorial for GeneRhythm gene clustering and analyzing on single-cell RNA-seq data](https://github.com/mcgilldinglab/GeneRhythm/blob/main/tutorial/scRNA-seq_mouse_embryo_blood.ipynb)

* [Tutorial for GeneRhythm gene clustering and differential frequency biomarker finding on single-cell ATAC-seq data](https://github.com/mcgilldinglab/GeneRhythm/blob/main/tutorial/scATAC-seq_mouse_atherosclerotic_plaque_immune_cells.ipynb)

* [Tutorial for GeneRhythm gene clustering and differential frequency biomarker finding on Spatial data](https://github.com/mcgilldinglab/GeneRhythm/blob/main/tutorial/Spatial_LIBD_human_dorsolateral_prefrontal_cortex.ipynb)
