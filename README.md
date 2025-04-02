# scGeneRhythm
## Overview
<img title="Model Overview" alt="Alt text" src="/figures/main.png">
Gene clustering and bio-marker finding in single-cell RNA sequencing play a pivotal role in unraveling a plethora of biological processes, from cell differentiation to disease progression and metabolic pathways. Traditional time-domain methods are instrumental in certain analyses, yet they may overlook intricate relationships. For instance, genes that appear distinct in the time domain might exhibit striking similarities in the frequency domain. Recognizing this, we present scGeneRhythm, an innovative deep learning technique that employs wavelets transformation. This approach captures the rich tapestry of gene expression from both the time and frequency domains. Through integrating frequency-domain data, scGeneRhythm not only refines gene grouping but also uncovers pivotal biological insights, such as nuanced gene rhythmicity. By deploying scGeneRhythm, we foster a richer, multi-dimensional understanding of gene expression dynamics, enriching the potential avenues of cellular and molecular biology research.

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


You may install scCross by the following command:

```
git clone https://github.com/mcgilldinglab/scGeneRhythm.git
cd scGeneRhythm
pip3 install -r requirements.txt 
```
## Tutorials

Please check the tutorial directory of the repository.

* [Tutorial for scGeneRhythm gene clustering and analyzing on single-cell RNA-seq data](https://github.com/mcgilldinglab/scGeneRhythm/blob/main/tutorial/scRNA-seq_mouse_embryo_blood.ipynb)

* [Tutorial for scGeneRhythm gene clustering and differential frequency biomarker finding on single-cell ATAC-seq data](https://github.com/mcgilldinglab/scGeneRhythm/blob/main/tutorial/scATAC-seq_mouse_atherosclerotic_plaque_immune cells.ipynb)

* [Tutorial for scGeneRhythm gene clustering and differential frequency biomarker finding on Spatial data](https://github.com/mcgilldinglab/scGeneRhythm/blob/main/tutorial/Spatial_LIBD_human_dorsolateral_prefrontal_cortex.ipynb)
