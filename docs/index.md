# GeneRhythm

## Overview

![Model Overview](_static/main.png)

Temporal gene expression programs underlie many genetic and cellular processes, yet most analytical approaches interrogate these dynamics primarily in the time domain, limiting sensitivity to rhythmic and frequency-specific regulation. Here, we present GeneRhythm, a deep learning framework that integrates wavelet-based time-frequency decomposition to model gene expression dynamics across biological conditions. GeneRhythm enables accurate gene clustering based on shared rhythmic patterns and identifies rhythm-differential genes whose dynamic behaviors differ between conditions despite minimal changes in mean expression, revealing coordinated oscillatory programs and phase-shifted regulation missed by differential expression and trajectory-based analyses. Beyond analytical inference, GeneRhythm translates gene expression dynamics into structured, playable musical scores, enabling direct auditory exploration of molecular dynamics, in which rhythmic patterns and temporal progression are mathematically derived from wavelet-resolved signals. By explicitly modeling rhythm and frequency and translating gene expression dynamics into structured musical representations, GeneRhythm provides a new lens for interrogating dysregulated temporal programs that underlie complex disease states.

## Key Capabilities

1. Utilize wavelet transformation to obatin frequency information of gene expression.
2. Acurately identify gene clusters with frequency information and deep generative model.
3. Acurately identify gene markers with differential analysis based on frequency information.
4. Expand the frequency information analysis to Spatial data.
5. Expand the frequency information analysis to Multi-omics data and get frequency primed genes.
6. Explore the bio-insight of the genes identified with frequency inforamtion.
7. Perform rhythmic perturbation to research diseases related drug and pathway targets.
8. Achieve rhythmic signal sonification from time-frequency features into musical parameters.

## Documentation

```{toctree}
:maxdepth: 2
:caption: Contents

installation
tutorials
```
