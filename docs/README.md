# Multi Modal Drug Embeddings

**Authors:** [James Bannon](https://github.com/jbannon)

**Contact:** [bhklab.jamesbannon@gmail.com](mailto:bhklab.jamesbannon@gmail.com)

**Description:** A procedure for multi-modal fusion for drug embedding

--------------------------------------

[![pixi-badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json&style=flat-square)](https://github.com/prefix-dev/pixi)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square)](https://github.com/astral-sh/ruff)
[![Built with Material for MkDocs](https://img.shields.io/badge/mkdocs--material-gray?logo=materialformkdocs&style=flat-square)](https://github.com/squidfunk/mkdocs-material)

![GitHub last commit](https://img.shields.io/github/last-commit/bhklab/multi-modal-drug-embeddings?style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/bhklab/multi-modal-drug-embeddings?style=flat-square)
![GitHub pull requests](https://img.shields.io/github/issues-pr/bhklab/multi-modal-drug-embeddings?style=flat-square)
![GitHub contributors](https://img.shields.io/github/contributors/bhklab/multi-modal-drug-embeddings?style=flat-square)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/bhklab/multi-modal-drug-embeddings?style=flat-square)

## Set Up

### Prerequisites

Pixi is required to run this project.
If you haven't installed it yet, [follow these instructions](https://pixi.sh/latest/)

### Installation

1. Clone this repository to your local machine
2. Navigate to the project directory
3. Set up the environment using Pixi:

```bash
pixi install
```

### Data Downloading

In the `data/rawdata/` folder create three directories named `LINCS`, `HDD` and `GEOM`. 

Next: 



1. From the [LINCS Project Page](https://orcestra.ca/annotations/69f36604f9dd88b7003f02fd) download the following files and place them in the created `LINCS` directory. 
    a. `colData.csv`
    b. `drug_metadata.tsv`
    c. `signatures.parquet`
    d. `signatures.tsv`
2. From the [HDD Project Page](https://orcestra.ca/annotations/6a3e65a98aaa8472dae547c8) download the following file and place it in the created `HDD` directory. 
    a. `colData.csv`
3. From the [GEOM Project Page](https://orcestra.ca/annotations/69f3650ef9dd88b7003f02f7) download the following files and place them in the created `GEOM` directory. 
    a. `colData.tsv`
    b. `WHIM_hp.parquet`
    c. `drug_metadata.tsv`


### Creating the Embeddings

In `pixi.toml` file, make sure the `scikit-learn` version is set to `">=1.5.0, <1.6"` by uncommenting line 39 and commenting out line 42 in the `pixi.toml` file.

Next navigate to the `workflow/scripts/` folder and execute

`pixi run "python3 generate_embeddings.py -config embedding_config.yaml"`


This will generate embeddings for all possible subsets and as such will take a while. 



### Analyzing the Embeddings

Change the version of `scikit-learn` being used by editing the `pixi.toml`. For this part of the code we need  `scikit-learn` version  `">=1.9.0,<2"`. 

Next execute 

`pixi run "python3 analyze_embeddings.py"` 

to generate the plots. 
