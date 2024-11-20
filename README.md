<div style="text-align: center">
<h1>Bayesian Ridge Regression for KinDEL Benchmark</h1>
</div>

This repository contains an implementation of **Bayesian Ridge Regression** for predicting enrichment scores of DNA-encoded library (DEL) compounds on kinase targets (DDR1 and MAPK14) from the **KinDEL Benchmark Dataset**. 

## Features
- Leverages Bayesian Ridge Regression.
- Incorporates molecular descriptors along with ChemBERTa's pre-trained representations.

---

## Installation

### Step 1: Create a Python environment
Use your preferred method to set up a Python environment. For example:

```bash
python -m venv env
source env/bin/activate
```

### Step 2: Install dependencies
Install the required Python libraries from requirements.txt:

```bash
pip install -r requirements.txt
```

## Training and Prediction
To run the Bayesian Ridge Regression implementation, execute the following command:

```bash
python predict.py --output-dir chemberta \
    --targets ddr1 mapk14 \
    --splits random disynthon \
    --split-indexes 1 2 3 4 5 \
    --representation chemberta
```

### Command-line Arguments
- output-dir: Directory to save results.
- targets: Specify one or more targets (ddr1, mapk14).
- splits: Data splits to use (random, disynthon).
- split-indexes: Indexes for the data splits (1 2 3 4 5).
- representation: Molecular representation used in the model. (circular, combined, chemberta)



## Results
To retrieve performance results following training, run the results.py script and specify the path to the output files:

```bash
python results.py --model-path [path]
```

## Licensing and Attribution
### Code Attribution
Most of the code in this repository is adapted from the [KinDEL project](https://github.com/insitro/kindel) developed by Insitro Inc. 
