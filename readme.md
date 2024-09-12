# Subset Selection Problem

## Python packages

```bash
pip install -r ./requirements.txt
```

gurobipy>=11.0.2

gymnasium>=0.29.1

matplotlib>=3.9.1

numpy<=1.26.0

openpyxl>=3.1.5

pandas>=2.2.2

scikit-learn>=1.5.2

scipy>=1.14.0

stable_baselines3>=2.3.2

torch>=2.3.1

numpy version can't be higher than 2.0.

## Files & Folders

* SSP-MV-Vect.ipynb: Main Notebook where the parameters can be choosen and where users can run the models. The dataset used are *indtrack7.txt* and *indtrack8.txt.*
* SSP_with_other_data.ipynb: Alternative Notebook with another dataset.
* Gurobi_SSP.ipynb: Notebook used to compute Gurobi's solutions.
* plot.ipynb: Notebook used to compare the results.
* training.py: Python file for the functions used by the DRL models for training and testing.
* environment.py: Python file for the DRL environments.
* Data: Folder for the datasets used during this project. Data from yfinance package were used but not saved here.
* Images: Folder for the plots generated during training and testing. Dataset used: *indtrack7.txt*. Hyperparameters used are in the file name.
* Plots: Folder for the plots used to compare the results.
* Sheets: Folder for the .xlsx files used to store the results.

## Results explained

![data](image/readme/1.png)

$\mu$ = Mean of each Ticker (column)
$\Sigma$ = Covariance matrix of the data

$\sigma$ = Risk trade-off

### Goal: Find $x$ which gives the highest $\sigma \mu^T x - ( 1 - \sigma) x^T \Sigma x$

For Deep Reinforcement Learning models, we will use 



![Plot](Images\A2C_ws291_init-1_rnnTrue_rdTrue_ccmcontribution_cc10_shFalse_lr1e-06_netarch16_16.png)

This figure only represents the **test**.
