# Basic Information:
This code is released for the papers (under review):

### ECML-PKDD'2023, CVTGAD: Simplified Transformer with Cross-View Attention for Unsupervised Graph Level Anomaly Detection

The article is in the period of review. Please do not quote or use it for other purposes

# Requirements
This code requires the following:

- Python==3.8
- Pytorch==1.10.1
- Pytorch Geometric==2.0.4
- Numpy==1.21.2
- Scikit-learn==1.0.2
- OGB==1.3.3
- NetworkX==2.7.1

# Hardware used for implementation
CPU: 15 vCPU Intel(R) Xeon(R) Platinum 8358P CPU @ 2.60GHz

GPU: A40(48GB)

RAM: 56GB

System Disk: 25GB

Data Disk: SSD 50GB

AutoDL platform: https://www.autodl.com/home

# Dataset
TUDataset
https://chrsmrrs.github.io/datasets/docs/datasets/

# Usage
Run anomaly detection on Tox21_PPAR-gamma datasets:
```
bash script/ad_PPAR-gamma.sh
```

# Reproducibility
Although we do our best to fix the random seeds, the results of running cannot be guaranteed to be 100% identical on different machines. The process of running is still full of randomness.
In order to ensure the reproducibility of the results, we have saved and sorted the results of all test epochs and adopt the best performance. If the result is still somewhat different from the report in the paper, you can try to fine-tune the hyperparameters, such as learning rate.


