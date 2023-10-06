# CVTGAD
This is the source code of ECML-PKDD'2023 paper "CVTGAD: Simplified Transformer with Cross-View Attention for Unsupervised Graph-level Anomaly Detection".

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
Although we do our best to fix the random seeds, the process and result of code running cannot be guaranteed to be 100% identical on different machines. The process of code running is still full of uncertainty. After a certain number of rounds of training, obtaining results based on the last state may not necessarily guarantee fairness in performance comparison. In order to ensure the reproducibility of the results, we have saved and sorted the results of all test epochs in the process of code running. In the later stages of training, the model and results tend to be stable. We adopt the best performance with it is similar to the performance of the top ranked ones. If the result is still somewhat different from the report in the paper, you can try to fine-tune the hyperparameters, such as learning rate.

# Cite
If you compare with, build on, or use aspects of this work, please cite the following:
```
@inproceedings{li2023cvtgad,
  title={CVTGAD: Simplified Transformer with Cross-View Attention for Unsupervised Graph-Level Anomaly Detection},
  author={Li, Jindong and Xing, Qianli and Wang, Qi and Chang, Yi},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={185--200},
  year={2023},
  organization={Springer}
}
```

