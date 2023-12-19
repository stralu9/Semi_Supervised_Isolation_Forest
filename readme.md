# Semi-Supervised Isolation Forest for Anomaly Detection

## Abstract

Anomaly detection algorithms attempt to find instances that deviate from the expected behavior. Because this is often tackled as an unsupervised task, anomaly detection models rely on exploiting intuitions about what constitutes anomalous behavior. These typically take the form of data-driven heuristics that measure the anomalousness of each instance. However, the unsupervised detectors are limited by the validity of their intuition. Because these are not universally true, one can improve the detectors’ performance by using a semi-supervised approach that exploits a few labeled instances. This paper proposes a novel semi-supervised tree ensemble based anomaly detection framework. We compare our proposed approach to several baselines and show that it performs comparably well to the best state-of-the-art neural networks on 15 benchmark datasets.

## Contents and usage

The folder contains:
- ssif.py, the file containing the model proposed in this paper;
- maxssif.py, the file containing the MaxSSIF model;
- ssif_ablation1.py, ssif_ablation2.py, ssif_ablation3.py, ssif_ablation4.py, the files containing the models for the ablation study;
- Notebook.ipynb, a notebook showing how to train SSIF and compute the anomaly scores on an artificial dataset;
- run_experiments.py, a function used to get the experimental results on benchmark datasets;
- SSAD.py, DevNet.py, DeepSAD_folder, REPEN.py, HIF.py, files implementing the baselines used for the comparison;
- Experiments.ipynb, a notebook showing how to run the experiments.

## Semi-Supervised Isolation Forest

Given a training dataset **X_train** with training labels **Y_train**(where y=0 for the unlabeled data, y=1 for the anomalies and y=-1 for the normals), and a test dataset **X_test**, the algorithm is applied as follows:

```python
from ssif import SSIF
from sklearn.metrics import roc_auc_score

# Train the model (for instance, here we use kNNO)
detector = SSIF(X_train,Y_train)

# Compute the anomaly scores in the training set
train_scores_ssif = detector.compute_anomaly_scores(X_train)

# Compute the anomaly scores in the test set
test_scores_ssif = detector.compute_anomaly_scores(X_test)

# Estimate the AUROC performance of the model
auc_performance = roc_auc_score(Y_test,test_scores_ssif)
```

## Dependencies

This folder requires the following python packages to be used:
- [Python 3](http://www.python.org)
- [Numpy](http://www.numpy.org)
- [Scipy](http://www.scipy.org)
- [Pandas](https://pandas.pydata.org/)
- [Matplot](https://matplotlib.org)
- [sklearn](https://scikit-learn.org/stable/)
- [anomatools](https://github.com/Vincent-Vercruyssen/anomatools)
- [PyOD](https://github.com/yzhao062/pyod)