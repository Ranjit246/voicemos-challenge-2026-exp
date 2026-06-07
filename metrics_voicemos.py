import numpy as np
import scipy.stats

# `true_mean_scores` and `predict_mean_scores` are both 1-d numpy arrays.

MSE = np.mean((true_mean_scores - predict_mean_scores)**2)
LCC = np.corrcoef(true_mean_scores, predict_mean_scores)[0][1]
SRCC = scipy.stats.spearmanr(true_mean_scores, predict_mean_scores)[0]
KTAU = scipy.stats.kendalltau(true_mean_scores, predict_mean_scores)[0]