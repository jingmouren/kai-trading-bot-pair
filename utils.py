import ast
import glob
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import statsmodels.api as sm
import warnings
from itertools import combinations
from joblib import Parallel, delayed
from networkx.algorithms.matching import max_weight_matching, maximal_matching
from scipy.stats import zscore
from sklearn import linear_model
from typing import *

__author__ = 'kq'

# Directories
HOME = os.environ['HOME'] + '/pairs/'

os.makedirs(HOME + '/adf/', exist_ok=True)
os.makedirs(HOME + '/betas/', exist_ok=True)
os.makedirs(HOME + '/spread/', exist_ok=True)

for method in ['max_weight', 'maximal', 'min_max', 'baseline']:
    os.makedirs(HOME + '{method}/figures/'.format(method=method), exist_ok=True)

# Settings
plt.style.use('seaborn-whitegrid')
warnings.filterwarnings("ignore")
