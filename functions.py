from matplotlib import pyplot as plt
import numpy as np
from numpy import array
import scipy
from scipy.optimize import minimize
import random
import math
import networkx as nx
import pennylane as qml
import itertools as it

def sigmoid(x):
  return np.exp(x) / (np.exp(x) + 1)

def prob_dist(params):
  return np.vstack([sigmoid(params), 1 - sigmoid(params)]).T

def calculate_entropy(distribution):

  total_entropy = []
  for i in distribution:
    total_entropy.append(-1*i[0]*np.log(i[0]) + -1*i[1]*np.log(i[1]))

  return total_entropy