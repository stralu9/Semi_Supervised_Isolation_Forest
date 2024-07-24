import numpy as np
import pandas as pd
import sys
import random as rn
from collections import Counter
import os
import warnings
import threading
from multiprocessing import Pool
import seaborn as sb
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import scipy
import warnings
warnings.filterwarnings('ignore')


def compute_labeled_component(X, Y, X_axis, ul):
    """
    This function computes the labeled component.

    Parameters
    ----------
    X : numpy array of shape (n_labeled_samples, n_features)
        The labeled samples at that splitting step.
    Y : numpy array of shape (n_samples)
        The corresponding labels.
    X_axis : numpy array of shape (n_thresholds)
        The thresholds for which the unlabeled component has to be estimated.
    ul : numpy array of shape (n_unlabeled_samples, n_features)
        The unlabeled samples at that splitting step.

    Returns
    -------
    labeled_component : numpy array of shape (n_thresholds)
        The estimated labeled component.
    """
    if Y[Y==1].size == 0: 
        
        min_normal = X[Y==-1].min()
        max_normal = X[Y==-1].max()
        
        dist_l = np.abs(min_normal - X_axis)
        dist_r = np.abs(max_normal - X_axis)
        dist = np.min(np.array([dist_r, dist_l]),axis=0)
                      
        dist[np.logical_or(dist <= max_normal, dist >= min_normal)] = 0
        return dist
            
    elif Y[Y==-1].size == 0:
        anomalies = X[Y==1]
        normals = ul
                      
        an_index = np.sort(np.searchsorted(X_axis, anomalies, side='right'))
        nor_index = np.sort(np.searchsorted(X_axis, normals, side='right'))
        
                      
        an_left = np.searchsorted(an_index, range(len(X_axis)),side='right')
        nor_left = np.searchsorted(nor_index,range(len(X_axis)), side='right')
        an_right = len(anomalies) - an_left
        nor_right = len(normals) - nor_left

        p_al = np.nan_to_num(an_left/(an_left+nor_left), nan=0, posinf=0, neginf=0)
        p_nl = 1 - p_al
        p_ar = np.nan_to_num(an_right/(an_right+nor_right), nan=0, posinf=0, neginf=0)
        p_nr = 1 - p_ar
        
        p_n = len(normals) / len(X)
        p_a = 1 - p_n
                      
        c_l = (an_left + nor_left) / len(X)
        c_r = 1 - c_l
        
        en_l = np.nan_to_num(- p_al * np.log2(p_al) - (p_nl) * np.log2(p_nl),nan=0,posinf=0,neginf=0)
        en_r = np.nan_to_num(- p_ar * np.log2(p_ar) - (p_nr) * np.log2(p_nr),nan=0,posinf=0,neginf=0)
        en_in = np.nan_to_num(- p_a * np.log2(p_a) - (p_n) * np.log2(p_n),nan=0,posinf=0,neginf=0)
        
        return (en_in - c_l * en_l - c_r * en_r)
    else:
        
        anomalies = X[Y==1]
        normals = X[Y==-1]
        an_index = np.sort(np.searchsorted(X_axis, anomalies, side='right'))
        nor_index = np.sort(np.searchsorted(X_axis, normals, side='right'))
        
                      
        an_left = np.searchsorted(an_index, range(len(X_axis)),side='right')
        nor_left = np.searchsorted(nor_index,range(len(X_axis)), side='right')
        an_right = len(anomalies) - an_left
        nor_right = len(normals) - nor_left

        p_al = np.nan_to_num(an_left/(an_left+nor_left), nan=0, posinf=0, neginf=0)
        p_nl = 1 - p_al
        p_ar = np.nan_to_num(an_right/(an_right+nor_right), nan=0, posinf=0, neginf=0)
        p_nr = 1 - p_ar
        
        p_n = len(normals) / len(X)
        p_a = 1 - p_n
                      
        c_l = (an_left + nor_left) / len(X)
        c_r = 1 - c_l
        
        en_l = np.nan_to_num(- p_al * np.log2(p_al) - (p_nl) * np.log2(p_nl),nan=0,posinf=0,neginf=0)
        en_r = np.nan_to_num(- p_ar * np.log2(p_ar) - (p_nr) * np.log2(p_nr),nan=0,posinf=0,neginf=0)
        en_in = np.nan_to_num(- p_a * np.log2(p_a) - (p_n) * np.log2(p_n),nan=0,posinf=0,neginf=0)
        
        return (en_in - c_l * en_l - c_r * en_r)


def compute_unlabeled_component(X,X_axis):
    """
    This function computes the unlabeled component.

    Parameters
    ----------
    X : numpy array of shape (n_unlabeled_samples, n_features)
        The unlabeled samples at that splitting step.
    X_axis : numpy array of shape (n_thresholds)
        The thresholds for which the unlabeled component has to be estimated.

    Returns
    -------
    unlabeled_component : numpy array of shape (n_thresholds)
        The estimated unlabeled component.
    """
    ul_index = np.sort(np.searchsorted(X_axis, X, side='right'))

    X_left = np.searchsorted(ul_index, range(len(X_axis)),side='right')
    X_right = len(X) - X_left
    
    matrix = np.repeat(X.reshape((len(X),1)), len(X_axis), axis=1)

    left = np.where(matrix < X_axis, matrix, np.nan)    
    right = np.where(matrix >= X_axis, matrix, np.nan)

    var_left = np.nanvar(left, axis=0)
    var_right = np.nanvar(right, axis=0)
    
    var_left[np.isnan(var_left)] = 0
    var_right[np.isnan(var_right)] = 0
       
    wb = X_left / X.size
    wf = 1 - wb
    
    V2w = wb * (var_left) + wf * (var_right)

    t1 = np.nan_to_num(wb*np.log(wb), nan=0,posinf=0, neginf=0)
    t2 = np.nan_to_num(wf*np.log(wf), nan=0,posinf=0, neginf=0)
    
    t3 =  np.nan_to_num(np.log(np.sqrt(V2w)), nan=0,posinf=0, neginf=0)
    
    return t1 +t2 - t3


def compute_split_distribution(X,Y,X_axis):
    """
    This function computes the split distribution for a feature.

    Parameters
    ----------
    X : numpy array of shape (n_samples,)
        The labeled samples along a given feature at that splitting step.
    Y : numpy array of shape (n_samples)
        The corresponding labels.
    X_axis : numpy array of shape (n_thresholds,)
        The thresholds for which the split distribution has to be estimated.

    Returns
    -------
    split_distribution : scipy.stats.rv_histogram
        The estimated split distribution.
    split_scores : numpy array of shape (n_thresholds-1,)
        The quality of the thresholds for which the split distribution is estimated. 
    """
    X_norm = np.nan_to_num((X - X.min()) / (X.max() - X.min()), nan=0.5)
    
    pp = Y[Y!=0].size/Y.size
    unlabeled_component = compute_unlabeled_component(X_norm[Y==0],X_axis) if Y[Y==0].size > 0 else np.zeros((X_axis.size))
    labeled_component = compute_labeled_component(X_norm[Y!=0], Y[Y!=0], X_axis, X_norm[Y==0]) if Y[Y!=0].size > 0 else np.zeros((X_axis.size))
    
    unlabeled_component = np.nan_to_num((unlabeled_component - unlabeled_component.min()) / (unlabeled_component.max() - unlabeled_component.min()),nan=0)
    labeled_component =  np.nan_to_num((labeled_component - labeled_component.min()) / (labeled_component.max() - labeled_component.min()),nan=0)
    
    split_scores = (1-pp)*unlabeled_component + (1+pp)*labeled_component
    
    if split_scores.max() != split_scores.min():
        split_distribution = scipy.stats.rv_histogram((split_scores,np.append(X_axis,1)), density=True)
        split_scores = (split_scores - split_scores.min()) / (split_scores.max() - split_scores.min())
    else:
        split_distribution = scipy.stats.uniform()
        split_scores[0:split_scores.size] = 1
    return split_distribution, split_scores

def c_factor(n) :
    if(n<2):
        n=2
    return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))
       
    
class SSIF(object):
    """
    SSIF class 
    The algorithm first constructs an ensemble of SSiTree. 
    Then, it computes the anomaly scores based on the (expected) path lengths in the ensemble.
    
    Parameters
    ----------
    ntrees : int, optional
        The number of SSiTree in the ensemble.
    sample : float, optional 
        The proportion of samples to draw from X to train each SSiTree.
        If sample is larger than 1, all samples will be used for all trees (no sampling).
    nattr : float, optional
        The proportion of features for which the split distribution is computed.
        If nattr is larger than 1, the split distribution will be computed for all feautres.
    max_depth : float, optional
        The height limit at which the tree construction phase stops given as a factor of the deafult value.
    seed : int, optional
        random_state is the seed used by the random number generator;
        If None, it is set to 9 by default
    """
    def __init__(self, contamination=0.1, max_depth=None, ntrees=100, seed=9, sample=None, nattr=None):
        self.ntrees = ntrees
        self.sample = sample
        self.max_depth = max_depth
        self.nattr = nattr
        self.ntrees = ntrees
        self.forest_seed = np.random.RandomState(seed=seed)
        self.contamination = contamination
        self.Trees = []
        
        
    def fit(self, Xtrain, ytrain, n_jobs=-1): 
        """
        The SSIF algorithm first constructs an ensemble of SSiTree. Then, it computes the anomaly scores based on the (expected) path lengths in the ensemble.
        
        Parameters
        ----------
        Xtrain : numpy array of shape (n_samples, n_features)
            The input samples.
        ytrain : numpy array of shape (n_samples)
            The input labels.
        n_jobs : int, optional
            The number of jobs to run in parallel for the `fit` method.
            If -1, then the number of jobs is set to the number of cores.
        """ 
        self.X = Xtrain
        self.Y = ytrain  
        
        
        if self.sample != None:
            self.sample = int(self.sample*Xtrain.shape[0])
        else:
            self.sample = min(max(128,int(Xtrain.shape[0]/3)), Xtrain.shape[0])
        self.c = c_factor(self.sample)
        
        if self.max_depth != None:
            self.max_depth = int(np.ceil(self.max_depth*2*max(np.log2(self.sample), 2)))
        else:
            self.max_depth = int(np.ceil(2*max(np.log2(self.sample), 2)))
        
        if self.nattr != None:
            self.nattr = int(max(1,self.nattr))
        else:
            self.nattr = min(Xtrain.shape[1],max(3,int(Xtrain.shape[1]/5)))    
            
        self.Trees = Parallel(n_jobs=n_jobs)(delayed(self.buildTree)(self.forest_seed.randint(0,100000)) for i in range(self.ntrees))             
        self.decision_scores_ = self.compute_anomaly_scores(Xtrain)
        self.min_score = np.min(self.decision_scores_)
        self.max_score = np.max(self.decision_scores_)
        self.decision_scores_ = (self.decision_scores_ - self.min_score) / (self.max_score - self.min_score)
        self.t_ = np.percentile(self.decision_scores_, q=int((1.0 - self.contamination) * 100))
        
    def predict(self, X):
        """
        Predict the labels for the input samples.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        labels : numpy array of shape (n_samples,)
            The predicted labels.
        """
        test_probs = self.predict_proba(X)
        return np.where(test_probs[:,1] >= 0.5, 1, -1)
    
    def predict_proba(self, X):
        """
        Predict the probabilities for the input samples.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        probs : numpy array of shape (n_samples, 2)
            The probabilities of being normal and anomalous for the input samples.
        """
        scores = self.compute_anomaly_scores(X)
        scores = np.clip((scores - self.min_score) / (self.max_score - self.min_score), 0, 1)
        probs = 1.0 - np.exp(np.log(0.5) * np.power(scores / self.t_, 2))
        return np.vstack([1-probs, probs]).T

    
    def buildTree(self, seed):
        """
        Wrapper funtion used for the parallel computation of the SSiTrees.

        Parameters
        ----------
        i : int
            Index of the tree.
        seed: int
            Seed of the tree.

        Returns
        -------
        t : SSiTree
            The constructed SSiTree.
        """  
        tree_seed = np.random.RandomState(seed=seed)
        ix = tree_seed.choice(range(len(self.X)), self.sample, replace=False)
        X_p = self.X[ix]
        Y_p = self.Y[ix]
        t =  SSiTree(X_p, Y_p, 0, self.max_depth, self.nattr, tree_seed)
        return t

            
    def compute_anomaly_scores(self, X):
        """
        Predict raw anomaly score of X using the fitted detector.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        S = np.zeros(len(X))
        
        for i in range(len(X)):
            paths = np.zeros(self.ntrees)
            for j in range(self.ntrees):
                pf = PathFactor(X[i], self.Trees[j], self.max_depth)
                paths[j] = pf.path
            S[i] = 2**(-np.mean(paths)/self.c)
        return S
    

class Node(object):
    """
    This class is used to represent the nodes of the SSiTree.

    Parameters
    ----------
    X : numpy array of shape (n_samples, n_features)
        The samples contained in the node.
    y : numpy array of shape (n_samples)
        The corresponding labels.
    q : int
        The feature used to split the data in the node. It is set to -1 if the node is a leaf.
    p : int
        The split values used to split the data in the node. It is set to -1 if the node is a leaf.
    e : int
        The height of the node in the tree.
    score : float
        The quality of the selected feature q and split value p.
    left : Node
        The left child of the node.
    right : Node
        The right child of the node.
    node_type : str, optional
        If the node is a leaf or an internal node.
    """
    def __init__(self, X, Y, q, p, e, score, left, right, node_type = '' ):
        self.e = e
        self.size = len(X)
        self.X = X #
        self.Y = Y
        self.q = q
        self.p = p
        self.left = left
        self.right = right
        self.ntype = node_type
        self.score = score    
            
        if Y.size > 0:
            self.perc_anomaly = Y[Y==1].size / Y.size
            self.perc_normal = Y[Y==-1].size / Y.size
        else: 
            self.perc_anomaly = self.perc_normal = 0

class SSiTree(object):
    """
    This class is used to represent an SSiTree.

    Parameters
    ----------
    X : numpy array of shape (n_samples, n_features)
        The samples contained in the tree.
    y : numpy array of shape (n_samples)
        The corresponding labels.
    e : int
        The height of the tree.
    l : int
        The height limit of the tree.
    nattr : int
        The number of features for which the split distribution is estimated.
    seed : int
        The seed of the tree.
    """
    def __init__(self,X,Y,e,l, nattr, seed):
        self.e = e # depth
        self.X = X 
        self.Y = Y 
        self.height = 0
        self.size = len(X) #  n objects
        self.l = l # depth inferior limit
        self.p = None
        self.q = None
        self.exnodes = -1
        self.eps = 10**-4
        self.nattr = nattr
        self.tree_seed = seed
        self.root = self.make_tree(X,Y,e,l)
              
    
    def make_tree(self,X,Y,e,l):
        """
        This function is used to recursively build the trees.

        Parameters
        ----------
        X : numpy array of shape (n_samples,n_features)
            The samples at that splitting step.
        Y : numpy array of shape (n_samples)
            The corresponding labels.
        e : int 
             The current height in the tree
        l : int
            The depth limit of the tree.

        Returns
        -------
        n : Node
            The built node.
        """   
        self.e = e
        seed = self.tree_seed
        if Y[Y!=1].size == 0 or len(np.unique(X, axis=0)) <= 1 or e >= l:            
            if e > self.height:
                self.height = e
            left = None
            right = None
            self.exnodes += 1
            return Node(X, Y, self.q, self.p, e, 1, left, right, node_type = 'exNode')        
        else: 
            
            num_bin = max(5, int(len(X)/10))
            X_axis = np.linspace(self.eps, 1-self.eps, num_bin-1)
            attr = np.array([len(np.unique(X[:,k])) for k in range(X.shape[1])], dtype=int)
            attr = np.where(attr > 1)[0].tolist()
            
            attr_morethanone = attr.copy()
            if Y[Y==1].size > 0:
                for i in attr.copy():
                    max_us = X[:, i][Y!=1].max()
                    min_us = X[:, i][Y!=1].min()

                    max_ss = X[:, i][Y==1].max()
                    min_ss = X[:, i][Y==1].min()

                    if max_ss <= max_us and min_us <= min_ss:
                        attr.remove(i)
                    else:
                        max_al = X[:, i][Y==1].max()
                        min_al = X[:, i][Y==1].min()

                        max_nl = X[:, i][Y!=1].max()
                        min_nl = X[:, i][Y!=1].min()
                        norm_max_al = (max_al - X[:,i].min()) / (X[:,i].max() - X[:,i].min())

                        norm_min_al = (min_al - X[:,i].min()) / (X[:,i].max() - X[:,i].min())

                        norm_max_nl = (max_nl - X[:,i].min()) / (X[:,i].max() - X[:,i].min())

                        norm_min_nl = (min_nl - X[:,i].min()) / (X[:,i].max() - X[:,i].min())

                        if norm_max_al > norm_max_nl and norm_max_nl > X_axis[num_bin-2]:
                            X_axis = np.insert(X_axis, num_bin-1, (norm_max_nl + norm_max_al) / 2)
                            X_axis = np.insert(X_axis, num_bin-1, norm_max_nl)
                            num_bin += 2
                        elif norm_max_al > norm_max_nl and norm_max_nl != 0: 
                            X_axis = np.insert(X_axis, num_bin-1, norm_max_nl-self.eps)
                            X_axis = np.sort(X_axis)
                            num_bin += 1
                        if norm_min_al < norm_min_nl and norm_min_nl < X_axis[0]:
                            X_axis = np.insert(X_axis, 0, (norm_min_nl + norm_min_al) / 2)
                            X_axis = np.insert(X_axis, 1, norm_min_nl)
                            num_bin += 2
                        elif norm_min_al < norm_min_nl and norm_min_nl != 1: 
                            X_axis = np.insert(X_axis, num_bin-1, norm_min_nl+self.eps)
                            X_axis = np.sort(X_axis)
                            num_bin += 1

            if len(attr) >= self.nattr:
                values = seed.choice(attr, self.nattr, replace=False)
            elif len(attr) > 0:
                values = seed.choice(attr, len(attr), replace=False)
            else:
                values = seed.choice(attr_morethanone, min(self.nattr,len(attr_morethanone)), replace=False)                
                  
            Y_axis={}  
                        
            for idx in values:
                Y_axis[idx] = []
                Y_axis[idx], _ = compute_split_distribution(X[:,idx], Y, X_axis)

            uniform = scipy.stats.uniform()
            KL_values = np.zeros((X.shape[1],))
            for idx in Y_axis.keys():
                p1 = Y_axis[idx].cdf(X_axis)
                
                p1 = p1[1:] - p1[:-1]
                
                u1 = uniform.cdf(X_axis)
                u1 = u1[1:] - u1[:-1]
                kl = np.sum(scipy.special.kl_div(p1,u1))
                
                KL_values[idx] = kl

            if KL_values.sum() <= 0:
                KL_values[values] = 1
               
            KL_norm = np.nan_to_num((KL_values - KL_values.min()) / (KL_values.max() - KL_values.min()),nan=1)
            KL_values /= KL_values.sum()
            cdf_kl = np.nan_to_num(np.cumsum(KL_values), nan=0)
            u = seed.uniform(0,1)
            
            self.q = np.searchsorted(cdf_kl, u) if sum(cdf_kl) != 0 else seed.choice(list(Y_axis.keys()),1,replace=False)[0]
                
            u = seed.uniform(0,1)
            split_value = Y_axis[self.q].ppf(u)
            self.p = split_value*(X[:,self.q].max() - X[:,self.q].min()) + X[:,self.q].min()
                
            w = np.where(X[:,self.q] < self.p,True,False)
            return Node(X, Y, self.q, self.p, e,1, left=self.make_tree(X[w],Y[w],e+1,l), right=self.make_tree(X[~w],Y[~w],e+1,l), node_type = 'inNode' )
  
    def get_node(self, path):
        node = self.root
        for p in path:
            if p == 'L' : node = node.left
            if p == 'R' : node = node.right
        return node


class PathFactor(object):
    """
    This class is used to find for a tree the height of the leaf where a sample ends up .

    Parameters
    ----------
    x : numpy array of shape (n_features)
        The considered sample.
    tree : SSiTree
        The considered tree.
    depth_limit : int
        The height limit of the tree.
    """
    def __init__(self,x, ssitree, depth_limit):
        self.path_list=[]
        self.x = x
        self.limit = depth_limit
        self.path = self.find_path(ssitree.root)

    
    def find_path(self, T, e=0, score=0):
        """
        This class is used to recursively find the leaf in the tree and modify the height depending on the available labels and on the splits quality.

        Parameters
        ----------
        T : Node
            The node at that point in the recursion.
        e : int
            The actual height in the tree.
        score : float
            The average quality of the splits traversed in the tree.

        Returns
        -------
        height : float
            The modified height of the sample.
        """
        if T.ntype == 'exNode':
            self.e = e
            self.score = score / e
            if self.e < self.limit:
                self.e = e
            else:
                self.e = e + c_factor(T.size)
            if self.e >= self.limit:
                return self.e
            else:
                return self.e*(1 + T.perc_normal - T.perc_anomaly)
        else:
            a = T.q
            if self.x[a] < T.p:
                self.path_list.append('L')
                return self.find_path(T.left, e+1, score+T.score)
            else:
                self.path_list.append('R')
                return self.find_path(T.right, e+1, score+T.score)