from tqdm import tqdm
import numpy as np
import os
import cv2
from PIL import Image
from sklearn.linear_model import Ridge
import torch.nn.functional as F

from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset
import json
import xml.etree.ElementTree as ET
from time import time
from utils.tools_for_broadcam import *
from utils.dataset_for_broadcam import *
from numpy import random
from scipy import linalg, sparse

def linear(data):
    return data
def orth(Wh):
    for i in range(0,Wh.shape[1]):
        wh = np.mat(Wh[:,i].copy()).T
        wh_sum = 0
        for j in range(i):
            wj = np.mat(Wh[:,j].copy()).T
            wh_sum += (wh.T.dot(wj))[0,0]*wj 
        wh -= wh_sum
        wh = wh/np.sqrt(wh.T.dot(wh))
        Wh[:,i] = np.ravel(wh)
    return Wh

class norm_maxmin():
    def __init__(self) -> None:
        self.max = 0
        self.min = 0
    def fit_transform(self, data):
        self.max = np.max(data)
        self.min = np.min(data)
        return (data-self.min)/(self.max-self.min)
    def transform(self,data):
        return (data-self.min)/(self.max-self.min)

class node_generator():
    def __init__(self, n_in_node, n_out_node_per_clss, whiten, pre_W, step) -> None:
        self.n_in_node = n_in_node
        self.n_H_node_per_clss = n_out_node_per_clss
        self.whiten = whiten
        self.pre_W = pre_W
        n_dim, n_class = pre_W.shape
        xita = np.zeros((n_in_node, 0))
        for i in range(n_class):
            xita_tmp = np.zeros((n_in_node, n_out_node_per_clss))
            pre_W_tmp = pre_W[:,i]
            idx = np.flip(np.argsort(pre_W_tmp))
            for j in range(n_out_node_per_clss):
                # xita_tmp[idx[j*step:(j+1)*step],j] = random.random((step))/step
                xita_tmp[idx[j*step:(j+1)*step],j] = np.ones((step))/step
            xita = np.column_stack((xita, (xita_tmp.T*pre_W[:,i]).T))
        if self.whiten:
            xita = orth(xita)
        self.xita = xita
    def sigmoid(self,data):
        return 1.0/(1+np.exp(-data))
    def linear(self,data):
        return data
    def fit_transform(self,data):
        data = data.dot(self.xita)
        self.offset_ = np.array(np.min(data, 0)).squeeze()
        self.scale_ = np.array(np.max(data, 0)).squeeze()
        return self.linear((data-self.offset_)/(self.scale_+1e-8))
        # return self.linear(data)
    def transform(self,data):
        data = data.dot(self.xita)
        return self.linear((data-self.offset_)/(self.scale_+1e-8))
        # return self.linear(data)
class node_generator_v0():
    def __init__(self, n_Z_node, n_H_node, whiten, pre_W) -> None:
        self.n_Z_node = n_Z_node
        self.n_H_node = n_H_node
        self.whiten = whiten
        self.pre_W = pre_W
        from scipy.sparse import rand
        # Wh = 2*random.random((n_Z_node, n_H_node))-1
        Wh = random.random((n_Z_node, n_H_node))
        b = random.random(n_H_node)
        if self.whiten:
            Wh = orth(Wh)
        self.Wh = np.mat(Wh)*0.1
        self.b = b*0.1
        test = 1

    def transform(self,Z):
        return linear(np.mat(Z).dot(self.Wh)+self.b)

class BLS():
    def __init__(self, alpha, enhance_times = 1, step = 2**8, whiten=False) -> None:
        self.alpha = alpha
        self.enhance_times = enhance_times
        self.whiten = whiten
        self.step = step
        from sklearn.preprocessing import StandardScaler
        self.norm_maxmin = norm_maxmin()
        self.normalize = StandardScaler()
        from sklearn.pipeline import make_pipeline
        self.ridge_z = make_pipeline(Ridge(alpha=alpha, normalize=True, fit_intercept=True))
        self.ridge = make_pipeline(Ridge(alpha=alpha, normalize=True, fit_intercept=True))
        
    def obtain_index_of_max_weights(self, w, k):
        index = w.argsort()[-k:][::-1] #获取前k个索引
        return index
    def fit(self, features, labels):
        features = self.normalize.fit_transform(features)
        Z = np.mat(features)
        Y = np.mat(labels)
        n_Z_node = Z.shape[1]
        n_class = Y.shape[1]
        step = self.step
        
        self.node_generators = []
        A = Z
        factor_z = 1/(self.normalize.scale_)
        n_class = Y.shape[1]
        n_dim = A.shape[1]
        pre_W = np.zeros((n_dim, n_class))
        ridge_step=2
        times = int(n_class/ridge_step)
        
        for i in range(times):
            self.ridge_z.fit(np.array(Z),np.array(Y[:,i*ridge_step:(i+1)*ridge_step]))
            pre_W[:,i*ridge_step:(i+1)*ridge_step] = self.ridge_z['ridge'].coef_.T
        pre_W = (pre_W.T*factor_z).T
        ## 在原始权重大于0的特征里面选择性进行排列组合，映射增强节点
        n_H_node_per_clss = int(np.sum(pre_W>0,0).min()/step)
        if self.enhance_times > 0:
            H = np.zeros((Z.shape[0], 0))
            for i in range(self.enhance_times):
                self.node_generators.append(node_generator(n_out_node_per_clss=n_H_node_per_clss,n_in_node=n_Z_node,whiten=self.whiten, pre_W =pre_W, step=self.step))
                H = np.column_stack((H,self.node_generators[i].fit_transform(Z)))
            A = np.column_stack((Z,H))
        ridge_step=2
        n_dim = A.shape[1]
        self.W = np.zeros((n_dim, n_class))
        times = int(n_class/ridge_step)
        intercept = np.zeros(n_class)
        for i in range(times):
            self.ridge.fit(np.array(A),np.array(Y[:,i*ridge_step:(i+1)*ridge_step]))
            intercept[i*ridge_step:(i+1)*ridge_step] = self.ridge['ridge'].intercept_
            self.W[:,i*ridge_step:(i+1)*ridge_step] = self.ridge['ridge'].coef_.T
        self.ridge['ridge'].coef_ = self.W.T
        self.ridge['ridge'].intercept_ = intercept
        
        Wz = self.W[:n_Z_node,:]
        coef = (Wz.T*factor_z).T
        n_H_node = n_H_node_per_clss*n_class
        for i in range(self.enhance_times):
            xita = self.node_generators[i].xita
            Wh = self.W[n_Z_node+(i*n_H_node):n_Z_node+((i+1)*n_H_node),:]
            factor_h = 1/self.node_generators[i].scale_
            coef += ((xita.T*factor_z).T*factor_h).dot(Wh)
        self.coef_ = coef.T

    def predict(self, features):
        features = self.normalize.transform(features)
        Z = np.mat(features)
        A = Z
        if self.enhance_times > 0:
            H = np.zeros((Z.shape[0], 0))
            for i in range(self.enhance_times):
                H = np.column_stack((H,self.node_generators[i].transform(Z)))
            A = np.column_stack((Z,H))
        Y = self.ridge.predict(np.array(A))
        return Y
    
    def pinv(self,A,alpha):
        return np.mat(alpha*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)   
    
class BroadCAM():
    def __init__(self, alpha, n_level, model, n_class, target_layer='[0+1]') -> None:
        ## "model" is the CNN model and it is required that the model has the function "def forward_map(self, x): ..." to extract feature maps from multi-stage.
        self.alpha = alpha
        self.n_level = n_level
        self.bls_merge = []
        self.target_layer = target_layer
        self.avg = torch.nn.AdaptiveAvgPool2d((1,1))
        self.dims_idx = np.zeros((self.n_level+1))
        bls = BLS(alpha=alpha)
        self.bls_merge.append(bls)
        self.model = model.cuda()
        self.n_class = n_class
    ## fit function
    def fit_from_feature(self, features_path, labels_path):
        ## features是一个列表每一个成员都是一个特征矩阵,如: features[0]的维度为 => n_sample x n_dimension
        ## Y是one-hot/multi-hot的标签矩阵,维度为 => n_sample x n_class
        features = torch.load(features_path)
        Y = torch.load(labels_path)
        for i in range(self.n_level):
            self.bls_merge[i].fit(features[i], Y)
        test = 1
    ## fit function
    def fit_from_feature_one_path(self, path):
        ## features是一个列表每一个成员都是一个特征矩阵,如: features[0]的维度为 => n_sample x n_dimension
        ## Y是one-hot/multi-hot的标签矩阵,维度为 => n_sample x n_class
        features_and_labels = torch.load(path)
        features = features_and_labels['features_bls']
        Y = features_and_labels['labels_bls']
        X = torch.zeros((Y.shape[0], 0))
        for i in range(self.n_level):
            if not (str(i) in self.target_layer):
                continue
            X = torch.cat((X,features[i]), 1)
        self.bls_merge[0].fit(X, Y)
    ## gen BroadCAM
    def gen_broadcam(self, img, is_flip=False, WSSS_or_WSOL='WSSS'):
        self.model.eval()
        input_size = img.size()[2:]
        broadcam = torch.zeros(self.n_class,input_size[0], input_size[1])
        with torch.no_grad():
            featuremaps = self.model.extract_map(img.cuda())
            bls_merge = self.bls_merge
        a,_,c,d = featuremaps[0].shape
        maps = torch.zeros((a,0,c,d)).cuda()
        for i in range(self.n_level):
            if not (str(i) in self.target_layer):
                continue
            maps = torch.cat((maps, featuremaps[i]), 1)
        blsW = ((bls_merge[0].coef_).T).astype('float32')
        blsW = torch.from_numpy(blsW.T).unsqueeze(2).unsqueeze(3).cuda()
        cam = F.conv2d(((F.relu(maps))), blsW)
        if is_flip:
            cam = (cam[0]+cam[1].flip(-1)).unsqueeze(0)
        broadcam = F.interpolate(cam, input_size, mode='bilinear', align_corners=False)
        broadcam[broadcam<0] = 0
        if WSSS_or_WSOL == 'WSSS':
            return broadcam[0].cpu().numpy()
        elif WSSS_or_WSOL == 'WSOL':
            return broadcam.cuda()