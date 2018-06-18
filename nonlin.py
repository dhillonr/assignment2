#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def softmax(Z):
    ps = np.exp(Z)             #compute e^z
    ps = ps/float(np.sum(ps))
    return ps

def relu(Z):
    return np.multiply(Z,Z>0)


def softmax_backprop (dA , A_prev):
    avg = np.sum(A_prev,axis = 0)
    numrtr = A_prev.dot(avg - A_prev)
    dnmntr = np.square(avg)
    dA_prev = np.multiply(dA,np.divide(numrtr,dnmntr))
    return dA_prev


def relu_backprop (dA, A_prev,W):
    dg = (A_prev>0)
    dA_prev = dA.dot(W.dot(dg))
    return dA_prev


def flatten_a_data(X):
    (m,a,b,c) = X.shape        #no of training examples = m
    X = np.reshape(X,(m,a*b*c))
    return X

def loss(A,Y):
    (m,n) = Y.shape
    return -np.sum(np.multiply(np.log(A),Y))/m


def convert_to_one_hot(Y, C):      # one hot conversion
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


def batchnorm_forward(x, gamma, beta, eps):

    N, D = x.shape

  #step1: calculate mean
    mu = 1./N * np.sum(x, axis = 0)

  #step2: subtract mean vector of every trainings example
    xmu = x - mu

  #step3: following the lower branch - calculation denominator
    sq = xmu ** 2

  #step4: calculate variance
    var = 1./N * np.sum(sq, axis = 0)

  #step5: add eps for numerical stability, then sqrt
    sqrtvar = np.sqrt(var + eps)

  #step6: invert sqrtwar
    ivar = 1./sqrtvar

  #step7: execute normalization
    xhat = xmu * ivar

  #step8: Nor the two transformation steps
    gammax = gamma * xhat

  #step9
    out = gammax + beta

  #store intermediate
    cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)

    return out, cache


def batchnorm_backward(dout, cache):

  #unfold the variables stored in cache
    xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache

  #get the dimensions of the input/output
    N,D = dout.shape

  #step9
    dbeta = np.sum(dout, axis=0)
    dgammax = dout #not necessary, but more understandable

  #step8
    dgamma = np.sum(dgammax*xhat, axis=0)
    dxhat = dgammax * gamma

  #step7
    divar = np.sum(dxhat*xmu, axis=0)
    dxmu1 = dxhat * ivar

  #step6
    dsqrtvar = -1. /(sqrtvar**2) * divar

  #step5
    dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar

  #step4
    dsq = 1. /N * np.ones((N,D)) * dvar

  #step3
    dxmu2 = 2 * xmu * dsq

  #step2
    dx1 = (dxmu1 + dxmu2)
    dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)

  #step1
    dx2 = 1. /N * np.ones((N,D)) * dmu

  #step0
    dx = dx1 + dx2

    return dx, dgamma, dbeta