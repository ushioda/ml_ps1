# -*- coding: utf-8 -*-

import numpy
import math

def batch(label, traindata, testdata, initial_w, eta, epsilon):

    N = len(label)
    M = len(testdata[:,0])
    dim = len(traindata[0,:])

    ww = initial_w

    ### loop

    slope = 1

    while slope > epsilon:
        delta = numpy.zeros(dim)
        for i, xx in enumerate(traindata):
            if label[i] * numpy.dot(xx,ww) <= 0:
                delta = delta - label[i] * xx
        delta = delta / N
        slope = numpy.linalg.norm(delta)
        ww = ww - eta * delta     

    return ww