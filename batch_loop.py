# -*- coding: utf-8 -*-

import numpy
import math
import pylab

### load the training data and labels

traindata = numpy.loadtxt("train2k.databw.35")
label = numpy.loadtxt("train2k.label.35")

### load test data

testdata = numpy.loadtxt("test200.databw.35")

### normalize each exmaple to have unit norm

norm_traindata = numpy.empty(traindata.size)
norm_traindata.shape = (len(traindata[:,0]),len(traindata[0,:]))

for i, row in enumerate(traindata):
    norm_traindata[i,:] = row / numpy.linalg.norm(row)
    
norm_testdata = numpy.empty(testdata.size)
norm_testdata.shape = (len(testdata[:,0]),len(testdata[0,:]))

for i, row in enumerate(testdata):
    norm_testdata[i,:] = row / numpy.linalg.norm(row)  

### sample size and data dimension

N = len(label)
M = len(norm_testdata[:,0])
dim = len(norm_traindata[0,:])

######################### training #########################

### set step size

eta = 1

### set convergence threshold

epsilon = 0.0001

### initial value of w

# w = numpy.zeros(dim)
w = numpy.loadtxt("w_4")

### loop

slope = 1

while  slope > epsilon:
    delta = numpy.zeros(dim)
    for i, x in enumerate(norm_traindata):
        if label[i] * numpy.dot(w, x) <= 0:
            delta = delta - label[i] * x
    delta = delta / N
    slope = numpy.linalg.norm(delta)
    w = w - eta * delta     

######################### test #########################


prediction_batch = numpy.empty(M)

for i, x in enumerate(norm_testdata):
    prediction_batch[i] = math.copysign(1, numpy.dot(x,w))

numpy.savetxt("test200.label.35.ushioda_batch_five", prediction_batch, fmt = '%i') 
print prediction_batch
  
numpy.savetxt("w_5", w, fmt='%10.7f', delimiter='\t')
print "done"