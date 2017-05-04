# -*- coding: utf-8 -*-

import numpy
import math
import pylab
from batch_func import batch

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
    

######################### training #########################

### set step size

eta = 0.1

### set convergence threshold

epsilon = 0.0001

### set number of cycles to go through data

cycle = 20

### initial value of w

w_initial = numpy.zeros(len(traindata[0,:]))

### loop

for j in range(cycle):
    w = batch(label, norm_traindata, norm_testdata, w_initial, eta, epsilon)
    if j == 0:
        w_0 = w
    if j == cycle - 1:
        w_end = w
    w_initial = w

if all(w_0 == w_end):
    print "same"
else:
    print "different"    

# print w
# numpy.savetxt("test200.label.35.ushioda_batch_multiple", w, fmt = '%i')

######################### test #########################


prediction_batch = numpy.empty(len(testdata))

for i, x in enumerate(norm_testdata):
    prediction_batch[i] = math.copysign(1, numpy.dot(x,w))

# numpy.savetxt("test200.label.35.ushioda_batch_multiple", prediction_batch, fmt = '%i') 
print prediction_batch