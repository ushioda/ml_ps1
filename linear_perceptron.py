# -*- coding: utf-8 -*-

import numpy
import math

# load the training data and labels

traindata = numpy.loadtxt("train2k.databw.35")
label = numpy.loadtxt("train2k.label.35")


# normalize each exmaple to have unit norm

norm_traindata = numpy.empty(traindata.size)
norm_traindata.shape = (len(traindata[:,0]),len(traindata[0,:]))

for i, row in enumerate(traindata):
    norm_traindata[i,:] = row / numpy.linalg.norm(row)
  
    
# create subsample of size 10

sub_data = norm_traindata[0:10,]
sub_label = label[0:10,]


# sample size and data dimension

N = len(sub_label)
dim = len(sub_data[0,:])


# initial value of w

w = numpy.zeros(dim)

# mistake recorder

mistake = numpy.empty(N)

# prediction recorder 

prediction = numpy.empty(N)

# loop 

for i, x in enumerate(sub_data):
    
    y_hat = math.copysign(1, numpy.dot(x,w))
    y = sub_label[i]
    prediction[i] = y_hat
    
    if y == y_hat:
        mistake[i] = 0
    elif y_hat == -1 and y == 1:
        w = w + x
        mistake[i] = 1   
    elif y_hat == 1 and y == -1:
        w = w - x
        mistake[i] = 1
        
        