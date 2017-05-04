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
    
### create subsample of size 10

# sub_data = norm_traindata[0:10,]
# sub_label = label[0:10,]


### sample size and data dimension

N = len(label)
M = len(norm_testdata[:,0])
dim = len(norm_traindata[0,:])


######################### training #########################


### initial value of w

w = numpy.zeros(dim)
# w = numpy.loadtxt("w_1")

### mistake recorder

mistake = numpy.empty(N)

### training loop 

for i, x in enumerate(norm_traindata):
    
    y_hat = math.copysign(1, numpy.dot(x,w))
    y = label[i]
    
    if y == y_hat:
        mistake[i] = 0
    elif y_hat == -1 and y == 1:
        w = w + x
        mistake[i] = 1   
    elif y_hat == 1 and y == -1:
        w = w - x
        mistake[i] = 1

# numpy.savetxt("w_1", w, fmt='%10.7f', delimiter='\t')       

######################### test #########################


prediction = numpy.empty(M)

for i, x in enumerate(norm_testdata):
    prediction[i] = math.copysign(1, numpy.dot(x,w))
  
print prediction

### export data
  
numpy.savetxt("test200.label.35.ushioda_online", prediction, fmt = '%i')


######################### plot #########################

# cum_mistake = numpy.cumsum(mistake)
# pylab.xlabel("Number of Examples Seen")
# pylab.ylabel("Cumulative Number of Mistakes")
# pylab.plot(cum_mistake)
# pylab.show()