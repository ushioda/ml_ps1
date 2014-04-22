# -*- coding: utf-8 -*-

import numpy 
from pylab import imshow, show

imagedata = numpy.loadtxt("test200.databw.35")

# which example do you want to see?

example = 85

# plot 

image = imagedata[example,:]

image.shape = (28, 28)

imshow(image, interpolation = "nearest")
show()