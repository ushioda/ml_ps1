# -*- coding: utf-8 -*-

import numpy

online = numpy.loadtxt("test200.label.35.ushioda")
batch = numpy.loadtxt("test200.label.35.ushioda_batch")

for i in range(len(online)):
    if not online[i] == batch[i]:
        print i, "online =", online[i], "batch =", batch[i]