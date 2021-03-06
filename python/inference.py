#!/usr/bin/env python
"""
classify.py is an out-of-the-box image classifer callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
"""
import numpy as np
import os
import sys
import argparse
import glob
import time

import caffe


def main(argv):
	model = 'deploy.prototxt';
        weights = 'pesos.caffemodel';

	net = caffe.Net(model, weights, 'test');
	image = imread('example_4.png');
	res = net.forward(image);
	prob = res;
	print prob

if __name__ == '__main__':
    main(sys.argv)
