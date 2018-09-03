#coding:utf-8

import numpy as np

if __name__ == '__main__':
	arr = np.ones((32,64,40,8))
	print arr.shape
	arr = arr.transpose((0,2,1,3))
	print arr.shape