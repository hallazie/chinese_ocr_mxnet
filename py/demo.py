#coding:utf-8

import numpy as np

if __name__ == '__main__':
	# arr = np.ones((32,64,40,8))
	# print arr.shape
	# arr = arr.transpose((0,2,1,3))
	# print arr.shape
	a = [1,2,3,4,5,6,4,3,2,1,4,5,6]
	b = []
	map(lambda x:b.append(x) if x>4 else 0, a)
	print b