#coding:utf-8

import numpy as np
import os
import random

from PIL import Image

img_w = 256
img_h = 32
seq_l = 20

def init_data():
	data_path = '../data/train/image/'
	label_path = '../data/train/label/'
	for _,_, fs in os.walk(data_path):
		random.shuffle(fs)
		fs = fs[:22]
		idx = 0
		while True:
			try:
				data = np.zeros((4, 1, img_w, img_h))
				label = np.zeros((4, seq_l))
				for j in range(4):
					f = fs[idx]
					img = Image.open(data_path+f)
					with open(label_path+f.split('.')[0]+'.dat', 'r') as l:
						lbl = [0 for i in range(20)]
						rdd = [e+1 for e in eval(l.readline().strip())]
						lbl[:len(rdd)] = rdd
					data[j] = np.array(img.convert('L')).reshape((1, img_w, img_h))
					label[j] = np.array(lbl).reshape((seq_l))
					idx += 1
				yield [data, label]
			except Exception as e:
				random.shuffle(fs)
				fs = fs[:22]
				idx = 0
				print 'iter stops, start again...'

if __name__ == '__main__':
	d = init_data()
	for i in range(100):
		print d.next()[0].shape
