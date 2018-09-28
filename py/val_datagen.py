# coding:utf-8

import random
import chardet
# import pygame
import os
import numpy as np

from PIL import Image

def gen(char_sheet, font, _id):
	idxs = [random.randint(0,len(char_sheet)) for e in range(random.randint(5,20))]
	text = [char_sheet[idx] for idx in idxs]
	imag = np.array(Image.new('RGB', [32,256], 'white'))
	cur_x, cur_y = 0, 0
	for i, c in enumerate(text):
		rtext = 255-pygame.surfarray.array2d(font.render(c, True, (0,0,0), (255,255,255)))
		w, h = rtext.shape
		try:
			imag[cur_x:cur_x+w,cur_y:cur_y+h,:] = np.minimum(imag[cur_x:cur_x+w,cur_y:cur_y+h,:], np.array([rtext,rtext,rtext]).transpose(1,2,0))
			cur_x += w
		except Exception as e:
			pass
	Image.fromarray(imag.astype('uint8').transpose((1,0,2))).save('../data/train/image/%s.png'%_id)
	with open('../data/train/label/%s.dat'%_id, 'w') as l:
		l.write(str(idxs))

def dataset():
	with open('../data/char_sheet.txt', 'r') as f:
		content = f.readline()
		print chardet.detect(content)
		char_sheet = list(content.decode('UTF-8-SIG'))

	idxs = [3483, 744, 290, 1439, 3269, 1263, 2562, 2244]
	print ''.join([char_sheet[i] for i in idxs])

	# pygame.init()
	# f1 = pygame.font.Font('../font/msyh.ttc',20)
	# f2 = pygame.font.Font('../font/simsun.ttc',20)
	# for i in range(5000):
	# 	try:
	# 		if i%2==0:
	# 			gen(char_sheet, f1, i)
	# 		else:
	# 			gen(char_sheet, f2, i)
	# 	except Exception as e:
	# 		print e

vobsize = 3593
batch_size = 4
model_prefix = '../params/ctc'
img_w = 256
img_h = 32
seq_l = 20
epoch_size = 256

def init_dataiter():
	data_path = '../data/train/image/'
	label_path = '../data/train/label/'
	for _,_, fs in os.walk(data_path):
		random.shuffle(fs)
		fs = fs[:epoch_size]
		data = np.zeros((len(fs), 1, img_w, img_h))
		label = np.zeros((len(fs), seq_l))
		for i, f in enumerate(fs):
			img = Image.open(data_path+f)
			with open(label_path+f.split('.')[0]+'.dat', 'r') as l:
				lbl = [-1 for i in range(20)]
				rdd = eval(l.readline().strip())
				lbl[:len(rdd)] = rdd
			data[i] = np.array(img.convert('L')).reshape((1, img_w, img_h))
			label[i] = np.array(lbl).reshape((seq_l))
		print 'data iter gen finished'
	print data.shape
	print label.shape

class T():
	def __init__(self, val):
		self.val = val
	def foo(self):
		self.val += 2
		return self
	def fee(self):
		self.val = self.val**2
		return self
	def fum(self):
		return self.val

def calc_norm(path):
	array_list = []
	norm, cnt = 0, 0
	for _,_,fs in os.walk(path):
		fs = sorted(fs)[:50]
		for f in fs:
			array = np.array(Image.open(path+f).convert('L'))
			# array = 255*(array-np.min(array))/(np.max(array)-np.min(array))
			array_list.append(array)
	for i in range(len(array_list)):
		for j in range(i):
			a1 = array_list[i]
			a2 = array_list[j]
			norm += np.linalg.norm(a1-a2)
			cnt += 1
		# print '%s finished'%(i)
	return norm/(cnt*(640*480))

if __name__ == '__main__':
	# init_dataiter()
	# t = T(3)
	# print t.foo().fee().fum()
	norm_sal = calc_norm('E:/Paper/feature_validation/data/norm/saliency_512/')
	norm_vgg = calc_norm('E:/Paper/feature_validation/data/norm/vgg_feature/')
	print 'norm sal: %s'%norm_sal
	print 'norm vgg: %s'%norm_vgg
	# # norm sal: 9251865169.961721
	# # norm vgg: 9803585796.040016
