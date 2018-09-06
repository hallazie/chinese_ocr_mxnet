#coding:utf-8

'''
	
	the total char num is : 3593

'''

import mxnet as mx
import numpy as np
import logging
import os
import random

logging.getLogger().setLevel(logging.DEBUG)

from PIL import Image
from collections import namedtuple

import model

vobsize = 3593
batch_size = 2
model_prefix = '../params/ctc'
ctx = mx.gpu(0)
img_w = 256
img_h = 32
seq_l = 20
epoch_size = 256

Batch = namedtuple('Batch', ['data'])

def arguments():
    symbol = model.ctc(vobsize+1)
    arg_names = symbol.list_arguments()
    arg_shapes, out_shapes, aux_shapes = symbol.infer_shape(data=(batch_size,1,img_w,img_h), label=(batch_size,seq_l))
    for name, shape in zip(arg_names, arg_shapes):
        print '%s\t\t%s'%(name, shape)
    print '%s\t\t%s'%('output', out_shapes[0])

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
				lbl = [0 for i in range(20)]
				rdd = [e+1 for e in eval(l.readline().strip())]
				lbl[:len(rdd)] = rdd
			data[i] = np.array(img.convert('L')).reshape((1, img_w, img_h))
			label[i] = np.array(lbl).reshape((seq_l))
		print 'data iter gen finished'
	return mx.io.NDArrayIter(data=data, label=label, batch_size=batch_size, shuffle=True, data_name='data', label_name='label')

def train():
	symbol = model.ctc(vobsize+1)
	data = mx.nd.normal(loc=0, scale=1, shape=(epoch_size, 1, img_w, img_h))
	label = mx.nd.normal(loc=0, scale=1, shape=(epoch_size, seq_l))
	dataiter = init_dataiter()
	# dataiter = mx.io.NDArrayIter(data=mx.nd.normal(loc=0, scale=1, shape=(8, 1, img_w, img_h)), label=mx.nd.normal(loc=0, scale=1, shape=(8, seq_l)), batch_size=2, shuffle=True, data_name='data', label_name='label')
	symbol = mx.mod.Module(symbol=symbol, context=ctx, data_names=('data',), label_names=('label',))
	symbol.bind(data_shapes=dataiter.provide_data, label_shapes=dataiter.provide_label)
	symbol.init_params(initializer=mx.init.Uniform(scale=.1))
	symbol.fit(
		dataiter,
		optimizer = 'rmsprop',
		optimizer_params = {'learning_rate':0.005},
		eval_metric = 'loss',
		batch_end_callback = mx.callback.Speedometer(batch_size, 5),
		epoch_end_callback = mx.callback.do_checkpoint(model_prefix, 1),
		num_epoch = 100,
	)

def test():
	symbol = model.ctc(vobsize, False)
	dataiter = mx.io.NDArrayIter(data=mx.nd.normal(loc=0, scale=1, shape=(1, 1, img_w, img_h)), label=mx.nd.normal(loc=0, scale=1, shape=(1, seq_l)), batch_size=2, shuffle=True)
	symbol = mx.mod.Module(symbol=symbol, context=mx.cpu(0), data_names=('data',), label_names=('label',))
	symbol.bind(for_training=False, data_shapes=dataiter.provide_data)
	symbol.init_params(initializer=mx.init.Uniform(scale=.1))
	symbol.forward(Batch([mx.nd.ones((2,1,256,64))]))
	out = symbol.get_outputs()[0].asnumpy()
	print out.shape

if __name__ == '__main__':
	train()
