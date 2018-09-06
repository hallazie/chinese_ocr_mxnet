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

# -----------------------------------------------------------------------------------------------------------------------

vobsize = 3593
batch_size = 2
model_prefix = '../params/ctc'
data_path = '../data/train/image/'
label_path = '../data/train/label/'
ctx = mx.gpu(0)
img_w = 256
img_h = 32
seq_l = 20
epoch_size = 256
epoch = 10
iterstop = 0

mod = namedtuple('mod', ['exc', 'symbol', 'data', 'label', 'arg_names', 'arg_dict', 'aux_dict', 'grd_dict'])

# -----------------------------------------------------------------------------------------------------------------------

def conv_block(data, num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), act_type='leaky', dilate=(0,0)):
	if dilate == (0,0):
		conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
	else:
		conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(2,2), dilate=(1,1))
	bn = mx.symbol.BatchNorm(data=conv)
	if act_type == 'leaky':
		act = mx.symbol.LeakyReLU(data=bn)
	elif act_type == 'none':
		act = bn
	else:
		act = mx.symbol.Activation(data=bn, act_type=act_type)
	return act

def pool_block(data, stride=(2,2), kernel=(2,2), pool_type='avg'):
	return mx.symbol.Pooling(data=data, stride=stride, kernel=kernel, pool_type=pool_type)

def ctc(vobsize, train=True):
	# 256 * 64 --> 64 * 8
	data = mx.symbol.Variable('data')
	label= mx.symbol.Variable('label')
	fweight = mx.symbol.Variable('f_weight')
	c1 = conv_block(data,32)
	c2 = conv_block(c1,32)
	p2 = pool_block(c2)
	c3 = conv_block(p2,64)
	c4 = conv_block(c3,64)
	p4 = pool_block(c4)
	c5 = conv_block(p4,128)
	c6 = conv_block(c5,128)
	c7 = conv_block(c6,128)
	p7 = pool_block(c7)
	c8 = conv_block(p7,192)
	c9 = conv_block(c8,192)
	tr = mx.symbol.transpose(c9, axes=(0,2,1,3))
	slc = []
	for i in range(32):
		sls = mx.symbol.slice(tr, begin=(None,i,None,None), end=(None,i+1,None,None))
		flt = mx.symbol.flatten(sls)
		fcn = mx.symbol.FullyConnected(flt, num_hidden=vobsize, flatten=False, weight=fweight, no_bias=True)
		slc.append(mx.symbol.expand_dims(fcn, axis=1))
	cat = mx.symbol.concat(*slc)
	out = mx.symbol.transpose(cat, axes=(1,0,2))
	if not train:
		return out
	ctc_loss = mx.symbol.contrib.ctc_loss(out, label)
	loss = mx.symbol.MakeLoss(ctc_loss)
	return loss

# -----------------------------------------------------------------------------------------------------------------------

def bind():
	symbol = ctc(vobsize+1)
	arg_names = symbol.list_arguments()
	aux_names = symbol.list_auxiliary_states()
	arg_shapes, output_shapes, aux_shapes = symbol.infer_shape(data = (batch_size, 1, img_w, img_h), label=(batch_size, seq_l))
	arg_array = [mx.nd.normal(shape=shape, ctx=ctx) for shape in arg_shapes]
	aux_array = [mx.nd.normal(shape=shape, ctx=ctx) for shape in aux_shapes]
	arg_dict = dict(zip(arg_names, arg_array))
	aux_dict = dict(zip(aux_names, aux_array))
	grd_dict = {}
	for name, shape in zip(arg_names, arg_shapes):
		print '%s\t\t%s'%(name, shape)
		if name in ['data', 'label']:
			continue
		grd_dict[name] = mx.nd.normal(shape=shape, ctx=ctx)
	exc = symbol.bind(
		ctx=ctx,
		args = arg_dict,
		args_grad = grd_dict,
		aux_states = aux_dict,
		grad_req = 'write'
		)
	[arg_names.remove(name) for name in ['data', 'label']]
	data, label = exc.arg_dict['data'], exc.arg_dict['label']
	return mod(exc = exc, symbol = symbol, data = data, label = label, arg_names = arg_names, arg_dict = arg_dict, aux_dict = aux_dict, grd_dict = grd_dict)

# -----------------------------------------------------------------------------------------------------------------------

def init_data():
	for _,_, fs in os.walk(data_path):
		fs = fs[:20]
		random.shuffle(fs)
		idx = 0
		while True:
			try:
				data = np.zeros((batch_size, 1, img_w, img_h))
				label = np.zeros((batch_size, seq_l))
				for j in range(batch_size):
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
				iterstop += 1
				print 'iter stops, start again...'

# -----------------------------------------------------------------------------------------------------------------------

def train():
	mod = bind()
	dataiter = init_data()
	optmzr = mx.optimizer.Adam(learning_rate=0.005, beta1=0.9, beta2=0.999, epsilon=1e-07)
	updater = mx.optimizer.get_updater(optmzr)
	step = 0
	while iterstop < epoch:
		data, label = dataiter.next()
		mod.data[:], mod.label[:] = data, label
		mod.exc.forward(is_train=True)
		mod.exc.backward()
		print mod.exc.outputs[0].asnumpy()

if __name__ == '__main__':
	train()