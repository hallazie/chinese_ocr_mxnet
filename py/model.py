#coding:utf-8

import mxnet as mx

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