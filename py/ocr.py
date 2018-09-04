#coding:utf-8

import mxnet as mx
import numpy as np
import logging

logging.getLogger().setLevel(logging.DEBUG)

from PIL import Image
from collections import namedtuple

import model

vobsize = 256
batch_size = 4
model_prefix = '../params/ctc'
ctx = mx.gpu(0)

Batch = namedtuple('Batch', ['data'])

def arguments():
    symbol = model.ctc(3000)
    arg_names = symbol.list_arguments()
    arg_shapes, out_shapes, aux_shapes = symbol.infer_shape(data=(4,1,256,64), label=(4,24))
    for name, shape in zip(arg_names, arg_shapes):
        print '%s\t\t%s'%(name, shape)
    print '%s\t\t%s'%('output', out_shapes[0])

def train():
	symbol = model.ctc(vobsize)
	data = mx.nd.normal(loc=0, scale=1, shape=(64, 1, 256, 64))
	label = mx.nd.normal(loc=0, scale=1, shape=(64, 24))
	dataiter = mx.io.NDArrayIter(
		data = data,
		label = label,
		batch_size = batch_size,
		shuffle = False,
		data_name='data',
		label_name='label'
	)
	symbol = mx.mod.Module(symbol=symbol, context=ctx, data_names=('data',), label_names=('label',))
	symbol.bind(data_shapes=dataiter.provide_data, label_shapes=dataiter.provide_label)
	symbol.init_params(initializer=mx.init.Uniform(scale=.1))
	symbol.fit(
		dataiter,
		optimizer = 'sgd',
		optimizer_params = {'learning_rate':0.005},
		eval_metric = 'loss',
		batch_end_callback = mx.callback.Speedometer(batch_size, 5),
		epoch_end_callback = mx.callback.do_checkpoint(model_prefix, 1),
		num_epoch = 10,
	)

def test():
	symbol = model.ctc(vobsize, False)
	dataiter = mx.io.NDArrayIter(data=mx.nd.normal(loc=0, scale=1, shape=(2, 1, 256, 64)), label=mx.nd.normal(loc=0, scale=1, shape=(2, 24)), batch_size=2, shuffle=True)
	symbol = mx.mod.Module(symbol=symbol, context=mx.cpu(0), data_names=('data',), label_names=('label',))
	symbol.bind(for_training=False, data_shapes=dataiter.provide_data)
	symbol.init_params(initializer=mx.init.Uniform(scale=.1))
	symbol.forward(Batch([mx.nd.ones((2,1,256,64))]))
	out = symbol.get_outputs()[0].asnumpy()
	print out.shape

if __name__ == '__main__':
	test()