#coding:utf-8

import mxnet as mx
import numpy as np
import logging

logging.getLogger().setLevel(logging.DEBUG)

from PIL import Image
from collections import namedtuple

import model

vobsize = 3000
batch_size = 4
model_prefix = '../params/ctc'
ctx = mx.gpu(0)

Batch = namedtuple('Batch', ['data'])

def train():
	symbol = model.ctc(vobsize)
	dataiter = mx.io.NDArrayIter(data=np.ones((500, 1, 256, 64)), label=np.ones((500, 24)), batch_size=batch_size, shuffle=True)
	symbol = mx.mod.Module(symbol=symbol, context=ctx, data_names=('data',), label_names=('softmax_label',))
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
	dataiter = mx.io.NDArrayIter(data=np.ones((1, 1, 256, 64)), label=np.ones((1, 24)), batch_size=1, shuffle=True)
	symbol = mx.mod.Module(symbol=symbol, context=mx.cpu(0), data_names=('data',), label_names=('softmax_label',))
	symbol.bind(for_training=False, data_shapes=dataiter.provide_data)
	symbol.init_params(initializer=mx.init.Uniform(scale=.1))
	symbol.forward(Batch([mx.nd.ones((1,1,256,64))]))
	out = symbol.get_outputs()[0][0].asnumpy()
	print out.shape



if __name__ == '__main__':
	test()