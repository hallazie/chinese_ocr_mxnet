#coding:utf-8

# @author: hallazie

'''
	
	the total char num is : 3593

'''

from mxnet import nd
from mxnet.gluon import nn

import mxnet.gluon as gluon
import mxnet as mx

vobsize = 100
batch_size = 2
model_prefix = '../params/ctc'
img_w = 256
img_h = 32
seq_l = 20
epoch_size = 32

lstm_num_hidden = 1024
lstm_num_layer = 1
ctx = mx.gpu(0)
epoch = 10
datasize = 128

class EncoderLayer(gluon.Block):
	def __init__(self, **kwargs):
		super(EncoderLayer, self).__init__(**kwargs)
		with self.name_scope():
			self.lstm = mx.gluon.rnn.LSTM(lstm_num_hidden, lstm_num_layer, bidirectional=True)
	def forward(self, x):
		x = x.transpose((0,3,1,2))
		x = x.flatten()
		x = x.split(num_outputs=32, axis=1)
		x = nd.concat(*[elem.expand_dims(axis=0) for elem in x], dim=0)
		x = self.lstm(x)
		x = x.transpose((1,0,2))
		return x

def backbone():
	net = nn.Sequential()
	net.add(
		gluon.nn.Conv2D(channels=32, kernel_size=(3,3), padding=(1,1), activation='relu'),
		gluon.nn.Conv2D(channels=32, kernel_size=(3,3), padding=(1,1),activation='relu'),
		gluon.nn.MaxPool2D(pool_size=2, strides=2),
		gluon.nn.Conv2D(channels=64, kernel_size=(3,3), padding=(1,1),activation='relu'),
		gluon.nn.Conv2D(channels=64, kernel_size=(3,3), padding=(1,1),activation='relu'),
		gluon.nn.MaxPool2D(pool_size=2, strides=2),
		gluon.nn.Conv2D(channels=128, kernel_size=(3,3), padding=(1,1),activation='relu'),
		gluon.nn.Conv2D(channels=128, kernel_size=(3,3), padding=(1,1),activation='relu'),
		gluon.nn.MaxPool2D(pool_size=2, strides=2),
		gluon.nn.Conv2D(channels=192, kernel_size=(3,3), padding=(1,1),activation='relu'),
		gluon.nn.Conv2D(channels=192, kernel_size=(3,3), padding=(1,1),activation='relu'),
		)
	net.hybridize()
	return net

def encoder():
	enc = gluon.nn.Sequential()
	enc.add(EncoderLayer())
	enc.add(gluon.nn.Dropout(0.3))
	return enc

def decoder():
	dec = mx.gluon.nn.Dense(units=vobsize+1, flatten=False)
	dec.hybridize()
	return dec

def assemble():
	net = gluon.nn.Sequential()
	with net.name_scope():
		net.add(backbone())
		net.add(encoder())
		net.add(decoder())
	return net

def dataiter():
	return mx.gluon.data.DataLoader(dataset, batch_size=batch_size, last_batch='discard', shuffle=True)

def train():
	net = assemble()
	net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
	exc = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate':0.005, })
	ctc_loss = gluon.loss.CTCLoss(weight=0.2)
	for e in range(epoch):
		loss = nd.zeros(1, ctx)
		for i in range(datasize):
			data = mx.nd.normal(loc=0, scale=1, shape=(batch_size, 1, img_w, img_h)).as_in_context(ctx)
			label = mx.nd.normal(loc=0, scale=1, shape=(batch_size, seq_l)).as_in_context(ctx)
			with mx.autograd.record():
				output = net(data)
				loss_ctc = ctc_loss(output, label)
				loss_ctc = (label!=-1).sum(axis=1)*loss_ctc
			loss_ctc.backward()
			loss = loss_ctc.mean().asnumpy()
			if i % 20 == 0:
				print 'epoch %s\t\tbatch %s\t\tloss=%s'%(e, i, loss)

'''
"D:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.11
cmake -G "Visual Studio 15 2017 Win64" -T cuda=9.1,host=x64 -DUSE_CUDA=1 -DUSE_CUDNN=1 -DUSE_NVRTC=1 -DUSE_OPENCV=1 -DUSE_OPENMP=1 -DUSE_BLAS=open -DUSE_LAPACK=1 -DUSE_DIST_KVSTORE=0 -DCUDA_ARCH_LIST=Common -DCUDA_TOOLSET=9.1 -DCUDNN_INCLUDE=E:\Env\cuDNN\cuda\include -DCUDNN_LIBRARY=E:\Env\cuDNN\cuda\lib\x64\cudnn.lib "E:\Env\incubator-mxnet"

'''

if __name__ == '__main__':
	train()
