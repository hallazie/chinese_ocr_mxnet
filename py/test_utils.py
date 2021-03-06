import mxnet as mx
import numpy as np

ctx = mx.gpu(0)

def check_ctc_loss(acts, labels, loss_truth):
    in_var = mx.sym.Variable('data')

    t1 = mx.symbol.reshape(in_var, (6,10))
    t2 = mx.symbol.FullyConnected(t1, num_hidden=5, no_bias=True)
    t3 = mx.symbol.reshape(t2, (3,2,5))

    labels_var = mx.sym.Variable('label')
    ctc = mx.sym.contrib.ctc_loss(t3, labels_var)
    loss = mx.symbol.MakeLoss(ctc)

    arg_names = loss.list_arguments()
    arg_shapes,_,_ = loss.infer_shape(data=(6,2,5), label=(2,3))
    for name, shape in zip(arg_names, arg_shapes):
        print '%s\t\t\t%s'%(name, shape)
    # acts_nd = mx.nd.array(acts, ctx=ctx)
    # weight_nd = mx.nd.array(np.ones((5,10)), ctx=ctx)
    # labels_nd = mx.nd.array(labels, ctx=ctx)
    # exe = loss.bind(ctx=ctx, args=[acts_nd, weight_nd, labels_nd])
    arg_array = [mx.nd.normal(shape=shape, ctx=ctx) for shape in arg_shapes]
    exe = loss.bind(ctx=ctx, args=arg_array)
    exe.forward(is_train=True)
    exe.backward()
    outTest = exe.outputs[0]
    # make sure losses calculated with both modes are the same
    print '%s,%s'%(outTest.asnumpy(), loss_truth)
    print '----------'

def test_ctc_loss():
    # Test 1: check that batches are same + check against Torch WarpCTC
    acts = np.array([
        [[1.2, 3.4, 1.2, -0.1, -2.34], [1.2, 3.4, 1.2, -0.1, -2.34]],
        [[0.1, 0.2, 0.3, 0.22, 0.123], [0.1, 0.2, 0.3, 0.22, 0.123]],
        [[-15, -14, -13, -12, -11], [-15, -14, -13, -12, -11]],
        [[1.2, 3.4, 1.2, -0.1, -2.34], [1.2, 3.4, 1.2, -0.1, -2.34]],
        [[0.1, 0.2, 0.3, 0.22, 0.123], [0.1, 0.2, 0.3, 0.22, 0.123]],
        [[-15, -14, -13, -12, -11], [-15, -14, -13, -12, -11]]
        ], dtype=np.float32)
    labels = np.array([[22,3,1], [3,4,1]])
    true_loss = np.array([4.04789, 4.04789], dtype=np.float32) # from Torch
    check_ctc_loss(acts, labels, true_loss)
    # Test 2:
    # acts2 = np.array([
    #     [[-5, -4, -3, -2, -1], [1.2, 3.4, 1.2, -0.1, -2.34]],
    #     [[-10, -9, -8, -7, -6], [0.1, 0.2, 0.3, 0.22, 0.123]],
    #     [[-15, -14, -13, -12, -11], [-15, -14.2, -13.5, -12.2, -11.22]]], dtype=np.float32)
    # labels2 = np.array([[2, 3, 1], [2, 0, 0]], dtype=np.float32)
    # true_loss = np.array([7.3557, 5.4091], dtype=np.float32) # from Torch
    # check_ctc_loss(acts2, labels2, true_loss)

def get_symbol():
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('label')
    loss = mx.symbol.contrib.CTCLoss(data, label)
    return mx.symbol.MakeLoss(loss)

def train():
    symbol = get_symbol()
    data = np.array([
        [[-5, -4, -3, -2, -1], [1.2, 3.4, 1.2, -0.1, -2.34]],
        [[-10, -9, -8, -7, -6], [0.1, 0.2, 0.3, 0.22, 0.123]],
        [[-15, -14, -13, -12, -11], [-15, -14.2, -13.5, -12.2, -11.22]]], dtype=np.float32)
    label = np.array([[1, 3], [2, 0]], dtype=np.float32)
    print np.ndim(data)
    print np.ndim(label)
    dataiter = mx.io.NDArrayIter(data=data, label=label, batch_size=2)
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

import model

def foo():
    symbol = model.ctc(512)
    data_nd = mx.nd.array(np.zeros((4,1,256,64)), ctx=ctx)
    label_nd = mx.nd.array(np.zeros((4,24)), ctx=ctx)
    arg_shapes, out_shapes, aux_shapes = symbol.infer_shape(data=(4,1,256,64), label=(4,24))
    arg_names = symbol.list_arguments()
    args = [mx.nd.array(np.zeros(shape), ctx=ctx) for shape in arg_shapes]
    auxs = [mx.nd.array(np.zeros(shape), ctx=ctx) for shape in aux_shapes]
    grds = []
    map(lambda x,y:grds.append(mx.nd.array(np.zeros(y))) if x not in ['data','label'] else 0, arg_names, arg_shapes)
    exe = symbol.bind(ctx=ctx, args=args, aux_states=auxs, args_grad=grds)
    exe.forward(is_train=True)
    exe.backward()
    outTest = exe.outputs[0]
    print '%s'%(outTest.asnumpy())

if __name__ == '__main__':
    in_var = mx.sym.Variable('data')
    labels_var = mx.sym.Variable('label')
    ctc = mx.sym.contrib.ctc_loss(in_var, labels_var)
    loss = mx.symbol.MakeLoss(ctc)
    arg_shapes,_,_ = loss.infer_shape(data=(6,2,1000), label=(2,3))
    arg_array = [mx.nd.normal(shape=shape, ctx=ctx) for shape in arg_shapes]
    exe = loss.bind(ctx=ctx, args=arg_array)
    exe.forward(is_train=True)
    exe.backward()
    outTest = exe.outputs[0]
    print '%s'%(outTest.asnumpy())
