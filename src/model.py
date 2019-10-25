import paddle
import paddle.fluid as fluid
from .metrics import *


def ensure_list(data):
    return data if isinstance(data, list) else [data]


def create_optimizer(name):
    learning_rate = 0.001
    if name == 'Adadelta':
        return fluid.optimizer.Adadelta(learning_rate)
    elif name == 'Adagrad':
        return fluid.optimizer.Adagrad(learning_rate)
    elif name == 'Adam':
        return fluid.optimizer.Adam(learning_rate)
    elif name == 'Adamax':
        return fluid.optimizer.Adamax(learning_rate)
    elif name == 'DecayedAdagrad':
        return fluid.optimizer.DecayedAdagrad(learning_rate)
    elif name == 'ExponentialMovingAverage':
        return fluid.optimizer.ExponentialMovingAverage(learning_rate)
    elif name == 'Ftrl':
        return fluid.optimizer.Ftrl(learning_rate)
    elif name == 'Lamb':
        return fluid.optimizer.LambOptimizer(learning_rate)
    elif name == 'LarsMomentum':
        return fluid.optimizer.LarsMomentum(learning_rate)
    elif name == 'ModelAverage':
        return fluid.optimizer.ModelAverage(learning_rate)
    elif name == 'Momentum':
        return fluid.optimizer.Momentum(learning_rate)
    elif name == 'Pipeline':
        return fluid.optimizer.PipelineOptimizer(learning_rate)
    elif name == 'RMSProp':
        return fluid.optimizer.RMSPropOptimizer(learning_rate)
    elif name == 'SGD':
        return fluid.optimizer.SGD(learning_rate)
    else:
        raise ValueError("unknown optimizer:" + name)


def create_loss(name, predict, label):
    if name == 'mse':
        return fluid.layers.mean(fluid.layers.square_error_cost(predict, label))
    elif name == 'mae':
        return fluid.layers.mean(fluid.layers.abs(fluid.layers.elementwise_sub(predict, label)))
    else:
        raise ValueError("unknown loss or metric:" + name)


def create_metric(name, predict, label):
    if name == 'psnr':
        return fluid.layers.mean(fluid.layers.py_func(psnr, predict, label))
    elif name == 'ssim':
        return fluid.layers.mean(fluid.layers.py_func(ssim, predict, label))
    else:
        return create_loss(name, predict, label)


class Model:
    def __init__(self, inputs, outputs, **kwargs):
        self.inputs = ensure_list(inputs)
        self.outputs = ensure_list(outputs)

    def compile(self, optimizer, loss, metrics=None, **kwargs):
        loss = ensure_list(loss)
        if len(loss) != len(self.outputs):
            raise ValueError("the number of loss must equal to outputs.")
        if isinstance(optimizer, str):
            optimizer = create_optimizer(optimizer)

        costs = [create_loss(name) if isinstance(name, str) else name for name in loss]
        loss = fluid.layers.sums(costs)

        metrics = [create_metric(name) if isinstance(name, str) else name for name in metrics]

        self.costs = costs
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer

        self.init_program = fluid.default_startup_program()
        self.train_program = fluid.default_main_program()
        self.test_program = self.train_program.clone(for_test=True)

        optimizer.minimize(loss)

        self.place = fluid.CUDAPlace(0)
        self.exe = fluid.Executor(place=self.place)

    def fit(self):
        pass

    def evaluate(self):
        pass

    def predict(self):
        pass

    def train_on_batch(self):
        pass

    def test_on_batch(self):
        pass

    def predict_on_batch(self):
        pass

    def fit_generator(self):
        pass

    def evaluate_generator(self):
        pass

    def predict_generator(self):
        pass

    def get_layer(self):
        pass
