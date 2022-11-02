
from collections import namedtuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function

QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])

_DEFAULT_FLATTEN = (1, -1) # channel-wise flatten 
_DEFAULT_FLATTEN_GRAD = (0, -1)


def _deflatten_as(x, x_full):
    shape = list(x.shape) + [1] * (x_full.dim() - x.dim())
    return x.view(*shape)

def calculate_qparams(x, num_bits, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, reduce_type='mean', keepdim=False, percentage=0.99999,
                      true_zero=False):
    with torch.no_grad():
        x_flat = x.flatten(*flatten_dims)
        if x_flat.dim() == 1:
            min_values = _deflatten_as(x_flat.topk(round((1-percentage) * x_flat.data.numel() + 0.5), largest=False)[0][-1], x)
            max_values = _deflatten_as(x_flat.topk(round((1-percentage) * x_flat.data.numel() + 0.5), largest=True)[0][-1], x)
        else:
            min_values = _deflatten_as(x_flat.topk(round((1-percentage) * x_flat.size(1) + 0.5), largest=False)[0].transpose(0,1)[-1], x)
            max_values = _deflatten_as(x_flat.topk(round((1-percentage) * x_flat.size(1) + 0.5), largest=True)[0].transpose(0,1)[-1], x)

        if reduce_dim is not None:
            if reduce_type == 'mean':
                min_values = min_values.mean(reduce_dim, keepdim=keepdim)
                max_values = max_values.mean(reduce_dim, keepdim=keepdim)
            else:
                min_values = min_values.min(reduce_dim, keepdim=keepdim)[0]
                max_values = max_values.max(reduce_dim, keepdim=keepdim)[0]

        range_values = max_values - min_values
        return QParams(range=range_values, zero_point=min_values,
                       num_bits=num_bits)

class UniformQuantize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, measurement=None, flatten_dims=_DEFAULT_FLATTEN,
                reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False, measure=False):

        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if measure:
            measurement(input, num_bits)
            return output 
        zero_point = measurement.running_zero_point
        qmin = -(2. ** (num_bits - 1)) if signed else 0.
        qmax = qmin + 2. ** num_bits - 1.
        scale = measurement.running_range / (qmax - qmin)

        min_scale = torch.tensor(1e-8).expand_as(scale).cuda() 
        scale = torch.max(scale, min_scale)

        with torch.no_grad():
            output.add_(qmin * scale - zero_point).div_(scale)
            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
            # quantize
            output.clamp_(qmin, qmax).round_()

            if dequantize:
                output.mul_(scale).add_(
                    zero_point - qmin * scale)  # dequantize
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None, None, None, None

class UniformQuantizeGrad(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, measurement=None, flatten_dims=_DEFAULT_FLATTEN_GRAD,
                reduce_dim=0, dequantize=True, signed=False, stochastic=True, measure=False):
        ctx.num_bits = num_bits
        ctx.measurement = measurement
        ctx.flatten_dims = flatten_dims
        ctx.stochastic = stochastic
        ctx.signed = signed
        ctx.dequantize = dequantize
        ctx.reduce_dim = reduce_dim
        ctx.measure = measure
        ctx.inplace = False
        return input

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            grad_input = quantize(grad_output, num_bits=ctx.num_bits,
                                  measurement=ctx.measurement, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim,
                                  dequantize=ctx.dequantize, signed=ctx.signed, stochastic=ctx.stochastic, inplace=False, measure=ctx.measure)
        return grad_input, None, None, None, None, None, None, None, None

def quantize(x, num_bits=None, measurement=None, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, dequantize=True, signed=False,
             stochastic=False, inplace=False, measure=False):
    if num_bits:
        return UniformQuantize().apply(x, num_bits, measurement, flatten_dims, reduce_dim, dequantize, signed, stochastic,
                                       inplace, measure)

    return x


def quantize_grad(x, num_bits=None, measurement=None, flatten_dims=_DEFAULT_FLATTEN_GRAD, reduce_dim=0, dequantize=True,
                  signed=False, stochastic=True, measure=False):
    if num_bits:
        return UniformQuantizeGrad().apply(x, num_bits, measurement, flatten_dims, reduce_dim, dequantize, signed,
                                           stochastic, measure)
    return x

class QuantMeasure(nn.Module):
    """docstring for QuantMeasure. 通过 一些预处理测量得到 在量化过程中的 scaling factor 以及 zero point 的位置， 这里的scaling factor, 为了方便起见，测量的是running range，而不是真正的scaling factor, 这里将其内容设置为仅仅修改相应的值，不进行forward的处理
    """

    def __init__(self, shape_measure=(1,), flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, reduce_type='extreme'):
        super(QuantMeasure, self).__init__()
        self.register_buffer('running_zero_point', torch.zeros(*shape_measure))
        self.register_buffer('running_range', torch.zeros(*shape_measure))
        self.register_buffer('num_measured', torch.zeros(1))
        self.flatten_dims = flatten_dims
        self.reduce_dim = reduce_dim
        self.reduce_type = reduce_type

    def forward(self, input, num_bits):
        qparams = calculate_qparams(
            input, num_bits=num_bits, flatten_dims=self.flatten_dims, reduce_dim=self.reduce_dim, reduce_type=self.reduce_type)
        with torch.no_grad():
            momentum = self.num_measured / (self.num_measured + 1)
            self.num_measured += 1
            self.running_zero_point.mul_(momentum).add_(
                qparams.zero_point * (1 - momentum))
            self.running_range.mul_(momentum).add_(
                qparams.range * (1 - momentum))

class QConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, measure=False):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.quantize_input = QuantMeasure(shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1))
        self.quantize_weight = QuantMeasure(shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1))
        self.quantize_output = QuantMeasure(shape_measure=(1, 1, 1, 1,), flatten_dims=(1, -1))
        self.measure = measure
        self.weight_grad = nn.Parameter(torch.zeros_like(self.weight))

    def forward(self, input, num_bits, num_grad_bits, lr):
        qinput = quantize(input, num_bits=num_bits, measurement=self.quantize_input, measure=self.measure)
        qweight = quantize(self.weight, num_bits=num_bits, measurement=self.quantize_weight, measure=self.measure)
        qweight = qweight - lr * self.weight_grad.detach()
        # base_output = F.conv2d(qinput, qweight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # grad_output = F.conv2d(qinput, self.weight_grad, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # output = base_output - lr * grad_output.detach()
        output = F.conv2d(qinput, qweight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        output = quantize_grad(output, num_bits=num_grad_bits, flatten_dims=(1, -1), measurement=self.quantize_output, measure=self.measure)

        return output
        
if __name__ == '__main__':
    x = torch.randn((128, 64, 56, 56))
    print(x.shape)
    x = x.cuda()
    model = QConv2d(64, 64, 3, stride=1, padding=1, bias=False, measure=True).cuda()
    out = model.forward(x, 8, 8)
    print(out.shape)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    loss = loss_fn(x, out)
    loss.backward()

