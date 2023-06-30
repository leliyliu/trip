from re import L
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, required
from typing import List, Optional, Type
import warnings 
from collections import defaultdict, abc as container_abcs
import ipdb


class TripOptimizer(Optimizer):
    r"""Implements Trip Optimizer interface, 
    实际上需要设置三套参数： 分别为 
    1. 正常参数逻辑， 采用传统方式进行梯度更新
    2. 权重参数 for convolution，其分为了前后两对的内容，其中偶数代表了原始的权重，而奇数 (i+1) 表示了权重的梯度，并将所有的梯度更新的 权重的梯度当中。
    """

    def __init__(self, params, weight_params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize=False, 
                 foreach: Optional[bool] = None, len_trainloader=0, zero_gr=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, foreach=foreach)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(TripOptimizer, self).__init__(params, defaults)

        self.weight_param_groups = []
        weight_param_groups = list(weight_params) 
        if len(weight_param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(weight_param_groups[0], dict):
            weight_param_groups = [{'params': weight_param_groups}]

        for weight_param_group in weight_param_groups:
            self.add_weight_param_group(weight_param_group)
        
        self.len_trainloader = len_trainloader
        self.zero_grad_rate = zero_gr
        self.zero_grad_count = 0 if self.len_trainloader == 0 else int(
            self.len_trainloader * self.zero_grad_rate)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)

    def add_weight_param_group(self, weight_param_group):
        r"""Add a weight param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
                specific optimization options.
        """
        assert isinstance(weight_param_group, dict), "param group must be a dict"

        params = weight_param_group['params']
        if isinstance(params, torch.Tensor):
            weight_param_group['params'] = [params] # 这里将 tensor 转换为了 list 
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            weight_param_group['params'] = list(params) # 这里实际上并没有改变任何的值

        for param in weight_param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " + torch.typename(param))
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in weight_param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " +
                                 name)
            else:
                weight_param_group.setdefault(name, default)

        params = weight_param_group['params']
        if len(params) != len(set(params)):
            warnings.warn("optimizer contains a parameter group with duplicate parameters; "
                          "in future, this will cause an error; "
                          "see github.com/pytorch/pytorch/issues/40967 for more information", stacklevel=3)

        param_set = set()
        for group in self.weight_param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(weight_param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.weight_param_groups.append(weight_param_group)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        params_with_grad 存储了所有 包含 梯度的参数
        d_p_list 存储了对应的梯度大小
        state 得到对应的state 状态
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()    

        # 传统的梯度的更新，不需要额外的开销，不需要使用在权重中的更新    
        for group in self.param_groups:            
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            has_sparse_grad = False

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    if p.grad.is_sparse:
                        has_sparse_grad = True

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])


            sgd(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                dampening=group['dampening'],
                nesterov=group['nesterov'],
                maximize=group['maximize'],
                has_sparse_grad=has_sparse_grad,
                foreach=group['foreach'])


            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        # ipdb.set_trace()

        for weight_group in self.weight_param_groups:
            weight_params = []
            grad_params = []
            grad_values = [] 
            momentum_buffer_list = []

            if not isinstance(weight_group['params'], list):
                raise TypeError('optimizer params need to be organized in ordered collections')
            for index, p in enumerate(weight_group['params']):
                if index % 2: # grad params 
                    grad_params.append(p)
                else:
                    grad_values.append(p.grad)
                    # weight_params.append(p)
                    state = self.state[p.grad]
                    if 'momentum_buffer2' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer2'])

            grad_update(params=grad_params, 
                        grads=grad_values,
                        momentum_buffer_list=momentum_buffer_list,
                        momentum=weight_group['momentum'],
                        lr=weight_group['lr'])

            for p, momentum_buffer in zip(grad_values, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer2'] = momentum_buffer

        return loss

    def zero_grad2(self, set_to_none: bool = False):
        # if self.zero_grad_count > 1:
        #     self.zero_grad_count = self.zero_grad_count - 1
        # else:
        #     if self.len_trainloader > 0:
        #         self.zero_grad_count = 0 if self.len_trainloader == 0 else int(
        #             self.len_trainloader * self.zero_grad_rate)
        #     for weight_group in self.weight_param_groups: 
        #         if not isinstance(weight_group['params'], list):
        #             raise TypeError('optimizer params need to be organized in ordered collections')
        #         for index, p in enumerate(weight_group['params']):
        #             if index % 2 == 0: 
        #                 p.grad.zero_()

        for weight_group in self.weight_param_groups: 
            if not isinstance(weight_group['params'], list):
                raise TypeError('optimizer params need to be organized in ordered collections')
            for index, p in enumerate(weight_group['params']):
                if index % 2 == 0: 
                    p.grad.zero_()

    @torch.no_grad()
    def weight_update(self):
        for weight_group in self.weight_param_groups: 
            weight_params = []
            grad_params = [] 
            momentum_buffer_list = []
            has_sparse_grad = False

            if not isinstance(weight_group['params'], list):
                raise TypeError('optimizer params need to be organized in ordered collections')
            for index, p in enumerate(weight_group['params']):
                if index % 2: # grad params 
                    grad_params.append(p)
                else:
                    weight_params.append(p)
                    state = self.state[p]
                    if 'momentum_buffer3' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer3'])

            # sgd(weight_params, 
            #     grad_params, 
            #     momentum_buffer_list,
            #     weight_decay=weight_group['weight_decay'],
            #     momentum=weight_group['momentum'],
            #     lr=weight_group['lr'],
            #     dampening=weight_group['dampening'],
            #     nesterov=weight_group['nesterov'],
            #     maximize=weight_group['maximize'],
            #     has_sparse_grad=has_sparse_grad,
            #     foreach=weight_group['foreach'])

            update_grad2weight(weight_params=weight_params, 
                               weight_grads=grad_params,
                               momentum_buffer_list=momentum_buffer_list,
                               momentum=weight_group['momentum'],
                               lr=weight_group['lr'])
            
            # update momentum_buffers in state
            for p, momentum_buffer in zip(weight_params, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer3'] = momentum_buffer


            for grad_param in grad_params:
                grad_param.zero_()

            # for weight_param in weight_params:
            #     weight_param.grad.zero_()

def update_grad2weight(weight_params: List[Tensor], 
                       weight_grads: List[Tensor],
                       momentum_buffer_list: List[Optional[Tensor]],
                        # weight_decay=weight_decay,
                       momentum:float,
                       lr:float):
    for i, param in enumerate(weight_params):
        grad = weight_grads[i]
        # grad = grad.add(param, alpha=0.01)

        # if momentum != 0:
        #     buf = momentum_buffer_list[i]

        #     if buf is None:
        #         buf = torch.clone(grad).detach()
        #         momentum_buffer_list[i] = buf
        #     else:
        #         buf.mul_(momentum).add_(grad)
            
        #     grad = buf

        param.add_(grad, alpha=-lr)

def grad_update(params: List[Tensor], 
                grads: List[Tensor],
                momentum_buffer_list: List[Optional[Tensor]],
                # weight_decay=weight_decay,
                momentum:float,
                lr:float):        
    for i, param in enumerate(params):
        grad = grads[i]

        # origin_shape = grad.shape
        # grad = grad.view(origin_shape[0], -1)
        # flatten_shape = grad.shape

        # max_change_index = 0
        # max_sum_suqare = -1

        # for kernel_index in range(flatten_shape[1]):
        #     sum_square = 0
        #     for channel_index in range(flatten_shape[0]):
        #         sum_square += grad[channel_index][kernel_index]**2
        #     if max_sum_suqare < sum_square:
        #         max_change_index = kernel_index
        #         max_sum_suqare = sum_square

        # for channel_index in range(flatten_shape[0]):
        #     for kernel_index in range(flatten_shape[1]):
        #         if kernel_index != max_change_index:
        #             grad[channel_index][kernel_index] = 0

        # grad = grad.reshape(origin_shape)
        
        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(grad).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(grad)
            
            grad = buf

        # if momentum != 0:
        #     if len(momentum_buf) < len(params):
        #         momentum_buf.append(grad)
        #     else:
        #         # sum_threshold = (1 - pow(1 / momentum, momentum_exp)) / (1 - momentum)
        #         (momentum_buf[i]).add_(grad * pow(momentum, -1 * momentum_exp), alpha=1 - dampening)

        # if len(momentum_buf) > i:
        #     momentum_sum = momentum_buf[i]
        #     grad = momentum_sum.mul_(pow(momentum, momentum_exp))

        param.add_(grad)

def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
        has_sparse_grad: bool = None,
        foreach: bool = None,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_sgd
    else:
        func = _single_tensor_sgd

    func(params,
         d_p_list,
         momentum_buffer_list,
         weight_decay=weight_decay,
         momentum=momentum,
         lr=lr,
         dampening=dampening,
         nesterov=nesterov,
         has_sparse_grad=has_sparse_grad,
         maximize=maximize)

def _single_tensor_sgd(params: List[Tensor],
                       d_p_list: List[Tensor],
                       momentum_buffer_list: List[Optional[Tensor]],
                       *,
                       weight_decay: float,
                       momentum: float,
                       lr: float,
                       dampening: float,
                       nesterov: bool,
                       maximize: bool,
                       has_sparse_grad: bool):

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        alpha = lr if maximize else -lr
        param.add_(d_p, alpha=alpha)

def _multi_tensor_sgd(params: List[Tensor],
                      grads: List[Tensor],
                      momentum_buffer_list: List[Optional[Tensor]],
                      *,
                      weight_decay: float,
                      momentum: float,
                      lr: float,
                      dampening: float,
                      nesterov: bool,
                      maximize: bool,
                      has_sparse_grad: bool):

    if len(params) == 0:
        return

    if has_sparse_grad is None:
        has_sparse_grad = any([grad.is_sparse for grad in grads])

    if weight_decay != 0:
        grads = torch._foreach_add(grads, params, alpha=weight_decay)

    if momentum != 0:
        bufs = []

        all_states_with_momentum_buffer = True
        for i in range(len(momentum_buffer_list)):
            if momentum_buffer_list[i] is None:
                all_states_with_momentum_buffer = False
                break
            else:
                bufs.append(momentum_buffer_list[i])

        if all_states_with_momentum_buffer:
            torch._foreach_mul_(bufs, momentum)
            torch._foreach_add_(bufs, grads, alpha=1 - dampening)
        else:
            bufs = []
            for i in range(len(momentum_buffer_list)):
                if momentum_buffer_list[i] is None:
                    buf = momentum_buffer_list[i] = torch.clone(grads[i]).detach()
                else:
                    buf = momentum_buffer_list[i]
                    buf.mul_(momentum).add_(grads[i], alpha=1 - dampening)

                bufs.append(buf)

        if nesterov:
            torch._foreach_add_(grads, bufs, alpha=momentum)
        else:
            grads = bufs

    alpha = lr if maximize else -lr
    if not has_sparse_grad:
        torch._foreach_add_(params, grads, alpha=alpha)
    else:
        # foreach APIs dont support sparse
        for i in range(len(params)):
            params[i].add_(grads[i], alpha=alpha)