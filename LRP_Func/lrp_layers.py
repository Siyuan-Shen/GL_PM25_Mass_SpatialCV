"""Layers for layer-wise relevance propagation.

Layers for layer-wise relevance propagation can be modified.

"""
import torch
from torch import nn
from .filter import relevance_filter
from torch.autograd import Variable

# Proportion of relevance scores that are allowed to pass.
top_k_percent = 0.04


class RelevancePropagationAdaptiveAvgPool2d(nn.Module):
    """Layer-wise relevance propagation for 2D adaptive average pooling.

    Attributes:
        layer: 2D adaptive average pooling layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(self, layer: torch.nn.AdaptiveAvgPool2d, eps: float = 1.0e-05) -> None:
        super().__init__()
        self.layer = layer
        self.eps = eps

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        a.requires_grad = True
        print('a.is_leaf?:', a.is_leaf)
        print('a_reqiure_grad:', a.requires_grad)
        z = self.layer.forward(a) + self.eps
       #z = Variable(z,requires_grad = True)
        #z = torch.flatten(z,1)
        #print('r.shape: ', r.shape, '\n z.shape: ', z.shape)
        s = (r / z).data
       #s = Variable(s,requires_grad = True)
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r


class RelevancePropagationAvgPool2d(nn.Module):
    """Layer-wise relevance propagation for 2D average pooling.

    Attributes:
        layer: 2D average pooling layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(self, layer: torch.nn.AvgPool2d, eps: float = 1.0e-05) -> None:
        super().__init__()
        self.layer = layer
        self.eps = eps

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        a.requires_grad = True
        print('a.is_leaf?:', a.is_leaf)
        print('a_reqiure_grad:', a.requires_grad)
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r


class RelevancePropagationMaxPool2d(nn.Module):
    """Layer-wise relevance propagation for 2D max pooling.

    Optionally substitutes max pooling by average pooling layers.

    Attributes:
        layer: 2D max pooling layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(self, layer: torch.nn.MaxPool2d, mode: str = "avg", eps: float = 1.0e-05) -> None:
        super().__init__()

        if mode == "avg":
            self.layer = torch.nn.AvgPool2d(kernel_size=(2, 2))
        elif mode == "max":
            self.layer = layer

        self.eps = eps

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        a.requires_grad = True
        print('a.is_leaf?:', a.is_leaf)
        print('a_reqiure_grad:', a.requires_grad)
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r


class RelevancePropagationConv2d(nn.Module):
    """Layer-wise relevance propagation for 2D convolution.

    Optionally modifies layer weights according to propagation rule. Here z^+-rule

    Attributes:
        layer: 2D convolutional layer.
        eps: a value added to the denominator for numerical stability.
    """

    def __init__(self, layer: torch.nn.Conv2d, mode: str = "z_plus", eps: float = 1.0e-5) -> None:
        super().__init__()
        self.layer = layer
        if mode == "z_plus":
            #self.layer.weight = torch.nn.Parameter(self.layer.weight.clamp(min=0.0))
            self.layer.weight = torch.nn.Parameter(self.layer.weight)
            #self.layer.bias = torch.nn.Parameter(torch.zeros_like(self.layer.bias))
        self.eps = eps

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        #r = relevance_filter(r, top_k_percent=top_k_percent)
        #print('a.requires_grad: ',a.requires_grad)
        # a.retain_grad()
        #a.requires_grad = True
        print('a.is_leaf?:', a.is_leaf)
        print('a_reqiure_grad:', a.requires_grad)
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        # print('z:',z,'\nz.requires_grad:',z.requires_grad)
        # print('s:',s,'\ns.requires_grad:',s.requires_grad)
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r


class RelevancePropagationLinear(nn.Module):
    """Layer-wise relevance propagation for linear transformation.

    Optionally modifies layer weights according to propagation rule. Here z^+-rule

    Attributes:
        layer: linear transformation layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(self, layer: torch.nn.Linear, mode: str = "z_plus", eps: float = 1.0e-05) -> None:
        super().__init__()

        self.layer = layer

        if mode == "z_plus":
            self.layer.weight = torch.nn.Parameter(self.layer.weight)
            #self.layer.weight = torch.nn.Parameter(self.layer.weight.clamp(min=0.0))
            self.layer.bias = torch.nn.Parameter(
                torch.zeros_like(self.layer.bias))

        self.eps = eps

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        #r = relevance_filter(r, top_k_percent=top_k_percent)
        print('a.is_leaf?:', a.is_leaf)
        z = self.layer.forward(a) + self.eps
        s = r / z
        c = torch.mm(s, self.layer.weight)
        r = (a * c).data
        return r


class RelevancePropagationFlatten(nn.Module):
    """Layer-wise relevance propagation for flatten operation.

    Attributes:
        layer: flatten layer.

    """

    def __init__(self, layer: torch.nn.Flatten) -> None:
        super().__init__()
        self.layer = layer

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        r = r.view(size=a.shape)
        return r


class RelevancePropagationReLU(nn.Module):
    """Layer-wise relevance propagation for ReLU activation.

    Passes the relevance scores without modification. Might be of use later.

    """

    def __init__(self, layer: torch.nn.ReLU) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r


class RelevancePropagationTanh(nn.Module):
    """Layer-wise relevance propagation for ReLU activation.

    Passes the relevance scores without modification. Might be of use later.

    """

    def __init__(self, layer: torch.nn.ReLU) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r


class RelevancePropagationBatchNorm(nn.Module):
    """Layer-wise relevance propagation for ReLU activation.

    Passes the relevance scores without modification. Might be of use later.

    """

    def __init__(self, layer: torch.nn.ReLU) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r


class RelevancePropagationDropout(nn.Module):
    """Layer-wise relevance propagation for dropout layer.

    Passes the relevance scores without modification. Might be of use later.

    """

    def __init__(self, layer: torch.nn.Dropout) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r


class RelevancePropagationIdentity(nn.Module):
    """Identity layer for relevance propagation.

    Passes relevance scores without modifying them.

    """

    def __init__(self, layer) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r
