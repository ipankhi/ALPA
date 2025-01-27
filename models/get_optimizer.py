import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np 


class ALPA(nn.Module):
    def __init__(self, gamma_neg=0, gamma_pos=0, clip=0.05, alpha=0, beta=0):
        super(ALPA, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.alpha = alpha
        self.beta = beta

        self.a_0 = -(3/2)
        self.a_1 = 3/2
        self.b_1 = 0
        self.c_0 = -1
        self.c_1 = 1
        self.d_1 = 0 

    def pade_approximation_plus(self, p, a_0, a_1, b_1):
        return (a_0 + a_1 * p) / (1 + b_1 * p)
    def pade_approximation_minus(self, p, c_0, c_1, d_1):
        return (c_0 + c_1 * p) / (1 + d_1 * p)

    def forward(self, x, y):
        """
        :param x: (batch_size, num_classes)
        :param y: (batch_size, num_classes)
        :return: loss
        """
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg =  (xs_neg + self.clip).clamp(max=1)
        
        # Basic Pade approximation polynomials 
        L_plus = self.pade_approximation_plus(xs_pos, self.a_0, self.a_1, self.b_1)
        L_minus = self.pade_approximation_minus(xs_neg, self.c_0, self.c_1, self.d_1)
        loss = (self.alpha* y * torch.pow((xs_neg),self.gamma_pos) * L_plus) +(self.beta * (1-y) * torch.pow((xs_pos), self.gamma_neg) * L_minus)

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

    return -torch.sum(loss)


def make_loss_optimizer(args, model):
    criterion = ALPA( args.gamma_neg, args.gamma_pos, args.clip, args.alpha, args.beta)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    return optimizer, criterion     
