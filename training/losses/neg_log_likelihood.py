import torch
import numpy as np
import math
import torch.nn as nn
import pdb

class NLLMixtureLaplaceandGaussian:
    """ Computes Negative Log Likelihood loss for a Mixture of Laplace distributions and Gaussian distributions. """
    def __init__(self, reduction='weighted_sum_normalized', ratio=1.0):
        """
        Args:
            reduction: str, type of reduction to apply to loss
            ratio:
        """
        super().__init__()
        self.reduction = reduction
        self.ratio = ratio

    #def __call__(self, gt_flow, est_flow, log_var_alpha,log_var_beta, weight_map_alpha,weight_map_beta, mask=None):
    def __call__(self, gt_flow, est_flow, log_var ,weight_map, mask=None):
        """
        Args:
            gt_flow: ground-truth flow field, shape (b, 2, H, W)
            est_flow: estimated flow field, shape (b, 2, H, W)
            log_var_alpha: estimated log variances for alpha, shape (b, nbr_components, H, W)
            log_var_beta: estimated log variances for beta, shape (b, nbr_components, H, W)
            weight_map_alpha: estimated weights map (alpha) for the mixture model, each corresponding to a log variance,
                        shape (b, nbr_components, H, W)
            weight_map_beta: estimated weights map (beta) for the mixture model, each corresponding to a log variance,
            shape (b, nbr_components, H, W)
            mask: valid mask, where the loss is computed. shape (b, 1, H, W)
        """
        # divide into two parts
        b,c,h,w = weight_map.shape
        b,c1,h1,w1 = log_var.shape
        assert (c==c1 and h==h1 and w==w1), "must have same shape" 

        weight_map = nn.Softmax(dim=1)(weight_map)
        weight_map_alpha = weight_map[:,:2:,:,:] # *! this part alpha is for Gaussian beta is for Lapalacian
        weight_map_beta = weight_map[:,2:,:,:] # *? 大的用拉普拉斯分布，小的用高斯分布
        log_var_alpha = log_var[:,:2:,:,:]
        log_var_beta = log_var[:,2:,:,:]

        b, _, h, w = gt_flow.shape
        l1 = torch.logsumexp(weight_map_alpha, 1, keepdim=True)
        # shape will be b,1,h,w

        reg = math.sqrt(2)*torch.sum(torch.abs((gt_flow - est_flow)*self.ratio), 1, keepdim=True)  # shape will be b,1,h,w
        exponent = weight_map_alpha - math.log(2) - log_var_alpha - reg * torch.exp(-0.5*log_var_alpha)
        l2 = torch.logsumexp(exponent, 1, keepdim=True)

        loss1 = l1 - l2
        
        PI = torch.tensor(np.pi).cuda()
        l3 = torch.logsumexp(weight_map_beta, 1, keepdim=True)

        reg1 = 0.5 * torch.sum((gt_flow - est_flow) ** 2, 1, keepdim=True)
        exponent1 = weight_map_beta - torch.log(2 * PI) - log_var_beta - reg1 * torch.exp(-log_var_beta)
        l4 = torch.logsumexp(exponent1, 1, keepdim=True)
        loss2 = l3 - l4
        

        loss = 0.8*loss1 + 0.2*loss2

        if mask is not None:
            mask = ~torch.isnan(loss.detach()) & ~torch.isinf(loss.detach()) & mask
        else:
            mask = ~torch.isnan(loss.detach()) & ~torch.isinf(loss.detach())

        if torch.isnan(loss.detach()).sum().ge(1) or torch.isinf(loss.detach()).sum().ge(1):
            # debugging
            print('mask or inf in the loss ! ')

        if self.reduction == 'mean':
            if mask is not None:
                loss = torch.masked_select(loss, mask).mean()
            else:
                loss = loss.mean()
            return loss
        elif 'weighted_sum' in self.reduction:
            if mask is not None:
                loss = loss * mask.float()
                L = 0
                for bb in range(0, b):
                    norm_const = float(h)*float(w) / (mask[bb, ...].sum().float() + 1e-6)
                    L += loss[bb][mask[bb, ...] != 0].sum() * norm_const
                if 'normalized' in self.reduction:
                    return L / b
                return L

            if 'normalized' in self.reduction:
                return loss.sum() / b
            return loss.sum()
        else:
            raise ValueError
        
class NLLMixtureLaplace:
    """ Computes Negative Log Likelihood loss for a Mixture of Laplace distributions. """
    def __init__(self, reduction='weighted_sum_normalized', ratio=1.0):
        """
        Args:
            reduction: str, type of reduction to apply to loss
            ratio:
        """
        super().__init__()
        self.reduction = reduction
        self.ratio = ratio

    def __call__(self, gt_flow, est_flow, log_var, weight_map, mask=None):
        """
        Args:
            gt_flow: ground-truth flow field, shape (b, 2, H, W)
            est_flow: estimated flow field, shape (b, 2, H, W)
            log_var: estimated log variances, shape (b, nbr_components, H, W)
            weight_map: estimated weights map (alpha) for the mixture model, each corresponding to a log variance,
                        shape (b, nbr_components, H, W)
            mask: valid mask, where the loss is computed. shape (b, 1, H, W)
        """
        b, _, h, w = gt_flow.shape
        l1 = torch.logsumexp(weight_map, 1, keepdim=True)
        # shape will be b,1,h,w

        reg = math.sqrt(2)*torch.sum(torch.abs((gt_flow - est_flow)*self.ratio), 1, keepdim=True)  # shape will be b,1,h,w
        exponent = weight_map - math.log(2) - log_var - reg * torch.exp(-0.5*log_var)
        l2 = torch.logsumexp(exponent, 1, keepdim=True)

        loss = l1 - l2

        if mask is not None:
            mask = ~torch.isnan(loss.detach()) & ~torch.isinf(loss.detach()) & mask
        else:
            mask = ~torch.isnan(loss.detach()) & ~torch.isinf(loss.detach())

        if torch.isnan(loss.detach()).sum().ge(1) or torch.isinf(loss.detach()).sum().ge(1):
            # debugging
            print('mask or inf in the loss ! ')

        if self.reduction == 'mean':
            if mask is not None:
                loss = torch.masked_select(loss, mask).mean()
            else:
                loss = loss.mean()
            return loss
        elif 'weighted_sum' in self.reduction:
            if mask is not None:
                loss = loss * mask.float()
                L = 0
                for bb in range(0, b):
                    norm_const = float(h)*float(w) / (mask[bb, ...].sum().float() + 1e-6)
                    L += loss[bb][mask[bb, ...] != 0].sum() * norm_const
                if 'normalized' in self.reduction:
                    return L / b
                return L

            if 'normalized' in self.reduction:
                return loss.sum() / b
            return loss.sum()
        else:
            raise ValueError
        
class NLLMixtureGaussian:
    """ Computes Negative Log Likelihood loss for a Mixture of Gaussian distributions. """
    def __init__(self, reduction='weighted_sum_normalized', ratio=1.0):
        """
        Args:
            reduction: str, type of reduction to apply to loss
            ratio:
        """
        super().__init__()
        self.reduction = reduction
        self.ratio = ratio

    def __call__(self, gt_flow, est_flow, log_var, weight_map, mask=None):
        """
        Args:
            gt_flow: ground-truth flow field, shape (b, 2, H, W)
            est_flow: estimated flow field, shape (b, 2, H, W)
            log_var: estimated log variances, shape (b, nbr_components, H, W)
            weight_map: estimated weights map (alpha) for the mixture model, each corresponding to a log variance,
                        shape (b, nbr_components, H, W)
            mask: valid mask, where the loss is computed. shape (b, 1, H, W)
        """
        b, _, h, w = gt_flow.shape
        PI = torch.tensor(np.pi).cuda()
        l1 = torch.logsumexp(weight_map, 1, keepdim=True)

        reg = 0.5 * torch.sum((gt_flow - est_flow) ** 2, 1, keepdim=True)
        exponent = weight_map - torch.log(2 * PI) - log_var - reg * torch.exp(-log_var)
        l2 = torch.logsumexp(exponent, 1, keepdim=True)
        loss = l1 - l2

        if self.reduction == 'mean':
            if mask is not None:
                loss = torch.masked_select(loss, mask).mean()
            else:
                loss = loss.mean()
            return loss
        elif 'weighted_sum' in self.reduction:
            if mask is not None:
                loss = loss * mask.float()
                L = 0
                for bb in range(0, b):
                    norm_const = float(h*w) / (mask[bb, ...].sum().float() + 1e-6)
                    L += loss[bb][mask[bb, ...] != 0].sum() * norm_const
                if 'normalized' in self.reduction:
                    return L / b
                return L

            if 'normalized' in self.reduction:
                return loss.sum() / b
            return loss.sum()
        else:
            raise ValueError

        
class NLLLaplace:
    """ Computes Negative Log Likelihood loss for a (single) Laplace distribution. """
    def __init__(self, reduction='weighted_sum_normalized', ratio=1.0):
        """
        Args:
            reduction: str, type of reduction to apply to loss
            ratio:
        """
        super().__init__()
        self.reduction = reduction
        self.ratio = ratio

    def __call__(self, gt_flow, est_flow, log_var, mask=None):
        """
        Args:
            gt_flow: ground-truth flow field, shape (b, 2, H, W)
            est_flow: estimated flow field, shape (b, 2, H, W)
            log_var: estimated log variance, shape (b, 1, H, W)
            mask: valid mask, where the loss is computed. shape (b, 1, H, W)
        """
        b, _, h, w = gt_flow.shape
        loss1 = math.sqrt(2) * torch.exp(-0.5 * log_var) * torch.abs(gt_flow - est_flow)
        # each dimension is multiplied
        loss2 = 0.5 * log_var
        loss = loss1 + loss2

        if mask is not None:
            mask = ~torch.isnan(loss.detach()) & ~torch.isinf(loss.detach()) & mask
        else:
            mask = ~torch.isnan(loss.detach()) & ~torch.isinf(loss.detach())

        if torch.isnan(loss.detach()).sum().ge(1) or torch.isinf(loss.detach()).sum().ge(1):
            print('mask or inf in the loss ! ')

        if self.reduction == 'mean':
            if mask is not None:
                loss = torch.masked_select(loss, mask).mean()
            else:
                loss = loss.mean()
            return loss
        elif 'weighted_sum' in self.reduction:
            if mask is not None:
                loss = loss * mask.float()
                L = 0
                for bb in range(0, b):
                    norm_const = float(h) * float(w) / (mask[bb, ...].sum().float() + 1e-6)
                    L += loss[bb][mask[bb, ...] != 0].sum() * norm_const
                if 'normalized' in self.reduction:
                    return L / b
                return L

            if 'normalized' in self.reduction:
                return loss.sum() / b
            return loss
        else:
            raise ValueError


class NLLGaussian:
    """ Computes Negative Log Likelihood loss for a (single) Gaussian distribution. """
    def __init__(self, reduction='weighted_sum_normalized', ratio=1.0):
        """
        Args:
            reduction: str, type of reduction to apply to loss
            ratio:
        """
        super().__init__()
        self.reduction = reduction
        self.ratio = ratio

    def __call__(self, gt_flow, est_flow, log_var, mask=None):
        """
        Args:
            gt_flow: ground-truth flow field, shape (b, 2, H, W)
            est_flow: estimated flow field, shape (b, 2, H, W)
            log_var: estimated log variance, shape (b, 1, H, W)
            mask: valid mask, where the loss is computed. shape (b, 1, H, W)
        """
        b, _, h, w = gt_flow.shape
        loss1 = torch.mul(torch.exp(-log_var), (gt_flow - est_flow) ** 2)
        loss2 = log_var
        loss = .5 * (loss1 + loss2)

        if self.reduction == 'mean':
            if mask is not None:
                loss = torch.masked_select(loss, mask).mean()
            else:
                loss = loss.mean()
            return loss
        elif 'weighted_sum' in self.reduction:
            if mask is not None:
                loss = loss * mask.float()
                L = 0
                for bb in range(0, b):
                    norm_const = float(h*w) / (mask[bb, ...].sum().float() + 1e-6)
                    L += loss[bb][mask[bb, ...] != 0].sum() * norm_const
                if 'normalized' in self.reduction:
                    return L / b
                return L
    
            if 'normalized' in self.reduction:
                return loss.sum() / b
            return loss.sum()
        else:
            raise ValueError


class NLLGaussianWithHuber:
    """ Computes Negative Log Likelihood loss for a (single) Gaussian distribution,
    using Huber distance. From https://github.com/brdav/refign/blob/main/models/losses.py"""
    def __init__(self, reduction='mean'):
        """
        Args:
            reduction: str, type of reduction to apply to loss
        """
        super().__init__()
        self.reduction = reduction

    def huber_distance(self, input, target):
        # factor 2 so it makes sense in probabilistic setup
        return 2.0 * nn.functional.smooth_l1_loss(input, target, reduction='none')

    def __call__(self, gt_flow, est_flow, log_var, mask=None):
        """
        Args:
            gt_flow: ground-truth flow field, shape (b, 2, H, W)
            est_flow: estimated flow field, shape (b, 2, H, W)
            log_var: estimated log variance, shape (b, 1, H, W)
            mask: valid mask, where the loss is computed. shape (b, 1, H, W)
        """
        b, _, h, w = gt_flow.shape
        dist = torch.sum(self.huber_distance(est_flow, gt_flow), 1, keepdim=True)
        loss1 = 0.5 * torch.exp(-log_var) * dist
        loss2 = log_var
        loss = loss1 + loss2

        if self.reduction == 'mean':
            if mask is not None:
                loss = torch.masked_select(loss, mask).mean()
            else:
                loss = loss.mean()
        else:
            raise ValueError
        return loss