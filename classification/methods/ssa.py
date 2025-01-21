"""
This file is based on the code from: https://openreview.net/forum?id=BllUWdpIOA
"""
import logging

import torch.jit
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

import numpy as np
from copy import deepcopy
from methods.base import TTAMethod
from models.model import ResNetDomainNet126
from augmentations.transforms_cotta import get_tta_transforms
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy, SymmetricCrossEntropy, SoftLikelihoodRatio

logger = logging.getLogger(__name__)

def to_float(t):
    return t.float() if torch.is_floating_point(t) else t

def update_model_probs(x_ema, x, momentum=0.9):
    return momentum * x_ema + (1 - momentum) * x

@ADAPTATION_REGISTRY.register()
class SSA(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.use_weighting = cfg.ROID.USE_WEIGHTING
        self.use_prior_correction = cfg.ROID.USE_PRIOR_CORRECTION
        self.use_consistency = cfg.ROID.USE_CONSISTENCY
        self.momentum_probs = cfg.ROID.MOMENTUM_PROBS
        self.temperature = cfg.ROID.TEMPERATURE
        self.batch_size = cfg.TEST.BATCH_SIZE
        self.class_probs_ema = 1 / self.num_classes * torch.ones(self.num_classes).cuda()
        self.tta_transform = get_tta_transforms(self.img_size, padding_mode="reflect", cotta_augs=False)

        # setup loss functions
        self.sce = SymmetricCrossEntropy()
        self.slr = SoftLikelihoodRatio()
        self.ent = Entropy()

        # copy and freeze the source model
        if isinstance(model, ResNetDomainNet126):
            for module in model.modules():
                for _, hook in module._forward_pre_hooks.items():
                    if isinstance(hook, WeightNorm):
                        delattr(module, hook.name)
        
        self.src_model = deepcopy(self.model)
        for param in self.src_model.parameters():
            param.detach_()
            
        # CMF
        self.dual_kf = self.cfg.SSA.DUAL_KF
        self.hidden_model = deepcopy(self.model)
        for param in self.hidden_model.parameters():
            param.detach_()
        self.cmf_parameters = {
            "alpha": cfg.SSA.ALPHA,
            "beta": cfg.SSA.BETA,
        }
        
        # SSA
        eps = cfg.SSA.EPS 
        ss = 1.0
        if "swin_" in self.cfg.MODEL.ARCH:
            ss = 2.0
        elif "vit_" in self.cfg.MODEL.ARCH:
            eps = 1e-11
        self.ssa_parameters = {
            "kappa": cfg.SSA.KAPPA,
            "S": ss * eps,
        }
        
        self.learnable_model_state = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.learnable_model_state[n] = p.clone().detach()
        self.k = 0
        self.buffer_size = cfg.SSA.CHUNK_SIZE
        self.full_flag = False
        self.register_buffer('gradient_buffer', torch.zeros(self.buffer_size).float().cuda())
        self.register_buffer('step_buffer', torch.zeros(self.buffer_size).float().cuda())
        
        self.steady_cond = np.sqrt(ss * 2) 
        self.steady_state = False
        
        # Monitoring
        self.accum_step = torch.zeros(1, requires_grad=False).float().cuda()
        self.num_accum = 0.0
        
        self.models = [self.src_model, self.model, self.hidden_model]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()
        
    def setup_optimizer(self):
        self.lr = self.cfg.OPTIM.LR
        if "d2v" in self.cfg.MODEL.ARCH:
            return torch.optim.SGD(self.params,
                                   lr=self.lr,
                                   momentum=self.cfg.OPTIM.MOMENTUM,
                                   dampening=self.cfg.OPTIM.DAMPENING,
                                   weight_decay=self.cfg.OPTIM.WD,
                                   nesterov=self.cfg.OPTIM.NESTEROV)
        else:
            return torch.optim.SGD(self.params,
                                   lr=self.lr,
                                   momentum=self.cfg.OPTIM.MOMENTUM,
                                   dampening=self.cfg.OPTIM.DAMPENING,
                                   weight_decay=self.cfg.OPTIM.WD,
                                   nesterov=self.cfg.OPTIM.NESTEROV)
    
    @torch.no_grad()
    def sde_buffering(self, value, step):
        if self.k < self.buffer_size:
            self.gradient_buffer[self.k] = value
            self.step_buffer[self.k] = step
        else:
            self.full_flag = True
            self.gradient_buffer[:-1] = self.gradient_buffer[1:].clone() 
            self.gradient_buffer[-1] = value
            self.step_buffer[:-1] = self.step_buffer[1:].clone() 
            self.step_buffer[-1] = step
        self.k += 1
        
    @torch.no_grad()
    def estimate_variance(self):
        observation = self.model.state_dict()
        prev_observation = self.prev_model.state_dict()
        
        total_numel = 0
        total_delta_g = 0.0
        for k in self.learnable_model_state.keys():
            fp32_param = to_float(observation[k].data)
            fp32_prev_param = to_float(prev_observation[k].data)
            
            delta_g = (fp32_prev_param - fp32_param) / self.lr # approximation
            
            total_delta_g += delta_g.sum()
            total_numel += delta_g.numel()
            
        delta_g = total_delta_g / total_numel
        
        local_time = self.buffer_size #self.step_buffer.sum()
        mean_delta_g = self.gradient_buffer.sum() / local_time
        var_t = (self.gradient_buffer - mean_delta_g).square().sum() + \
            (delta_g - mean_delta_g).square()
        var_t = var_t / (local_time+1)
        
        return var_t
    
    @torch.no_grad()
    def bayesian_filtering(
        self,
        bwe=True
    ): 
        
        # 1. Inference variance
        if bwe:
            step = 1.0
            var_t = self.estimate_variance()
            if self.full_flag:
                S = self.ssa_parameters["S"]
                proposal_step = torch.sqrt(S / var_t) / self.lr
                # print(proposal_step, self.steady_state)
                
                # if proposal_step < self.steady_cond or self.steady_state:
                self.steady_state = True
                    
                if self.steady_state:
                    step = proposal_step
            
        # 2. Inference mean
        src_model, model, hidden_model = self.models
        prev_model = self.prev_model
        
        total_numel = 0
        total_delta_g = 0.0
        zip_models = zip(src_model.parameters(), model.parameters(), hidden_model.parameters(), prev_model.parameters())
        for models_param in zip_models:
            src_param, param, hidden_param, prev_param = models_param
            if param.requires_grad:
                param_ = param.data
                prev_param_ = prev_param.data
                src_param_ = src_param.data
                
                if bwe:
                    # (CMF) Prediction step
                    if self.dual_kf:
                        hidden_param_ = hidden_param.data
                        predicted_hidden_param = self.cmf_parameters["alpha"] * hidden_param_ + (1 - self.cmf_parameters["alpha"]) * src_param_
                    # Prediction step
                    if self.full_flag:
                        predicted_param = (1 - step) * prev_param_ + step * param_
                    else:
                        predicted_param = param_
                    
                    # (CMF) Update step
                    if self.dual_kf:
                        updated_hidden_param = self.cmf_parameters["beta"] * predicted_hidden_param + (1 -  self.cmf_parameters["beta"]) * predicted_param
                        hidden_param.data = updated_hidden_param
                    else:
                        updated_hidden_param = src_param_
                    # Update step
                    updated_param = (1 - self.ssa_parameters["kappa"]) * predicted_param + self.ssa_parameters["kappa"] * updated_hidden_param 
                    param.data = updated_param
                
                # new statistics
                delta_g = (prev_param_ - param_) / self.lr
                total_delta_g += delta_g.sum()
                total_numel += delta_g.numel()
                
        if total_numel > 0:
            delta_g = total_delta_g / total_numel
            self.sde_buffering(delta_g, step)
        else:
            raise ValueError()
        
        local_time = self.buffer_size
        mean_delta_g = self.gradient_buffer.sum() / local_time
        var_t = (self.gradient_buffer - mean_delta_g).square().sum() / local_time
        # print(var_t.item())
        
        return model, hidden_model, var_t
    
    def loss_calculation(self, x, roid_loss=False):
        imgs_test = x[0]
        outputs = self.model(imgs_test)
        
        if roid_loss:
            if self.use_weighting:
                with torch.no_grad():
                    # calculate diversity based weight
                    weights_div = 1 - F.cosine_similarity(self.class_probs_ema.unsqueeze(dim=0), outputs.softmax(1), dim=1)
                    weights_div = (weights_div - weights_div.min()) / (weights_div.max() - weights_div.min())
                    mask = weights_div < weights_div.mean()

                    # calculate certainty based weight
                    weights_cert = - self.ent(logits=outputs)
                    weights_cert = (weights_cert - weights_cert.min()) / (weights_cert.max() - weights_cert.min())

                    # calculate the final weights
                    weights = torch.exp(weights_div * weights_cert / self.temperature)
                    weights[mask] = 0.

                    self.class_probs_ema = update_model_probs(x_ema=self.class_probs_ema, x=outputs.softmax(1).mean(0),
                                                            momentum=self.momentum_probs)

            # calculate the soft likelihood ratio loss
            loss_out = self.slr(logits=outputs)

            # weight the loss
            if self.use_weighting:
                loss_out = loss_out * weights
                loss_out = loss_out[~mask]
        else:
            loss_out = self.ent(logits=outputs)
        loss = loss_out.sum() / self.batch_size

        if roid_loss:      
            # calculate the consistency loss
            if self.use_consistency:
                outputs_aug = self.model(self.tta_transform(imgs_test[~mask]))
                loss += (self.sce(x=outputs_aug, x_ema=outputs[~mask]) * weights[~mask]).sum() / self.batch_size

        return outputs, loss

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        self.prev_model = deepcopy(self.model)
        for param in self.prev_model.parameters():
            param.detach_()
        
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss = self.loss_calculation(x)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs, loss = self.loss_calculation(x)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()    

        with torch.no_grad():
            self.model, self.hidden_model, step = self.bayesian_filtering()
            if step is not None:
                self.accum_step += step
                self.num_accum += 1

            if self.use_prior_correction:
                prior = outputs.softmax(1).mean(0)
                smooth = max(1 / outputs.shape[0], 1 / outputs.shape[1]) / torch.max(prior)
                smoothed_prior = (prior + smooth) / (1 + smooth * outputs.shape[1])
                outputs *= smoothed_prior

        return outputs

    def reset(self):
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()
        self.class_probs_ema = 1 / self.num_classes * torch.ones(self.num_classes).to(self.device)
        
    def collect_params(self):
        """Collect the affine scale + shift parameters from normalization layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def configure_model(self):
        """Configure model."""
        self.model.eval()
        self.model.requires_grad_(False)
        # re-enable gradient for normalization layers
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
