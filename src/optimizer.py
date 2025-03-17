import torch
import math

class AdamWAlignOptimizer(torch.optim.Optimizer):
    """
    ADAM-WALIGN optimizer that integrates key concepts from ADOPT and cautious optimization
    to enhance update stability and convergence integrity.
    
    The optimizer reorders momentum updates and normalization, aligns adjustments via 
    gradient-specific masks, and refines update magnitude.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0.0, align_threshold=0.1, caution_factor=0.5):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= align_threshold <= 1.0:
            raise ValueError(f"Invalid align_threshold value: {align_threshold}")
        if not 0.0 <= caution_factor <= 1.0:
            raise ValueError(f"Invalid caution_factor value: {caution_factor}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, 
                        weight_decay=weight_decay, 
                        align_threshold=align_threshold,
                        caution_factor=caution_factor)
        super(AdamWAlignOptimizer, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(AdamWAlignOptimizer, self).__setstate__(state)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get parameters
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamWAlignOptimizer does not support sparse gradients')
                
                # Get optimizer parameters
                lr = group['lr']
                beta1, beta2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']
                align_threshold = group['align_threshold']
                caution_factor = group['caution_factor']
                
                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Previous gradient for alignment calculation
                    state['prev_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                # Get state variables
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                prev_grad = state['prev_grad']
                state['step'] += 1
                step = state['step']
                
                # Perform weight decay before momentum update (different from Adam)
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)
                
                # Calculate alignment mask based on current and previous gradients
                # This is a key feature of ADAM-WALIGN
                grad_norm = torch.norm(grad.data, p=2)
                prev_grad_norm = torch.norm(prev_grad, p=2)
                
                # Avoid division by zero
                if grad_norm > 0 and prev_grad_norm > 0:
                    cosine_sim = torch.sum(grad.data * prev_grad) / (grad_norm * prev_grad_norm)
                    # Create alignment mask where gradients are consistent
                    align_mask = (cosine_sim > align_threshold).float()
                else:
                    align_mask = torch.ones_like(grad.data)
                
                # Update biased first moment estimate (momentum)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # Compute bias-corrected moment estimates
                exp_avg_corrected = exp_avg / bias_correction1
                exp_avg_sq_corrected = exp_avg_sq / bias_correction2
                
                # Calculate the denominator
                denom = exp_avg_sq_corrected.sqrt().add_(eps)
                
                # Apply cautious factor to step size based on alignment
                # This combines ADOPT's resilience with cautious optimization
                # Convert align_mask to scalar if it's a tensor with single element
                if isinstance(align_mask, torch.Tensor) and align_mask.numel() == 1:
                    align_mask_scalar = align_mask.item()
                else:
                    # For multi-element tensors, use a uniform step size based on mean alignment
                    align_mask_scalar = align_mask.mean().item() if isinstance(align_mask, torch.Tensor) else align_mask
                
                step_size = lr * (1.0 - (1.0 - align_mask_scalar) * caution_factor)
                
                # Update parameters with alignment-aware step
                p.data.addcdiv_(exp_avg_corrected, denom, value=-step_size)
                
                # Store current gradient for next iteration's alignment calculation
                state['prev_grad'].copy_(grad.data)
        
        return loss
