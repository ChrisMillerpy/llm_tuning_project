import torch.nn as nn
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

# LoRA implementation
class LoRALinear(nn.Module):
    """
    Implements a LoRA (Low-Rank Adaptation) linear layer.

    LoRA modifies a pre-trained linear layer by introducing trainable low-rank matrices.
    It enables fine-tuning large models efficiently by only training small additional parameters.

    Attributes:
        original_linear (nn.Linear): The original frozen linear layer.
        r (int): The rank of the LoRA adaptation.
        alpha (int): The scaling factor for LoRA.
        A (torch.nn.Parameter): Low-rank trainable matrix A.
        B (torch.nn.Parameter): Low-rank trainable matrix B.
    """
    def __init__(self, original_linear: nn.Linear, r: int, alpha: int = None):
        super().__init__()
        assert isinstance(original_linear, nn.Linear)
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
        in_dim = original_linear.in_features
        out_dim = original_linear.out_features
        self.r = r
        self.alpha = alpha if alpha else r

        device = original_linear.weight.device
        self.A = nn.Parameter(torch.empty(r, in_dim, device=device))
        self.B = nn.Parameter(torch.zeros(out_dim, r, device=device))
        
        # Initialise A with He initialization
        nn.init.kaiming_normal_(self.A, nonlinearity="linear")

    def forward(self, x):
        """
        Forward pass of the LoRA-modified linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with LoRA adaptation.
        """
        base_out = self.original_linear(x)
        lora_out = (x @ self.A.T) @ self.B.T
        return base_out + lora_out * (self.alpha / self.r)

def add_LoRA(model, lora_rank):
    # Make sure we have passed a Qwen Model
    assert isinstance(model, Qwen2ForCausalLM)

    # Inject LoRA to every layer of the model
    for layer in model.model.layers:
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=lora_rank)
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=lora_rank)
    
    # Return the model
    return model