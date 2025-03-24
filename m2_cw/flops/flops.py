from dataclasses import dataclass

@dataclass
class FlopsConfig:
    hidden_features: int = 4864
    vocabulary_size: int = 13
    embedding_dimension: int = 896
    sequence_length: int = 512
    attention_heads: int = 14
    transformer_layers: int = 24
    lookup_table: bool = False
    mode: str = "training"
    generation_length: int = 20 * 13
    batch_size: int = 4
    lora_rank: int = 4

@dataclass
class InferenceConfig(FlopsConfig):
    sequence_length: int = 512
    mode: str = "inference"
    generation_length: int = 20 * 13
    batch_size: int = 1
    lora_rank: int = 4

@dataclass
class TrainConfig(FlopsConfig):
    sequence_length: int = 512
    mode: str = "training"
    batch_size: int = 4
    lora_rank: int = 4


def flops_matrix_mult(matrix_1_dim, matrix_2_dim):
    """
    Calculate FLOPs for a matrix multiplication.
    """
    a, b = matrix_1_dim
    c, d = matrix_2_dim
    assert b == c

    flops = a * d * (2 * b - 1)
    return flops


def flops_linear(in_features: int, out_features: int, sequence_length: int, bias: bool = True) -> int:
    """
    Calculate FLOPs for a linear (fully connected) layer.
    WX + B
    """
    m1 = (out_features, in_features) # W
    m2 = (in_features, sequence_length) # X
    flops = flops_matrix_mult(m1, m2) # W@X

    if bias:
        flops += out_features * sequence_length # + B
    return flops

def flops_silu(in_features: int, sequence_length: int) -> int:
    """
    Estimate FLOPs for SiLU (Sigmoid Linear Unit) activation.
    """
    return 13 * in_features * sequence_length

def flops_mlp(config: FlopsConfig) -> int:
    """
    Compute FLOPs for a feed-forward MLP block with SiLU activation.
    """
    flops = 0
    ed, hf, sl = config.embedding_dimension, config.hidden_features, config.sequence_length
    flops += flops_linear(ed, hf, sl, bias=False)
    flops += flops_linear(ed, hf, sl, bias=False)
    flops += hf * sl
    flops += flops_linear(hf, ed, sl, bias=False)
    flops += flops_silu(ed, sl)
    return flops

def flops_rmsnorm(in_features: int, sequence_length: int) -> int:
    """
    Estimate FLOPs for RMSNorm.
    """
    return (3 * in_features + 11) * sequence_length

def flops_lora(config: FlopsConfig) -> int:
    """
    Estimate FLOPs for a LoRA (Low-Rank Adaptation) layer.
    """
    flops = 0
    if config.lora_rank == 0:
        return flops
    ed, r, sl, heads = config.embedding_dimension, config.lora_rank, config.sequence_length, config.attention_heads
    hd = ed / heads # Calculate the dimension of each head

    datam = (hd, sl) # Data matrix
    LoRA1 = (r, hd) # LoRA matrix 1
    resultm = (r, sl) # result = LoRA1 @ datam
    LoRA2 = (hd, r) # LoRA matrix 2
    flops += flops_matrix_mult(LoRA1, datam) # LoRA1 @ datam
    flops += flops_matrix_mult(LoRA2, resultm) # LoRA2 @ resultm

    return flops

def flops_self_attention(config: FlopsConfig) -> int:
    """
    Compute FLOPs for a multi-head self-attention block.
    """
    flops = 0
    ed, sl, heads, mode = config.embedding_dimension, config.sequence_length, config.attention_heads, config.mode

    for _ in range(heads):
        head_dim = ed // heads

        if mode == "training":
            flops += flops_linear(head_dim, head_dim, sl) # Q
            flops += flops_lora(config) # LoRA Q
            flops += flops_linear(head_dim, head_dim, sl) # K
            flops += flops_lora(config) # LoRA K
            flops += flops_linear(head_dim, head_dim, sl) # V
        else:
            # Reduced computation because we have cached K, Q, V vectors from the last stage
            flops += flops_linear(head_dim, head_dim, 1) # Q_N+1
            flops += flops_lora(config) # LoRA Q
            flops += flops_linear(head_dim, head_dim, 1) # K_N+1 
            flops += flops_lora(config) # LoRA K
            flops += flops_linear(head_dim, head_dim, 1) # V_N+1

        flops += 2 * head_dim * sl # Adding lora result back onto K and Q matrices

        if mode == "training":
            flops += flops_matrix_mult((sl, head_dim), (head_dim, sl)) # K^T Q
            flops += sl * sl + 10 # Normalise by 1 / sqrt(d)
        else:
            flops += sl * (2 * head_dim - 1) # Only compute the new attention values
            flops += sl + 10 # Normalise only the relevant new ones

        # Softmax the attentions matrix for each column
        flops += sl * sl # add the attention mask to each value in the attention matrix
        flops += sl * ((10 * sl) + (sl - 1) + (sl)) # softmax columns num_cols * (exp each element + add them together + divide each one)
        flops += flops_matrix_mult((sl, sl), (sl, head_dim)) # attention @ values

    # recombination linear layer after each attention head is done
    flops += flops_linear(ed, ed, sl, bias=False)
    return flops

def flops_transformer_block(config: FlopsConfig) -> int:
    """
    Compute FLOPs for a full transformer block, including self-attention and MLP.
    """
    ed, sl = config.embedding_dimension, config.sequence_length
    flops = 0
    flops += flops_rmsnorm(ed, sl)
    flops += flops_self_attention(config)
    flops += ed * sl
    flops += flops_rmsnorm(ed, sl)
    flops += flops_mlp(config)
    flops += ed * sl
    return flops

def flops_embedding(config: FlopsConfig) -> int:
    """
    Estimate FLOPs for computing token embeddings.
    """
    if config.lookup_table:
        return 0
    return config.sequence_length * config.embedding_dimension * (2 * config.vocabulary_size - 1)

def flops_qwen(config: FlopsConfig) -> int:
    """
    Estimate total FLOPs for a QWen-like transformer model.
    """
    flops = 0
    mode = config.mode

    # Config for a full pass with no KV
    first_pass_config = config
    first_pass_config.mode = "training"

    # First pass through the network
    flops += flops_embedding(first_pass_config) # Embedding

    for _ in range(config.transformer_layers): # 24 Transformer Blocks
        flops += flops_transformer_block(first_pass_config)

    flops += flops_rmsnorm(first_pass_config.embedding_dimension, first_pass_config.sequence_length) # Normalises
    flops += flops_linear(first_pass_config.embedding_dimension, first_pass_config.vocabulary_size, first_pass_config.sequence_length) # LM Head

    if mode == "training": # If training, we just do batch size forward passes and then one backpass
        return flops * (config.batch_size + 2) # Return flops for each sequence in batch and +2 for backpass

    elif mode == "inference": # If inferencing we then run KV-Cached forward passes with reduced computational cost
        second_pass_config = config
        second_pass_config.mode = "inference"

        for _ in range(second_pass_config.generation_length - 1): # The remaining generation passes
            flops += flops_embedding(second_pass_config) # embedding
            for _ in range(second_pass_config.transformer_layers): # kv-cache transformer layers
                flops += flops_transformer_block(second_pass_config)
            
            flops += flops_rmsnorm(second_pass_config.embedding_dimension, 1) # Noramlise just the final token because we just want to predict only next token
            flops += flops_linear(second_pass_config.embedding_dimension, second_pass_config.vocabulary_size, 1) # Pass through LM_head

        return flops

    else:
        raise ValueError(f"Invalid mode: {mode}")
