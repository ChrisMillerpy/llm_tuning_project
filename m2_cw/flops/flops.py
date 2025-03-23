### DEFAULT VALUES
hidden_features = 4864
vocabulary_size = 13
embedding_dimension = 896
sequence_length = 512
attention_heads = 14
transformer_layers = 24
lookup_table = False
mode = "training"
generation_length = 20 * 13
batch_size = 1
lora_rank = 4

def flops_linear(in_features,
                 out_features,
                 sequence_length, 
                 bias=True):
    """
    Compute the number of floating-point operations (FLOPs) for a linear (fully connected) layer.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        sequence_length (int): Length of the input sequence.
        bias (bool): Whether the layer includes a bias term.

    Returns:
        int: Total FLOPs for the linear layer.
    """
    flops = 0
    flops += sequence_length * out_features * (2 * in_features - 1) # Flops for inputs * weights
    if bias: flops += out_features * sequence_length # Flops for bias
    return flops

def flops_silu(in_features,
               sequence_length):
    """
    Estimate FLOPs for SiLU (Sigmoid Linear Unit) activation.

    Args:
        in_features (int): Number of input features.
        sequence_length (int): Length of the input sequence.

    Returns:
        int: Total FLOPs for SiLU activation.
    """
    flops = 13 * in_features * sequence_length
    return flops

def flops_mlp(in_features=embedding_dimension,
              hidden_features=hidden_features,
              out_features=embedding_dimension,
              sequence_length=sequence_length):
    """
    Compute the FLOPs for a feed-forward MLP block with SiLU-based activation.

    Args:
        in_features (int): Number of input features.
        hidden_features (int): Number of hidden units in the MLP.
        out_features (int): Number of output features.
        sequence_length (int): Length of the input sequence.

    Returns:
        int: Total FLOPs for the MLP block.
    """
    flops = 0
    flops += flops_linear(in_features=in_features, out_features=hidden_features, sequence_length=sequence_length, bias=False) # up projection flops
    flops += flops_linear(in_features=in_features, out_features=hidden_features, sequence_length=sequence_length, bias=False) # gate projection flops
    flops += hidden_features * sequence_length # up proj * gate proj flops
    flops += flops_linear(in_features=hidden_features, out_features=out_features, sequence_length=sequence_length, bias=False) # down projection flops
    flops += flops_silu(in_features=out_features, sequence_length=sequence_length) # Flops of swiglu activation
    return flops

def flops_rmsnorm(in_features,
                  sequence_length):
    """
    Estimate FLOPs for RMSNorm normalization.

    Args:
        in_features (int): Number of input features.
        sequence_length (int): Length of the input sequence.

    Returns:
        int: Total FLOPs for RMSNorm.
    """
    flops = ( 3 * in_features + 11 ) * sequence_length
    return flops

def flops_self_atte(embedding_dimension=embedding_dimension,
                    attention_heads=attention_heads,
                    sequence_length=sequence_length,
                    mode=mode,
                    lora_rank=lora_rank):
    """
    Compute FLOPs for a multi-head self-attention block.

    Args:
        embedding_dimension (int): Dimension of the embedding vectors (should be divisible by attention_heads).
        attention_heads (int): Number of attention heads.
        sequence_length (int): Length of the input sequence.

    Returns:
        int: Total FLOPs for the self-attention block.
    """
    flops = 0
    for _ in range(attention_heads):
        head_features = embedding_dimension // attention_heads
        if mode == "training":
            flops += flops_linear(in_features=head_features,
                                out_features=head_features,
                                sequence_length=sequence_length,
                                bias=True) # flops for queries linear layer

            flops += flops_lora(embedding_dimension=embedding_dimension,
                                lora_rank=lora_rank,
                                sequence_length=sequence_length) # flops for lora on queries

            flops += flops_linear(in_features=head_features,
                                out_features=head_features,
                                sequence_length=sequence_length,
                                bias=True) # flops for keys linear layer

            flops += flops_lora(embedding_dimension=embedding_dimension,
                                lora_rank=lora_rank,
                                sequence_length=sequence_length) # flops for lora on keys

            flops += flops_linear(in_features=head_features,
                                out_features=head_features,
                                sequence_length=sequence_length,
                                bias=True) # flops for values linear layer
        elif mode == "inference":
            flops += flops_linear(in_features=head_features, 
                                  out_features=head_features,
                                  sequence_length=1,
                                  bias=True) # Flops for calculating the next query

            flops += flops_lora(embedding_dimension=embedding_dimension,
                                sequence_length=sequence_length,
                                lora_rank=lora_rank) # flops for lora on queries

            flops += flops_linear(in_features=head_features,
                                  out_features=head_features,
                                  sequence_length=1,
                                  bias=True) # Flops for next key

            flops += flops_lora(embedding_dimension=embedding_dimension,
                                lora_rank=lora_rank,
                                sequence_length=sequence_length) # flops for lora on queries

            flops += flops_linear(in_features=head_features,
                                  out_features=head_features,
                                  sequence_length=1,
                                  bias=True) # Flops for next query

        flops += 2 * head_features * sequence_length # flops for adding the rotary positional embeddings
        if mode == "training":
            flops += sequence_length * sequence_length * ( 2 * head_features - 1 ) # flops for K^T Q
        elif mode == "inference":
            flops += sequence_length * (2 * head_features - 1) # flops for only the new query and key dot products
        if mode == "training":
            flops += sequence_length * sequence_length + 10 # flops for sqrt(D) then N**2 divisions
        elif mode == "inference":
            flops += sequence_length + 10 # flops for normalising the new query and key dot products
        flops += sequence_length * sequence_length # flops for adding the causal self attention mask
        flops += sequence_length * ( 10 * sequence_length + sequence_length - 1 + sequence_length ) # flops for softmax on columns
        flops += head_features * 1 * (2 * embedding_dimension - 1) # flops for one attention score * values
    
    flops += flops_linear(in_features=embedding_dimension, out_features=embedding_dimension, sequence_length=sequence_length, bias=False)

    return flops

def flops_transformer(embedding_dimension=embedding_dimension,
                    hidden_features=hidden_features,
                    sequence_length=sequence_length,
                    attention_heads=attention_heads,
                    mode=mode,
                    lora_rank=lora_rank):
    """
    Compute the FLOPs for a full transformer block, including attention and MLP.

    Args:
        embedding_dimension (int): Dimension of the embedding vectors (should be divisible by `attention_heads`).
        hidden_features (int): Hidden dimension of the MLP block.
        sequence_length (int): Length of the input sequence.
        attention_heads (int): Number of attention heads.

    Returns:
        int: Total FLOPs for the transformer block.
    """
    flops = 0
    flops += flops_rmsnorm(in_features=embedding_dimension, sequence_length=sequence_length) # input norm flops
    flops += flops_self_atte(embedding_dimension=embedding_dimension, sequence_length=sequence_length, attention_heads=attention_heads, mode=mode, lora_rank=lora_rank) # self attention flops
    flops += embedding_dimension * sequence_length # Flops for adding the residual

    flops += flops_rmsnorm(in_features=embedding_dimension, sequence_length=sequence_length) # post attention norm flops
    flops += flops_mlp(in_features=embedding_dimension, hidden_features=hidden_features, out_features=embedding_dimension, sequence_length=sequence_length) # Flops for mlp
    flops += embedding_dimension * sequence_length # flops for adding the residual

    return flops

def flops_lora(embedding_dimension=embedding_dimension,
               lora_rank=lora_rank,
               sequence_length=sequence_length):
    """
    Placeholder for FLOPs of a LoRA (Low-Rank Adaptation) layer.

    Returns:
        int: Placeholder value or computed FLOPs in a real implementation.
    """
    flops = 0
    if lora_rank > 0:
        flops += flops_linear(in_features=embedding_dimension, out_features=lora_rank, sequence_length=sequence_length, bias=False) # First matrix multiplication
        flops += flops_linear(in_features=lora_rank, out_features=embedding_dimension, sequence_length=sequence_length, bias=False) # Second matrix multiplication
        flops += 2 * embedding_dimension * sequence_length # multiplying by alpha and adding the lora offset at the end
    return flops

def flops_embedding(embedding_dimension=embedding_dimension,
                    sequence_length=sequence_length,
                    vocabulary_size=vocabulary_size,
                    lookup_table=False):
    """
    Estimate FLOPs for computing token embeddings via a vocabulary matrix.

    Args:
        embedding_dimension (int): Dimension of the embedding vectors.
        sequence_length (int): Length of the input sequence.
        vocabulary_size (int): Size of the vocabulary.
        lookup_table (bool): If true, treats the embedding as free wrt flops. If false, treats the embedding as a matrix multiplication.

    Returns:
        int: Total FLOPs for embedding computation.
    """
    if lookup_table:
        return 0

    flops = sequence_length * embedding_dimension * (2 * vocabulary_size - 1) # flops for onehot token matrix @ vocab matrix to get embedding of sequence
    return flops

def flops_qwen(embedding_dimension=embedding_dimension,
               hidden_features=hidden_features,
               sequence_length=sequence_length,
               attention_heads=attention_heads,
               transformer_layers=transformer_layers,
               vocabulary_size=vocabulary_size,
               lookup_table=lookup_table,
               mode=mode,
               generation_length=generation_length,
               batch_size=batch_size,
               lora_rank=lora_rank):
    """
    Estimate total FLOPs for a QWen-like transformer model end-to-end.

    Args:
        embedding_dimension (int): Size of the embedding and transformer input/output.
        hidden_features (int): Hidden layer size in the MLPs.
        sequence_length (int): Length of the input sequence.
        attention_heads (int): Number of attention heads.
        transformer_layers (int): Number of transformer layers.
        vocabulary_size (int): Vocabulary size for the output logits.
        lookup_table (bool): If true, treats the embedding as free wrt flops. If false, treats the embedding as a matrix multiplication.
        mode (str): [`training`, `inference`], Whether or not the model is being used for training or inference.
        generation_length (int): Number of tokens we are asking the model to predict.
        batch_size (int): Number of sequences in the batch.

    Returns:
        int: Total FLOPs for the model.
    """
    if mode == "training":
        flops = 0
        flops += flops_embedding(embedding_dimension=embedding_dimension, sequence_length=sequence_length, vocabulary_size=vocabulary_size, lookup_table=lookup_table) # flops for converting tokens into embedding vectors
        
        for _ in range(transformer_layers):
            flops += flops_transformer(embedding_dimension=embedding_dimension,
                                    hidden_features=hidden_features,
                                    sequence_length=sequence_length,
                                    attention_heads=attention_heads,
                                    mode=mode,
                                    lora_rank=lora_rank)
        
        flops += flops_rmsnorm(in_features=embedding_dimension, sequence_length=sequence_length) # flops for one more normalisation step

        flops += flops_linear(in_features=embedding_dimension, out_features=vocabulary_size, sequence_length=sequence_length, bias=True) # lm_head flops

        return flops * 3 * batch_size
    elif mode == "inference":
        flops = 0

        flops += flops_embedding(embedding_dimension=embedding_dimension, sequence_length=sequence_length, vocabulary_size=vocabulary_size, lookup_table=lookup_table) # flops for converting tokens into embedding vectors
        
        for _ in range(transformer_layers):
            flops += flops_transformer(embedding_dimension=embedding_dimension,
                                    hidden_features=hidden_features,
                                    sequence_length=sequence_length,
                                    attention_heads=attention_heads,
                                    mode=mode)
        
        flops += flops_rmsnorm(in_features=embedding_dimension, sequence_length=sequence_length) # flops for one more normalisation step

        flops += flops_linear(in_features=embedding_dimension, out_features=vocabulary_size, sequence_length=1, bias=True) # lm_head flops

        for t in range(generation_length - 1):
            flops += flops_embedding(embedding_dimension=embedding_dimension,
                                     sequence_length=1,
                                     vocabulary_size=vocabulary_size,
                                     lookup_table=lookup_table)
            
            for _ in range(transformer_layers):
                flops += flops_transformer(embedding_dimension=embedding_dimension,
                                           hidden_features=hidden_features,
                                           sequence_length=sequence_length,
                                           attention_heads=attention_heads,
                                           mode=mode)
            
            flops += flops_linear(in_features=embedding_dimension, out_features=vocabulary_size, sequence_length=1, bias=True) # Flops for passing final embedding vector through lm_head

        return flops
    else:
        raise ValueError(f"Mode `{mode}` not valid.")

def printsf(value,
            sf=3,
            prefix=None,
            add_newline=False):
    """
    Prints the given value formatted to 3 significant figures.

    Parameters:
        value (float or int): The numeric value to be printed with 3 significant figures.
    """
    if prefix is None:
        print(f"{value:.{sf}g}")
    else:
        print(f"{prefix}: {value:.{sf}g}")
    if add_newline:
        print("")