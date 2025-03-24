import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
import copy


class TokenConverter():
    def __init__(self, token_map: dict):
        self.token_map = token_map
        self.to_lookup = self._make_to_lookup()
        self.back_lookup = self._make_back_lookup()
    
    def _make_to_lookup(self):
        max_key = max(self.token_map.keys())
        lookup = torch.full((max_key + 1,), -1)  # use -1 or some default for unmapped
        for k, v in self.token_map.items():
            lookup[k] = v

        return lookup
    
    def _make_back_lookup(self):
        max_key = max(self.token_map.values())
        lookup = torch.full((max_key + 1,), -1)
        for k, v in self.token_map.items():
            lookup[v] = k
        return lookup
    
    def to(self, tokens):
        """
        Converts to the tokens of system 2
        """
        if isinstance(tokens, BatchEncoding):
            new_input_ids = self.to_lookup[tokens.input_ids]
            new_tokens = copy.deepcopy(tokens)
            new_tokens["input_ids"] = new_input_ids
            return new_tokens
        elif isinstance(tokens, torch.Tensor):
            new_tokens = self.to_lookup[tokens]
            return new_tokens
    
    def back(self, tokens):
        """
        Converts to the tokens of system 1
        """
        if isinstance(tokens, BatchEncoding):
            new_input_ids = self.back_lookup[tokens.input_ids]
            new_tokens = copy.deepcopy(tokens)
            new_tokens["input_ids"] = new_input_ids
            return new_tokens
        elif isinstance(tokens, torch.Tensor):
            new_tokens = self.back_lookup[tokens]
            return new_tokens

def convert_tokens_to(desired_type: str,
                      tokens, 
                      token_map):

    # Make lookup for the conversion
    if desired_type == "forecast":
        max_key = max(token_map.keys())
        lookup = torch.full((max_key + 1,), -1)  # use -1 or some default for unmapped
        for k, v in token_map.items():
            lookup[k] = v

    elif desired_type == "qwen":
        max_key = max(token_map.values())
        lookup = torch.full((max_key + 1,), -1)
        for k, v in token_map.items():
            lookup[v] = k
    if isinstance(tokens, BatchEncoding):
        new_input_ids = lookup[tokens.input_ids]
        new_tokens = copy.deepcopy(tokens)
        new_tokens["input_ids"] = new_input_ids
        return new_tokens
    elif isinstance(tokens, torch.Tensor):
        new_tokens = lookup[tokens]
        return new_tokens


def reduce_embedding(model, tokenizer, valid_words):
    # Lists for tokens and embedding vectors
    qwen_tokens = []
    new_tokens = []
    embedding_vectors = []

    # New token counter
    token = 0
    for word in valid_words:
        # Get qwen's token for the word
        qwen_token_tensor = tokenizer(word, return_tensors="pt", add_special_tokens=False).input_ids[0]
        qwen_token = qwen_token_tensor.item()
        qwen_tokens.append(qwen_token)

        # define my token for the word
        new_tokens.append(token)
        token += 1

        # Get qwens embedding vector for the word
        embedding_vector = model.model.embed_tokens(qwen_token_tensor)
        embedding_vectors.append(embedding_vector)

    # make the token map keys = qwen, values = forecast
    token_map = {qwen: forecast for qwen, forecast in zip(qwen_tokens, new_tokens)}


    # Make the embedding vectors into torch parameters
    embedding_vectors = torch.concat(embedding_vectors)
    embedding_vectors = nn.Parameter(embedding_vectors)


    # make new embedding that only has tokens from our allowed words list
    new_embedding = nn.Embedding(num_embeddings=len(new_tokens), embedding_dim=model.model.embed_tokens.embedding_dim)
    new_embedding.weight = embedding_vectors
    model.model.embed_tokens = new_embedding

    return model, token_map


def reduce_head(model, token_map):
    lm_head = model.lm_head
    new_weight = []
    for qwen, forecast in token_map.items():
        weights = lm_head.weight[qwen, :].unsqueeze(0)
        new_weight.append(weights)


    new_weight = torch.concat(new_weight, dim=0)
    new_weight = nn.Parameter(new_weight)

    new_lm_head = nn.Linear(in_features=lm_head.in_features, out_features=new_weight.shape[0], bias=False)
    new_lm_head.weight = new_weight

    model.lm_head = new_lm_head

    return model


def reduce_vocabulary(model, tokenizer, valid_words=list("0123456789,;")):
    model, token_map = reduce_embedding(model, tokenizer, valid_words)
    model = reduce_head(model, token_map)
    model.config.vocab_size = len(valid_words)
    return model, token_map


def load_qwen(small_vocabulary=False):
    """
    Loads the Qwen2.5-0.5B-Instruct language model and tokenizer with a frozen base model 
    and a trainable bias parameter in the language modeling (LM) head.

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The Qwen language model with all parameters frozen 
              except for a trainable bias in the LM head.
            - tokenizer (transformers.PreTrainedTokenizer): The corresponding tokenizer for the model.
    """
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    if small_vocabulary:
        model, token_map = reduce_vocabulary(model, tokenizer)

    # Freeze all parameters except LM head bias
    for param in model.parameters():
        param.requires_grad = False

    # Add trainable bias to logits
    assert model.lm_head.bias is None
    model.lm_head.bias = torch.nn.Parameter(
        torch.zeros(model.config.vocab_size, device=model.device)
    )
    model.lm_head.bias.requires_grad = True

    if small_vocabulary:
        return model, tokenizer, token_map
    else:
        return model, tokenizer

