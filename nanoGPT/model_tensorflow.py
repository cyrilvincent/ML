"""
Full definition of a GPT Language Model in TensorFlow/Keras.
Translated from the PyTorch implementation by Andrej Karpathy (nanoGPT).

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
   https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
   https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# LayerNorm with optional bias
# ---------------------------------------------------------------------------

class LayerNorm(layers.Layer):
    """LayerNorm with an optional bias (Keras doesn't expose bias=False directly)."""

    def __init__(self, ndim: int, bias: bool, **kwargs):
        super().__init__(**kwargs)
        self.ndim = ndim
        self.use_bias = bias

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight",
            shape=(self.ndim,),
            initializer="ones",
            trainable=True,
        )
        if self.use_bias:
            self.bias_term = self.add_weight(
                name="bias",
                shape=(self.ndim,),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.bias_term = None
        super().build(input_shape)

    def call(self, x):
        # mean and variance over the last axis
        mean, variance = tf.nn.moments(x, axes=[-1], keepdims=True)
        x_norm = (x - mean) / tf.sqrt(variance + 1e-5)
        x_norm = x_norm * self.weight
        if self.bias_term is not None:
            x_norm = x_norm + self.bias_term
        return x_norm

    def get_config(self):
        config = super().get_config()
        config.update({"ndim": self.ndim, "bias": self.use_bias})
        return config


# ---------------------------------------------------------------------------
# Causal Self-Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout_rate = config.dropout
        self.block_size = config.block_size

        # Q, K, V projections packed in a single linear
        self.c_attn = layers.Dense(3 * config.n_embd, use_bias=config.bias)
        # Output projection
        self.c_proj = layers.Dense(config.n_embd, use_bias=config.bias)

        self.attn_dropout = layers.Dropout(config.dropout)
        self.resid_dropout = layers.Dropout(config.dropout)

    def build(self, input_shape):
        # Causal mask: lower-triangular matrix of shape (1, 1, block_size, block_size)
        mask = tf.linalg.band_part(
            tf.ones((self.block_size, self.block_size), dtype=tf.float32), -1, 0
        )
        # Shape: (1, 1, T, T) — broadcast over batch and heads
        self.causal_mask = tf.reshape(mask, (1, 1, self.block_size, self.block_size))
        super().build(input_shape)

    def call(self, x, training=False):
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        C = self.n_embd
        hs = C // self.n_head  # head size

        # Project to Q, K, V
        qkv = self.c_attn(x)                              # (B, T, 3*C)
        q, k, v = tf.split(qkv, 3, axis=-1)               # each (B, T, C)

        # Reshape to (B, n_head, T, hs)
        def reshape_for_attn(t):
            t = tf.reshape(t, (B, T, self.n_head, hs))
            return tf.transpose(t, perm=[0, 2, 1, 3])

        q = reshape_for_attn(q)
        k = reshape_for_attn(k)
        v = reshape_for_attn(v)

        # Attention scores: (B, n_head, T, T)
        scale = 1.0 / math.sqrt(hs)
        att = tf.matmul(q, k, transpose_b=True) * scale   # (B, nh, T, T)

        # Apply causal mask (set future positions to -inf)
        mask_slice = self.causal_mask[:, :, :T, :T]
        att = att * mask_slice + (1.0 - mask_slice) * tf.constant(-1e9, dtype=att.dtype)

        att = tf.nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, training=training)

        y = tf.matmul(att, v)                              # (B, nh, T, hs)

        # Re-assemble heads: (B, T, C)
        y = tf.transpose(y, perm=[0, 2, 1, 3])
        y = tf.reshape(y, (B, T, C))

        # Output projection + residual dropout
        y = self.resid_dropout(self.c_proj(y), training=training)
        return y


# ---------------------------------------------------------------------------
# MLP (Feed-Forward block)
# ---------------------------------------------------------------------------

class MLP(layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.c_fc   = layers.Dense(4 * config.n_embd, use_bias=config.bias, activation="gelu")
        self.c_proj = layers.Dense(config.n_embd, use_bias=config.bias)
        self.dropout = layers.Dropout(config.dropout)

    def call(self, x, training=False):
        x = self.c_fc(x)
        x = self.c_proj(x)
        x = self.dropout(x, training=training)
        return x


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class Block(layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def call(self, x, training=False):
        x = x + self.attn(self.ln_1(x), training=training)
        x = x + self.mlp(self.ln_2(x), training=training)
        return x


# ---------------------------------------------------------------------------
# GPT Config
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int  = 1024
    vocab_size: int  = 50304   # GPT-2 vocab 50257, padded to nearest multiple of 64
    n_layer:    int  = 12
    n_head:     int  = 12
    n_embd:     int  = 768
    dropout:    float = 0.0
    bias:       bool = True    # True: like GPT-2; False: slightly better & faster


# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

class GPT(keras.Model):

    def __init__(self, config: GPTConfig, **kwargs):
        super().__init__(**kwargs)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Embeddings
        self.wte = layers.Embedding(config.vocab_size, config.n_embd)   # token embeddings
        self.wpe = layers.Embedding(config.block_size, config.n_embd)   # position embeddings
        self.drop = layers.Dropout(config.dropout)

        # Transformer blocks
        self.h = [Block(config, name=f"block_{i}") for i in range(config.n_layer)]

        # Final layer norm
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

        # Language model head
        # NOTE: weight tying (wte.weight == lm_head.kernel) is set after the first
        # forward pass (weights are created lazily in Keras). See `tie_weights()`.
        self.lm_head = layers.Dense(config.vocab_size, use_bias=False)

        print(f"GPT model initialised — weights will be reported after first forward pass.")

    # ------------------------------------------------------------------
    # Weight tying: share token embedding weights with lm_head
    # ------------------------------------------------------------------
    def tie_weights(self):
        """Call once after the first forward pass to tie wte and lm_head weights."""
        self.lm_head.kernel.assign(tf.transpose(self.wte.embeddings))
        # From this point on you need to keep them in sync manually during training,
        # or override `train_step` to copy after each optimizer step.

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def call(self, idx, targets=None, training=False):
        """
        idx     : int tensor of shape (B, T)
        targets : int tensor of shape (B, T), optional
        Returns (logits, loss)  —  loss is None when targets is not provided.
        """
        B = tf.shape(idx)[0]
        T = tf.shape(idx)[1]

        tf.debugging.assert_less_equal(
            T,
            self.config.block_size,
            message=f"Sequence length exceeds block_size ({self.config.block_size})",
        )

        pos = tf.range(0, T, dtype=tf.int32)          # shape (T,)

        # Embeddings
        tok_emb = self.wte(idx)                        # (B, T, n_embd)
        pos_emb = self.wpe(pos)                        # (T, n_embd) — broadcasts over B
        x = self.drop(tok_emb + pos_emb, training=training)

        # Transformer blocks
        for block in self.h:
            x = block(x, training=training)

        x = self.ln_f(x)

        if targets is not None:
            # Training / evaluation: compute logits for all positions
            logits = self.lm_head(x)                  # (B, T, vocab_size)
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                targets, logits, from_logits=True
            )
            # Mask padding tokens (ignore_index=-1 equivalent)
            mask = tf.cast(tf.not_equal(targets, -1), dtype=loss.dtype)
            loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
        else:
            # Inference: only compute logits for the last token (efficiency)
            logits = self.lm_head(x[:, -1:, :])       # (B, 1, vocab_size)
            loss = None

        return logits, loss

    # ------------------------------------------------------------------
    # Parameter count
    # ------------------------------------------------------------------
    def get_num_params(self, non_embedding=True):
        n_params = sum(tf.size(w).numpy() for w in self.trainable_weights)
        if non_embedding:
            n_params -= tf.size(self.wpe.embeddings).numpy()
        return n_params

    # ------------------------------------------------------------------
    # Crop block size (model surgery)
    # ------------------------------------------------------------------
    def crop_block_size(self, block_size: int):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        # Trim position embedding table
        new_wpe = tf.Variable(self.wpe.embeddings[:block_size], trainable=True)
        self.wpe.embeddings.assign(new_wpe)

    # ------------------------------------------------------------------
    # Optimizer configuration (AdamW with weight decay)
    # ------------------------------------------------------------------
    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple,
    ):
        """
        Returns a tf.keras.optimizers.AdamW instance.
        TensorFlow's AdamW applies weight decay globally; selective decay
        (skip biases & LayerNorm) is handled via exclude_from_weight_decay.
        """
        # Collect names of parameters that should NOT be decayed
        no_decay_names = []
        for w in self.trainable_weights:
            name = w.name
            if any(skip in name for skip in ["bias", "layer_norm", "ln_", "wpe", "wte"]):
                no_decay_names.append(name)

        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta_1=betas[0],
            beta_2=betas[1],
            exclude_from_weight_decay=no_decay_names,
        )
        print(f"AdamW configured — lr={learning_rate}, wd={weight_decay}, betas={betas}")
        return optimizer

    # ------------------------------------------------------------------
    # Load from HuggingFace GPT-2 checkpoint
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, model_type: str, override_args: Optional[dict] = None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}
        assert all(k == "dropout" for k in override_args)

        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pretrained GPT: {model_type}")

        config_args = {
            "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        config_args["bias"] = True
        if "dropout" in override_args:
            config_args["dropout"] = override_args["dropout"]

        config = GPTConfig(**config_args)
        model = cls(config)

        # Trigger weight creation with a dummy forward pass
        dummy = tf.zeros((1, 1), dtype=tf.int32)
        model(dummy)

        # Load HuggingFace weights
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Mapping: HF name → (TF weight, transpose?)
        # Weights that were stored as Conv1D in OpenAI's original code need to be transposed.
        transposed_suffixes = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        def _copy(tf_var, pt_tensor, transpose=False):
            data = pt_tensor.detach().numpy()
            if transpose:
                data = data.T
            tf_var.assign(data)

        # Token & position embeddings
        _copy(model.wte.embeddings,   sd_hf["transformer.wte.weight"])
        _copy(model.wpe.embeddings,   sd_hf["transformer.wpe.weight"])
        # Final layer norm
        _copy(model.ln_f.weight,      sd_hf["transformer.ln_f.weight"])
        if model.ln_f.bias_term is not None:
            _copy(model.ln_f.bias_term, sd_hf["transformer.ln_f.bias"])

        for i, block in enumerate(model.h):
            prefix = f"transformer.h.{i}"
            # LayerNorm 1
            _copy(block.ln_1.weight,      sd_hf[f"{prefix}.ln_1.weight"])
            if block.ln_1.bias_term is not None:
                _copy(block.ln_1.bias_term, sd_hf[f"{prefix}.ln_1.bias"])
            # LayerNorm 2
            _copy(block.ln_2.weight,      sd_hf[f"{prefix}.ln_2.weight"])
            if block.ln_2.bias_term is not None:
                _copy(block.ln_2.bias_term, sd_hf[f"{prefix}.ln_2.bias"])
            # Attention
            _copy(block.attn.c_attn.kernel, sd_hf[f"{prefix}.attn.c_attn.weight"], transpose=True)
            if config.bias:
                _copy(block.attn.c_attn.bias, sd_hf[f"{prefix}.attn.c_attn.bias"])
            _copy(block.attn.c_proj.kernel, sd_hf[f"{prefix}.attn.c_proj.weight"], transpose=True)
            if config.bias:
                _copy(block.attn.c_proj.bias, sd_hf[f"{prefix}.attn.c_proj.bias"])
            # MLP
            _copy(block.mlp.c_fc.kernel,   sd_hf[f"{prefix}.mlp.c_fc.weight"], transpose=True)
            if config.bias:
                _copy(block.mlp.c_fc.bias, sd_hf[f"{prefix}.mlp.c_fc.bias"])
            _copy(block.mlp.c_proj.kernel, sd_hf[f"{prefix}.mlp.c_proj.weight"], transpose=True)
            if config.bias:
                _copy(block.mlp.c_proj.bias, sd_hf[f"{prefix}.mlp.c_proj.bias"])

        # lm_head shares weights with wte (weight tying)
        model.lm_head.kernel.assign(tf.transpose(model.wte.embeddings))
        print("Weights loaded successfully.")
        return model

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------
    def generate(
        self,
        idx: tf.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> tf.Tensor:
        """
        idx : int tensor of shape (B, T) — conditioning token indices.
        Returns the extended sequence of shape (B, T + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = (
                idx
                if tf.shape(idx)[1] <= self.config.block_size
                else idx[:, -self.config.block_size:]
            )
            # Forward pass (inference mode — no targets)
            logits, _ = self(idx_cond, training=False)       # (B, 1, vocab_size)
            logits = logits[:, -1, :] / temperature           # (B, vocab_size)

            # Top-k filtering
            if top_k is not None:
                k = min(top_k, logits.shape[-1])
                top_values, _ = tf.math.top_k(logits, k=k)
                # Threshold: smallest value among the top-k
                threshold = top_values[:, -1:]                # (B, 1)
                logits = tf.where(logits < threshold, tf.fill(tf.shape(logits), -1e9), logits)

            # Sample from the distribution
            probs = tf.nn.softmax(logits, axis=-1)            # (B, vocab_size)
            idx_next = tf.random.categorical(
                tf.math.log(probs), num_samples=1, dtype=tf.int32
            )                                                  # (B, 1)

            idx = tf.concat([idx, idx_next], axis=1)

        return idx


# ---------------------------------------------------------------------------
# MFU estimation (Model FLOPs Utilisation)
# ---------------------------------------------------------------------------

def estimate_mfu(model: GPT, fwdbwd_per_iter: int, dt: float) -> float:
    """Estimate MFU in units of A100 bfloat16 peak FLOPS."""
    N = model.get_num_params()
    cfg = model.config
    L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
    flops_per_token  = 6 * N + 12 * L * H * Q * T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter   = flops_per_fwdbwd * fwdbwd_per_iter
    flops_achieved   = flops_per_iter / dt
    flops_promised   = 312e12   # A100 bfloat16 peak = 312 TFLOPS
    return flops_achieved / flops_promised
