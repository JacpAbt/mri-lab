# -- VENDORED FROZEN INSTRUMENT -- do not edit behavior ------------------------
# source: mri_compressor/model_utils.py
# source sha256: d93d46c87cf2a27cac9578b1df2ea743a6f7898831942d87865199b4c4f3cc39
# vendored: 2026-09-01 by scripts/vendor.py (0 patch(es) applied)
# The golden-row tests (tests/golden/) are this file's license to exist: any
# change here requires the full golden lane to pass first. See VENDORING.md.
# ------------------------------------------------------------------------------
"""
Model loading and hook infrastructure.
Handles both GPT-2 style (fc1 -> GELU -> fc2) and Llama/Qwen style (gate_proj * up_proj -> down_proj).

MEMORY OPTIMIZATION (v2):
  - Model loaded in bfloat16 instead of float32 (saves 50% VRAM)
  - Statistics computed in float32 on CPU after collection
  - Streaming layer-by-layer collection mode for large models
"""

import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
import re
import gc


@dataclass
class MLPInfo:
    """Describes the MLP structure for one layer."""
    layer_idx: int
    is_gated: bool          # True for SwiGLU/GeGLU, False for standard GELU MLP
    gate_proj: Optional[nn.Linear]   # gate_proj in gated MLP, None for standard
    up_proj: Optional[nn.Linear]     # up_proj in gated MLP, fc1 in standard
    down_proj: nn.Linear             # down_proj / fc2
    activation_fn: str               # "silu", "gelu", "relu", etc.
    intermediate_size: int


@dataclass 
class AttentionInfo:
    """Describes attention structure for one layer."""
    layer_idx: int
    num_heads: int
    num_kv_heads: int       # for GQA models
    head_dim: int
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    o_proj: nn.Linear


class ModelInspector:
    """
    Architecture-agnostic model inspector.
    Detects MLP type, activation function, and provides uniform access to internals.
    """
    
    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "auto",
                 revision: Optional[str] = None,
                 attn_impl: Optional[str] = None,
                 quant: Optional[str] = None):
        print(f"Loading model: {model_name}"
              + (f" (revision={revision})" if revision else ""))

        # Determine dtype: bf16 for analysis (good enough precision, 50% memory saving)
        if dtype == "auto":
            # Use bf16 for models > 1B params, float32 for small ones
            dtype = torch.bfloat16
        elif dtype == "float32":
            dtype = torch.float32
        else:
            dtype = torch.bfloat16

        # Attention backend selection.
        # Priority:
        #   1. explicit `attn_impl` argument
        #   2. `FDGC_ATTN_IMPL` environment variable
        #   3. flash_attention_2 if installed
        #   4. sdpa otherwise
        # Recognized values: "flash_attention_2", "sdpa", "eager".
        # Use "sdpa" or "eager" when FA2 is incompatible with a specific
        # model (e.g. Qwen3.5 hybrid attention, where FA2's varlen path
        # crashes on size-1 trailing batches).
        _extra_kwargs = {}
        chosen_attn = attn_impl or os.environ.get("FDGC_ATTN_IMPL")
        if chosen_attn:
            _extra_kwargs["attn_implementation"] = chosen_attn
            print(f"  Attention backend: {chosen_attn} (forced)")
        else:
            try:
                import flash_attn  # noqa: F401
                _extra_kwargs["attn_implementation"] = "flash_attention_2"
                print(f"  Using Flash Attention 2")
            except ImportError:
                _extra_kwargs["attn_implementation"] = "sdpa"
                print(f"  Flash Attention not installed, using SDPA")

        if revision is not None:
            _extra_kwargs["revision"] = revision

        # 8-bit / 4-bit loading for models past the card's bf16 ceiling. Any
        # measurement on a quantized model must first pass the precision-
        # validation cell (same model bf16 vs quantized, same statistic).
        if quant in ("8bit", "4bit"):
            from transformers import BitsAndBytesConfig
            _extra_kwargs["quantization_config"] = (
                BitsAndBytesConfig(load_in_8bit=True) if quant == "8bit"
                else BitsAndBytesConfig(load_in_4bit=True,
                                        bnb_4bit_compute_dtype=torch.bfloat16,
                                        bnb_4bit_quant_type="nf4"))
        elif quant is not None:
            raise ValueError(f"quant must be None/8bit/4bit, got {quant!r}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
            **_extra_kwargs,
        )
        self.model.eval()
        tok_kwargs = {"trust_remote_code": True}
        if revision is not None:
            tok_kwargs["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.dtype = dtype
        self._detect_architecture()
    
    def _detect_architecture(self):
        """Auto-detect model architecture and build layer maps."""
        self.mlp_layers: List[Optional[MLPInfo]] = []
        self.attn_layers: List[Optional[AttentionInfo]] = []
        self.num_layers = 0
        # True for state-space / linear-attention stacks (Mamba, Mamba-2,
        # RWKV, …) whose blocks have an SSM "mixer" instead of a standard
        # MLP. The FDGC discriminative pipeline (step1/step2/step8) only
        # needs get_layers() + the residual stream, so these models work
        # without per-layer MLP/attention metadata.
        self.is_ssm = False

        # Try to find the per-block module list. `backbone.layers` covers
        # Mamba/Mamba-2 (Mamba2ForCausalLM → model.backbone.layers).
        layers = None
        for attr in ['transformer.h', 'model.layers', 'gpt_neox.layers',
                     'backbone.layers',
                     # multimodal wrappers (Gemma-3, Llama-vision): the text
                     # stack sits under a language_model submodule
                     'model.language_model.layers', 'language_model.layers',
                     'model.text_model.layers']:
            obj = self.model
            try:
                for part in attr.split('.'):
                    obj = getattr(obj, part)
                layers = obj
                self.layer_path = attr
                break
            except AttributeError:
                continue

        if layers is None:
            raise ValueError(f"Cannot detect layer structure for {self.model_name}")

        self.num_layers = len(layers)
        print(f"  Found {self.num_layers} blocks via '{self.layer_path}'")

        for i, layer in enumerate(layers):
            try:
                self._parse_mlp(i, layer)
            except ValueError:
                # SSM block (e.g. Mamba-2 mixer): no standard MLP. Record a
                # placeholder so list indices stay aligned with block indices.
                self.is_ssm = True
                self.mlp_layers.append(None)
            self._parse_attention(i, layer)

        first_mlp = next((m for m in self.mlp_layers if m is not None), None)
        if first_mlp is not None:
            arch_type = "Gated MLP (SwiGLU)" if first_mlp.is_gated else "Standard MLP"
            print(f"  Architecture: {arch_type}")
            print(f"  Activation: {first_mlp.activation_fn}")
            print(f"  Intermediate size: {first_mlp.intermediate_size}")
        else:
            print("  Architecture: state-space / linear (no standard MLP blocks)")
        first_attn = next((a for a in self.attn_layers if a is not None), None)
        if first_attn:
            print(f"  Attention heads: {first_attn.num_heads} "
                  f"(KV heads: {first_attn.num_kv_heads})")
        else:
            print("  Attention: all linear (no pruneable heads)")
        print(f"  Model dtype: {self.dtype}")
    
    def _parse_mlp(self, idx: int, layer):
        """Parse MLP structure from a transformer layer."""
        mlp = None
        for attr in ['mlp', 'feed_forward']:
            if hasattr(layer, attr):
                mlp = getattr(layer, attr)
                break
        
        if mlp is None:
            raise ValueError(f"Cannot find MLP in layer {idx}")
        
        # Check for gated architecture
        gate_proj = getattr(mlp, 'gate_proj', None)
        up_proj = getattr(mlp, 'up_proj', None)
        down_proj = getattr(mlp, 'down_proj', None)
        
        if gate_proj is not None and up_proj is not None:
            self.mlp_layers.append(MLPInfo(
                layer_idx=idx,
                is_gated=True,
                gate_proj=gate_proj,
                up_proj=up_proj,
                down_proj=down_proj,
                activation_fn=self._detect_activation(mlp),
                intermediate_size=gate_proj.out_features,
            ))
        else:
            fc1 = getattr(mlp, 'c_fc', None) or getattr(mlp, 'fc1', None) or getattr(mlp, 'dense_h_to_4h', None)
            fc2 = getattr(mlp, 'c_proj', None) or getattr(mlp, 'fc2', None) or getattr(mlp, 'dense_4h_to_h', None)
            
            if fc1 is None or fc2 is None:
                raise ValueError(f"Cannot find fc1/fc2 in layer {idx} MLP")
            
            self.mlp_layers.append(MLPInfo(
                layer_idx=idx,
                is_gated=False,
                gate_proj=None,
                up_proj=fc1,
                down_proj=fc2,
                activation_fn=self._detect_activation(mlp),
                intermediate_size=fc1.out_features if hasattr(fc1, 'out_features') else fc1.nf,
            ))
    
    def _parse_attention(self, idx: int, layer):
        """Parse attention structure from a transformer layer."""
        attn = None
        for attr in ['self_attn', 'attn', 'attention', 'self_attention']:
            if hasattr(layer, attr):
                attn = getattr(layer, attr)
                break

        if attn is None:
            # Generic fallback: child must have BOTH an input projection (q_proj/c_attn) AND
            # an output projection (o_proj/c_proj) — requiring both prevents false-positives
            # on Gated DeltaNet or other linear-attention modules that have q_proj but no o_proj.
            for _, mod in layer.named_children():
                has_in = hasattr(mod, 'q_proj') or hasattr(mod, 'c_attn')
                has_out = hasattr(mod, 'o_proj') or hasattr(mod, 'c_proj')
                if has_in and has_out:
                    attn = mod
                    break

        if attn is None:
            # Hybrid model (e.g. Qwen3.5 DeltaNet layers): no pruneable standard attention
            # in this layer — store None and continue gracefully.
            print(f"  Layer {idx}: linear/non-standard attention, skipping head pruning.")
            self.attn_layers.append(None)
            return
        
        q_proj = getattr(attn, 'q_proj', None) or getattr(attn, 'c_attn', None)
        k_proj = getattr(attn, 'k_proj', None)
        v_proj = getattr(attn, 'v_proj', None)
        o_proj = getattr(attn, 'o_proj', None) or getattr(attn, 'c_proj', None)
        
        if k_proj is None and hasattr(attn, 'c_attn'):
            q_proj = k_proj = v_proj = attn.c_attn
        
        num_heads = getattr(attn, 'num_heads', None) or getattr(attn, 'num_attention_heads', None)
        num_kv_heads = getattr(attn, 'num_key_value_heads', num_heads)
        head_dim = getattr(attn, 'head_dim', None)
        
        if head_dim is None and num_heads is not None:
            hidden_size = self.model.config.hidden_size
            head_dim = hidden_size // num_heads
        
        self.attn_layers.append(AttentionInfo(
            layer_idx=idx,
            num_heads=num_heads or 12,
            num_kv_heads=num_kv_heads or 12,
            head_dim=head_dim or 64,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            o_proj=o_proj,
        ))
    
    def _detect_activation(self, mlp) -> str:
        """Detect the activation function used in the MLP."""
        mlp_str = str(type(mlp)).lower()
        module_str = str(mlp).lower()
        
        if 'silu' in module_str or 'swiglu' in mlp_str:
            return 'silu'
        elif 'gelu' in module_str or 'gelu' in mlp_str:
            return 'gelu'
        elif 'relu' in module_str:
            return 'relu'
        
        config = self.model.config
        act_fn = getattr(config, 'hidden_act', None) or getattr(config, 'activation_function', 'gelu')
        return act_fn
    
    def get_layers(self):
        """Return the nn.ModuleList of transformer layers."""
        obj = self.model
        for part in self.layer_path.split('.'):
            obj = getattr(obj, part)
        return obj
    
    @property
    def _text_config(self):
        """Multimodal configs (Gemma-3, vision-language wrappers) nest the
        text-stack params under `text_config`; single-modality configs are
        their own text config."""
        return getattr(self.model.config, "text_config", self.model.config)

    @property
    def hidden_size(self) -> int:
        return self._text_config.hidden_size
    
    @property
    def is_gated(self) -> bool:
        """Whether the model uses gated MLP (SwiGLU/GeGLU). False for
        SSM/linear stacks with no standard MLP."""
        first_mlp = next((m for m in self.mlp_layers if m is not None), None)
        return first_mlp.is_gated if first_mlp is not None else False

    @property
    def vocab_size(self) -> int:
        return self._text_config.vocab_size


class ActivationCollector:
    """
    Hook-based collector for MLP intermediate activations.
    Captures activations at the point between the activation function and down_proj.
    """
    
    def __init__(self, inspector: ModelInspector):
        self.inspector = inspector
        self.hooks = []
        self.collected: Dict[int, List[torch.Tensor]] = {}
    
    def _make_hook(self, layer_idx: int):
        """Create a forward hook for capturing post-activation MLP intermediates."""
        def hook_fn(module, input, output):
            if isinstance(input, tuple):
                act = input[0]
            else:
                act = input
            
            if layer_idx not in self.collected:
                self.collected[layer_idx] = []
            # Store on CPU in float32 for analysis precision
            self.collected[layer_idx].append(act.detach().float().cpu())
        return hook_fn
    
    def register_hooks(self, layer_indices: Optional[List[int]] = None):
        """Register hooks on down_proj / fc2 to capture post-activation intermediates."""
        if layer_indices is None:
            layer_indices = list(range(self.inspector.num_layers))
        
        for idx in layer_indices:
            mlp_info = self.inspector.mlp_layers[idx]
            hook = mlp_info.down_proj.register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)
        
        return self
    
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def clear(self):
        self.collected = {}
    
    def get_concatenated(self, layer_idx: int) -> torch.Tensor:
        """Return all collected activations for a layer, concatenated along batch*seq dim."""
        tensors = self.collected[layer_idx]
        return torch.cat(tensors, dim=0).reshape(-1, tensors[0].shape[-1])


def run_forward_passes(inspector: ModelInspector, dataset, batch_size: int = 4, max_batches: int = 32):
    """Run forward passes without any hooks — just to warm up / benchmark."""
    from .data_utils import get_dataloader
    dataloader = get_dataloader(dataset, batch_size=batch_size)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(inspector.device)
            inspector.model(input_ids=input_ids)


def collect_single_layer(
    inspector: ModelInspector, dataset, layer_idx: int,
    batch_size: int = 4, max_batches: int = 32,
    preserve_shape: bool = False,
) -> torch.Tensor:
    """
    MEMORY-EFFICIENT: Collect activations for a SINGLE layer, then release.
    
    Args:
        preserve_shape: If True, return (total_batch, seq_len, D). 
                       If False, return (N, D) where N = total_batch * seq_len.
    Returns:
        Activation tensor on CPU in float32.
    """
    from .data_utils import get_dataloader
    
    collected = []
    
    def hook_fn(module, input, output):
        x = input[0] if isinstance(input, tuple) else input
        collected.append(x.detach().float().cpu())
    
    mlp_info = inspector.mlp_layers[layer_idx]
    hook = mlp_info.down_proj.register_forward_hook(hook_fn)
    
    dataloader = get_dataloader(dataset, batch_size=batch_size)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(inspector.device)
            inspector.model(input_ids=input_ids)
    
    hook.remove()
    
    result = torch.cat(collected, dim=0)  # (total_batch, seq_len, D)
    
    if not preserve_shape:
        result = result.reshape(-1, result.shape[-1])  # (N, D)
    
    # Free intermediate list
    del collected
    gc.collect()
    
    return result


class AttentionPatternCollector:
    """
    Collects attention weight matrices for head importance analysis.
    """
    
    def __init__(self, inspector: ModelInspector):
        self.inspector = inspector
        self.hooks = []
        self.attention_weights: Dict[int, List[torch.Tensor]] = {}
    
    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_w = output[1]  # (batch, num_heads, seq, seq)
                if layer_idx not in self.attention_weights:
                    self.attention_weights[layer_idx] = []
                self.attention_weights[layer_idx].append(attn_w.detach().float().cpu())
        return hook_fn
    
    def register_hooks(self, layer_indices: Optional[List[int]] = None):
        if layer_indices is None:
            layer_indices = list(range(self.inspector.num_layers))
        
        layers = self.inspector.get_layers()
        for idx in layer_indices:
            for attr in ['self_attn', 'attn', 'attention']:
                if hasattr(layers[idx], attr):
                    attn_module = getattr(layers[idx], attr)
                    hook = attn_module.register_forward_hook(self._make_hook(idx))
                    self.hooks.append(hook)
                    break
        return self
    
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def clear(self):
        self.attention_weights = {}