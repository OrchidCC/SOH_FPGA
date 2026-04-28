"""
FPGA-Aware Mixed-Precision Quantization Framework for PINNs.

Pipeline:
  1. Profile: per-layer weight/activation distributions
  2. Sensitivity: per-layer output sensitivity via perturbation
  3. DSP map: (w_bits, a_bits) → MAC/DSP for target FPGA
  4. Search: Pareto-optimal mixed-precision assignment
  5. Quantize: apply chosen bit-widths
  6. Validate: predicted vs actual accuracy
"""

import copy
import math
from dataclasses import dataclass, field
from itertools import product as iterproduct
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ─── FPGA Hardware Spec ──────────────────────────────────────────────

@dataclass
class DSPSpec:
    """FPGA DSP multiplier specification."""
    name: str
    a_port_bits: int  # wider port
    b_port_bits: int  # narrower port

    def mac_per_dsp(self, w_bits: int, a_bits: int) -> int:
        """Max parallel MACs per DSP for given operand widths.

        Packing rule: multiple operands of `op_bits` packed into one port,
        separated by guard bands of `product_bits = w_bits + a_bits` to
        prevent carry contamination between adjacent products.
        """
        prod = w_bits + a_bits
        # Weight in A-port (wider), activation broadcast via B-port
        n_wA = 1 + (self.a_port_bits - w_bits) // prod if w_bits <= self.a_port_bits and a_bits <= self.b_port_bits else 0
        # Weight in B-port (narrower), activation broadcast via A-port
        n_wB = 1 + (self.b_port_bits - w_bits) // prod if w_bits <= self.b_port_bits and a_bits <= self.a_port_bits else 0
        return max(n_wA, n_wB)


DSP48E2 = DSPSpec(name="DSP48E2", a_port_bits=27, b_port_bits=18)


# ─── Layer Profiling ─────────────────────────────────────────────────

@dataclass
class LayerProfile:
    """Per-layer statistics collected from calibration data."""
    index: int
    name: str
    in_features: int
    out_features: int
    n_params: int
    weight_abs_max: float
    weight_std: float
    activation_abs_max: float  # input activation to this layer
    activation_std: float
    is_input_raw: bool  # True for first layer (input not from Sin)


@dataclass
class LayerSensitivity:
    """How much a unit perturbation at this layer changes the final output."""
    index: int
    sensitivity: float  # output_MSE_change / perturbation_MSE


@dataclass
class LayerQuantError:
    """Actual quantization MSE at this layer for given bit-width."""
    index: int
    weight_bits: int
    act_bits: int
    weight_quant_mse: float
    act_quant_mse: float
    combined_mse: float


def profile_layers(model: nn.Module, calib_x: torch.Tensor) -> List[LayerProfile]:
    """Profile each Linear layer's weight and input activation distributions."""
    profiles = []
    h = calib_x.detach()
    layer_idx = 0

    for mod in _all_modules(model):
        if isinstance(mod, nn.Linear):
            w = mod.weight.detach()
            profiles.append(LayerProfile(
                index=layer_idx,
                name=f"L{layer_idx}({mod.in_features}→{mod.out_features})",
                in_features=mod.in_features,
                out_features=mod.out_features,
                n_params=w.numel(),
                weight_abs_max=w.abs().max().item(),
                weight_std=w.std().item(),
                activation_abs_max=h.abs().max().item(),
                activation_std=h.std().item(),
                is_input_raw=(layer_idx == 0),
            ))
            with torch.no_grad():
                h = mod(h)
            layer_idx += 1
        else:
            with torch.no_grad():
                h = _apply_non_linear(mod, h)

    return profiles


def measure_sensitivity(model: nn.Module, calib_x: torch.Tensor,
                        noise_std: float = 0.01) -> List[LayerSensitivity]:
    """Measure each layer's output sensitivity via noise injection."""
    with torch.no_grad():
        fp32_out = _forward_full(model, calib_x)

    n_layers = sum(1 for m in _all_modules(model) if isinstance(m, nn.Linear))
    sensitivities = []

    for target in range(n_layers):
        with torch.no_grad():
            out_features = list(m for m in _all_modules(model) if isinstance(m, nn.Linear))[target].out_features
            noise = torch.randn(calib_x.shape[0], out_features) * noise_std
            noisy_out = _forward_with_noise(model, calib_x, target, noise)
            output_mse = (fp32_out - noisy_out).pow(2).mean().item()
            noise_mse = noise.pow(2).mean().item()
            sens = output_mse / (noise_mse + 1e-15)

        sensitivities.append(LayerSensitivity(index=target, sensitivity=sens))

    return sensitivities


def measure_quant_errors(model: nn.Module, calib_x: torch.Tensor,
                         bit_options: List[int]) -> Dict[Tuple[int, int, int], LayerQuantError]:
    """Measure actual weight and activation quantization MSE per layer per bit-width."""
    # Collect per-layer inputs
    layer_inputs = []
    h = calib_x.detach()
    linears = []
    for mod in _all_modules(model):
        if isinstance(mod, nn.Linear):
            layer_inputs.append(h.clone())
            linears.append(mod)
            with torch.no_grad():
                h = mod(h)
        else:
            with torch.no_grad():
                h = _apply_non_linear(mod, h)

    errors = {}
    for li, (linear, inp) in enumerate(zip(linears, layer_inputs)):
        for wb in bit_options:
            # Weight quantization MSE
            w_q = _quantize_symmetric(linear.weight.detach(), wb)
            b_q = _quantize_symmetric(linear.bias.detach(), wb) if linear.bias is not None else None
            with torch.no_grad():
                out_fp = nn.functional.linear(inp, linear.weight, linear.bias)
                out_q = nn.functional.linear(inp, w_q, b_q)
            w_mse = (out_fp - out_q).pow(2).mean().item()

            for ab in bit_options:
                # Activation quantization MSE
                if li == 0:
                    a_mse = 0.0  # raw input, no Sin quantization
                else:
                    qmax_a = 2 ** (ab - 1) - 1
                    a_q = torch.round(inp * qmax_a) / qmax_a
                    a_mse = (inp - a_q).pow(2).mean().item()

                errors[(li, wb, ab)] = LayerQuantError(
                    index=li, weight_bits=wb, act_bits=ab,
                    weight_quant_mse=w_mse, act_quant_mse=a_mse,
                    combined_mse=w_mse + a_mse,
                )

    return errors


# ─── Search ──────────────────────────────────────────────────────────

@dataclass
class QuantConfig:
    """Per-layer bit-width assignment."""
    layer_configs: List[Tuple[int, int]]  # [(w_bits, a_bits), ...]
    predicted_mse: float
    avg_mac_per_dsp: float
    total_weight_bits: int


def search_mixed_precision(
    profiles: List[LayerProfile],
    sensitivities: List[LayerSensitivity],
    quant_errors: Dict[Tuple[int, int, int], LayerQuantError],
    dsp: DSPSpec,
    bit_options: List[int] = None,
    mse_threshold: float = None,
    min_mac_per_dsp: float = 2.0,
) -> List[QuantConfig]:
    """Search for Pareto-optimal mixed-precision configurations.

    Returns configs sorted by avg_mac_per_dsp descending (most efficient first),
    filtered by min_mac_per_dsp.
    """
    if bit_options is None:
        bit_options = [4, 5, 6, 8]

    n_layers = len(profiles)
    total_macs = sum(p.in_features * p.out_features for p in profiles)

    # Enumerate all per-layer options
    layer_options = []
    for li in range(n_layers):
        opts = []
        for wb in bit_options:
            for ab in bit_options:
                mac = dsp.mac_per_dsp(wb, ab)
                err = quant_errors.get((li, wb, ab))
                if err is None:
                    continue
                opts.append((wb, ab, mac, err.combined_mse))
        layer_options.append(opts)

    # Brute-force search (feasible for 5 layers × ~16 options = 16^5 ≈ 1M)
    results = []
    for combo in iterproduct(*layer_options):
        pred_mse = 0.0
        weighted_mac = 0.0
        total_wbits = 0

        for li, (wb, ab, mac, layer_mse) in enumerate(combo):
            pred_mse += layer_mse * sensitivities[li].sensitivity
            layer_macs = profiles[li].in_features * profiles[li].out_features
            weighted_mac += mac * layer_macs / total_macs
            total_wbits += profiles[li].n_params * wb

        if weighted_mac < min_mac_per_dsp:
            continue
        if mse_threshold is not None and pred_mse > mse_threshold:
            continue

        cfg = QuantConfig(
            layer_configs=[(wb, ab) for wb, ab, _, _ in combo],
            predicted_mse=pred_mse,
            avg_mac_per_dsp=weighted_mac,
            total_weight_bits=total_wbits,
        )
        results.append(cfg)

    # Pareto filter: keep only configs where no other config is both
    # faster AND more accurate
    results.sort(key=lambda c: (-c.avg_mac_per_dsp, c.predicted_mse))
    pareto = []
    best_mse = float('inf')
    for cfg in results:
        if cfg.predicted_mse < best_mse:
            pareto.append(cfg)
            best_mse = cfg.predicted_mse

    return pareto


# ─── Quantize ────────────────────────────────────────────────────────

def apply_quantization(model: nn.Module, config: QuantConfig) -> nn.Module:
    """Apply per-layer weight quantization according to config."""
    q_model = copy.deepcopy(model)
    layer_idx = 0
    for mod in _all_modules(q_model):
        if isinstance(mod, nn.Linear):
            wb, ab = config.layer_configs[layer_idx]
            with torch.no_grad():
                mod.weight.copy_(_quantize_symmetric(mod.weight.detach(), wb))
                if mod.bias is not None:
                    mod.bias.copy_(_quantize_symmetric(mod.bias.detach(), wb))
            layer_idx += 1
    q_model.eval()
    return q_model


def forward_mixed_precision(model: nn.Module, x: torch.Tensor,
                            config: QuantConfig) -> torch.Tensor:
    """Forward pass with per-layer activation quantization."""
    h = x
    layer_idx = 0
    for mod in _all_modules(model):
        if isinstance(mod, nn.Linear):
            h = mod(h)
            layer_idx += 1
        elif _is_activation(mod):
            h = _apply_non_linear(mod, h)
            # Quantize activation output (except after last layer)
            if layer_idx < len(config.layer_configs):
                _, ab = config.layer_configs[layer_idx - 1]
                qmax = 2 ** (ab - 1) - 1
                h = torch.round(h * qmax) / qmax
        elif isinstance(mod, nn.Dropout):
            pass
    return h


# ─── Validate ────────────────────────────────────────────────────────

def validate_config(model: nn.Module, config: QuantConfig,
                    test_loader, eval_fn) -> Dict[str, float]:
    """Quantize model with config and evaluate on test data."""
    q_model = apply_quantization(model, config)
    q_model.eval()

    true_all, pred_all = [], []
    with torch.no_grad():
        for x, _, y, _ in test_loader:
            pred = forward_mixed_precision(q_model, x, config)
            true_all.append(y.numpy())
            pred_all.append(pred.numpy())

    true_all = np.concatenate(true_all)
    pred_all = np.concatenate(pred_all)
    metrics = eval_fn(pred_all, true_all)
    return {
        'MAE': metrics[0], 'MAPE': metrics[1],
        'MSE': metrics[2], 'RMSE': metrics[3],
    }


# ─── Full Pipeline ───────────────────────────────────────────────────

def run_pipeline(
    model: nn.Module,
    calib_x: torch.Tensor,
    test_loader,
    eval_fn,
    dsp: DSPSpec = DSP48E2,
    bit_options: List[int] = None,
    top_k: int = 10,
) -> Dict:
    """Run full FPGA-aware quantization pipeline.

    Args:
        model: trained PINN model (eval mode)
        calib_x: calibration input tensor
        test_loader: test data loader
        eval_fn: function(pred, true) -> [MAE, MAPE, MSE, RMSE]
        dsp: FPGA DSP specification
        bit_options: bit-widths to search over
        top_k: number of Pareto configs to validate

    Returns:
        dict with profiles, sensitivities, pareto configs, and validation results
    """
    if bit_options is None:
        bit_options = [4, 5, 6, 8]

    # Step 1: Profile
    profiles = profile_layers(model, calib_x)

    # Step 2: Sensitivity
    sensitivities = measure_sensitivity(model, calib_x)

    # Step 3 & 4: Measure errors and search
    quant_errors = measure_quant_errors(model, calib_x, bit_options)
    pareto = search_mixed_precision(
        profiles, sensitivities, quant_errors, dsp, bit_options)

    # Step 5 & 6: Validate top-k
    validated = []
    for cfg in pareto[:top_k]:
        actual = validate_config(model, cfg, test_loader, eval_fn)
        validated.append({
            'config': [(w, a) for w, a in cfg.layer_configs],
            'predicted_mse': cfg.predicted_mse,
            'avg_mac_per_dsp': cfg.avg_mac_per_dsp,
            'weight_kb': cfg.total_weight_bits / 8 / 1024,
            'actual_mape': actual['MAPE'],
            'actual_rmse': actual['RMSE'],
        })

    return {
        'profiles': [{'name': p.name, 'params': p.n_params,
                       'w_max': p.weight_abs_max, 'w_std': p.weight_std,
                       'a_max': p.activation_abs_max, 'a_std': p.activation_std}
                      for p in profiles],
        'sensitivities': [{'name': profiles[s.index].name, 'sensitivity': s.sensitivity}
                          for s in sensitivities],
        'pareto_configs': validated,
    }


# ─── Internal helpers ────────────────────────────────────────────────

def _all_modules(model):
    """Yield all leaf modules in forward order for Solution_u architecture."""
    from models.Model import Sin
    for mod in model.encoder.net:
        yield mod
    for mod in model.predictor.net:
        yield mod


def _is_activation(mod):
    from models.Model import Sin
    return isinstance(mod, (Sin, nn.Tanh, nn.ReLU))


def _apply_non_linear(mod, h):
    from models.Model import Sin
    if isinstance(mod, Sin):
        return torch.sin(h)
    elif isinstance(mod, nn.Dropout):
        return h
    else:
        return mod(h)


def _forward_full(model, x):
    h = x
    for mod in _all_modules(model):
        if isinstance(mod, nn.Linear):
            h = mod(h)
        else:
            h = _apply_non_linear(mod, h)
    return h


def _forward_with_noise(model, x, target_layer, noise):
    h = x
    li = 0
    for mod in _all_modules(model):
        if isinstance(mod, nn.Linear):
            h = mod(h)
            if li == target_layer:
                h = h + noise[:h.shape[0]]
            li += 1
        else:
            h = _apply_non_linear(mod, h)
    return h


def _quantize_symmetric(tensor, bits):
    if bits >= 32:
        return tensor.clone()
    qmax = 2 ** (bits - 1) - 1
    abs_max = tensor.abs().max().item()
    if abs_max <= 0:
        return tensor.clone()
    scale = abs_max / qmax
    return torch.round(tensor / scale) * scale
