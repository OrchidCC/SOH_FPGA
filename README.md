# PINN-FPGA

Co-designed quantized Physics-Informed Neural Network FPGA accelerator for battery State-of-Health estimation. Combines per-channel GridWarp QAT (W4A4) with a correction-free neuron-parallel MAC datapath on Xilinx Zynq-7020.

## Structure

```
models/
  Model.py             PINN model definition
  Compare_Models.py    Baselines for comparison
  checkpoints/         Best QAT checkpoint per dataset/cell
data/                  HUST / MIT / TJU / XJTU battery datasets
quantization/          W4A4 QAT, per-channel GridWarp, ablations
scripts/               Training and figure-plotting entry points
utils/                 Dataloader and helpers
results/               Aggregated JSON results and figures
rtl/
  src/                 Verilog (mac3_pack, layer_stage, pinn_dataflow)
  tb/                  Testbenches
  reports/             Synthesis & implementation reports (Zynq-7020)
  run_vivado.tcl       Vivado build script
```

## Quick Start

### Training

```bash
# Train on all datasets
python scripts/train_all_datasets.py

# Run W4A4 ablations
python quantization/w4a4_ablation_final.py

# PCR vs PTQ sweep
python scripts/run_all_pcr_vs_ptq.py
```

### FPGA

```bash
cd rtl
vivado -mode batch -source run_vivado.tcl
```

## License

MIT
