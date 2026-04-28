# =============================================================================
# run_vivado.tcl — Non-project mode: synth + impl for PINN dataflow
# =============================================================================

set part "xc7z020clg400-1"

# ── Read sources ──
read_verilog -sv src/mac3_pack.v
read_verilog -sv src/layer_stage.v
read_verilog -sv src/pinn_dataflow.v

# ── Synthesis (out-of-context: no I/O buffers) ──
puts "=========================================="
puts "  Running Synthesis (out-of-context)..."
puts "=========================================="
synth_design -top pinn_dataflow -part $part -mode out_of_context -flatten_hierarchy rebuilt

# ── Post-synth reports ──
report_utilization -file vivado_proj/synth_utilization.rpt
report_timing_summary -file vivado_proj/synth_timing.rpt
write_checkpoint -force vivado_proj/post_synth.dcp

puts "\n=== Post-Synthesis Utilization ==="
report_utilization

# ── Clock constraint ──
create_clock -period 10.000 -name clk [get_ports clk]

# ── Optimize ──
opt_design

# ── Place ──
puts "\n=========================================="
puts "  Running Placement..."
puts "=========================================="
place_design -directive Explore

# ── Route ──
puts "\n=========================================="
puts "  Running Routing..."
puts "=========================================="
route_design -directive Explore

# ── Post-impl reports ──
file mkdir vivado_proj
report_utilization -file vivado_proj/impl_utilization.rpt
report_timing_summary -file vivado_proj/impl_timing.rpt
report_power -file vivado_proj/impl_power.rpt
write_checkpoint -force vivado_proj/post_impl.dcp

puts "\n=========================================="
puts "  Implementation Complete!"
puts "=========================================="

puts "\n=== Post-Implementation Utilization ==="
report_utilization

puts "\n=== Timing Summary ==="
report_timing_summary

puts "\n=== Power Estimate ==="
report_power

exit 0
