#!/bin/zsh

# Define dictionary (associative array)
declare -A my_dict=(
  ["BL"]="data/Baseline"
  ["D1"]="data/D1"
  ["D2"]="data/D2"
  ["D3"]="data/D3"
)

out_dir="docs/experiments"
output_list=()

# Iterate over dictionary keys
for key value in ${(kv)my_dict}; do
  # Default set of experiments
  typeset -A experiments=(
    "${key}_2M"          "$value/2M"
    "${key}_2M_ens64"    "$value/2M_ens64"
    "${key}_nomask_2M"   "$value/2M"
  )

  typeset -A flags=(
    "${key}_2M_ens64"    "--n-ensemble=64"
    "${key}_nomask_2M"   "--masked=no"
  )

  # Add experiments based on key conditions
  [[ "$key" != "D3" ]] && experiments+=("${key}_extreme_1M" "$value/2M_extreme")
  [[ "$key" != "D2" ]] && experiments+=(
    "${key}_4M"      "$value/4M"
    "${key}_4M_1322" "$value/4M"
    # "${key}_2M_ds4"  "$value/ds_4/2M"
  )

  # Store experiment order in an indexed array
  experiment_order=("${(@k)experiments}")

  # Run experiments
  for exp_name exp_path in ${(kv)experiments}; do
    exp_flag="${flags[$exp_name]:-}"  # Get flag or empty string
    python3 corrdiff_plotgen.py "$exp_path" "$out_dir/$exp_name" ${exp_flag:+$exp_flag}
    output_list+=("$exp_name")
  done
done
python3 plot_prcp_metric.py

# Save output_list as JSON
output_json=$(printf '%s\n' "${output_list[@]}" | jq -R . | jq -s .)
echo $output_json > "$out_dir/list.json"

echo "JSON file '$out_dir/list.json' created successfully."
