#!/bin/zsh

# Define a dictionary (associative array)
declare -A my_dict=(
  ["BL"]="data/Baseline/"
  ["D1"]="data/D1/"
)

output_list=()

# Iterate over each key-value pair in the dictionary
for key value in ${(kv)my_dict}; do
  out_dir="docs/experiments"

  # Experiments
  nomask_2M="${key}_nomask_2M"
  masked_2M="${key}_2M"
  masked_4M="${key}_4M"
  masked_2M_ds4="${key}_2M_ds4"

  python3 corrdiff_plotgen.py "$value/2M" "$out_dir/$nomask_2M" --masked="no"
  python3 corrdiff_plotgen.py "$value/2M" "$out_dir/$masked_2M"
  python3 corrdiff_plotgen.py "$value/4M" "$out_dir/$masked_4M"
  python3 corrdiff_plotgen.py "$value/ds_4/2M" "$out_dir/$masked_2M_ds4"

  output_list+=("$nomask_2M")
  output_list+=("$masked_2M")
  output_list+=("$masked_4M")
  output_list+=("$masked_2M_ds4")
done

# Convert the output_list to a JSON array and save to file
output_json=$(printf '%s\n' "${output_list[@]}" | jq -R . | jq -s .)
echo $output_json > "$out_dir/list.json"

echo "JSON file '$out_dir/list.json' created successfully."
