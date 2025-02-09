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
  nomask="${key}_2M_1823"
  masked="${key}_msk_2M_1823"
  masked_4M="${key}_msk_4M_1823"

  python3 corrdiff_plotgen.py "$value/2M" "$out_dir/$nomask" --masked="no"
  python3 corrdiff_plotgen.py "$value/2M" "$out_dir/$masked"
  python3 corrdiff_plotgen.py "$value/4M" "$out_dir/$masked_4M"

  output_list+=("$nomask")
  output_list+=("$masked")
  output_list+=("$masked_4M")
done

# Convert the output_list to a JSON array and save to file
output_json=$(printf '%s\n' "${output_list[@]}" | jq -R . | jq -s .)
echo $output_json > "$out_dir/list.json"

echo "JSON file '$out_dir/list.json' created successfully."
