#!/bin/zsh

# Define a dictionary (associative array)
declare -A my_dict=(
  ["Baseline"]="data/Baseline/"
  ["D1"]="data/D1/"
)

output_list=()

# Iterate over each key-value pair in the dictionary
for key value in ${(kv)my_dict}; do
  out_dir="docs/experiments"
  out_nomask="${key}_nomask"
  out_masked="${key}_masked"

  python3 corrdiff_plotgen.py "$value" "$out_dir/$out_nomask" --masked="no"
  python3 corrdiff_plotgen.py "$value" "$out_dir/$out_masked"

  output_list+=("$out_nomask")
  output_list+=("$out_masked")
done

# Convert the output_list to a JSON array and save to file
output_json=$(printf '%s\n' "${output_list[@]}" | jq -R . | jq -s .)
echo $output_json > "$out_dir/list.json"

echo "JSON file '$out_dir/list.json' created successfully."