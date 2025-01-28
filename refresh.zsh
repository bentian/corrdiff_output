#!/bin/zsh

# Define a dictionary (associative array)
declare -A my_dict=(
  ["Baseline"]="data/Baseline/"
  ["D1"]="data/D1/"
)

# Iterate over each key-value pair in the dictionary
for key value in ${(kv)my_dict}; do
  out_nomask="docs/experiments/${key}_nomask"
  out_masked="docs/experiments/${key}_masked"

  echo "Running corrdiff_plotgen.py with data: $value and output: $out_nomask"
  python3 corrdiff_plotgen.py "$value" "$out_nomask" --masked="no"

  echo "Running corrdiff_plotgen.py with data: $value and output: $out_masked"
  python3 corrdiff_plotgen.py "$value" "$out_masked"
done
