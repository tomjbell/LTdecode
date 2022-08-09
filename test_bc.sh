#!/bin/bash

dirname=$(python3 generate_input_files.py -fn="10_qubit_graphs_ordered_num_in_class" -p="$PWD/data/uib_data" -bs=100)
echo "$dirname"
echo "Data generated"
echo "analysing graphs"
python3 script_blue_crystal.py -arix=7 -p="$PWD/$dirname"

