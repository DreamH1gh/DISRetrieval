#!/bin/bash
set -e

DATA_DIR="data/qasper"

echo "Running Stage 1 with env_rst_parser"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate RSTparser
python rst_tree_build/tree_build.py --data_path "$DATA_DIR" --stage first


echo "Running summarize using vllm"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm
python rst_tree_build/vllm_summrize.py \
    --tree_path "$DATA_DIR/doc_data/subtree" \
    --save_path "$DATA_DIR/subtree_vllm.json" \
    --tau 0


echo "Running Stage 2 with env_rst_parser"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate RSTparser
python rst_tree_build/tree_build.py --data_path "$DATA_DIR" --stage second


echo "Running summarize using vllm"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm
python rst_tree_build/vllm_summrize.py \
    --tree_path  "$DATA_DIR/doc_data/highlevel_tree" \
    --save_path "$DATA_DIR/highlevel_vllm.json" \
    --tau 0


echo "Running final Stage with env_rst_parser"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate RSTparser
python rst_tree_build/tree_build.py --data_path "$DATA_DIR" --stage final


echo "All done!"