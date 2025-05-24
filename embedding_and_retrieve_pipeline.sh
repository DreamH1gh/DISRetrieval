set -e

DATA_DIR="data/qasper"

echo "build and retrieve"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rag
python tree_retrieve/build_and_retrieve.py \
    --data_path "$DATA_DIR"\
    --save_path qasper_result.json \
    --max_node_depth 10000 \
    --max_len 400 \
    --leaf_topk 5 \
    --method rest_topk