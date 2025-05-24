export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:16:8

## train_percent represents the percentage of the total data that is used for training
touch log_10_precentage
nohup python -u driver/TrainTest.py  --config_file config/new.train  --use_cuda --train_percent 0.1 > log_10_percentage 2>&1 &
tail -f log_10_percentage

