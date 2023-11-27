# ----------------------------------------------trainer-Copy1----------------------------------
# export CUDA_LAUNCH_BLOCKING=1 LOCAL_RANK=0
# deepspeed --module training.trainer-Copy1 \
#     --input-model EleutherAI/pythia-2.8b \
#     --deepspeed ./config/a100_config.json \
#     --epochs 2 \
#     --local-output-dir ./output/1 \
#     --per-device-train-batch-size 6 \
#     --per-device-eval-batch-size 6 \
#     --logging-steps 10 \
#     --save-steps 200 \
#     --save-total-limit 20 \
#     --eval-steps 50 \
#     --warmup-steps 50 \
#     --test-size 200 \
#     --lr 5e-6 \
#     --bf16 true

# ---------------------------------------------trainer-Copy2-----------------------------------

# export OMP_NUM_THREADS=3 
# export TORCH_CPP_LOG_LEVEL=INFO NCCL_DEBUG=INFO

# #--nproc_per_node=2
# LOCAL_RANK=4
NCCL_P2P_LEVEL=NVL && python -m torch.distributed.launch \
    --nproc_per_node 2 --nnodes 3 --node_rank 2\
    --master_addr "100.200.0.3" --master_port 23456\
    training/trainer-Copy2.py \
    --input-model EleutherAI/pythia-2.8b \
    --deepspeed ./config/a100_config.json \
    --epochs 2 \
    --local-output-dir ./output/1 \
    --per-device-train-batch-size 6 \
    --per-device-eval-batch-size 6 \
    --logging-steps 1 \
    --save-steps 200 \
    --save-total-limit 20 \
    --eval-steps 50 \
    --warmup-steps 50 \
    --test-size 200 \
    --lr 5e-6 \
    --bf16 true --local_rank 0

# ----------------------------------------------trainer-Copy3----------------------------------
# --node_rank 0
# export TORCH_CPP_LOG_LEVEL=INFO NCCL_DEBUG=INFO NCCL_P2P_LEVEL=NVL
# TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.run --nproc_per_node 2 \
#                                     training/trainer-Copy3.py --local-output-dir ./output/1 \
#                                     --bf16 true --epochs 2 \
#                                     --per-device-train-batch-size 6 \
#                                     --per-device-eval-batch-size 6 \
#                                     --logging-steps 1 \
#                                     --save-steps 200 \
#                                     --save-total-limit 20 \
#                                     --eval-steps 50 \
#                                     --warmup-steps 50 \
#                                     --test-size 200 \
#                                     --lr 5e-6 \
#                                     --bf16 true


# accelerate launch --multi_gpu \
#                 --machine_rank=2 --main_process_ip="10.41.0.5" --main_process_port=23456\
#                 --num_processes=6 --num_machines=3\
#                 training/trainer-Copy5.py \
#                 --input-model EleutherAI/pythia-2.8b  \
#                 --deepspeed ./config/a100_config.json \
#                 --epochs 2  --local-output-dir ./output/1 \
#                 --per-device-train-batch-size 6  \
#                 --per-device-eval-batch-size 6   \
#                 --logging-steps 1 \
#                 --save-steps 200  --save-total-limit 20 \
#                 --eval-steps 50  --warmup-steps 50 \
#                 --test-size 200  --lr 5e-6 --bf16 true --local_rank 0


# accelerate launch --multi_gpu \
#                 --machine_rank=2 --main_process_ip="10.41.0.5" --main_process_port=23456\
#                 --num_processes=6 --num_machines=3\
#                 DOLLY.py \