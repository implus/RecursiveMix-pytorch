# Code for 'RecursiveMix: Mixed Training with History'

RecursiveMix (RM), which uses the historical input-prediction-label triplet to enhance the generalization of Deep Vision Models

## Requirements

Experiment Environment
- python == 3.6
- pytorch == 1.7.1+cu101
- torchvision == 0.8.2

## Usage

### 1. Train the model
For example, to reproduce the results of RM in CIFAR-10 (97.65% Top-1 acc in averaged 3 runs, logs are provided in logs/):
```python
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29500 main.py \
            --name 'your_experiment_log_path' \
            --model_file 'pyramidnet' \
            --model_name 'pyramidnet_200_240' \
            --data 'cifar10' \
            --data_dir '/path/to/CIFAR10' \
            --epoch 300 \
            --batch_size 64 \
            --lr 0.25 \
            --scheduler 'step' \
            --schedule 150 225 \
            --weight_decay 1e-4 \
            --nesterov \
            --num_workers 8 \
            --save_model \
            --aug 'recursive_mix' \
            --aug_alpha 0.5 \
            --aug_omega 0.1
```

### 2. Test the model
```python
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29500 main.py \
            --name 'your_experiment_log_path' \
            --batch_size 64 \
            --model_file 'pyramidnet' \
            --model_name 'pyramidnet_200_240' \
            --data 'cifar10' \
            --data_dir '/path/to/CIFAR10' \
            --num_workers 8 \
            --evaluate \
            --resume 'best'
```