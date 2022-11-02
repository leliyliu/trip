export CUDA_VISIBLE_DEVICES=$1
python train.py \
--save_folder ./logs \
--arch cifar100_resnet_74 \
--workers 4 \
--dataset cifar100 \
--datadir /home/shared/cifar-100 \
--num_bits 8 \
--num_grad_bits 8 \
--experiment_name "naive-integer" \
--scenario_name "integer-training"