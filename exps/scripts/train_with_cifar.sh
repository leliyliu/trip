export CUDA_VISIBLE_DEVICES=$1 
python train_with_cifar.py \
    --datadir /home/shared/cifar-10 \
    --num_bits 8 \
    --num_grad_bits 8 \
    --momentum 0.0 \
    --weight_decay 0.0 \
    --experiment_name "trip_train_with_cifar" \
    --scenario_name "trip-training" 