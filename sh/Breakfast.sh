export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1

TOTAL_NUM_GPUS=4

PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "Using port: $PORT"

declare -a datasets=("Breakfast")

declare -A dataset_classes
dataset_classes["HMDB51"]=51
dataset_classes["UCF101"]=101
dataset_classes["SSV2"]=174
dataset_classes["Breakfast"]=10

declare -A dataset_num_frames
dataset_num_frames["HMDB51"]=8
dataset_num_frames["UCF101"]=8
dataset_num_frames["SSV2"]=8
dataset_num_frames["Breakfast"]=32

declare -A dataset_epochs
dataset_epochs["HMDB51"]=10
dataset_epochs["UCF101"]=10
dataset_epochs["SSV2"]=7
dataset_epochs["Breakfast"]=20

declare -A dataset_lr
dataset_lr["HMDB51"]=3e-3
dataset_lr["UCF101"]=5e-3
dataset_lr["SSV2"]=2e-4
dataset_lr["Breakfast"]=1e-3

declare -A dataset_bs
dataset_bs["HMDB51"]=8
dataset_bs["UCF101"]=8
dataset_bs["SSV2"]=8
dataset_bs["Breakfast"]=2

declare -A dataset_update_freq
dataset_update_freq["HMDB51"]=1
dataset_update_freq["UCF101"]=1
dataset_update_freq["SSV2"]=16
dataset_update_freq["Breakfast"]=8

declare -a tuning_configs=("ours")


for dataset in "${datasets[@]}"
do

    nb_classes=${dataset_classes[$dataset]}
    data_path="/data/dataset/zhukai/$dataset"

    num_frames=${dataset_num_frames[$dataset]}

    real_dataset=$dataset
    if [[ $dataset == "Breakfast" ]]; then
        real_dataset="Kinetics_sparse"
    fi

    echo "Training on dataset: $dataset, Number of classes: $nb_classes"

    for tuning_config in "${tuning_configs[@]}"
    do

        output_dir="./output/$tuning_config/$dataset/$tuning_config-f$num_frames"
        log_dir="./logs/$tuning_config/$dataset/$tuning_config-f$num_frames"

        run_time_log_dir="${log_dir}/run_time_logs-ViM-$tuning_config-$dataset.log"
        mkdir -p $log_dir

        finetune_path="/data/ckpt/zhukai/VideoMamba-Checkpoints/videomamba_m16_k400_mask_ft_f${num_frames}_res224.pth"

        lr=${dataset_lr[$dataset]}
        epochs=${dataset_epochs[$dataset]}
        batch_size=${dataset_bs[$dataset]}
        update_freq=${dataset_update_freq[$dataset]}

        python3 -m torch.distributed.run \
            --nproc_per_node=$TOTAL_NUM_GPUS \
            --master_port=$PORT \
            main_video.py \
            --enable_deepspeed \
            --model videomamba_middle \
            --finetune $finetune_path \
            --data_path $data_path \
            --data_set $real_dataset \
            --nb_classes $nb_classes \
            --log_dir $log_dir \
            --output_dir $output_dir \
            --batch_size $batch_size \
            --num_sample 2 \
            --input_size 224 \
            --short_side_size 224 \
            --save_ckpt_freq 100 \
            --num_frames $num_frames \
            --num_workers 20 \
            --warmup_epochs 5 \
            --tubelet_size 1 \
            --epochs $epochs \
            --update_freq $update_freq \
            --lr $lr \
            --layer_decay 0.8 \
            --drop_path 0.4 \
            --opt adamw \
            --opt_betas 0.9 0.999 \
            --weight_decay 0.05 \
            --test_num_segment 8 \
            --test_num_crop 6 \
            --dist_eval \
            --test_best \
            --bf16 \
            --$tuning_config \
            --ours_num 3 \
        > "${run_time_log_dir}" 2>&1

        echo "Finished running $tuning_config with lr=$lr on $dataset, runtime logs are saved in $run_time_log_dir"

    done

    echo "Completed all configurations for dataset: $dataset"
    echo "-----------------------------------------------------"
done

echo "All training tasks completed!"
