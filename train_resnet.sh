#GPU
gpu=1

#CSV files
csv_train=/home/hthieu/AICityChallenge2019/resnet/data/veri_type_id.csv
csv_test=/home/hthieu/AICityChallenge2019/resnet/data/veri_type_id_test.csv

#Pre-trained model
checkpoint=/home/hthieu/AICityChallenge2019/checkpoint/resnet_v1_101.ckpt

#Experiment ID
exp_id=vehi_type_classify

#Model Config
model=resnet_v1_101

#Dataset directory
data_root=/home/hthieu/AICityChallenge2019/data
data_name=VeRi

#Output folder
src_model_dir=/home/hthieu/AICityChallenge2019/track2_resnet_experiments/pretrained_${model}_${exp_id}

#Training Configuration
train=1
test=1

clean_log=1
clean_dir=0

data_dir=$data_root/$data_name

gray2rgb=0
image_size=224

split_train=train_images
split_test=test
feat_name=global_pool

#Learning Cofig
num_iters_cls=7000
solver_cls=adam
lr_cls=0.0001
weight_decay=0.00002

batch_size_cls=64

#Traning Source Only
if [[ $train -eq 1 ]]; then
    if [ ! -d $src_model_dir ]; then #Create source model path
        mkdir -p $src_model_dir
    fi

    if [[ $clean_log -eq 1 ]]; then #Delete Log
        rm -rf $src_model_dir/events.*
    fi

    if [[ $clean_dir -eq 1 ]]; then
        rm -rf $src_model_dir/*
    fi

    ./train_classifier.py \
        --gpu_id $gpu \
        --solver $solver_cls \
        --learning_rate $lr_cls \
        --weight_decay $weight_decay \
        --num_iters $num_iters_cls \
        --model $model \
        --checkpoint $checkpoint \
        --model_path $src_model_dir \
        --dataset $data_name \
        --split $split_train \
        --dataset_dir $data_dir \
        --gray2rgb $gray2rgb \
        --batch_size $batch_size_cls \
        --image_size $image_size \
        --csv_file $csv_train 
fi


#Test Target Only
if [[ $test -eq 1 ]]; then
    python3 test_classifier.py \
        --gpu_id $gpu \
        --model $model \
        --model_path $src_model_dir \
        --dataset $data_name\
        --dataset_dir $data_dir \
        --split $split_test \
        --gray2rgb $gray2rgb \
        --batch_size $batch_size_cls \
        --image_size $image_size \
        --csv_file $csv_test
fi
