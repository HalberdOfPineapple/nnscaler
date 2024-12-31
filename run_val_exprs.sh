#!/usr/bin/bash

iter_list=(10 20 30 40 50 60 70 80 90 100 110)

for iter_idx in ${iter_list[@]}
do
    echo "-----------------------------------------------------------------------------------------"
    echo "Running validation for iter_idx: $iter_idx of phi_mf_mb_4k_re_131072 (minfer_phi, H100_8)"
    python /scratch/nnscaler/experiments/minfer_phi/validation.py --gpu_set H100_8 --expr_dir minfer_phi --expr_name phi_mf_mb_4k_re_131072 --epoch_idx 0 --iter_idx $iter_idx
done

for iter_idx in ${iter_list[@]}
do
    echo "-----------------------------------------------------------------------------------------"
    echo "Running validation for iter_idx: $iter_idx of phi_mf_mb_4k_mcontrol_131072 (minfer_phi, H100_8)"
    python /scratch/nnscaler/experiments/minfer_phi/validation.py --gpu_set H100_8 --expr_dir minfer_phi --expr_name phi_mf_mb_4k_mcontrol_131072 --epoch_idx 0 --iter_idx $iter_idx
done


iter_list=(10 20 30 40 50 60 70 80 90 100 110)
for iter_idx in ${iter_list[@]}
do
    echo "-----------------------------------------------------------------------------------------"
    echo "Running validation for iter_idx: $iter_idx of phi_lc_131072 (minfer_phi, H100_4)"
    python /scratch/nnscaler/experiments/minfer_phi/validation.py --gpu_set H100_4 --expr_dir minfer_phi --expr_name phi_lc_131072 --epoch_idx 0 --iter_idx $iter_idx
done