now=$(date +"%Y%m%d_%H%M%S")
job=rrc_pfv_adam6e-4_40e_l2_branchbce_subabs_dualmax_C
work_path=$(dirname $0)
srun --mpi=pmi2 -p $1 -J $job -n1 --gres=gpu:1 --ntasks-per-node=1 \
python -u test.py --config $work_path/config.yaml \
--result-path=$work_path/test_results \
--result_sha=test_mm \
--load-path=$work_path/ckpt_best.pth.tar \
2>&1|tee $work_path/Test-img_${now}.log

