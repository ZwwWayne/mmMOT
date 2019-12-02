now=$(date +"%Y%m%d_%H%M%S")
job=pp0.1_p_adam6e-4_40e_l2_branchbce_mul_s5nogn
work_path=$(dirname $0)
srun --mpi=pmi2 -p AD -J $job -n1 --gres=gpu:1 --ntasks-per-node=1 -w SH-IDC1-10-5-36-234 \
python -u eval_seq.py --config $work_path/config.yaml \
--load-path=$work_path/ckpt_best.pth.tar \
--result-path=$work_path/results \
--result_sha=Tracking_3part \
2>&1|tee $work_path/Eval-split-${now}.log  &
