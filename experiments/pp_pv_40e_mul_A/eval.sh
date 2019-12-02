now=$(date +"%Y%m%d_%H%M%S")
work_path=$(dirname $0)
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:1 --ntasks-per-node=1 \
python -u eval_seq.py --config $work_path/config.yaml \
--load-path=./pretrain_models/pp_pv_40e_mul_A-gpu.pth \
--result-path=$work_path/results \
--result_sha=all \
2>&1|tee $work_path/Eval-pts-${now}.log 
