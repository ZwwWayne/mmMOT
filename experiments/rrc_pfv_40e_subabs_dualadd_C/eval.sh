now=$(date +"%Y%m%d_%H%M%S")
work_path=$(dirname $0)
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:1 --ntasks-per-node=1 \
python -u eval_seq.py --config $work_path/config.yaml \
--load-path=$work_path/ckpt_best.pth.tar \
--result-path=$work_path/results \
--result_sha=all \
2>&1|tee $work_path/Eval-split-${now}.log
