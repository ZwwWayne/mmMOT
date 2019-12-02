now=$(date +"%Y%m%d_%H%M%S")
work_path=$(dirname $0)
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:1 --ntasks-per-node=1 \
python -u main.py --config $work_path/config.yaml \
--result-path=$work_path/results \
2>&1|tee $work_path/T-${now}.log  &
 #--load-path=$work_path/ckpt.pth.tar \
 #--recover
