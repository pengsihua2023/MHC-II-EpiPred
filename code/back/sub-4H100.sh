#!/bin/bash
#SBATCH --job-name=Batch2-dearly-0.0001           # Job name
#SBATCH --partition=gpu_p             # Partition (queue) name
#SBATCH --gres=gpu:H100:4             # Requests 4 GPU devices
#SBATCH --nodes=1                     # Requests 1 node
#SBATCH --ntasks=4                    # Total number of tasks
#SBATCH --ntasks-per-node=4           # Number of tasks per node
#SBATCH --cpus-per-task=16            # Number of CPU cores per task
#SBATCH --mem=512gb                   # Job memory request
#SBATCH --time=7-00:00:00             # Time limit hrs:min:sec
#SBATCH --output=Batch2-Drop-0.4-98.%j.out      # Standard output log
#SBATCH --error=Batch2-Drop-0.4-98.%j.err       # Standard error log
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sp96859@uga.edu   # Where to send mail

cd $SLURM_SUBMIT_DIR

# 加载模块
ml Miniconda3/23.5.2-0
ml CUDA/12.4.0  # 使用CUDA 12.4，与PyTorch 2.4.0+cu124匹配

echo "Checking allocated GPUs:"
nvidia-smi

echo "Activating Conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/sp96859/.conda/envs/ESM2

# 检查环境
echo "Python path:"
which python
echo "Pip path:"
which pip
echo "PyTorch version and CUDA info:"
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

# 设置CUDA库路径
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
# export NCCL_SOCKET_TIMEOUT=1800  # 设置 30 分钟

export NCCL_P2P_DISABLE=1

export TORCH_DISTributed_TIMEOUT=3600
export NCCL_DEBUG=INFO

# 设置 Triton 缓存到本地存储，避免使用 NFS
export TRITON_CACHE_DIR=/tmp/triton_cache
mkdir -p /tmp/triton_cache  # 确保目录存在

# 启动DeepSpeed
deepspeed --num_gpus=4 Grok3-With-LN-II-650M-H100-V10.py
