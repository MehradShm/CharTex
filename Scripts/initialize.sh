
salloc --mem=125G --time=6:0:0 --cpus-per-task=32 --gpus-per-node=a100:4 --account=ctb-enamul --partition=c-enamul

module load gcc/12.3 python scipy-stack arrow

mkdir $SLURM_TMPDIR/work
python -m venv $SLURM_TMPDIR/work/env
source $SLURM_TMPDIR/work/env/bin/activate

mkdir $SLURM_TMPDIR/work/codes
cp /home/msm97/scratch/Codebases/Test_torch/codes/* $SLURM_TMPDIR/work/codes
cd $SLURM_TMPDIR/work/codes

pip install --no-index -r /home/msm97/scratch/Codebases/Test_torch/reqs.txt