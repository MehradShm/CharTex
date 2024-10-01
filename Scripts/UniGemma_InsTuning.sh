#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --account=ctb-enamul
#SBATCH --partition=c-enamul
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=110G
#SBATCH --output=jobs_output/%x-%j.out

module load gcc/12.3 python scipy-stack arrow

mkdir $SLURM_TMPDIR/work

python -m venv $SLURM_TMPDIR/work/env
source $SLURM_TMPDIR/work/env/bin/activate
pip install --no-index -r /home/msm97/projects/def-enamul/msm97/Codebases/new.txt
cp /home/msm97/projects/def-enamul/msm97/Codebases/Test_torch/trainer.py $SLURM_TMPDIR/work/env/lib/python3.11/site-packages/pytorch_lightning/trainer

mkdir $SLURM_TMPDIR/uni_subset
mkdir $SLURM_TMPDIR/work/codes
mkdir $SLURM_TMPDIR/work/models
mkdir $SLURM_TMPDIR/work/results
cp /home/msm97/projects/def-enamul/msm97/Codebases/ChartInstruct/* $SLURM_TMPDIR/work/codes
cd $SLURM_TMPDIR/work/codes


mkdir $SLURM_TMPDIR/work/models/unichart
mkdir $SLURM_TMPDIR/work/models/gemma2-2B

unzip -q /home/msm97/projects/def-enamul/pretraining_project_shared_data/'UniChart Images.zip' -d $SLURM_TMPDIR
mv $SLURM_TMPDIR/content/'UniChart Images' $SLURM_TMPDIR/content/tmp

cp /home/msm97/scratch/datasets/fixed_format_instructions.json $SLURM_TMPDIR

cd $SLURM_TMPDIR
mkdir $SLURM_TMPDIR/tmp
tar -xzf /home/msm97/scratch/datasets/WebImages.tar.gz -C $SLURM_TMPDIR/tmp
cp /home/msm97/projects/def-enamul/msm97/Codebases/test_codes/move_images.py $SLURM_TMPDIR
python $SLURM_TMPDIR/move_images.py



cp -r /home/msm97/scratch/models/Unichart/* $SLURM_TMPDIR/work/models/unichart
cp -r /home/msm97/scratch/models/models--google--gemma-2-2b/* $SLURM_TMPDIR/work/models/gemma2-2B

python $SLURM_TMPDIR/work/codes/UniGemma_InsTuning.py

/localscratch/msm97.25219646.0/content/tmp