#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --account=ctb-enamul
#SBATCH --partition=c-enamul
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=99G
#SBATCH --output=jobs_output/%x-%j.out

module load gcc/12.3 python scipy-stack arrow

mkdir $SLURM_TMPDIR/work
python -m venv $SLURM_TMPDIR/work/env
source $SLURM_TMPDIR/work/env/bin/activate

mkdir $SLURM_TMPDIR/work/codes
mkdir $SLURM_TMPDIR/work/models
cp -r /home/msm97/projects/def-enamul/msm97/Codebases/Test_torch/codes_v6/* $SLURM_TMPDIR/work/codes
mkdir /home/msm97/scratch/model_checkpoints/OCQA/1

mkdir $SLURM_TMPDIR/work/models/unichart
mkdir $SLURM_TMPDIR/work/models/gemma2-2B

pip install --no-index -r /home/msm97/projects/def-enamul/msm97/Codebases/new.txt
cp /home/msm97/projects/def-enamul/msm97/Codebases/Test_torch/trainer.py $SLURM_TMPDIR/work/env/lib/python3.11/site-packages/pytorch_lightning/trainer

cp -r /home/msm97/scratch/models/Unichart/* $SLURM_TMPDIR/work/models/unichart
cp -r /home/msm97/scratch/models/models--google--gemma-2-2b/* $SLURM_TMPDIR/work/models/gemma2-2B

mkdir $SLURM_TMPDIR/data
mkdir $SLURM_TMPDIR/files


# Chart-to-text 
cd $SLURM_TMPDIR
cp /home/msm97/scratch/datasets/Benchmarks/Chart-to-text/statista/statista_data.json $SLURM_TMPDIR/data
unzip -q /home/msm97/scratch/datasets/Chart-to-text-main.zip -d $SLURM_TMPDIR/files
python /home/msm97/projects/def-enamul/msm97/Codebases/ChartInstruct/benchmark_scripts/CTT_move_images.py
##statista
python /home/msm97/projects/def-enamul/msm97/Codebases/ChartInstruct/benchmark_finetuning/UniGemma_CTT_statista_finetune.py --model-path /home/msm97/scratch/model_checkpoints/ins_tuning/model-checkpoint-epoch=2-104000