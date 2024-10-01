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

cp -r /home/msm97/scratch/models/Unichart/* $SLURM_TMPDIR/work/models/unichart
cp -r /home/msm97/scratch/models/models--google--gemma-2-2b/* $SLURM_TMPDIR/work/models/gemma2-2B

mkdir $SLURM_TMPDIR/data
mkdir $SLURM_TMPDIR/files



unzip -q /home/msm97/scratch/datasets/Benchmarks/CQA/CQABench.zip -d $SLURM_TMPDIR/data
cd $SLURM_TMPDIR
python /home/msm97/projects/def-enamul/msm97/Codebases/ChartInstruct/benchmark_finetuning/UniGemma_CQA_finetune.py --model-path /home/msm97/scratch/model_checkpoints/ins_tuning/model-checkpoint-epoch=2-104000