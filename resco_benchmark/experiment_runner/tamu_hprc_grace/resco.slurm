#!/bin/bash
#SBATCH --partition=medium
#SBATCH --export=NONE

#SBATCH --time=24:00:00
#SBATCH --job-name=RESCO
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=7500M
#SBATCH --output=slurm_logs/RESCO.%j

#SBATCH --account=132824382529

#First Executable Line
echo "Retrieving RESCO: $(date)"
rsync -aq ~/RESCO.7z $TMPDIR
cd $TMPDIR
ml Miniconda3/24.11.1 GCCcore/12.3.0 p7zip/17.04
7za x $TMPDIR/RESCO.7z
echo "Retrieving env: $(date)"
rsync -aq $SCRATCH/resco.tar.gz $TMPDIR
echo "Extracting env: $(date)"
mkdir -p $TMPDIR/conda/resco
tar -xzf $TMPDIR/resco.tar.gz -C $TMPDIR/conda/resco
echo "Loading env: $(date)"
source $TMPDIR/conda/resco/bin/activate
echo "Cleaning env: $(date)"
conda-unpack
rm $TMPDIR/RESCO.7z
rm $TMPDIR/resco.tar.gz
mkdir -p $SCRATCH/results/SLURM$SLURM_JOBID
export PYTHONPATH="$TMPDIR/COOM:$PYTHONPATH"
export PYTHONPATH="$TMPDIR/RESCO:$PYTHONPATH"
cd $TMPDIR/RESCO/resco_benchmark/
echo "Starting RESCO: $(date)"
echo "Use shell script"