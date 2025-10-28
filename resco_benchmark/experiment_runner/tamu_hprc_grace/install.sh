ml Miniconda3/24.11.1 GCCcore/13.3.0 git/2.45.1
git clone git@github.com:jault/RESCO.git
git clone git@github.com:hyintell/COOM.git
conda create --yes -p "$SCRATCH/conda/resco" python==3.10
source activate "$SCRATCH/conda/resco"
python -m pip install --yes ~/RESCO/.[cl]
python -m pip install --yes ~/COOM/.
conda clean --all --yes
python -m pip uninstall --yes resco_benchmark
python -m pip uninstall --yes COOM
ml Miniconda3/24.11.1 GCCcore/12.3.0 p7zip/17.04
cp ~/RESCO/resco_benchmark/experiment_runner/tamu_hprc_grace/* ~/
rm -rf ~/RESCO/.git ~/COOM/.git ~/COOM/assets
7za a -mx=9 RESCO.7z RESCO COOM
rm -rf ~/RESCO ~/COOM
python -m pip install conda-pack
conda pack -p "$SCRATCH/conda/resco"