rm ~/RESCO.7z
ml Miniconda3/24.11.1 GCCcore/13.3.0 git/2.45.1
git clone git@github.com:jault/RESCO.git ~/RESCO
git clone git@github.com:hyintell/COOM.git ~/COOM
cp ~/RESCO/resco_benchmark/experiment_runner/tamu_hprc_grace/resco.slurm ~/
ml Miniconda3/24.11.1 GCCcore/12.3.0 p7zip/17.04
rm -rf ~/RESCO/.git ~/COOM/.git ~/COOM/assets
7za a -mx=9 RESCO.7z RESCO COOM
rm -rf ~/RESCO ~/COOM