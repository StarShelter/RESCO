sed -i '$d' resco.slurm
echo "python main.py @cologne1 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @cologne1 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @ingolstadt1 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @ingolstadt1 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @cologne3 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @cologne3 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @ingolstadt7 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @ingolstadt7 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @cologne8 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @cologne8 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @ingolstadt21 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @ingolstadt21 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @cologne1 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @cologne1 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @ingolstadt1 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @ingolstadt1 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @cologne3 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @cologne3 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @ingolstadt7 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @ingolstadt7 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @cologne8 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @cologne8 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @ingolstadt21 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @ingolstadt21 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @cologne1 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @cologne1 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @ingolstadt1 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @ingolstadt1 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @cologne3 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @cologne3 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @ingolstadt7 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @ingolstadt7 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @cologne8 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @cologne8 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @ingolstadt21 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @ingolstadt21 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @cologne1 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @cologne1 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @ingolstadt1 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @ingolstadt1 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @cologne3 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @cologne3 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @ingolstadt7 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @ingolstadt7 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @cologne8 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @cologne8 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @ingolstadt21 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @ingolstadt21 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @cologne1 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @cologne1 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @ingolstadt1 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @ingolstadt1 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @cologne3 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @cologne3 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @ingolstadt7 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @ingolstadt7 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @cologne8 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @cologne8 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
sed -i '$d' resco.slurm
echo "python main.py @ingolstadt21 @IPPO episodes:1400 home_log:True;mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID" | tee -a resco.slurm > log.txt
echo python main.py @ingolstadt21 @IPPO episodes:1400
sbatch ./resco.slurm
sleep 10
