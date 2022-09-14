#!/bin/bash
#SBATCH -J Diffusion # A single job name for the array
#SBATCH -c 1 # Number of cores
#SBATCH -p shared # Partition
#SBATCH --mem 4000 # Memory request (6Gb)
#SBATCH -t 0-8:00 # Maximum execution time (D-HH:MM)
#SBATCH -o Diff_%A_%a.out # Standard output
#SBATCH -e Diff_%A_%a.err # Standard error

start=`date +%s`

# Set the configurable variables
DIFF_PATH=/n/home10/knavarro/packages/Diffusion/
INPUT_FOLDER=$SCRATCH/guenette_lab/Users/$USER/Diff
jobid=${SLURM_ARRAY_TASK_ID}
#JOBNAME="PrepTraining"
JOBNAME="PrepDataPrepTraining"
TYPE="Data"

# Create the directory
cd $SCRATCH/guenette_lab/Users/$USER/
mkdir -p $JOBNAME/$TYPE/jobid_"${jobid}"
cd $JOBNAME/$TYPE/jobid_"${jobid}"

# Setup Python and run
echo "Python" 2>&1 | tee -a log_nexus_"${jobid}".txt
source $DIFF_PATH/setup.sh

# Merge
echo "Merge all.h5" 2>&1 | tee -a log_nexus_"${jobid}".txt
python $DIFF_PATH/Merge.py $INPUT_FOLDER/$TYPE/* 2>&1 | tee -a log_nexus_"${jobid}".txt

# Filter
echo "Running Filter Stage" 2>&1 | tee -a log_nexus_"${jobid}".txt
python $DIFF_PATH/PrepData.py 2>&1 | tee -a log_nexus_"${jobid}".txt

# Prep Training
echo "Running Prep Training" 2>&1 | tee -a log_nexus_"${jobid}".txt
python $DIFF_PATH/PrepTraining.py 2>&1 | tee -a log_nexus_"${jobid}".txt


# # Plane XY
# echo "Running Training XY" 2>&1 | tee -a log_nexus_"${jobid}".txt
# python $DIFF_PATH/Training_XY.py 2>&1 | tee -a log_nexus_"${jobid}".txt

# # Plane XZ
# echo "Running Training XZ" 2>&1 | tee -a log_nexus_"${jobid}".txt
# python $DIFF_PATH/Training_XZ.py 2>&1 | tee -a log_nexus_"${jobid}".txt

# # Plane YZ
# echo "Running Training YZ" 2>&1 | tee -a log_nexus_"${jobid}".txt
# python $DIFF_PATH/Training_YZ.py 2>&1 | tee -a log_nexus_"${jobid}".txt




echo; echo; echo;

echo "FINISHED....EXITING" 2>&1 | tee -a log_nexus_"${jobid}".txt

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds | tee -a log_nexus_"${jobid}".txt