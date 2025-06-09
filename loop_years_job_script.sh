#!/bin/bash


# Define the range for the loop
# start_year=1998
# end_year=2010

# Loop through the years
# for year in $(seq $start_year $end_year); do

# Define the list of years explicitly
years=(2000 2001 2004 2005 2006 2007 2012 2013 2015 2017 2019)

system_name='slurm'  # Change this to 'slurm' if using SLURM
# system_name='slurm'

# Job script file
job_script="run_gpu.${system_name}"

# Loop through the specified years
for year in "${years[@]}"; do
    # Update beginyears_endyears and Estimation_years dynamically
    beginyears_endyears="[$year]"
    Estimation_years="[[$year]]"

    # Create a temporary modified script
    modified_script="modified_job_script_${year}.${system_name}"
    cp $job_script $modified_script

    # Use sed to replace variables in the script
    sed -i "s/^beginyears_endyears=.*/beginyears_endyears=${beginyears_endyears}/" $modified_script
    sed -i "s/^Estimation_years=.*/Estimation_years=${Estimation_years}/" $modified_script
    if [ "$system_name" == "slurm" ]; then
        sed -i "s/^#SBATCH --job-name=.*/#SBATCH --job-name=\"V6.02.03a data ${year}\"/" $modified_script
    else
        sed -i "s/^#BSUB -J .*/#BSUB -J \"V6.02.03 Annual data ${year}\"/" $modified_script
    fi
    #  sed -i "s/^#SBATCH --job-name=.*/#SBATCH --job-name=\"V6.02.03a data ${year}\"/" $modified_script
    # sed -i "s/^#BSUB -J .*/#BSUB -J \"V6.02.03 Annual data ${year}\"/" $modified_script

    # Update the pause_time calculation
    sed -i "s/^pause_time=\$((RANDOM % 30 .*/pause_time=\$((RANDOM % 30 + (${year} - ${start_year}) * 180))/" $modified_script

    # Submit the modified script using bsub
    echo "Submitting job for year $year..."
    #sbatch $modified_script
    if [ "$system_name" == "slurm" ]; then
        sbatch $modified_script
    else
        bsub < $modified_script
    fi
    

    # Optional: Clean up temporary script after submission
    

    # Pause for 90 seconds before the next submission
    # echo "Waiting for 10 seconds before the next job..."
    # sleep 1

    rm $modified_script
done
