#!/bin/bash

# Define the range for the loop
start_radius=0
end_radius=200
radius_bin=10

# Job script file
job_script="run_gpu.bsub"

# Loop through the years
for (( radius=$start_radius; radius<=$end_radius; radius+=$radius_bin )); do
    # Update beginyears_endyears and Estimation_years dynamically
    Buffer_size="[$radius]"

    # Create a temporary modified script
    modified_script="modified_job_script_${Buffer_size}.bsub"
    cp $job_script $modified_script

    # Use sed to replace variables in the script
    sed -i "s/^Buffer_size=.*/Buffer_size=${Buffer_size}/" $modified_script
    sed -i "s/^#BSUB -J .*/#BSUB -J \"V6.02.03 Annual Model BLISCO ${radius}\"/" $modified_script

    # Update the pause_time calculation
    sed -i "s/^pause_time=\$((RANDOM % 30 .*/pause_time=\$((RANDOM % 30 + (${radius} - ${start_radius}) * 12))/" $modified_script

    # Submit the modified script using bsub
    echo "Submitting job for radius $radius..."
    bsub < $modified_script

    # Optional: Clean up temporary script after submission
    

    # Pause for 90 seconds before the next submission
    echo "Waiting for 10 seconds before the next job..."
    sleep 10

    rm $modified_script
done