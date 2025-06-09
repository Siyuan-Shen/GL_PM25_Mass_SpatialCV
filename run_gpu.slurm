#!/bin/bash
#SBATCH --job-name="_Lambda1-12800_Lambda2-25d6_padding_replicate_layer0_311_avgpool_ELU Estimation"
#SBATCH --partition=general
#SBATCH --mem=600G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --exclude=c2-gpu-[002,003,009,014,011,015]
#SBATCH --output=job_output/job-%j-output.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s.siyuan@wustl.edu
#SBATCH --account=compute-rvmartin
#SBATCH -vvv


# If you're using containers with Slurm
# Uncomment this line if needed:
#SBATCH --container-image=syword/python3-pytorch:2024.05
#SBATCH --container-mounts=/storage1/fs1/rvmartin/Active/s.siyuan:/my-projects,/storage1/fs1/rvmartin2/Active/s.siyuan:/my-projects2
#SBATCH --container-workdir=/my-projects2/Projects/MLCNN_PM25_2021/code/Training_Evaluation_Estimation/PM25/V6.02.03-test

# Optional: mimic host exclusion if needed via constraints or scheduler filters

# Run your code
cd /my-projects2/Projects/MLCNN_PM25_2021/code/Training_Evaluation_Estimation/PM25/V6.02.03-test
# Random pause

pause_time=$((RANDOM % 30 + 100))
echo "Pausing for $pause_time seconds..."
sleep $pause_time

echo "Starting the job at $(date)"

start_time=$(date +%s)
# Buffer_size=[80] 
# sed -i '/\[BLCO\-CrossValidation\]/,/^\[/{/Buffer_size/s/=.*/= '"$Buffer_size"'/}' config.toml
beginyears_endyears=[2021]
Estimation_years=[[2021]]
sed -i '/\[Estimation\-Settings\.Training_Settings\]/,/^\[/{/beginyears/s/=.*/= '"$beginyears_endyears"'/;/endyears/s/=.*/= '"$beginyears_endyears"'/;}' config.toml
sed -i '/\[Estimation\-Settings\.Map_Estimation_Settings\]/,/^\[/{/Estiamtion_trained_beginyears/s/=.*/= '"$beginyears_endyears"'/}' config.toml
sed -i '/\[Estimation\-Settings\.Map_Estimation_Settings\]/,/^\[/{/Estiamtion_trained_endyears/s/=.*/= '"$beginyears_endyears"'/}' config.toml
sed -i '/\[Estimation\-Settings\.Map_Estimation_Settings\]/,/^\[/{/Estimation_years/s/=.*/= '"$Estimation_years"'/}' config.toml
sed -i '/\[Estimation\-Settings\.Visualization_Settings\]/,/^\[/{/Map_Plot_YEARS/s/=.*/= '"$beginyears_endyears"'/}' config.toml
sed -i '/\[OfficialData\-Settings\.Padding_fine_Global_Mapdata\]/,/^\[/{/Padding_fine_Global_Mapdata_Years/s/=.*/= '"$beginyears_endyears"'/}' config.toml
sed -i '/\[OfficialData\-Settings\.Crop_fine_Mapdata_regions\]/,/^\[/{/Crop_fine_Mapdata_regions_Years/s/=.*/= '"$beginyears_endyears"'/}' config.toml
sed -i '/\[OfficialData\-Settings\.Convert_fine2coarse_Mapdata\]/,/^\[/{/Convert_fine2coarse_Mapdata_Years/s/=.*/= '"$beginyears_endyears"'/}' config.toml

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Configuration updated. Elapsed time for configuration update: $elapsed_time seconds."

cd /my-projects2/Projects/MLCNN_PM25_2021/code/Training_Evaluation_Estimation/PM25/V6.02.03-test

echo "Entered the directory ......."

python3 main.py

