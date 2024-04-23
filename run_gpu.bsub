#!/bin/bash
#BSUB -q general
#BSUB -R 'gpuhost rusage[mem=300GB] span[hosts=1]'
#BSUB -G compute-rvmartin
#BSUB -gpu "num=1:gmodel=NVIDIAA100_SXM4_80GB"
#BSUB -a 'docker(syword/python3-pytorch:2023.12)'
#BSUB -J "_Optimal_model v2.0.0 2001-2019 _OldVersion-Geo-GC-OldMethods"
#BSUB -m 'compute1-exec-393.ris.wustl.edu compute1-exec-395.ris.wustl.edu compute1-exec-396.ris.wustl.edu compute1-exec-399.ris.wustl.edu compute1-exec-400.ris.wustl.edu compute1-exec-401.ris.wustl.edu'
#BSUB -g /s.siyuan/Test
#BSUB -N
#BSUB -u s.siyuan@wustl.edu
#BSUB -o job_output/job-%J-output.txt

/bin/true

cd /my-projects/Projects/MLCNN_PM25_2021/code/Training_Testing_Evaluation/v2.0.0/
python3 main_1.py

