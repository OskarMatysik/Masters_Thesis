#!/bin/bash

#SBATCH -c20
#SBATCH --mem=45gb
#SBATCH --time=96:00:00
#SBATCH --mail-user=268748@student.pwr.edu.pl
#SBATCH --job-name=Deffuant_Weisbuch_Calibration_ML

d="$1"
mu="$2"
N="$3"
topology="$4"

cd Masters_Thesis
uv run main.py --d "$d" --mu "$mu" --N "$N" --topology "$topology"