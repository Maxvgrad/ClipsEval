#!/bin/bash -l

#SBATCH -A bolt
#SBATCH --job-name="bolt-internship-convert-results"
#SBATCH --time=1:00:00
#SBATCH --partition=main
#SBATCH --output=not_tracked_dir/slurm/%j_slurm_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G

#----------------------------------------
# Parse Arguments
#----------------------------------------
wandb_id=None
resume_optim=False
resume=""  # No resume
timestamp=$(date +%Y-%m-%d_%H-%M-%S)
output_dir="not_tracked_dir/output_${model_name}_${timestamp}"

annotation_file=""
result_files=()

while getopts "a:r:" opt; do
  case ${opt} in
    a )
      annotation_file=$OPTARG
      ;;
    r )
      result_files+=("$OPTARG")  # Collect result files in an array
      ;;
    \? )
      echo "Usage: $0 -a <annotation_file> -r <result_file1> [-r <result_file2> ...]"
      exit 1
      ;;
  esac
done

# Check that required arguments are provided
if [[ -z "$annotation_file" || ${#result_files[@]} -eq 0 ]]; then
  echo "Error: Both annotation_file and at least one result_file are required."
  echo "Usage: $0 -a <annotation_file> -r <result_file1> [-r <result_file2> ...]"
  exit 1
fi

#----------------------------------------
# Environment Setup
#----------------------------------------
module load miniconda3
conda activate clips_eval

#----------------------------------------
# Debug Information
#----------------------------------------

echo "output_dir=$output_dir"
echo "resume=$resume"
echo "resume_optim=$resume_optim"
echo "wandb_id=$wandb_id"


for result_file in "${result_files[@]}"; do
  # Construct output file path by prefixing "output_" and changing extension to .json
  output_file="${output_dir}/output_$(basename "${result_file%.*}").json"

  echo "Processing result file: $result_file"
  echo "Output will be saved to: $output_file"

  # Run the evaluation script
  python eval.py \
    --annotation_file "$annotation_file" \
    --result_file "$result_file" \
    --output_file "$output_file"
done