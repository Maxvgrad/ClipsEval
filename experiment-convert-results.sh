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

echo "annotation_file=$annotation_file"
echo "result_files=$result_files"
echo "output_file=$output_file"

for result_file in "${result_files[@]}"; do
  result_file_dir="$(dirname "$result_file")"

  # Construct output file path by prefixing "output_" and changing extension to .json
  output_file="${result_file_dir}/output_$(basename "${result_file%.*}").json"

  echo "Processing result file: $result_file"
  echo "Output will be saved to: $output_file"

  # Run the evaluation script
  python eval.py \
    --annotation_file "$annotation_file" \
    --result_file "$result_file" \
    --output_file "$output_file"

done