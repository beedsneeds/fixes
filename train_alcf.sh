#!/bin/bash --login
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:15:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -M bkodanda@stevens.edu
#PBS -j oe
#PBS -o /eagle/lc-mpi/bharadhwaj/Megatron-DeepSpeed/myscripts/logs/
#PBS -A lc-mpi

cd /eagle/lc-mpi/bharadhwaj/Megatron-DeepSpeed
export PBS_O_WORKDIR="$(realpath .)" 
module use /soft/modulefiles
module load conda/2025-09-25
conda activate base
source ${PBS_O_WORKDIR}/venvs/polaris/2025-09-25/bin/activate

chmod +x pretrain_gpt_alcf.py

HERE=$(python3 -c 'import os; print(os.getcwd())') && export HERE
GIT_BRANCH=$(git branch --show-current) && export GIT_BRANCH

# shellcheck disable=SC1090
source ${PBS_O_WORKDIR}/deps/ezpz/src/ezpz/bin/utils.sh || exit
ezpz_setup_env

if  command -v "ezpz-test"; then
    log_message INFO "${GREEN}✓${RESET} ezpz is already installed."
    # printf "[!! %s] ezpz is already installed.\n" "$(printGreen "INFO")"
else
    log_message WARNING "${RED}✗${RESET} ezpz is not installed."
    log_message INFO "Installing ezpz..."
    python3 -m pip install "git+https://github.com/saforem2/ezpz" || exit
fi

export TRAIN_ITERS=10              # Very few iterations for debugging


#####################################
# AuroraGPT-7B
#
# Main production script for training
# AuroraGPT-7B @ ALCF
#####################################
train_aGPT() {

    # 1. Navigate into `$PBS_O_WORKDIR`
    # cd "${PBS_O_WORKDIR}" || exit
    HERE=$(python3 -c 'import os; print(os.getcwd())') && export HERE
    GIT_BRANCH=$(git branch --show-current) && export GIT_BRANCH

    # 2. source `ALCF/helpers.sh` for Megatron-DeepSpeed setup
    source "${HERE}/ALCF/helpers.sh" || {
      log_message ERROR "Unable to source ALCF/helpers.sh."
      log_message ERROR "Please ensure you are in the correct directory."
      return 1
    }

    # 3. call `setup` from `./ALCF/helpers.sh`
    setup "$@" || exit
    # export run_cmd="${run_cmd}"
    echo "${run_cmd[@]}" | tee -a "${OUTPUT_LOG}"

    # 4. Tell user where to find output
    printf "Output will be saved to %s\n" "${OUTPUT_LOG}" | tee -a "${OUTPUT_LOG}"
    # printf "[!! %s] View output at:\n %s\n" "$(printBlue "NOTE")" "$(printYellow "${OUTPUT_LOG}")" | tee -a "${OUTPUT_LOG}"

    # 5. Evaluate ${run_cmd} and append outputs to ${OUTPUT_LOG}
    # eval "${run_cmd[*]}" |& tee -a "${OUTPUT_LOG}"
    if [[ "${DEBUG:-}" ]]; then
        set -x
        bash -c "${run_cmd[*]}" |& tee -a "${OUTPUT_LOG}"
        set +x
    else
        bash -c "${run_cmd[*]}" |& tee -a "${OUTPUT_LOG}"
    fi
}


# Kill any existing MPI processes
ezpz_kill_mpi || exit

train_aGPT "$@"
