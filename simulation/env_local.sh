# Local adaptation: conda env `policyik` + standalone Isaac Sim 5.1.0 in $HOME (instead of pip isaacsim).
# usage: source env_local.sh   (from PolicyIK/simulation)
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate policyik
export ISAACSIM_PATH="$HOME/isaac-sim-standalone-5.1.0-linux-x86_64"
source "$ISAACSIM_PATH/setup_conda_env.sh"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
