#!/bin/bash

# Read each argument from the sessionargs.txt file
while IFS= read -r line; do
  # Extract the key and value from each line
  key=$(echo "$line" | cut -d ' ' -f 1)
  value=$(echo "$line" | cut -d ' ' -f 2-)
  
  # Assign the value to the corresponding variable
  case $key in
    mode)
      mode=$value
      ;;
    session_id)
      session_id=$value
      ;;
    node_id)
      node_id=$value
      ;;
    role)
      role=$value
      ;;
    coordinator_runtime)
      coordinator_runtime=$value
      ;;
    manager_runtime)
      manager_runtime=$value
      ;;
    follower_runtime)
      follower_runtime=$value
      ;;
    nfollowers)
      nfollowers=$value
      ;;
    model_dir)
      model_dir=$value
      ;;
    model)
      model=$value
      ;;
    nsamples)
      nsamples=$value
      ;;
    niters)
      niters=$value
      ;;
    proposal)
      proposal=$value
      ;;
    lkernel)
      lkernel=$value
      ;;
    recycling)
      recycling=$value
      ;;
    integrator)
      integrator=$value
      ;;
    step_size)
      step_size=$value
      ;;
    hmc_steps)
      hmc_steps=$value
      ;;
    seed)
      seed=$value
      ;;
    verbose)
      verbose=$value
      ;;
  esac
done < session_args.txt

# Extract CondorSMC
unzip condorsmc.zip -d condorsmc
rm condorsmc.zip
mkdir config
if [ "$role" = "follower" ]; then
    mv follower_config.cfg config/config.cfg
elif [ "$role" = "manager" ]; then
    mv manager_config.cfg config/config.cfg
else
    echo "Invalid role: $role"
fi

python3 -V
# Source Python virtual environment
source condorsmc/condorsmc/bin/condorsmc_env/bin/activate
python3 -V

echo
echo

ls condorsmc/
echo
echo

ls condorsmc/condorsmc

echo
echo


# Install CondorSMC
cd ./condorsmc
python3 -m pip install .
cd ../

# Run CondorSMC
python3 -m condorsmc \
    --session-id "$session_id" \
    --node-id "$node_id" \
    --mode "$mode" \
    --role "$role" \
    --coordinator-runtime "$coordinator_runtime" \
    --manager-runtime "$manager_runtime" \
    --follower-runtime "$follower_runtime" \
    --nfollowers "$nfollowers" \
    --model-dir "$model_dir" \
    --model "$model" \
    --nsamples "$nsamples" \
    --niters "$niters" \
    --proposal "$proposal" \
    --lkernel "$lkernel" \
    --recycling "$recycling" \
    --integrator "$integrator" \
    --step-size "$step_size" \
    --hmc-steps "$hmc_steps" \
    --seed "$seed" \
    --verbose
