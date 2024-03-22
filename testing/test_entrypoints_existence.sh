#! /bin/bash

# Array of commands to run
commands=(
  # main
  "gandlf --version"
  # subcommands
#  "gandlf anonymizer --help"
#  "gandlf collect-stats --help"
#  "gandlf config-generator --help"
  "gandlf construct-csv --help"
#  "gandlf debug-info --help"
#  "gandlf deploy --help"
#  "gandlf generate-metrics --help"
#  "gandlf optimize-model --help"
#  "gandlf patch-miner --help"
#  "gandlf preprocess --help"
#  "gandlf recover-config --help"
#  "gandlf run --help"
#  "gandlf update-version --help"
#  "gandlf verify-install --help"
  # old entrypoints
  "gandlf_anonymizer --help"
  "gandlf_collectStats --help"
  "gandlf_configGenerator --help"
  "gandlf_constructCSV --help"
  "gandlf_debugInfo --help"
  "gandlf_deploy --help"
  "gandlf_generateMetrics --help"
  "gandlf_optimizeModel --help"
  "gandlf_patchMiner --help"
  "gandlf_preprocess --help"
  "gandlf_recoverConfig --help"
  "gandlf_run --help"
  "gandlf_updateVersion --help"
  "gandlf_verifyInstall --help"
)

for cmd in "${commands[@]}"; do
  echo "Running '$cmd'..."
  output=$($cmd 2>&1)
  status=$?
  # Suppressing stdout in successful case, only showing command being run
  if [ $status -ne 0 ]; then
    echo "Command '$cmd' failed with the following output:"
    echo "$output"
    exit $status
  fi
done

echo "All entrypoints succeeded."
