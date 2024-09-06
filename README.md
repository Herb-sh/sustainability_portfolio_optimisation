# sustainability_portfolio_optimisation


# Best practice, use an environment rather than install in the base env
conda create -n portfolio_optimisation
conda activate portfolio_optimisation
# If you want to install from conda-forge
conda config --env --add channels conda-forge
# The actual install command
conda install numpy