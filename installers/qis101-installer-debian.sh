cd $HOME
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
conda deactivate
conda update -n base conda -y
conda create -y -n qis101 python=3.12
conda activate qis101
python.exe -m pip install --upgrade pip
pip install numba matplotlib pyqt6 pygame
pip install sympy scipy scikit-learn pandas jupyter ipympl tqdm
pip install networkx pulp numexpr
pip install selenium webdriver-manager beautifulsoup4
pip install mayavi configobj vtk==9.4
pip install 'jax[cpu]'
pip install 'qiskit[all]' qiskit-aer qiskit-algorithms
pip install qiskit-ibm-runtime qiskit-ibm-catalog qiskit-experiments
pip install qiskit-dynamics qiskit-finance qiskit-nature
pip install qiskit-machine-learning qiskit-optimization
pip install numpy==2.2
pip install certifi>=2025.4.26
code --install-extension ms-vscode.cpptools
code --install-extension ms-vscode.cpptools-extension-pack
code --install-extension ms-vscode.powershell
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-toolsai.jupyter
code --install-extension ms-toolsai.jupyter-renderers
code --install-extension ms-python.isort
code --install-extension visualstudioexptteam.vscodeintellicode
code --install-extension visualstudioexptteam.intellicode-api-usage-examples
code --install-extension streetsidesoftware.code-spell-checker
code --install-extension ms-vscode.cmake-tools
code --install-extension davidanson.vscode-markdownlint
code --install-extension esbenp.prettier-vscode
code --install-extension mechatroner.rainbow-csv
code --install-extension emmanuelbeziat.vscode-great-icons
code --install-extension james-yu.latex-workshop
code --install-extension cschlosser.doxdocgen
code --install-extension redhat.vscode-yaml
code --install-extension charliermarsh.ruff
echo 'y' | jupyter lab --generate-config
echo 'c.ServerApp.use_redirect_file = False' >> $HOME/.jupyter/jupyter_lab_config.py

