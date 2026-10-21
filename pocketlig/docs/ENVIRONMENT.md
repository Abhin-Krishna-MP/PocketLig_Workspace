# Environment
1. Pick ONE CUDA version that both your PyTorch build and the TargetDiff repo support (read its environment.yaml / README).
2. Install torch from https://pytorch.org/get-started/previous-versions/ for that CUDA, then PyTorch Geometric and extensions
   from https://data.pyg.org/whl/ for that exact torch+CUDA pair.
3. `conda env create -f environment.yml` for mmseqs2, apbs, openbabel and the pip packages.
4. Run `python -m pytest -q`. All tests must pass before real data is touched.
5. Record every version in docs/VERSIONS.md (`pip freeze`, `nvidia-smi`, `python --version`).
TargetDiff has its own environment (torch_scatter etc.); run its sampler in a separate env if versions conflict.
