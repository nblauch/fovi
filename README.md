# `fovi`

Welcome to the `fovi` codebase, a PyTorch library for implementing foveated vision. This library provides tools for foveated sampling and an interface to deep vision models, including CNNs and ViTs. 


## 🛠️ Install

First, create a fresh conda environment:
```
   conda create -n fovi python=3.9
   conda activate fovi
```

To download pretrained models from our repo, you will need `git lfs`, which is used for "large file storage". Then clone the repo and enter it:
```
conda install git-lfs
git lfs install
git clone https://github.com/nblauch/fovi.git
cd fovi
```

Now, for installing our package. The easiest installation is without `ffcv`, as `ffcv` rquires Python 3.9 and other harder dependencies. Installing without it will allow you to use everything in our code-base except the training functionality that leverages `ffcv`. If you want training functionality with `ffcv`, see below. You could also use your own training scripts with our models. 

For the easy install, with your new environment activated, just do:
```
# from within the fovi repo
pip install -e . # this will automatically install fovi/requirements.txt
```

To install with `ffcv` to allow fast training, we first follow the instructions to install `ffcv-ssl`, which has stricter requirements, and then install `fovi` and its requirements. With your `fovi` conda environment activated, do:
```
conda install cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision torchaudio pytorch-cuda numba -c pytorch -c nvidia -c conda-forge
pip install git+https://github.com/facebookresearch/FFCV-SSL.git
# from within the fovi repo
pip install -e .
```

To use flash attention, install per the typical approach:
```
pip install packaging ninja
pip install flash-attn --no-build-isolation
```

## 📝 Example notebooks

`notebooks/step0_sensor_manifold` : explore the basic concepts involved in our foveated sensor

`notebooks/step1_sampling.ipynb` : learn how to do foveated sampling from images

`notebooks/step2_knnconv.ipynb` : learn how to build kNN-convolutional neural networks to process foveated sensor outputs

`notebooks/step3_dinov3.ipynb` : work with a state-of-the-art foveated vision system based on the DINOv3 ViT model, adapted to handle foveated inputs. 

`notebooks/step4_get_activations.ipynb`: use hooks to extract intermediate activations from a model, and explore the Trainer class

## 📚 Documentation

The docs are hosted at: https://nblauch.github.io/fovi_docs/index.html

We can automatically generate comprehensive documentation from the source code and docstrings. The documentation includes:

- **API Reference**: Complete documentation of all functions, classes, and modules
- **User Guide**: Installation, quickstart, and usage examples
- **Developer Guide**: Contributing guidelines and development setup

To do so:

```bash
# Install documentation dependencies
pip install -r requirements-docs.txt

# Generate documentation
python scripts/generate_docs.py

# View the documentation
open docs/_build/html/index.html

# View documentation on a remote cluster (need to forward the port separately, this is done automatically in VScode/Cursor)
python -m http.server 8000 --directory docs/_build/html
```

## 🏛️ Citation
Blauch, N.M., Alvarez, G.A., Konkle, T. (2025). FOVI: A biologically-inspired foveated interface for deep vision models. arXiv.
