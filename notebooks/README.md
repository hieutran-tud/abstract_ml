# Notebooks

The notebooks are executable demonstrations of the core library. Install the optional notebook dependencies from the repository root:

```bash
python -m pip install -e ".[notebooks]"
cd notebooks
jupyter lab
```

## Demonstrations

- `experiment.ipynb` — synthetic regression and classification.
- `experiment_gan.ipynb` — a two-dimensional GAN experiment.
- `gan_mnist.ipynb` and `gan_mnist_1.ipynb` — MNIST GAN variants.
- `mnist_classification.ipynb` — MNIST classification.

The MNIST notebooks expect the four uncompressed IDX files under `data/mnist/` at the repository root. The files are ignored by Git; download them separately before running those notebooks.

Training cells can be CPU-intensive. The short synthetic notebook is the recommended smoke test.
