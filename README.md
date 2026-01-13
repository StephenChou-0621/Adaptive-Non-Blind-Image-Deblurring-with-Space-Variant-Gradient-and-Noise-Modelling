# Implementation of Adaptive Non-Blind Image Deblurring with Space-Variant Gradient and Noise Modelling

## Environment

This implementation is based on **traditional (non-learning) image processing algorithms**,  
and does **not require GPU or deep learning frameworks**.

### Tested Environment
- OS: Linux
- Python: 3.10.19

### Python Dependencies
The following Python packages are required:
- numpy
- scipy
- opencv-python
- scikit-image
- matplotlib
- PyWavelets

### Setup
```bash
conda create -n deblur python=3.10.19 -y
conda activate deblur
pip install -r requirements.txt
```

## How to run
1. To test different blur kernels, run:
```bash
bash RUN.sh
```

2. To test different noise standard deviations, set the parameter `std_noise_set` in `main.py`, and then run:
```bash
bash RUN.sh
```

3. To test different noise exponents, set the parameter `alpha_n_set` in `main.py`, and then run
```bash
bash RUN.sh
```

## How to test your own data
### Data Preparation
- Put your blur kernels (`*.png`) into:
`./data/kernels/`
- Put the blurred images (`*.png`) into:
`./data/blur/kernel*/`
according to which kernel is applied.
- Put the sharp (ground truth) images (`*.png`) into:
`./data/sharp/`

### Notes
- The number of images in `./data/blur/kernel*/` and `./data/sharp/` must be identical.
- The directories under `./data/blur/` should be named as:
`kernel1/`, `kernel2/`, `kernel3/`, ...

## Results

All experimental results (i.e. deblurring performance under different noise exponents, kernels, and noise standard deviations) are saved in:
`./all_results/`

### Directory Organization

- Noise exponents:
`alpha03/`, `alpha05/`, `alpha07/`

- Blur kernels:
`k1/`, `k2/`, ..., `k8/`

- Noise standard deviations:
 `std001/`, `std002/`, `std003/`
 
### Evaluation

You can evaluate the restored images by running:

```bash
python eval.py
```
The SSIM and PSNR values will be saved in:
`output.csv`

