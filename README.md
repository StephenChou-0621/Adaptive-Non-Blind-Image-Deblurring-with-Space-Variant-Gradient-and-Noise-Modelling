# Adaptive Non-Blind Image Deblurring with Space-Variant Gradient and Noise Modelling

This repository provides the **reference implementation** of the image deblurring method proposed in the manuscript:

**Adaptive Non-Blind Image Deblurring with Space-Variant Gradient and Noise Modelling**, submitted to *The Visual Computer*.

The proposed approach is based on **traditional (non-learning) image processing methods** and focuses on **non-blind motion deblurring** with space-variant gradient and noise distribution modelling. The implementation does **not require GPUs or deep learning frameworks**.

<!-- ## Permanent Link and DOI

This repository is publicly available on GitHub.

Upon acceptance of the manuscript, an archived release of this code will be created and
assigned a **permanent DOI via Zenodo**, ensuring long-term accessibility and
reproducibility of the reported results. -->

## Tested Environment
- OS: Linux
- Python: 3.10.19

## Python Dependencies
The following Python packages are required:
- numpy
- scipy
- opencv-python
- scikit-image
- matplotlib
- PyWavelets

## Setup
```bash
conda create -n deblur python=3.10.19 -y
conda activate deblur
pip install -r requirements.txt
```

## Dataset and Blur Kernels

All datasets and blur kernels used in the experiments are **publicly available benchmarks** that have been widely adopted in the image restoration literature.

### Datasets

- **Set12**  
  A commonly used benchmark consisting of 12 grayscale images for evaluating image restoration methods. The Set12 dataset has been extensively used in denoising and deblurring studies.

### Blur Kernels

- **Levin Motion Blur Kernels**  
  The motion blur kernels are taken from the benchmark introduced by Levin et al. in their seminal work on image deblurring. These kernels have been widely used as standard test cases in motion deblurring experiments.


## How to run

### 1. Test with Different Blur Kernels
    To evaluate the proposed method under different blur kernels, run:
    ```bash
    bash RUN.sh
    ```

### 2. Test with Different Noise Standard Deviations
    Set the parameter `std_noise_set` in `main.py` (e.g., 0.01, 0.02, ...) and run:
    ```bash
    bash RUN.sh
    ```

### 3. Test with Different Noise Exponents
    Set the parameter `alpha_n_set` in `main.py` (e.g., 0.1 to 0.9) and run:
    ```bash
    bash RUN.sh
    ```


## How to test your own data
### Data Preparation
- Place blur kernels (`*.png`) in:
`./data/kernels/`
- Place blurred images (`*.png`) in:
`./data/blur/kernel*/`
according to the applied kernel.
- Place ground-truth images (`*.png`) in:
`./data/sharp/`

### Notes
- The number of images in `./data/blur/kernel*/` and `./data/sharp/` must be identical.
- The directories under `./data/blur/` should be named as:
`kernel1/`, `kernel2/`, `kernel3/`, ...

## Results

All experimental results (i.e., deblurring performance under different noise exponents, blur kernels, and noise standard deviations) are saved in:
`./all_results/`

### Directory Organization

- Noise exponents:
`alpha03/`, `alpha05/`, `alpha07/`

- Blur kernels:
`k1/`, `k2/`, ..., `k8/`

- Noise standard deviations:
 `std001/`, `std002/`, `std003/`
 
### Evaluation

To evaluate the restored images, run:

```bash
python eval.py
```

The PSNR and SSIM values will be saved in:
`output.csv`


## Related Publication and Citation

This code accompanies the manuscript:  
**Adaptive Non-Blind Image Deblurring with Space-Variant Gradient and Noise Modelling**,  
submitted to The *Visual Computer*.

If you use this code or the associated dataset in your research, **please cite the following paper**:

```bibtex
@article{YourPaper2026Deblurring,
  title   = {Adaptive Non-Blind Image Deblurring with Space-Variant Gradient and Noise Modelling},
  author = {Chou, Shih-Hsuan and Chang, Je-Yuan and Liao, Chun-Lin and Tsai, Zse-Hong and Ding, Jian-Jiun},
  journal = {The Visual Computer},
  year    = {2026},
}
```

## License
This project is released under the MIT License for academic and research use.