# Adaptive Non-Blind Image Deblurring with Space-Variant Gradient and Noise Modelling

This repository provides the **reference implementation** of the image deblurring method proposed in the manuscript:

**Adaptive Non-Blind Image Deblurring with Space-Variant Gradient and Noise Modelling**,  
submitted to *The Visual Computer*.

The proposed approach is based on **traditional (non-learning) image processing methods** and focuses on **non-blind motion deblurring** with space-variant gradient and noise distribution modelling. The implementation does **not require GPUs or deep learning frameworks**.

---

## Permanent Link and DOI

This repository is publicly available on GitHub and provides full access to the source code required to reproduce all experiments reported in the manuscript.

An archived release of this repository will be created on Zenodo and assigned a permanent DOI upon acceptance of the manuscript, ensuring long-term accessibility and reproducibility.

Code DOI (Zenodo): to be assigned upon acceptance of the manuscript.

The archived version will correspond to the implementation used for the experiments reported in the final accepted manuscript.

---

## Tested Environment
- OS: Linux
- Python: 3.10.19

---

## Python Dependencies
The following Python packages are required:
- numpy
- scipy
- opencv-python
- scikit-image
- matplotlib
- PyWavelets

All dependencies are standard scientific Python libraries and can be installed via `pip`.

---

## Setup
```bash
conda create -n deblur python=3.10.19 -y
conda activate deblur
pip install -r requirements.txt
```

---

## Dataset and Blur Kernels
All datasets and blur kernels used in this work are **publicly available benchmark resources** that are widely adopted in the image restoration and deblurring literature.  
No proprietary, private, or restricted data are involved.

### Datasets

- **Set12**  
  A widely used benchmark dataset consisting of 12 grayscale images for evaluating image restoration methods. The Set12 dataset has been extensively used in denoising and deblurring studies.

### Blur Kernels

- **Levin Motion Blur Kernels**  
  A set of motion blur kernels introduced by Levin *et al.* in their seminal work on image deblurring.  
  These kernels are commonly used as standard test cases in non-blind motion deblurring experiments and enable fair comparison with existing methods.
  

---

## Code Structure and Key Components

- `main.py`  
  Entry point for all experiments.  

- `./src/`  
  Contains the core implementation of the proposed adaptive non-blind image deblurring algorithm.
  
The implementation follows the mathematical formulation described in Section 4 of the manuscript.

---

## How to Run Experiments
### 1. Evaluation with Different Blur Kernels
  To evaluate the proposed method under different blur kernels, run:
```bash
bash RUN.sh
```

### 2. Evaluation with Different Noise Standard Deviations
  Set the parameter `std_noise_set` in `main.py` (e.g., 0.01, 0.02, ...) and run:
```bash
bash RUN.sh
```

### 3. Evaluation with Different Noise Exponents
  Set the parameter `alpha_n_set` in `main.py` (e.g., 0.1 to 0.9) and run:
```bash
bash RUN.sh
```

---

## Testing on Custom Data
### Data Preparation
- Place blur kernels (`*.png`) in:  
`./data/kernels/`
- Place blurred images (`*.png`) in:  
`./data/blur/kernel*/`  
according to the applied kernel.
- Place ground-truth images (`*.png`) in:  
`./data/sharp/`

### Notes
- The number of blurred images in `./data/blur/kernel*/` and ground-truth images in `./data/sharp/` must be identical.
- Subdirectories under `./data/blur/` should be named as:  
`kernel1/`, `kernel2/`, `kernel3/`, ...

---

## Results
This repository distinguishes between **user-generated results** and the **reference experimental results reported in the manuscript**.

### User-Generated Results
When users run the code on their own data or custom experimental settings, the restored images are saved to:  
`./results/`

This directory is intended for user experiments and does not affect the reference results reported in the manuscript.

### Reference Results (Manuscript Results)
All experimental results reported in the manuscript, including restored images generated under different noise exponents, blur kernels, and noise standard deviations, are stored in:  
`./all_results/`

These results correspond to the experiments described in the paper and are used for quantitative evaluation.

### Directory Organization of `./all_results/`

- Noise exponents:  
`alpha03/`, `alpha05/`, `alpha07/`

- Blur kernels:  
`k1/`, `k2/`, ..., `k8/`

- Noise standard deviations:  
 `std001/`, `std002/`, `std003/`
 
### Evaluation
The evaluation script `eval.py` computes quantitative metrics (PSNR and SSIM) using the restored images stored in the `./all_results/` directory.  
To reproduce the quantitative results reported in the manuscript, run:

```bash
python eval.py
```

The evaluation results will be saved to:  
`output.csv`

---

## Related Publication and Citation

This code is directly related to the following manuscript:  
**Adaptive Non-Blind Image Deblurring with Space-Variant Gradient and Noise Modelling**,  
submitted to The *Visual Computer*.

If you use this code or the associated dataset in your research, **please cite the following paper**:

```bibtex
@unpublished{Chou2026AdaptiveDeblurring,
  title  = {Adaptive Non-Blind Image Deblurring with Space-Variant Gradient and Noise Modelling},
  author = {Chou, Shih-Hsuan and Chang, Je-Yuan and Liao, Chun-Lin and Tsai, Zse-Hong and Ding, Jian-Jiun},
  note   = {Manuscript submitted to The Visual Computer},
  year   = {2026}
}
```

## License
This project is released under the MIT License for academic and research use.