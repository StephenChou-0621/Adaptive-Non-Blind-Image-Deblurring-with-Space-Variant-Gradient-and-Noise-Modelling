## Implementation of "Adaptive Non-Blind Image Deblurring with Space-Variant Gradient and Noise Modelling"

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


# How to run
1. To test different blur kernels, type "bash RUN.sh".
2. To test different noise stds, set the parameter "std_noise_set" in main.py, and then type "bash RUN.sh".
3. To test different noise exponents, set the parameter "alpha_n_set" in main.py, and then type "bash RUN.sh".

# How to test your own data
put your blur kernels (*.png) into "./data/kernels/" directory, and the blurred images (*.png) into "./data/blur/kernel*" directory according to which kernel is applied to the blurred image, and sharp images (*.png) into "./data/sharp" directory as ground truth image.
Note that the numbers of images in "./data/blur/kernel*" and "data/sharp" should be identical.
Note that "./data/blur/kernel*" should be named as "./data/blur/kernel1", "./data/blur/kernel2", and so on.

# results
Our experiment results (i.e. deblurring performance under different noise exponents, kernels, and noise stds) are saved in "./all_results" directory. There are three for different noise exponents, named as "alpha03", "alpha05", "alpha07"; eight for different kernels, named as "k1" to "k8"; and three for different stds, named as "std001", "std002", and "std003".
You can evaluate the qualities of the restored images in "./all_results/" by typing "python eval.py", and the ssim and psnr values will be saved in "output.csv".

