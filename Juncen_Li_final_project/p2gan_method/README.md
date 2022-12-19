# P&sup2;-GAN Fast Style Transfer

## Using P100
- `srun --partition=gpu --nodes=1 --pty --gres=gpu:p100:1 --ntasks=1 --mem=4GB --time=04:00:00 /bin/bash`


## Requirement

numpy==1.18.1
opencv-python==4.1.0.25 -> pip3 install opencv-contrib-python==4.1.0.25
tensorflow==1.15.4


## Create and using TF1.15.0_env
- `module load anaconda3/2022.01`
- `module load cuda/10.0`
- `conda create --name TF1.15.0_env python=3.7`
- `source activate TF1.15.0_env`
- `export LD_LIBRARY_PATH=$HOME/.conda/envs/TF1.15.0_env/lib:$LD_LIBRARY_PATH`
- `conda install -c conda-forge cudatoolkit=10.0 cudnn=7.6 -y`
- `pip install tensorflow-gpu==1.15`

## Other instruction
conda uninstall cudnn
conda list cudnn

## Dependence

* opencv-python
* tensorflow 1.x

This project was implemented in tensorflow, and used `slim` API, which was removed in tensorflow 2.x, thus you need running it on tensorflow 1.x.

## Training

**Dataset**
It doesn't need a very large dataset, Pascal VOC Dataset is good enough. For a dataset like VOC2007, it contains about 10K images, training 2~3 epochs can gain a good result.

**Pre-trained VGG Model**
The VGG16 model we used is here: [tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg), you need to download the weights file before training.

**Training Command**
After you prepared the pre-trained VGG model and dataset, you can train a model by the command as follows:

```bash
python train.py --model model_path --style style_image_path --dataset dataset_path

```
* `--model`: path to save the trained model.
* `--style`: path to the style image.
* `--dataset`: path to dataset.

For example:

```bash
python train.py --model model_save/Van-Gogh-The-Starry-Night --style style/Van-Gogh-The-Starry-Night.jpg --dataset VOCdevkit/VOC2007/JPEGImages/ --lambda 5e-6
```
Further, we added an argument `--lambda` in this example, it's the hyper parameter between 1e-6~1e-7 to balance content and style.

**Training Control**
If you want to change the optimizer configuration, you need to edit `train.py`.

In each iteration while the model is training, `cfg.json` will be reloaded, thus some configuration can be set on training time:
* `epoch_lim`: how many training epochs will take.
* `preview`: allow to render preview images while training.
* `view_iter`: if `preview` valued `true`, render preview images at that iteration.


## Testing

Choose a model and run the command to test the model:

```bash
python render.py --model model_path --inp input_path --oup output_path --size number [--cpu true]
```
* `--model`: Choose a mode.
* `--inp`: Path to input images.
* `--onp`: Path to save synthetic images.
* `--size`: Processing image size.
* `--cpu`: Optional, set `true` will processing by CPU.

For example:

```bash
python render.py --model model_save/Van-Gogh-The-Starry-Night --inp content/ --oup output/Van-Gogh-The-Starry-Night --size 256
```

Note that `--inp` must be a directory, and `render.py` will process all images(extension with `jpg`, `bmp`, `png` and `jpeg`) under the directory. And the output directory must exist.





