<p align="center"><img src="https://github.com/yuanze-lin/IllumiCraft/blob/main/assets/IllumiCraft.png" alt="icon" width="150" height="150" style="vertical-align:middle; margin-right:5px;" /></p>

# IllumiCraft: Unified Geometry and Illumination Diffusion for Controllable Video Generation (NeurIPS 2025) <br />

Official implementation of "IllumiCraft: Unified Geometry and Illumination Diffusion for Controllable Video Generation" 


[![PDF](https://img.shields.io/badge/PDF-Download-orange?style=flat-square&logo=adobeacrobatreader&logoColor=white)](https://arxiv.org/pdf/2506.03150)
[![arXiv](https://img.shields.io/badge/arXiv-2506.03150-b31b1b.svg)](https://arxiv.org/abs/2506.03150)
[![Project Page](https://img.shields.io/badge/Project%20Page-Visit%20Now-00d45c?style=flat-square&logo=googlechrome&logoColor=white)](https://yuanze-lin.me/IllumiCraft_page/)
[![YouTube Video](https://img.shields.io/badge/YouTube%20Video-FF0000?style=flat-square&logo=youtube&logoColor=white)](https://youtu.be/qAV58sADEzo)
[![Model](https://img.shields.io/badge/🤗%20Model-Download-yellow?style=flat-square)](https://huggingface.co/YuanzeLin/Illumicraft-checkpoints)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-Download-yellow?style=flat-square)](https://huggingface.co/datasets/YuanzeLin/IllumiCraft)

[Yuanze Lin](https://yuanze-lin.me/), [Yi-Wen Chen](https://wenz116.github.io/), [Yi-Hsuan Tsai](https://sites.google.com/site/yihsuantsai/), [Ronald Clark](https://www.ron-clark.com/), [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/)

## 💡 Method 

![image](https://github.com/yuanze-lin/IllumiCraft/blob/main/assets/framework.png)

## 📣 News
- [x] Release the training code.
- [x] Release IllumiCraft dataset.
- [x] Release the model and the inference code.
- [x] Set up the project page.

## ❤️ Support IllumiCraft

If you find this repository useful, please consider giving it a star ⭐.


## 🚀 Installation 
```bash
git clone https://github.com/yuanze-lin/IllumiCraft.git
cd IllumiCraft

conda create -n illumicraft python=3.10 -y
conda activate illumicraft
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118
conda env update -n illumicraft -f environment.yml
```

## 📂 Dataset Preparation

Download the IllumiCraft training dataset and demo examples:

```bash
python download_illumicraft_dataset.py
```

The script will automatically download the dataset from Hugging Face and organize it into two parts, `train` and `demo_examples`, for **training** and **inference**, respectively:

```text
dataset/
├── train/
└── demo_examples/
```

The training dataset will be stored in:

```text
dataset/train/
├── foreground_videos/
├── background_videos/
├── tracking_videos/
├── lighting_videos/
├── videos/
├── prompt.txt
├── videos.txt
├── foreground_videos.txt
├── background_videos.txt
├── tracking_videos.txt
└── lighting_videos.txt
```

Use `dataset/train/` as the `DATA_ROOT` in `train.sh` and `dataset/demo_examples/` as the `DATA_ROOT` in `inference.sh`.

## 📥 Download Pretrained Weights

Before running training and inference, download both the base **Wan2.1-Fun-1.3B-Control** model and the released **IllumiCraft** checkpoint:

```bash
python download_weights.py
```

The script will automatically download the checkpoints to:

```text
checkpoints/
├── Wan2.1-Fun-1.3B-Control/
└── illumicraft_weights/
```

After downloading, verify that the model paths in `inference.sh` are correctly configured:

```bash
WAN_MODEL_PATH="checkpoints/Wan2.1-Fun-1.3B-Control"
ILLUMICRAFT_CKPT_PATH="checkpoints/illumicraft_pretrained_weights"
```

`WAN_MODEL_PATH` points to the base Wan2.1 model and is shared by both ``train.sh`` and ``inference.sh``. `ILLUMICRAFT_CKPT_PATH` points to the pretrained IllumiCraft checkpoint used during inference.

## 🏋️ Training

Edit the following fields in `train.sh`:

```bash
DATA_ROOT=/path/to/train_dataset
WAN_MODEL_PATH=/path/to/Wan2.1-Fun-1.3B-Control

DATA_ROOT=/path/to/train
```

Launch training:

```bash
bash train.sh
```

## 🎥 Inference

Run video generation using a trained IllumiCraft checkpoint.

Edit the following fields in `inference.sh`:

```bash
WAN_MODEL_PATH=/path/to/Wan2.1-Fun-1.3B-Control
ILLUMICRAFT_CKPT_PATH=/path/to/illumicraft_pretrained_weights

DATA_ROOT=/path/to/demo_examples
```

Launch inference:

```bash
bash inference.sh
```

#### Outputs

By default, for each sample, IllumiCraft generates:

```text
sample_bg.mp4            # background-conditioned generation
sample_bg_concat.mp4     # foreground | background | generated

sample_nobg.mp4          # generation without background
sample_nobg_concat.mp4   # foreground | generated
```

`background.txt` and `light.txt` are paired line-by-line. If they are not provided, only the no-background generation is produced.

### 🎭 Foreground Video Generation

We provide `foreground_video_example.py` as a reference script for generating foreground videos from an RGB video and its corresponding mask video, where foreground pixels are `(255, 255, 255)` and background pixels are `(0, 0, 0)`.

> **Note:** `foreground_video_example.py` is provided only as a reference code snippet for generating foreground videos from an RGB video and its corresponding binary mask video.
> 
## 🎬 Sample Results
<img width="600" align="left" alt="image" src="https://github.com/user-attachments/assets/aeb594c5-c32b-4ffa-bcda-0723e7612187" />

![image](https://github.com/yuanze-lin/IllumiCraft/blob/main/examples/14.gif)
<img width="600" align="left" alt="image" src="https://github.com/user-attachments/assets/372d8fec-db53-4c35-b668-76055472e96b"/>

![image](https://github.com/yuanze-lin/IllumiCraft/blob/main/examples/2.gif)
<img width="600" align="left" alt="image" src="https://github.com/user-attachments/assets/8dfc7346-b322-48f5-82b3-fafaad513edd" />

![image](https://github.com/yuanze-lin/IllumiCraft/blob/main/examples/4.gif)
<img width="600" align="left" alt="image" src="https://github.com/user-attachments/assets/a8e9f972-9b6d-4423-a90d-c7a88687d2dd" />

![image](https://github.com/yuanze-lin/IllumiCraft/blob/main/examples/5.gif)
<img width="600" align="left" alt="image" src="https://github.com/user-attachments/assets/8c145d55-2b70-4582-8620-52bdfcec3c60" />

![image](https://github.com/yuanze-lin/IllumiCraft/blob/main/examples/7.gif)
<img width="600" align="left" alt="image" src="https://github.com/user-attachments/assets/96c9fa52-2ed6-4658-99d7-13f8d55b040b" />

![image](https://github.com/yuanze-lin/IllumiCraft/blob/main/examples/8.gif)
<img width="600" align="left" alt="image" src="https://github.com/user-attachments/assets/0d9e9040-0fa1-4412-a21b-7de593c7cf60" />

![image](https://github.com/yuanze-lin/IllumiCraft/blob/main/examples/10.gif)
<img width="600" align="left" alt="image" src="https://github.com/user-attachments/assets/6500634d-e8a6-40d2-a090-38f40b014546" />

![image](https://github.com/yuanze-lin/IllumiCraft/blob/main/examples/11.gif)

<!-- <img width="600" align="left" alt="image" src="https://github.com/user-attachments/assets/45fcec9b-ec34-40a5-8809-e261c79e48a1"/>

![image](https://github.com/yuanze-lin/IllumiCraft/blob/main/examples/1.gif)
<img width="600" align="left" alt="image" src="https://github.com/user-attachments/assets/372d8fec-db53-4c35-b668-76055472e96b"/>

![image](https://github.com/yuanze-lin/IllumiCraft/blob/main/examples/3.gif) -->

## ❓ FAQ

#### Q: Why do you use background videos during training but background images during inference?

During training, we only use the **first frame** of each background video. Therefore, a background image is sufficient during inference. If you have a background image, you can simply repeat it to create a background video with the same length as the input foreground video.

We originally used background videos in the dataset for training because we also explored background-video-conditioned video generation.

#### Q: Why does inference use both `foreground_prompt.txt` and `lighting_prompt.txt`?

##### 🏋️ Training

- `prompt.txt` describes the **entire video**, including both foreground and background content.

##### 🎥 Inference

- `foreground_prompt.txt` describes the **foreground object and its appearance**.
- `lighting_prompt.txt` describes the **background scene and lighting conditions** associated with the selected background image.

Since the background images used during inference are independently collected and can be freely replaced with custom images, they are not paired with the foreground videos. Therefore, `lighting_prompt.txt` is used to provide scene and illumination information that is not contained in `foreground_prompt.txt`.

> **Note:**
> For paired data (e.g., formal evaluation), where the foreground, background, caption, and ground-truth video correspond to the same scene, a single caption describing the entire scene can be stored in `prompt.txt`.
>
> For arbitrary background image customization, we recommend using `foreground_prompt.txt` to describe the foreground and `lighting_prompt.txt` to describe the background scene and lighting conditions.

## 📚 Citation

If you find IllumiCraft useful for your research, please consider citing:

```bibtex
@article{lin2026illumicraft,
  title={Illumicraft: Unified geometry and illumination diffusion for controllable video generation},
  author={Lin, Yuanze and Chen, Yi-Wen and Tsai, Yi-Hsuan and Clark, Ronald and Yang, Ming-Hsuan},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={27798--27829},
  year={2026}
}
```

