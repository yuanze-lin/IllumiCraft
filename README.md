<p align="center"><img src="https://github.com/yuanze-lin/IllumiCraft/blob/main/assets/IllumiCraft.png" alt="icon" width="150" height="150" style="vertical-align:middle; margin-right:5px;" /></p>

# IllumiCraft: Unified Geometry and Illumination Diffusion for Controllable Video Generation  <br />

Official implementation of "IllumiCraft: Unified Geometry and Illumination Diffusion for Controllable Video Generation" 


[![PDF](https://img.shields.io/badge/PDF-Download-orange?style=flat-square&logo=adobeacrobatreader&logoColor=white)](https://arxiv.org/pdf/2506.03150)
[![arXiv](https://img.shields.io/badge/arXiv-2506.03150-b31b1b.svg)](https://arxiv.org/abs/2506.03150)
[![Project Page](https://img.shields.io/badge/Project%20Page-Visit%20Now-00d45c?style=flat-square&logo=googlechrome&logoColor=white)](https://yuanze-lin.me/IllumiCraft_page/)
[![YouTube Video](https://img.shields.io/badge/YouTube%20Video-FF0000?style=flat-square&logo=youtube&logoColor=white)](https://youtu.be/qAV58sADEzo)

[Yuanze Lin](https://yuanze-lin.me/), [Yi-Wen Chen](https://wenz116.github.io/), [Yi-Hsuan Tsai](https://sites.google.com/site/yihsuantsai/), [Ronald Clark](https://www.ron-clark.com/), [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/)

## :low_brightness: Method 

![image](https://github.com/yuanze-lin/IllumiCraft/blob/main/assets/framework.png)

## :mega:  News
- [x] Release the training code.
- [x] Release IllumiCraft dataset.
- [x] Release the model and the inference code.
- [x] Set up the project page.

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

The IllumiCraft dataset is shown with the following structure:

```text
dataset/
├── foreground_videos/
├── background_videos/
├── tracking_videos/
├── lighting_videos/
├── prompt.txt
├── videos.txt
├── foreground_videos.txt
├── background_videos.txt
├── tracking_videos.txt
└── lighting_videos.txt
```

Update the dataset path in `train.sh` before training.

## 🏋️ Training

Edit the following fields in `train.sh`:

```bash
DATA_ROOT=/path/to/train_dataset
MODEL_PATH=/path/to/Wan2.1-Fun-1.3B-Control
OUTPUT_PATH=checkpoints/illumicraft_weights
```

Launch training:

```bash
bash train.sh
```

### 📥 Download Pretrained Weights

Before running inference, download the released IllumiCraft checkpoint:

```bash
python download_illumicraft_weights.py
```

The pretrained weights will be automatically downloaded to:

```text
checkpoints/illumicraft_weights/
```
After downloading, update the checkpoint path in `inference.sh`:


## 🎥 Inference

Run video generation using a trained IllumiCraft checkpoint.

Edit the following fields in `inference.sh`:

```bash
MODEL_PATH=/path/to/Wan2.1-Fun-1.3B-Control
CHECKPOINT_PATH=/path/to/illumicraft_checkpoint

VALIDATION_IMAGES=/path/to/foreground_video.mp4
VALIDATION_BACKGROUNDS=/path/to/background_video.mp4
TRACKING_MAP_PATH=/path/to/tracking_video.mp4
HDR_MAP_PATH=/path/to/lighting_video.mp4

OUTPUT_DIR=outputs
```

Launch inference:

```bash
bash inference.sh
```

## :snowboarder: Results
<img width="600" align="left" alt="image" src="https://github.com/user-attachments/assets/aeb594c5-c32b-4ffa-bcda-0723e7612187" />

![image](https://github.com/yuanze-lin/IllumiCraft/blob/main/examples/14.gif)
<img width="600" align="left" alt="image" src="https://github.com/user-attachments/assets/372d8fec-db53-4c35-b668-76055472e96b"/>

![image](https://github.com/yuanze-lin/IllumiCraft/blob/main/examples/2.gif)
<img width="600" align="left" alt="image" src="https://github.com/user-attachments/assets/45fcec9b-ec34-40a5-8809-e261c79e48a1"/>

![image](https://github.com/yuanze-lin/IllumiCraft/blob/main/examples/1.gif)
<img width="600" align="left" alt="image" src="https://github.com/user-attachments/assets/372d8fec-db53-4c35-b668-76055472e96b"/>

![image](https://github.com/yuanze-lin/IllumiCraft/blob/main/examples/3.gif)
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

