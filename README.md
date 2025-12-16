# Café People Detection

This project fine-tunes a pre-trained Faster R-CNN model from TorchVision to more accurately and precisely detect people in a café environment under varying lighting conditions.

Data: Images (train/valid/test) and annotations can be accessed [here](https://drive.google.com/drive/folders/1lL5gz42IVmu0zyoa9fdtz9qmFBzUhwdq?usp=sharing).

Results: We ran one experiment. Records of loss across each epoch during training and model performance (AP) during testing can be found in `results.txt`. Output from the model's transformation of test images can be found [here](https://drive.google.com/drive/folders/16-ytg9w3wHKGIG4bZ55ZbwlH6dkFp0eG?usp=sharing). 

> [!NOTE]
> Checkpoint files outputted from our fine-tuning experiment were too large to upload to the repo. You can download the last checkpoint [here](https://drive.google.com/file/d/1xLiZOzVR_JEC1QjCp2BizvvekKs6h4rn/view?usp=sharing).

## Setup
1. Clone this repository:
```
git clone https://github.com/katiehur5/cafe-people-detection.git
cd cafe-people-detection
```
2.  Install required Python packages using the requirements.txt file:
> Requires Python 3.10–3.12.

> [!TIP]
> Using a virtual environment is recommended but not required.
```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
3. Train the model on custom dataset. You can skip this step by downloading our [last checkpoint's weights](https://drive.google.com/file/d/1xLiZOzVR_JEC1QjCp2BizvvekKs6h4rn/view?usp=sharing) and putting them in a `outputs/checkpoints/` folder.
> [!NOTE]
> Training on CPU may take a long time; checkpoints are saved per epoch.
```
python src/train.py
```
4. Test the model on 15 test images and check people counts.
```
python src/eval.py --ckpt outputs/checkpoints/model_epoch_10.pth
```
5. Evaluate model's performance.
```
python src/eval_map.py --ckpt outputs/checkpoints/model_epoch_10.pth
```