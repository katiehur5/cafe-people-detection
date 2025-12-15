# Café People Detection

This project fine-tunes a pre-trained Faster R-CNN model from TorchVision to more accurately and precisely detect people in a café environment under varying lighting conditions.

> [!NOTE]
> Training on CPU may take a long time; checkpoints are saved per epoch.

> [!TIP]
> Using a virtual environment is recommended but not required.

> Requires Python 3.10–3.12.

> If `python` points to Python 2 on your system, use `python3` instead.

## Setup
1. Clone this repository:
```
git clone https://github.com/katiehur5/cafe-people-detection.git
cd cafe-people-detection
```
2.  Install required Python packages using the requirements.txt file:
```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
3. Train the model on custom dataset.
```
python src/train.py
```
4. Test the model on 15 test images and check people counts.
```
python src/eval.py --ckpt outputs/checkpoints/model_epoch_10.pth
```