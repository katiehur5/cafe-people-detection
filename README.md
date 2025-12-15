# Café People Detection

This project fine-tunes a pre-trained Faster R-CNN model from TorchVision to more accurately and precisely detect people in a café environment under varying lighting conditions.

Results: We ran one experiment. Records of loss across each epoch during training and model outputs during testing can be found in `results.txt`.

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
3. Train the model on custom dataset.
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