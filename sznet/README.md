## Welcome to SafeZoneNet (SZNet)
This is a project being worked on by the ULAS HiPR's Payload team.

## Objective
Design and train a neural network that can identify safe landing zones for a high-powered rocket.

## Steps
1. Segment an image captured by an onboard camera into different classes (e.g. forest, buildings, field, water, residential, etc.)
2. Rank the segments by safety and rocket retrievability (e.g. field > forest > water > buildings > residential)
3. Choose the optimal landing zone
4. Calculate approximate coordinates of chosen landing zone using altitude, camera FOV and potentially IMU data.

## Goals
1. Model is small and efficient enough to run on our chosen hardware (Raspberry Pi 5 4GB and Coral TPU (4 TOPS @ int8))
2. Model is fast enough to continuously run inference on frames returned from the camera and update safe landing zones
3. Model performance is good enough to demonstrate a successful proof-of-concept

## Project Layout
- `experiments/`
    - Experiments comparing model performance with different architectures, depths, widths, image resolutions and hyperparameters. Typically in Jupyter Notebook format.
- `utils/`
    - Auxiliary code used in data pre-processing or inference.
- `scripts/`
    - Ancillary scripts written as part of experimenting deemed useful enough to be included in the codebase.

## Tech Decisions
### ML Framework
We have chosen PyTorch as our ML framework. This is due to its widespread adoption in the industry and frequent use in sample implementations of model architectures. 


## Dataset
We have chosen the OpenEarthMap dataset. [Add more info]

The original dataset has 8 classes: Bareland, Rangeland, Developed Space, Road, Tree, Water, Agricultural Land, Building. 

<!-- For our purposes, we really only need to distinguish between 3 superclasses: Safe & Retrievable (Bareland, Rangeland, Agricultural Land), Safe & Unretrievable (Tree, Water) and Unsafe (Developed Space, Road, Building). To improve model performance and efficiency we merged the existing classes into our superclasses to create our dataset.  -->

For our purposes, we really only need to distinguish between 3 superclasses: Safe (Bareland, Rangeland, Agricultural Land, Tree, Water) and Unsafe (Developed Space, Road, Building). To improve model performance and efficiency we merged the existing classes into our superclasses to create our dataset. 

1, 2, 7, 5, 6 = 1
3, 4, 8 = 2

```python
import cv2
import numpy as np
import Image
def merge_classes(image_path: str):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    np.where(image == 1 or image == 2 or image == 4 or image == 7, 1, 0)
    np.where(image == 5 or image == 6, 2, 0)
    np.where(image == 3 or image == 8, 3, 0)
    im = Image.fromarray(image)
    im.save(image_path)
```