## Welcome to SafeZoneNet (SZNet)
This is a project being worked on by the ULAS HiPR's Payload team.

## Objective
Design and train a neural network that can identify safe landing zones for a high-powered rocket.

## Steps
1. Segment an image captures by an onboard camera into different classes (e.g. forest, buildings, field, water, residential, etc.)
2. Rank the segments by safety and rocket retrievability (e.g. field > forest > water > buildings > residential)
3. Choose the optimimal landing zone
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
- `scipts/`
    - Ancillary scripts written as part of experimenting deemed useful enough to be included in the codebase.

## Tech Decisions
### ML Framework
We have chosen PyTorch as our ML framework. This is due to its widespread adoption in the industry and frequent use in sample implementations of model architectures. 
