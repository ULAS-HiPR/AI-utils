# AgriNet

A RGB to NIR image translation model for agricultural aerial evaluation on vegetation and moisture data.

## Datasets

nirscene0 - 477 images
multiple scenes ranging in conditions, this was used as it is a good balance for generalisation.

## Tools

### Training

CLI tool available via `python train.py --help` for more information.

```man
usage: train.py [-h] --name NAME --data_dir DATA_DIR --batch_size BATCH_SIZE
                --epochs EPOCHS [--lr LR] [--ext EXT] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           Name of experiment, used for logging and saving
                        checkpoints and weights
  --data_dir DATA_DIR   Path to data directory. must contain train and test
                        folders with images
  --batch_size BATCH_SIZE
                        Batch size for training
  --epochs EPOCHS       Number of epochs to train
  --lr LR               Learning rate for training
  --ext EXT             Extension of the images
  --seed SEED           Random seed for training
```

#### Unit tests
  
  ```bash
  agrinet $ python -m unittest discover -q tests
  ```
