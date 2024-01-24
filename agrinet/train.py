import argparse

import tensorflow as tf
from utils.LogManager import LogManager
from utils.DataLoader import load_image_train, load_image_test

BUFFER_SIZE = 400


def main(args):
    logger = LogManager.get_logger("AGRINET TRAIN")

    # Gathering training and testing images
    logger.info("Building data pipeline...")

    train_dataset = tf.data.Dataset.list_files(args.data_dir + "/train/*." + args.ext)
    train_dataset = train_dataset.map(
        load_image_train, num_parallel_calls=tf.data.AUTOTUNE
    )
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(args.batch_size)
    logger.debug("Train dataset contains {} images".format(len(train_dataset)))

    test_dataset = tf.data.Dataset.list_files(args.data_dir + "/test/*." + args.ext)
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(args.batch_size)
    logger.debug("Test dataset contains {} images".format(len(test_dataset)))

    logger.info("Building model...")

    logger.info("Starting training...")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to data directory",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
        required=True,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train",
        required=True,
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for training"
    )
    parser.add_argument(
        "--ext", type=str, default="jpg", help="Extension of the images"
    )
    parser.add_argument("--seed", type=int, default=10, help="Random seed for training")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
