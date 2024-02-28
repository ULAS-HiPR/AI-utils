import argparse

import tensorflow as tf
from utils.DataLoader import load_image_test
from utils.LogManager import LogManager
from utils.Metrics import CGANMetrics
from utils.Model import Discriminator, Generator

BUFFER_SIZE = 400


def main(args):
    logger = LogManager.get_logger("AGRINET TEST")
    logger.info("Building data pipeline...")

    test_dataset = tf.data.Dataset.list_files(args.data_dir + "/test/*." + args.ext)
    test_dataset = test_dataset.map(
        load_image_test, num_parallel_calls=tf.data.AUTOTUNE
    )
    test_dataset = test_dataset.shuffle(BUFFER_SIZE)
    test_dataset = test_dataset.batch(args.batch_size)
    logger.debug(
        "Test dataset contains {} images".format(len(test_dataset) * args.batch_size)
    )

    logger.info("Building model...")

    generator = Generator()
    discriminator = Discriminator()

    try:
        discriminator.load_weights(f"{args.exp}/discriminator_weights")
        generator.load_weights(f"{args.exp}/generator_weights")
        logger.debug("{} parameter model loaded".format(generator.count_params()))

    except Exception as e:
        logger.critical("Error while loading weights: {}".format(e))

    logger.info("Testing model...")
    results = CGANMetrics(save_path=f"{args.exp}/metrics" if args.save else None)

    for test_input, test_target in test_dataset:
        prediction = generator(test_input, training=False)
        out = discriminator([prediction, test_target], training=False)
        results.update(out, prediction, test_input, test_target)

    mmd = results.get_metric("MMD")
    psnr = results.get_metric("PSNR")

    logger.debug(f"Mean MMD: {mmd:.2f}\tMean PSNR: {psnr:.2f}")
    logger.info("Results saved to {}".format(f"{args.exp}/metrics"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="Experiment name")
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Path to the data"
    )
    parser.add_argument(
        "--ext", type=str, default="jpg", help="Extension of the images"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--save", type=bool, default=False, help="Save results to file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
