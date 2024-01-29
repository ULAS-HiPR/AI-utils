import argparse

import tensorflow as tf
from utils.LogManager import LogManager
from utils.Model import Generator


def main(args):
    logger = LogManager.get_logger("AGRINET INFERENCE")
    logger.info("Loading generator...")

    generator = Generator()

    try:
        saved_model = tf.saved_model.load(args.weights)
        generator.load_weights(saved_model)
    except Exception as e:
        logger.critical("Error while loading weights: {}".format(e))

    logger.info("Model loaded")

    logger.info("Loading input image...")
    input = None

    try:
        input = tf.io.read_file(args.input)
    except Exception as e:
        logger.critical("Error while reading input image: {}".format(e))

    if input is None:
        logger.critical("Input image is empty")
    else:
        logger.info("Input image loaded")

    input = tf.image.decode_jpeg(input)
    input = tf.image.resize(input, [256, 256])
    input = (input / 127.5) - 1  # Normalize the images to [-1, 1]
    input = tf.expand_dims(input, axis=0)
    input = tf.cast(input, tf.float32)

    output = generator(input, training=True)
    output = tf.cast(output[0], tf.uint8)
    output = tf.image.encode_jpeg(output)

    try:
        tf.io.write_file("results/output.jpg", output)
    except Exception as e:
        logger.critical("Error while saving utput image: {}".format(e))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
