import argparse

import numpy as np

from utils.LogManager import LogManager
from utils.Model import Generator
from utils.DataLoader import load_single, save_result
from utils.EnviroIndices import parse_indices


def main(args):
    logger = LogManager.get_logger("AGRINET INFERENCE")
    logger.info("Warming up...")

    generator = Generator()
    try:
        generator.load_weights(f"{args.exp}/generator_weights")
        logger.debug("Model loaded")

    except Exception as e:
        logger.critical("Error while loading weights: {}".format(e))

    input = None

    try:
        input = load_single(args.input)
        logger.debug("Input loaded")
    except Exception as e:
        logger.critical("Error while loading input: {}".format(e))

    input = np.expand_dims(input, 0)
    prediction = generator(input, training=False)[0]
    save_result(input[0], prediction, f"{args.exp}/results/output.jpg")
    logger.info("Output saved")

    ndvi, ndwi, ndvi_regions, ndwi_regions = parse_indices(
        input[0], prediction, regions=True
    )

    ndvi = np.stack([np.zeros_like(ndvi), ndvi, np.zeros_like(ndvi)], axis=-1)
    ndwi = np.stack([np.zeros_like(ndwi), np.zeros_like(ndwi), ndwi], axis=-1)
    save_result(ndvi, ndwi, f"{args.exp}/results/indices.jpg")
    logger.info("Indices saved")

    ndvi_regions = ndvi_regions * 255
    ndwi_regions = ndwi_regions * 255
    save_result(ndvi_regions, ndwi_regions, f"{args.exp}/results/regions.jpg")
    logger.info("Regions saved")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="Experiment name")
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
