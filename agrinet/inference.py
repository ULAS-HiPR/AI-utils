from utils.LogManager import LogManager


def main(args):
    logger = LogManager.get_logger("AGRINET INFERENCE")
    logger.info("Warming up...")


def parse_args():
    pass  # TODO


if __name__ == "__main__":
    args = parse_args()
    main(args)
