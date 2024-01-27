import argparse
import os
import time

import tensorflow as tf
from tensorflow.summary import create_file_writer
from utils.DataLoader import load_image_test, load_image_train
from utils.LogManager import LogManager
from utils.Model import (
    Discriminator,
    Generator,
    discriminator_loss,
    discriminator_optimizer,
    generator_loss,
    generator_optimizer,
)

BUFFER_SIZE = 400


def main(args):
    logger = LogManager.get_logger("AGRINET TRAIN")

    # training logs
    log_dir = os.path.join("./", args.name, "logs")
    summary_writer = create_file_writer(log_dir)

    # Gathering training
    logger.info("Building data pipeline...")

    train_dataset = tf.data.Dataset.list_files(args.data_dir + "/train/*." + args.ext)
    train_dataset = train_dataset.map(
        load_image_train, num_parallel_calls=tf.data.AUTOTUNE
    )
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(args.batch_size)
    logger.debug("Train dataset contains {} images".format(len(train_dataset)))

    test_dataset = tf.data.Dataset.list_files(args.data_dir + "/train/*." + args.ext)
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(args.batch_size)
    logger.debug("Test dataset contains {} images".format(len(test_dataset)))

    logger.info("Building model...")
    generator = Generator()
    discriminator = Discriminator()

    @tf.function
    def train_step(input_image, target, step):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=True)

            disc_real_output = discriminator([input_image, target], training=True)
            disc_generated_output = discriminator(
                [input_image, gen_output], training=True
            )

            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
                disc_generated_output, gen_output, target
            )
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(
            gen_total_loss, generator.trainable_variables
        )
        discriminator_gradients = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables
        )

        generator_optimizer.apply_gradients(
            zip(generator_gradients, generator.trainable_variables)
        )
        discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, discriminator.trainable_variables)
        )

        txt_gen_total_loss = tf.convert_to_tensor(gen_total_loss)
        txt_gen_gan_loss = tf.convert_to_tensor(gen_gan_loss)
        txt_gen_l1_loss = tf.convert_to_tensor(gen_l1_loss)
        txt_disc_loss = tf.convert_to_tensor(disc_loss)

        with summary_writer.as_default():
            tf.summary.scalar("gen_total_loss", txt_gen_total_loss, step=step)
            tf.summary.scalar("gen_gan_loss", txt_gen_gan_loss, step=step)
            tf.summary.scalar("gen_l1_loss", txt_gen_l1_loss, step=step)
            tf.summary.scalar("disc_loss", txt_disc_loss, step=step)

            tf.summary.image("input_image", input_image, step=step)
            tf.summary.image("target", target, step=step)
            tf.summary.image("gen_output", gen_output, step=step)

    checkpoint_dir = f"./{args.name}/training_checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )

    logger.info("Starting training...")

    def fit(train_ds, test_ds, steps):
        start = time.time()

        for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
            train_step(input_image, target, step)

            # Checkpoint frequency is 50 steps
            if (step + 1) % 50 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)
                logger.debug("Checkpoint saved at {}".format(checkpoint_prefix))

            # Logging frequency is 10 steps
            if (step + 1) % 10 == 0:
                logger.debug(
                    "Epoch {} completed, {:.2f}s elapsed.".format(
                        step + 1, time.time() - start
                    )
                )

    fit(train_dataset, test_dataset, steps=args.epochs)
    logger.info("Training finished")

    # day-month-year-hour-minute
    filetime = time.strftime("%d%m%y_%H%M")

    tf.saved_model.save(generator, "./{}/generator_{}".format(args.name, filetime))
    tf.saved_model.save(
        discriminator, "./{}/discriminator_{}".format(args.name, filetime)
    )

    logger.debug(
        "Model and weights saved at {} and {} respectively".format(
            "./{}/generator_{} ".format(args.name, filetime),
            " ./{}/discriminator_{}".format(args.name, filetime),
        )
    )

    try:
        generator.save_weights("./{}/generator_weights_{}".format(args.name, filetime))
        discriminator.save_weights(
            "./{}/discriminator_weights_{}".format(args.name, filetime)
        )

    except Exception as e:
        logger.error("Error while saving weights : {}".format(e))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        help="Name of experiment, used for logging and saving checkpoints and weights",
        required=True,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to data directory. must contain train and test folders with images",
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
