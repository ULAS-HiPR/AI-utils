import sys
import os

# fixes "ModuleNotFoundError: No module named 'utils'"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# flake8: noqa
import unittest
import tensorflow as tf
from utils.DataLoader import load_image_test, load_image_train, resize, save_result


class TestImageDataFunctions(unittest.TestCase):
    def setUp(self):
        # Set up any necessary resources or configurations for tests
        pass

    def tearDown(self):
        # Clean up after tests
        pass

    def test_load_image_train(self):
        # Condition: input_image.shape != real_image.shape from the train dataset
        image_file = "assets/green-field.jpg"
        input_image, real_image = load_image_train(image_file)

        self.assertIsInstance(input_image, tf.Tensor)
        self.assertIsInstance(real_image, tf.Tensor)
        self.assertEqual(input_image.shape, real_image.shape)
        self.assertEqual(input_image.shape, (256, 256, 3))

    def test_load_image_test(self):
        # Condition: input_image.shape == real_image.shape from the test dataset
        image_file = "assets/green-field.jpg"

        input_image, real_image = load_image_test(image_file)

        self.assertIsInstance(input_image, tf.Tensor)
        self.assertIsInstance(real_image, tf.Tensor)
        self.assertEqual(input_image.shape, real_image.shape)
        self.assertEqual(input_image.shape, (256, 256, 3))

    def test_resize(self):
        # Condition: input_image.shape == real_image.shape
        input_image = tf.constant([[[1, 2, 3], [4, 5, 6]]], dtype=tf.float32)
        real_image = tf.constant([[[7, 8, 9], [10, 11, 12]]], dtype=tf.float32)
        height, width = 128, 128
        resized_input, resized_real = resize(input_image, real_image, height, width)

        self.assertIsInstance(resized_input, tf.Tensor)
        self.assertIsInstance(resized_real, tf.Tensor)
        self.assertEqual(resized_input.shape, resized_real.shape)
        self.assertEqual(resized_input.shape, (128, 128, 3))

    def save_result_rgb(self):
        input_img = tf.constant([[[1, 2, 3], [4, 5, 6]]], dtype=tf.float32)
        output_img = tf.constant([[[7, 8, 9], [10, 11, 12]]], dtype=tf.float32)
        path = "assets/result.jpg"

        try:
            res = save_result(input_img, output_img, path)
            assert res.shape == (256, 512, 3)
        except:
            assert False, "save_result() raised an exception!"

    def save_result_grayscale(self):
        input_img = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
        output_img = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
        path = "assets/result.jpg"

        try:
            res = save_result(input_img, output_img, path)
            assert res.shape == (256, 512, 3)

        except:
            assert False, "save_result() raised an exception!"


if __name__ == "__main__":
    unittest.main()
