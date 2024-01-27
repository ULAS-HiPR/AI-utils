import unittest
from io import StringIO
from unittest.mock import patch

from utils.LogManager import LogManager


class TestLogManager(unittest.TestCase):
    def setUp(self):
        # Redirect stdout to capture log messages
        self.mock_stdout = StringIO()
        patch("sys.stdout", self.mock_stdout).start()

    def tearDown(self):
        # Clean up and restore stdout
        patch.stopall()

    def test_singleton_instance(self):
        with self.assertRaises(Exception) as context:
            log_manager1 = LogManager()
            log_manager2 = LogManager()
            del log_manager1, log_manager2

        self.assertEqual(str(context.exception), "This class is a singleton!")

    def test_get_logger(self):
        log_manager = LogManager()
        logger = log_manager.get_logger("test_logger")
        self.assertEqual(logger.name, "utils.LogManager")


if __name__ == "__main__":
    unittest.main()
