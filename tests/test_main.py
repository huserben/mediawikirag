import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from main import main


class TestMain(unittest.TestCase):
    def test_main_runs(self):
        # This test just checks that main() runs without error
        try:
            main()
        except Exception as e:
            self.fail(f"main() raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
