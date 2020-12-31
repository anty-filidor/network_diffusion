import os
import pathlib
import random
import shutil
import string
import unittest

from network_diffusion import utils


class TestUtils(unittest.TestCase):
    """Test class for utils script."""

    def test_read_mlx(self) -> None:
        """Test for reading mlx file."""
        print(utils.get_absolute_path())
        data = utils.read_mlx(
            os.path.join(
                utils.get_absolute_path(), "tests/data/bankwiring.mpx"
            )
        )
        expected_data = {
            "type": ["multiplex"],
            "layers": [
                ["horseplay", "UNDIRECTED"],
                ["arguments", "UNDIRECTED"],
                ["friendship", "UNDIRECTED"],
                ["antagonist", "UNDIRECTED"],
                ["help", "DIRECTED"],
                ["job_trading", "DIRECTED"],
            ],
            "edge attributes": [["job_trading", "number", "numeric"]],
            "actors": [
                ["I1"],
                ["I3"],
                ["W1"],
                ["W2"],
                ["W3"],
                ["W4"],
                ["W5"],
                ["W6"],
                ["W7"],
                ["W8"],
                ["W9"],
                ["S1"],
                ["S2"],
                ["S4"],
            ],
            "edges": [
                ["W1", "W2", "horseplay"],
                ["W1", "W3", "horseplay"],
                ["W1", "W4", "horseplay"],
                ["W1", "W5", "horseplay"],
                ["W1", "S1", "horseplay"],
                ["W2", "W3", "horseplay"],
                ["W2", "W4", "horseplay"],
                ["W2", "S1", "horseplay"],
                ["W3", "W4", "horseplay"],
                ["W3", "W5", "horseplay"],
                ["W3", "S1", "horseplay"],
                ["W4", "W5", "horseplay"],
                ["W4", "S1", "horseplay"],
                ["W5", "W7", "horseplay"],
                ["W5", "S1", "horseplay"],
                ["W6", "W7", "horseplay"],
                ["W6", "W8", "horseplay"],
                ["W6", "W9", "horseplay"],
                ["W7", "W8", "horseplay"],
                ["W7", "W9", "horseplay"],
                ["W7", "S4", "horseplay"],
                ["W8", "W9", "horseplay"],
                ["W8", "S4", "horseplay"],
                ["W9", "S4", "horseplay"],
                ["I1", "W1", "horseplay"],
                ["I1", "W2", "horseplay"],
                ["I1", "W3", "horseplay"],
                ["I1", "W4", "horseplay"],
                ["W4", "W5", "arguments"],
                ["W4", "W6", "arguments"],
                ["W4", "W7", "arguments"],
                ["W4", "W9", "arguments"],
                ["W5", "W6", "arguments"],
                ["W5", "S1", "arguments"],
                ["W6", "W7", "arguments"],
                ["W6", "W8", "arguments"],
                ["W6", "W9", "arguments"],
                ["W6", "S1", "arguments"],
                ["W6", "S4", "arguments"],
                ["W7", "W8", "arguments"],
                ["W7", "W9", "arguments"],
                ["W7", "S4", "arguments"],
                ["W8", "W9", "arguments"],
                ["W8", "S1", "arguments"],
                ["W8", "S4", "arguments"],
                ["W9", "S1", "arguments"],
                ["S1", "S4", "arguments"],
                ["W1", "S1", "friendship"],
                ["I1", "W3", "friendship"],
                ["W1", "W3", "friendship"],
                ["W1", "W4", "friendship"],
                ["W3", "W4", "friendship"],
                ["W3", "S1", "friendship"],
                ["W4", "S1", "friendship"],
                ["W7", "W8", "friendship"],
                ["W7", "W9", "friendship"],
                ["W7", "S1", "friendship"],
                ["W8", "W9", "friendship"],
                ["W8", "S4", "friendship"],
                ["W9", "S4", "friendship"],
                ["W1", "W3", "help"],
                ["W1", "W9", "help"],
                ["W1", "S1", "help"],
                ["W2", "W3", "help"],
                ["W2", "W4", "help"],
                ["W2", "S1", "help"],
                ["W3", "W2", "help"],
                ["W4", "W1", "help"],
                ["W4", "W3", "help"],
                ["W4", "W6", "help"],
                ["W5", "W3", "help"],
                ["W6", "W3", "help"],
                ["W8", "W6", "help"],
                ["W6", "W7", "help"],
                ["W6", "W8", "help"],
                ["W6", "W9", "help"],
                ["W7", "S4", "help"],
                ["W8", "W7", "help"],
                ["W8", "W9", "help"],
                ["W9", "S4", "help"],
                ["S1", "W7", "help"],
                ["S2", "W6", "help"],
                ["S4", "W4", "help"],
                ["S4", "W8", "help"],
                ["I1", "I3", "antagonist"],
                ["I1", "W2", "antagonist"],
                ["I3", "W5", "antagonist"],
                ["I3", "W6", "antagonist"],
                ["I3", "W7", "antagonist"],
                ["I3", "W8", "antagonist"],
                ["I3", "W9", "antagonist"],
                ["I3", "S4", "antagonist"],
                ["W2", "W7", "antagonist"],
                ["W2", "W8", "antagonist"],
                ["W2", "W9", "antagonist"],
                ["W4", "W5", "antagonist"],
                ["W5", "W6", "antagonist"],
                ["W5", "W7", "antagonist"],
                ["W5", "W8", "antagonist"],
                ["W5", "W9", "antagonist"],
                ["W5", "S1", "antagonist"],
                ["W5", "S2", "antagonist"],
                ["W6", "W7", "antagonist"],
                ["W1", "S1", "job_trading", "2"],
                ["W2", "S4", "job_trading", "4"],
                ["W2", "S1", "job_trading", "12"],
                ["W6", "S2", "job_trading", "2"],
                ["W5", "S4", "job_trading", "7"],
                ["W8", "S4", "job_trading", "20"],
                ["W7", "S4", "job_trading", "2"],
            ],
        }
        self.assertEqual(
            data, expected_data, "Bankwiring file read incorrectly!"
        )

    def test_create_directory_new(self) -> None:
        """
        Test for creating directory.

        Scenario 1 - directory doesn't exists
        """
        # mock up data
        dir = os.path.join(
            pathlib.Path(__file__).parent,
            "".join(random.choices(string.ascii_letters, k=5)),
        )
        if os.path.isdir(dir):
            shutil.rmtree(dir)

        # execute desired function
        utils.create_directory(dir)

        # check correctness of execution
        self.assertEqual(
            os.path.isdir(dir), True, "Unable to create directory!"
        )

        # clean up
        shutil.rmtree(dir)

    def test_create_directory_exists(self) -> None:
        """
        Test for creating directory.

        Scenario 2 - directory exists.
        """
        # mock up data
        dir = os.path.join(
            pathlib.Path(__file__).parent,
            "".join(random.choices(string.ascii_letters, k=5)),
        )
        os.mkdir(dir)
        file_name = "".join(random.choices(string.ascii_letters, k=3)) + ".txt"
        with open(os.path.join(dir, file_name), "a") as f:
            f.write("random text")
            f.close()

        # execute function
        utils.create_directory(dir)

        # check correctness of execution
        self.assertEqual(
            os.path.isdir(dir), True, f"Directory {dir} should exists!"
        )
        self.assertEqual(
            os.path.isfile(os.path.join(dir, file_name)),
            True,
            f"File {file_name} should exists!",
        )

        # clean up
        shutil.rmtree(dir)


if __name__ == "__main__":
    unittest.main()
