"""Test the util functions."""

from pathlib import Path

from lbg_tools import data


def test_add_data_directory():
    """Test adding a new directory to the data directory list."""
    # Before we add the test directory, this test file shouldn't be in there
    files = [str(file) for file in data.files]
    assert __file__ not in files

    # Add test directory and check that this file is now in the list of files
    test_dir = Path(__file__).parent
    data.add_directory(test_dir)
    files = [str(file) for file in data.files]
    assert __file__ in files
