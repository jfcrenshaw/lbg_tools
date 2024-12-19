"""Test library class"""

from pathlib import Path

from lbg_tools import library


def test_add_data_directory() -> None:
    """Test adding a new directory to library."""
    # Before we add the test directory, this test file shouldn't be in there
    files = [str(file) for file in library.files]
    assert __file__ not in files

    # Add test directory and check that this file is now in the list of files
    test_dir = Path(__file__).parent
    library.add_directory(test_dir)
    files = [str(file) for file in library.files]
    assert __file__ in files


def test_completeness_files() -> None:
    """It should find a completeness file for every LSST LBG dropout sample."""
    files = [file.stem for file in library.completeness_files]
    for band in "ugriz":
        assert f"completeness_{band}" in files, f"Missing file for {band} band"


def test_bandpass_files() -> None:
    """It should find a bandpass file for every LSST band."""
    bands = [file.stem.split("_")[1] for file in library.bandpass_files]
    for band in "ugrizy":
        assert band in bands, f"Missing file for {band} band"


def test_bands() -> None:
    """Test the function that returns band names.

    Currently hardcoded assuming that we have completeness for ugriz and
    bandpasses for ugrizy.
    """
    cbands = sorted(library.get_bands())
    assert cbands == sorted(list("ugriz"))

    bbands = sorted(library.get_bands(completeness=False, bandpass=True))
    assert bbands == sorted(list("ugrizy"))

    union = sorted(library.get_bands(bandpass=True))
    assert union == bbands

    intersection = sorted(library.get_bands(bandpass=True, union=False))
    assert intersection == cbands
