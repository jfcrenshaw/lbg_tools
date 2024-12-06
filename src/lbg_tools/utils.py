"""Utilities for this package."""

from pathlib import Path


class DataDirectory:
    """Object to hold paths to data files."""

    directories = [Path(__file__).parent / "data"]

    def add_directory(self, path: str | Path) -> None:
        """Add directory to list of data directories.

        Parameters
        ----------
        path : str or pathlib.Path
            Path of new data directory
        """
        # Cast to pathlib object
        path = Path(path).resolve()

        # Raise error if path doesn't exist
        if not path.exists():
            raise ValueError(f"Directory {path} does not exist.")

        # Add new directory to the front of the list
        self.directories.insert(0, path)

    @property
    def files(self) -> list[Path]:
        """List of all data files."""
        # Loop over every path in the data directories
        files: list[Path] = []
        for path in self.directories:
            files.extend(path.glob("*"))

        return files

    @property
    def completeness_files(self) -> list[Path]:
        """List of all completeness files."""
        return [file for file in self.files if "completeness_" in file.stem]

    @property
    def bands(self) -> list[str]:
        """List of bands for which a completeness file exists."""
        return [file.stem.split("_")[-1] for file in self.completeness_files]


# Instantiate for use in other modules
data = DataDirectory()
