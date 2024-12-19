"""Class to locate data files and related info."""

from pathlib import Path


class Library:
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
            files.extend([file for file in path.glob("**/*") if file.is_file()])

        return files

    @property
    def completeness_files(self) -> list[Path]:
        """List of all completeness files."""
        return [file for file in self.files if "completeness_" in file.stem]

    @property
    def bandpass_files(self) -> list[Path]:
        """List of all completeness files."""
        return [file for file in self.files if "bandpass_" in file.stem]

    def get_bands(
        self,
        completeness: bool = True,
        bandpass: bool = False,
        union: bool = True,
    ) -> list[str]:
        """Get list of available bands.

        Parameters
        ----------
        completeness : bool, optional
            Whether to index on completeness files. Default is True.
        bandpass : bool, optional
            Whether to index on bandpass files. Default is False.
        union : bool, optional
            Whether to take the union of bandpasses indexed from each list.
            If False, the intersection is taken instead. Default is True.

        Returns
        -------
        list[str]
            List of bands
        """
        sets = []
        if completeness:
            sets.append(
                set([file.stem.split("_")[1] for file in self.completeness_files])
            )
        if bandpass:
            sets.append(set([file.stem.split("_")[1] for file in self.bandpass_files]))

        if union:
            return list(set.union(*sets))
        else:
            return list(set.intersection(*sets))


# Instantiate for use in other modules
library = Library()
