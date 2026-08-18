"""Class to load completeness info."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from .library import library


@lru_cache(maxsize=None)
def _load_tables(
    band: str,
    directories: tuple[Path, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a completeness table from disk and condition it.

    Cached, because the result depends only on the band and where the data
    directories are: the depth enters the completeness only as ``m - m5_det``,
    so every depth shares one table. Conditioning dominates the cost of
    building a :class:`Completeness`, which in turn dominates the cost of
    building a :class:`~lbg_tools.TomographicBin`, so anything that scans a
    grid of depths or magnitude cuts was paying it once per grid point.

    Parameters
    ----------
    band : str
        Name of dropout band.
    directories : tuple of pathlib.Path
        Data directories to search, as a tuple so that this is hashable and so
        that adding a directory invalidates the cache rather than being missed.

    Returns
    -------
    pandas.DataFrame
        The table as read from disk.
    pandas.DataFrame
        The table forced monotonic in magnitude and unimodal in redshift.

    Raises
    ------
    ValueError
        If no table for this band exists in any data directory.
    RuntimeError
        If more than one does.

    Notes
    -----
    Both frames are shared between every caller. Do not mutate them.
    """
    files: list[Path] = []
    for directory in directories:
        files.extend(list(directory.glob(f"**/completeness_{band}.dat")))
    if len(files) == 0:
        raise ValueError(
            f"completeness_{band}.dat not found in any data directory.\n"
            "Perhaps you need to run `from lbg_tools import data` and "
            "`library.add_directory('path/to/files')` before creating the "
            "completeness object."
        )
    if len(files) > 1:  # pragma: no cover
        raise RuntimeError(
            f"Found {len(files)} files with the name 'completeness_{band}.dat' "
            "in the data directories, and I don't know which one to pick! "
            "You must remove all but one of these files, or use unique "
            "band names."
        )

    table0 = pd.read_csv(
        files[0],
        sep="  ",
        header=5,
        engine="python",
        dtype=np.float64,
    )
    table0.index = table0.index.to_numpy(dtype=float)
    table0.columns = table0.columns.to_numpy(dtype=float)

    # Force completeness to be monotonic along magnitude axis
    # and unimodal along the redshift axis. This ensures extrapolation
    # trends towards zero. Both steps adjust their argument in place, so they
    # get a copy: without it table0 would be the conditioned table too, which
    # it is named and documented not to be.
    table = Completeness._force_z_unimodality(
        Completeness._force_mag_monotonicity(table0.copy())
    )

    return table0, table


class Completeness:
    """Completeness function object."""

    def __init__(
        self,
        band: str,
        m5_det: float,
        allow_extrap: bool = True,
        extrap_bright: bool = True,
        validate_deep: bool = True,
    ) -> None:
        """Create completeness function.

        Parameters
        ----------
        band : str
            Name of dropout band
        m5_det : float
            5-sigma limiting depth in the detection band
        allow_extrap : bool, optional
            Whether to allow extrapolation. If False, a ValueError is raised
            for values outside of the domain of the completeness table.
            Default is True.
        extrap_bright : bool, optional
            Whether to linearly extrapolate on the bright end. If False,
            completeness at the bright end is flat. Default is True.
        validate_deep : bool, optional
            Whether to perform quick validation that extrapolation to deep values
            results in zero completeness. This only matters if allow_extrap == True.
            Default is True.
        """
        # Save params
        self._band = band
        self._m5_det = m5_det
        self.extrap_bright = extrap_bright

        # Load the completeness table. Shared between every object with this
        # band, so do not mutate self.table0 or self.table.
        self.table0, self.table = _load_tables(band, tuple(library.directories))

        # Create interpolators
        self._interpolator = RegularGridInterpolator(
            (self.table.index.to_numpy(), self.table.columns.to_numpy()),
            self.table.values,
            method="linear",
            bounds_error=not allow_extrap,
            fill_value=None,
        )

        # Do a quick check that deep values have zero completeness
        if allow_extrap and validate_deep:
            vals = self(m5_det + 10, self.table.index)
            assert np.allclose(
                vals, 0
            ), "Extrapolation to deep values does not yield zero!"

    @property
    def band(self) -> str:  # pragma: no cover
        """Name of the dropout band"""
        return self._band

    @property
    def m5_det(self) -> float:
        """5-sigma depth in the detection band."""
        return self._m5_det

    @staticmethod
    def _force_mag_monotonicity(table: pd.DataFrame) -> pd.DataFrame:
        """Force completeness to be monotonic along magnitude axis.

        Parameters
        ----------
        table : pd.DataFrame
            Table to adjust

        Returns
        -------
        pd.DataFrame
            Adjusted table
        """
        # Loop over rows...
        for i in range(table.shape[0]):
            # And columns...
            for j in range(table.shape[1]):
                # If an element is zero, set everything at fainter mags to zero
                # if np.isclose(table.iloc[i, j], 0):
                #    table.iloc[i, :] = 0
                #    break
                # Else replace elements with max of vals to the right
                table.iloc[i, j] = table.iloc[i, j:].max()

        return table

    @staticmethod
    def _force_unimodality(array: np.ndarray) -> np.ndarray:
        """Force the array to be unimodal.

        Parameters
        ----------
        array : np.ndarray
            Array to adjust

        Returns
        -------
        np.ndarray
            Adjusted array
        """
        # Create monotonic trend from both ends
        front_monotonic = array.copy()
        back_monotonic = array.copy()
        for i in range(len(array)):
            front_monotonic[i] = front_monotonic[: i + 1].max()
            back_monotonic[i] = back_monotonic[i:].max()

        # Create the new array
        new_array = front_monotonic.copy()
        mask = np.isclose(new_array, new_array.max())
        new_array[mask] = back_monotonic[mask]

        return new_array

    @staticmethod
    def _force_z_unimodality(table: pd.DataFrame) -> pd.DataFrame:
        """Force completeness to be unimodal along the redshift axis.

        Parameters
        ----------
        table : pd.DataFrame
            Table to adjust

        Returns
        -------
        pd.DataFrame
            Adjusted table
        """
        # Loop over columns
        for j in range(table.shape[1]):
            table.iloc[:, j] = Completeness._force_unimodality(
                table.iloc[:, j].to_numpy()
            )

        return table

    def __call__(
        self,
        m: float | np.ndarray,
        z: float | np.ndarray,
    ) -> float | np.ndarray:
        """Estimate completeness.

        Parameters
        ----------
        m : float or np.ndarray
            Apparent magnitude in the detection band
        z : float or np.ndarray
            Redshift

        Returns
        -------
        float or np.ndarray
            Completeness fraction
        """
        # Calculate mag - m5
        dm = m - self.m5_det

        # Clip dm so that brighter mags aren't extrapolated towards 1
        if not self.extrap_bright:
            dm = np.clip(dm, self.table.columns.min(), None)

        # Linear interpolation
        completeness = self._interpolator((z, dm))

        # Make sure linear extrapolation bottoms out at 0
        if self.extrap_bright:
            completeness = np.clip(completeness, 0, 1)
        else:
            # (not clipping at 1 because nothing should extrapolate upwards,
            #  which will be checked in a unit test)
            completeness = np.clip(completeness, 0, None)

        return completeness
