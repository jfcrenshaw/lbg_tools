"""Class to load completeness info."""

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from .library import library


class Completeness:
    """Completeness function object."""

    def __init__(
        self,
        band: str,
        m5_det: float,
    ) -> None:
        """Create completeness function.

        Parameters
        ----------
        band : str
            Name of dropout band
        m5_det : float
            5-sigma limiting depth in the detection band
        """
        # Save params
        self._band = band
        self._m5_det = m5_det

        # Load the completeness table
        files = []
        for directory in library.directories:
            files.extend(list(directory.glob(f"**/completeness_{band}.dat")))
        if len(files) == 0:
            raise ValueError(
                f"completeness_{band}.dat not found in any data directory.\n"
                "Perhaps you need to run `from lbg_tools import data` and "
                "`library.add_directory('path/to/files')` before creating the "
                "completeness object."
            )
        if len(files) > 1:
            raise RuntimeError(
                f"Found {len(files)} files with the name 'completeness_{band}.dat' "
                "in the data directories, and I don't know which one to pick! "
                "You must remove all but one of these files, or use unique "
                "band names."
            )
        else:
            self.table0 = pd.read_csv(
                files[0],
                sep="  ",
                header=5,
                engine="python",
                dtype=np.float64,
            )
            self.table0.index = self.table0.index.to_numpy(dtype=float)
            self.table0.columns = self.table0.columns.to_numpy(dtype=float)

        # Force completeness to be monotonic along magnitude axis
        # and unimodal along the redshift axis. This ensures extrapolation
        # trends towards zero
        self.table = self._force_z_unimodality(
            self._force_mag_monotonicity(self.table0)
        )

        # Create interpolators
        self._interpolator = RegularGridInterpolator(
            (self.table.index.to_numpy(), self.table.columns.to_numpy()),
            self.table.values,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    @property
    def band(self) -> str:
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
                if np.isclose(table.iloc[i, j], 0):
                    table.iloc[i, :] = 0
                    break
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

    def _force_z_unimodality(self, table: pd.DataFrame) -> pd.DataFrame:
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
            table.iloc[:, j] = self._force_unimodality(table.iloc[:, j].to_numpy())

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
        dm = np.clip(dm, self.table.columns.min(), None)

        # Linear interpolation
        completeness = self._interpolator((z, dm))

        # Make sure linear extrapolation bottoms out at 0
        # (not clipping at 1 because nothing should extrapolate upwards,
        #  which will be checked in a unit test)
        completeness = np.clip(completeness, 0, None)

        # Make unimodal along both directions
        completeness = np.atleast_2d(completeness)
        for i in range(completeness.shape[0]):
            completeness[i] = self._force_unimodality(completeness[i])
        for j in range(completeness.shape[1]):
            completeness[:, j] = self._force_unimodality(completeness[:, j])

        return completeness.squeeze()
