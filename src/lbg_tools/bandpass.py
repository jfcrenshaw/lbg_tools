"""Class representing bandpasses and enabling magnitude calculations."""

import numpy as np
from scipy.integrate import simpson

from .library import library


class Bandpass:
    def __init__(self, band: str) -> None:
        """Create bandpass object.

        Parameters
        ----------
        band : str
            Name of the bandpass.
        """
        self._band = band

        # Load the bandpass data
        files = []
        for directory in library.directories:
            files.extend(list(directory.glob(f"**/bandpass_{band}*.dat")))
        if len(files) == 0:
            raise ValueError(
                f"Bandpass data for {band} band not found in any data directory.\n"
                "Perhaps you need to run `from lbg_tools import data` and "
                "`library.add_directory('path/to/files')` before creating the "
                "bandpass object."
            )
        if len(files) > 1:
            raise RuntimeError(
                f"Found {len(files)} files with name 'bandpass_{band}*.dat' "
                "in the data directories, and I don't know which one to pick! "
                "You must remove all but one of these files, or use unique "
                "band names."
            )
        else:
            # Load values from file
            wavelen, throughput = np.genfromtxt(files[0], unpack=True)

            # Convert wavelength nm -> angstrom
            wavelen *= 10

            # Save values
            self._wavelen = wavelen
            self._throughput = throughput

    @property
    def band(self) -> str:
        """Name of the bandpass"""
        return self._band

    @property
    def wavelen(self) -> np.ndarray:
        """Wavelength grid, in angstroms."""
        return self._wavelen

    @property
    def throughput(self) -> np.ndarray:
        """Bandpass throughput (value between 0 and 1).

        This is the same as the bands ''transmission'', and is defined
        as the fraction of incident intensity at a given wavelength, incident
        on the top of the atmosphere, that is recorded by the telescope/camera.
        """
        return self._throughput

    @property
    def fwhm(self) -> float:
        """Bandpass full-width-at-half-max, in angstroms."""
        w, t = self.wavelen, self.throughput
        w_left = w[np.argmax(t > 0.5 * t.max())]
        w_right = w[-np.argmax(t[::-1] > 0.5 * t.max())]
        return w_right - w_left

    @property
    def mean_wavelength(self) -> float:
        """Photon-weighted mean wavelength of the bandpass, in angstroms"""
        num = simpson(self.wavelen**2 * self.throughput, x=self.wavelen)
        den = simpson(self.wavelen * self.throughput, x=self.wavelen)
        return num / den

    def calc_eff_wavelength(self, wavelen: np.ndarray, flambda: np.ndarray) -> float:
        """Calculate the photon-weighted effective wavelength of the bandpass.

        This differs from mean_wavelength in that it is weighted by the
        SED of an object.

        Parameters
        ----------
        wavelen : np.ndarray
            Wavelength grid of the object, in angstroms.
        flambda : np.ndarray
            SED of the object in erg / s / cm^2 / angstrom

        Returns
        -------
        float
            Effective wavelength in angstroms
        """
        # Re-sample SED on throughput grid
        flambda_r = np.interp(self.wavelen, wavelen, flambda)

        # Calculate relevant ratio
        num = simpson(self.wavelen**2 * flambda_r * self.throughput, x=self.wavelen)
        den = simpson(self.wavelen * flambda_r * self.throughput, x=self.wavelen)

        return num / den

    @property
    def zero_point(self) -> float:
        """AB magnitude zero point for the bandpass."""
        # Create the AB reference spectrum
        ab_ref = 0.109 * self.wavelen**-2  # erg / s / cm^2 / angstrom

        # Calculate zero point
        num = simpson(ab_ref * self.throughput * self.wavelen, x=self.wavelen)
        den = simpson(self.throughput * self.wavelen, x=self.wavelen)
        zp = 2.5 * np.log10(num / den)

        return zp

    def calc_magnitude(self, wavelen: np.ndarray, flambda: np.ndarray) -> float:
        """Calculate observed magnitude in the bandpass.

        Parameters
        ----------
        wavelen : np.ndarray
            Wavelength grid of the object, in angstroms.
        flambda : np.ndarray
            SED of the object in erg / s / cm^2 / angstrom

        Returns
        -------
        float
            Observed AB magnitude
        """
        # Re-sample SED on throughput grid
        flambda_r = np.interp(self.wavelen, wavelen, flambda)

        # Calculate magnitude
        num = simpson(flambda_r * self.throughput * self.wavelen, x=self.wavelen)
        den = simpson(self.throughput * self.wavelen, x=self.wavelen)
        mag = -2.5 * np.log10(num / den) + self.zero_point

        return mag
