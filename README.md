
# lbg_tools

[![PyPI](https://img.shields.io/pypi/v/lbg_tools?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/lbg_tools/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/jfcrenshaw/lbg_tools/smoke-test.yml)](https://github.com/jfcrenshaw/lbg_tools/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/jfcrenshaw/lbg_tools/branch/main/graph/badge.svg)](https://codecov.io/gh/jfcrenshaw/lbg_tools)

Tools for forecasting Lyman-break Galaxies.

## Installation

For regular installation, use PyPI: `pip install lbg_tools`

For dev installation, create a python environment, activate it, then run `bash .setup_dev.sh`.

## Cosmologies

In multiple places you are able to specify a cosmology object for luminosity distance and differential comoving volume calculations.
The default cosmology is Astropy's Planck18, but you can use any Astropy cosmology.
You can also use a CCL cosmology, but to do so you must install CCL yourself.

## Citations

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

If you use this package, please cite Crenshaw et al. 2025:

```bibtex
@ARTICLE{crenshaw2025,
       author = {{Crenshaw}, John Franklin and {Leistedt}, Boris and {Graham}, Melissa Lynn and {Payerne}, Constantin and {Connolly}, Andrew J. and {Gawiser}, Eric and {Karim}, Tanveer and {Malz}, Alex I. and {Newman}, Jeffrey A. and {Ricci}, Marina and {The LSST Dark Energy Science Collaboration}},
        title = "{Quantifying the Impact of LSST $u$-band Survey Strategy on Photometric Redshift Estimation and the Detection of Lyman-break Galaxies}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2025,
        month = mar,
          eid = {arXiv:2503.06016},
        pages = {arXiv:2503.06016},
          doi = {10.48550/arXiv.2503.06016},
archivePrefix = {arXiv},
       eprint = {2503.06016},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv250306016C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

This project was automatically generated using the LINCC-Frameworks
[python-project-template](https://github.com/lincc-frameworks/python-project-template).
