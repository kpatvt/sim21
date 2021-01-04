=====
sim21
=====


.. image:: https://img.shields.io/pypi/v/sim21.svg
        :target: https://pypi.python.org/pypi/sim21

.. image:: https://img.shields.io/travis/kpatvt/sim21.svg
        :target: https://travis-ci.com/kpatvt/sim21

.. image:: https://readthedocs.org/projects/sim21/badge/?version=latest
        :target: https://sim21.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


Open source simulator for typical process/chemical/refinery engineering applications with rigorous thermodynamics.
Revival of legacy Sim42 simulator for Python 3 with custom thermodynamics.

* Learn more about legacy `Sim42 <https://web.archive.org/web/20050204025650/http://manual.sim42.org/>`_
* Free software: MIT license
* Documentation: https://sim21.readthedocs.io.

Under continual development, breaking changes are to be expected.

Status
------
* Many examples with output now available with PR/SRK EOS Thermo:
    * Air Separation Unit (ASU)
    * Natural Gas Separation Train
    * Refrigeration Loops
    * Nitrogen Rejection
    * Compressors/Expanders with given efficiencies and performance curves
    * Many examples of wide and narrow boiling distillation/absorption/stripping (DeC1, DeC2, DeC3s)
    * Towers with pumparounds/efficiencies
    * Superfractionator example with C3= Splitter
    * Ejector
    * Heat Exchangers with multiple sides (Coldboxes)
    * Basic Controllers

* Cubic EOS Thermodynamics provider is largely complete
    Pathological cases remain where flashes can fail
    Implement Newton based solver to speed up flashes
    No support for critical properties/phase envelopes yet

* Tower/Distillation Column model is fully functional (Supporting Pumparounds, Bypass, Efficiencies and Side-Strippers)
* Some reactor models (Equilibrium/PFR) work partially
* Some unusual behavior especially when changing components on the fly
* Cleaned up some exception handling

TODO
----

* Support component addition/deletion on the fly
* Cleanup command line interface
* Provide Python based alternate interface to run simulator to integrate better with Python ecosystem
* Revamp web/HTML interface to use for models

Features
--------

* Updated for Python 3.x and removed major incompatibilities
* Removed obsolete references and updated to use modern replacements (numpy, scipy)
* Thermodynamics:
    * Equation of State (EOS)
        * Peng-Robinson (PR) /Soave-Redlich-Kwong (RK)
    * IAPWS97 implementation based on `XSteam <https://github.com/KurtJacobson/XSteam>`_
    * Two-phase flashes for common specifications using inside-out method
* Builtin database of 400+ common components using `ChemSep <http://www.chemsep.com/>`_ database


Next Steps
----------

* Fully implement basic thermodynamic provider with component database
* Massive code cleanup (Use snake_case instead of CamelCase, Exception Handling)

Credits
-------

* Kiran Pashikanti <kpatvt@gmail.com>

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
