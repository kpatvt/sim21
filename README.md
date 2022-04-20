sim21
-----
Open source simulator for typical process/chemical/refinery engineering applications with rigorous thermodynamics.
Revival of legacy Sim42 simulator for Python 3 with custom thermodynamics.

* Learn more about legacy Sim42 https://web.archive.org/web/20050204025650/http://manual.sim42.org/
* Free software: MIT license
* Documentation: https://sim21.readthedocs.io.

Under continual development, breaking changes are to be expected.

Status
------
* Nov 11, 2021
    * Not dead yet - Just busy with other things
    * Code requires a significant rewrite
      * Most of the current code moved to a separate 'old' package.
      * All tests still pass

* Feb 2, 2021
    * Implemented Activity Coefficient Models: NRTL, Wilson with some sample interations

* Jan 23, 2021
    * Finished implementation of a seperate steam/water Thermo using IFC-97
    * Included example of a very simple steam cycle flowsheet
* Jan 21, 2021
    * Signficant speed ups in the distillation simulation, now quite fast
    * Other smaller speed-ups
    * Runtime is about 2-3x faster over all examples

* Jan 20, 2021
    * All example cases now converge with some massaging
    * Fixed a nasty bug that was preventing one phase results in Flash unitops

* Jan 12, 2021
    * Many more test cases working including PipeSegment
    * All column examples now work with different types of specs including recoveries,
      deg. of subcooling, etc.

Highlights
----------
* Independent steam model with high accuracy IFC-97 Steam model
    * Sample steam cycle included in flowsheet examples

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
    * Additional examples found where the PH-solver fails dramatically - Needs improvement.
    * Implement Newton based solver to speed up flashes
    * No support for critical properties/phase envelopes yet

* Tower/Distillation Column model is fully functional (Supporting Pumparounds, Bypass, Efficiencies and Side-Strippers)

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
    * IAPWS97 implementation based on https://github.com/KurtJacobson/XSteam
    * Two-phase flashes for common specifications using inside-out method
* Builtin database of 400+ common components using ChemSep: <http://www.chemsep.com/> database


Next Steps
----------

* Fully implement basic thermodynamic provider with component database
* Massive code cleanup (Use snake_case instead of CamelCase, Exception Handling)
