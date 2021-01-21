# conversion reactor example
# calculation of propane adiabatic flame temperature

# This example highlights a slightly tricky issue with the Ig Cp correlations
# Once the Cp correlations are extended out of their range, the values can go negative or be unrealistic

# This causes an unusual maxima in the enthalpy as a function of T at high temperatures, which causes
# enthalpy searches to fail since the enthalpy is no longer a monotonic function of T.

# The current fix in the thermo is to essentially extrapolate outside the valid range for each component
# Is that the best answer aside from getting more IG Cp data at high temperatures? Not sure.
# Without good quality IG Cp data, the calculations at high temp are suspect

$thermo = Sim21Thermo.SRK
/ -> $thermo
thermo + PROPANE OXYGEN NITROGEN CARBON_DIOXIDE WATER

units SI

#create reactor
r = ConvRxn.ConvReactor()
cd /r
In.P = 1 atm
In.T = 25 C
In.MoleFlow = 1
In.Fraction = 1 6 24 0 0
DeltaP = 0
OutQ = 0

#create reaction
NumberRxn = 1
Rxn0.Formula = PropaneAFT:4*WATER+3*"CARBON DIOXIDE"-!PROPANE-5*OXYGEN

Out

copy /r
paste /
/rClone.Out
