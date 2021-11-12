# TEALARC Liquefaction process

units SI

# set up thermo
$thermo = Sim21Thermo.Peng-Robinson

/ -> $thermo
thermo + METHANE ETHANE PROPANE N-BUTANE NITROGEN

# Start with the feed composition
ngfeed1 = Stream.Stream_Material()
ngfeed1.In.T = 30 C
ngfeed1.In.P = 4000 kPa
ngfeed1.In.MoleFlow = 36700 kgmole/h
ngfeed1.In.Fraction = 0.897 0.055 0.018 0.001 0.029
ngfeed1.Out

# This is the final condition we want the ng in
ngfinal = Stream.Stream_Material()
ngfinal.In.VapFrac = 0
ngfinal.In.T = -163 C

# Can use this to get the overall the heat curve
cool_ng = Heater.Cooler()
cool_ng.In -> ngfeed1.Out
cool_ng.Out -> ngfinal.In

# What's the pressure of the final product?
ngfinal.Out



