# Example of a steam cycle using the steam package

units SI

# set up thermo
$thermo = Sim21Thermo.Steam97

/ -> $thermo
thermo + WATER

# Let's specify a sat'd vapor from the boiler
boiler_out = Stream.Stream_Material()
boiler_out.In.VapFrac = 1
boiler_out.In.P = 125.0 bar
boiler_out.In.Fraction = 1
boiler_out.Out

# Super heat it a bit
sup_heat = Heater.Heater()
boiler_out.Out -> sup_heat.In
sup_heat.DeltaP = 0
sup_heat.Out.T = 335.0
sup_heat.Out

# Now setup a Expander
exp_in = Stream.Stream_Material()
sup_heat.Out -> exp_in.In

exp = Compressor.Expander()
exp.Efficiency = .75
exp.Out.P = 0.1 bar
exp.In -> exp_in.Out

# Echo the expander properties
exp.Out

# Now let's condense it all
cond = Heater.Cooler()
exp.Out -> cond.In
cond.DeltaP = 0
cond.Out.VapFrac = 0
cond.Out

# Now let's pump it up to go back to boiler
p1 = Pump.Pump()
cond.Out -> p1.In
p1.Out.P = 125.0 bar
p1.Efficiency = .75
p1.Out

# Now we heat it up!
boil = Heater.Heater()
p1.Out -> boil.In
boil.Out -> boiler_out.In

# boil.Out and boiler_out.In should be the same
boil.Out

# Now the loop is fully specified ... except for the flows
# Let's say we want 5 MW of energy from the turbine,
# What's it the flow required?
# So we just specify the Power generated
exp.OutQ.Energy = 5e6 W

# And now the flow rates should be calculated
exp.Out

# Duties for the boiler, condenser and pump should be available
boil.InQ
cond.OutQ
p1.InQ

# No Design specs required to a basic calc ...
# We'd like to hit 30 C on the expander outlet temp
# Let's try a few manual approaches

exp.Out.P = 0.07 bar
exp.Out

exp.Out.P = 0.03 bar
exp.Out

exp.Out.P = 0.045 bar
exp.Out

exp.Out.P = 0.0425 bar
exp.Out

# That's close enough, ideally you would use a controller here..
