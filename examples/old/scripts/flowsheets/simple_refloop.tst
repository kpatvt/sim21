# A simple refrigeration loop example
# This is refrigeration with a user at one level
# Demonstrates how a controller is not required to solve this problem

units Field
$thermo = Sim21Thermo.Peng-Robinson
/ -> $thermo
thermo + ETHANE PROPANE ISOBUTANE N-BUTANE


# s1 is the stream coming out the condenser
# Notice we only provide the composition of the stream and flash conditions
# Flow will be calculated backwards
s1 = Stream.Stream_Material()
s1.In.T = 140 F
s1.In.Fraction = 0.010 0.969 0.017 0.003
s1.In.VapFrac = 0

# Expansion Valve
s2 = Stream.Stream_Material()
v1 = Valve.Valve()
s1.Out -> v1.In
v1.Out -> s2.In
v1.Out.P = 88

# Chiller
s3 = Stream.Stream_Material()
h2 = Heater.Heater()
s2.Out -> h2.In
h2.DeltaP.DP = 5 psi
h2.Out.T = 44 F
s3.In -> h2.Out

# Flash
s4 = Stream.Stream_Material()
s5 = Stream.Stream_Material()

f1 = Flash.SimpleFlash()
s3.Out -> f1.In
s4.In -> f1.Vap
s5.In -> f1.Liq0

# Ref. User
s6 = Stream.Stream_Material()
h_user = Heater.Heater()
h_user.In -> s5.Out

# Specify the duty consumed by the User
# Typically this would require a controller to vary feed rate to get
# to this value, but since the solver works backwards, that is not
# needed
h_user.InQ.Energy = 82.74e6 Btu/hr
h_user.DeltaP.DP = 9 psi
h_user.Out.VapFrac = 1
h_user.Out -> s6.In

# Note that vapor stream coming out of the flash is calculated as well
s4.Out
s6.Out

# Now combine both streams and feed to compressor
m1 = Mixer.Mixer()
s4.Out -> m1.In0
s6.Out -> m1.In1

# Compressor, note we don't define the discharge pressure
# That will be calculated backward from the condenser
s8 = Stream.Stream_Material()
comp1 = Compressor.Compressor()
m1.Out -> comp1.In
comp1.Efficiency = .8
comp1.Out -> s8.In

# Condenser, we connect it back to stream S1 (which is defined at the bubble point at 140F)
# The bubble point pressure will then set the H1 pressure in and out which will fully define
# the compressor
h1 = Heater.Heater()
s8.Out -> h1.In
h1.DeltaP.DP = 5 psi
h1.Out -> s1.In

# Now show us what s8 (comp1 discharge) and comp1 work is is
s8.Out
comp1.TotalQ.Out

# The flowsheet is fully solved now ...
# Let's see what happens when efficiency of the compressor
comp1.Efficiency = .75

s8.Out
comp1.TotalQ.Out

# Surprise, duty goes up!

# Let's change the duty on the user and see what the results are
h_user.InQ.Energy = 41.37e6 Btu/hr

s8.Out
comp1.TotalQ.Out

