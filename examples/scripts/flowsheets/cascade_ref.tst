# Cascade refrigeration example

units SI

# set up thermo

$thermo = Sim21Thermo.Peng-Robinson
/ -> $thermo
thermo + METHANE ETHYLENE PROPYLENE

# The Propylene Loop

c3_cond = Heater.Cooler()
c3_cond.Out.T = 35 C
c3_cond.DeltaP.DP = 0
c3_cond.Out.VapFrac = 0
c3_cond.Out.Fraction = 0 0 1
c3_cond.Out

c3_vlv = Valve.Valve()
c3_vlv.In -> c3_cond.Out
c3_vlv.Out.T = -30 C
c3_vlv.Out

c3_htr = Heater.Heater()
c3_htr.In -> c3_vlv.Out
c3_htr.DeltaP = 0
c3_htr.Out.VapFrac = 1
c3_htr.Out

c3_compr1 = Compressor.Compressor()
c3_compr1.In -> c3_htr.Out
c3_compr1.Efficiency = .8
c3_compr1.Out.P = 530 kPa
c3_compr1.Out

c3_compr2 = Compressor.Compressor()
c3_compr2.In -> c3_compr1.Out
c3_compr2.Efficiency = .8
c3_compr2.Out -> c3_cond.In

c3_compr2.Out

# The loop is fully defined, now specifying the duty required
# in c3_htr will fully specify the loop
c3_htr.InQ.Energy

# Specify the Duty
c3_htr.InQ.Energy = 1e3 W

# Now the loop should be solved
c3_cond.Out



# The ethylene loop
c2_cond = Heater.Cooler()
c2_cond.Out.T = -25 C
c2_cond.DeltaP.DP = 0
c2_cond.Out.VapFrac = 0
c2_cond.Out.Fraction = 0 1 0
c2_cond.Out

c2_vlv = Valve.Valve()
c2_vlv.In -> c2_cond.Out
c2_vlv.Out.T = -105 C
c2_vlv.Out

c2_htr = Heater.Heater()
c2_htr.In -> c2_vlv.Out
c2_htr.DeltaP = 0
c2_htr.Out.VapFrac = 1
c2_htr.Out

c2_compr1 = Compressor.Compressor()
c2_compr1.In -> c2_htr.Out
c2_compr1.Efficiency = .8
c2_compr1.Out.P = 1000 kPa
c2_compr1.Out

c2_compr2 = Compressor.Compressor()
c2_compr2.In -> c2_compr1.Out
c2_compr2.Efficiency = .8
c2_compr2.Out -> c2_cond.In

c2_compr2.Out

# The loop is fully defined, now specifying the duty required
# in c2_htr will fully specify the loop
c2_htr.InQ.Energy

# Specify the Duty
c2_htr.InQ.Energy = 1e3 W

# Now the loop should be solved
c2_cond.Out

# The methane loop
c1_cond = Heater.Cooler()
c1_cond.Out.T = -100 C
c1_cond.DeltaP.DP = 0
c1_cond.Out.VapFrac = 0
c1_cond.Out.Fraction = 1 0 0
c1_cond.Out

c1_vlv = Valve.Valve()
c1_vlv.In -> c1_cond.Out
c1_vlv.Out.T = -165 C
c1_vlv.Out

c1_htr = Heater.Heater()
c1_htr.In -> c1_vlv.Out
c1_htr.DeltaP = 0
c1_htr.Out.VapFrac = 1
c1_htr.Out

c1_compr1 = Compressor.Compressor()
c1_compr1.In -> c1_htr.Out
c1_compr1.Efficiency = .8
c1_compr1.Out.P = 1500 kPa
c1_compr1.Out

c1_compr2 = Compressor.Compressor()
c1_compr2.In -> c1_compr1.Out
c1_compr2.Efficiency = .8
c1_compr2.Out -> c1_cond.In

c1_compr2.Out

# The loop is fully defined, now specifying the duty required
# in c1_htr will fully specify the loop
c1_htr.InQ.Energy

# Specify the Duty
c1_htr.InQ.Energy = 1e3 W

# Now the loop should be solved
c1_cond.Out

# First delete the spec on the htrs in the C2, C3 levels
c2_htr.InQ.Energy = None
c3_htr.InQ.Energy = None

# Connect the heaters up
c2_htr.InQ -> c1_cond.OutQ
c3_htr.InQ -> c2_cond.OutQ


# Now let's see we're at
c1_compr1.TotalQ.Out
c1_compr2.TotalQ.Out

c2_compr1.TotalQ.Out
c2_compr2.TotalQ.Out

c3_compr1.TotalQ.Out
c3_compr2.TotalQ.Out

# Change the Duty at C1 level
c1_htr.InQ.Energy = 2e3 W

# Now let's see we're at
c1_compr1.TotalQ.Out
c1_compr2.TotalQ.Out

c2_compr1.TotalQ.Out
c2_compr2.TotalQ.Out

c3_compr1.TotalQ.Out
c3_compr2.TotalQ.Out

# 2x the Duty! - not surprising, how about we change the efficiencies of the compressors at the C1 level
hold
c1_compr1.Efficiency = .9
c1_compr2.Efficiency = .9
go

# Now let's see we're at, a decent improvement
c1_compr1.TotalQ.Out
c1_compr2.TotalQ.Out

c2_compr1.TotalQ.Out
c2_compr2.TotalQ.Out

c3_compr1.TotalQ.Out
c3_compr2.TotalQ.Out

# Now what happens if we improve the efficiency at the C3 level but not the C1 level.
hold

# Go back the old values here
c1_compr1.Efficiency = .8
c1_compr2.Efficiency = .8

c3_compr1.Efficiency = .9
c3_compr2.Efficiency = .9

go

# Now let's see we're at
c1_compr1.TotalQ.Out
c1_compr2.TotalQ.Out

c2_compr1.TotalQ.Out
c2_compr2.TotalQ.Out

c3_compr1.TotalQ.Out
c3_compr2.TotalQ.Out
