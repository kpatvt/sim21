# A Simple Air Separation Unit Example
# Similar to example found on ChemSep website
# Converges quickly, but a good estimate of the tear is required since it's a coupled column

units SI

# set up thermo
$thermo = Sim21Thermo.Peng-Robinson

/ -> $thermo
thermo + NITROGEN OXYGEN ARGON

/MaxNumIterations = 50
# /RecycleDetails = 1

air_feed = Stream.Stream_Material()
air_feed.In.T = 30 C
air_feed.In.P = 1.01315 bar
air_feed.In.MassFlow = 100 kg/s
air_feed.In.Fraction = 0.7812 0.2095 0.0093
air_feed.Out

comp1 = Compressor.Compressor()
air_feed.Out -> comp1.In
comp1.Out.P = 1.98 bar
comp1.Efficiency = .8

h1 = Heater.Heater()
comp1.Out -> h1.In
h1.DeltaP.DP = 0.2 bar
h1.Out.T = 40 C

comp2 = Compressor.Compressor()
h1.Out -> comp2.In
comp2.Out.P = 3.46 bar
comp2.Efficiency = .8

h2 = Heater.Heater()
comp2.Out -> h2.In
h2.DeltaP.DP = 0.2 bar
h2.Out.T = 40 C

comp3 = Compressor.Compressor()
h2.Out -> comp3.In
comp3.Out.P = 6.35 bar
comp3.Efficiency = .8

h3 = Heater.Heater()
comp3.Out -> h3.In
h3.DeltaP.DP = 0.2 bar
h3.Out.T = 30 C

spl1 = Split.Splitter()
h3.Out -> spl1.In
spl1.Out0.MassFlow = 85.0 kg/s
spl1.Out1

h4 = Heater.Heater()
spl1.Out0 -> h4.In
h4.DeltaP.DP = 0.4 bar
h4.Out.T = -173.8 C

h5 = Heater.Heater()
spl1.Out1 -> h5.In
h5.DeltaP.DP = 0.2 bar
h5.Out.T = -133.2 C

h4.Out
h5.Out

col1 = Tower.Tower()
col1.Stage_0 + 44  # 45 stages total

# Feed Stage is Btm Stage
cd col1.Stage_45
f = Tower.Feed()
/h4.Out -> f.Port

l = Tower.LiquidDraw()
l.Port.P = 5.7 bar
cd /

# Overhead Specs
cd col1.Stage_0
cond = Tower.EnergyFeed(0)
l = Tower.LiquidDraw()
l.Port.P = 5.6 bar
# l.Port.MassFlow = 23.1 kg/s
l.Port.Fraction.ARGON = 1e-7

# reflux = Tower.StageSpecification('Reflux')
# reflux.Value = 20
cd ..

TryToSolve = 1

/ovhd_liq_col1 = Stream.Stream_Material()
/ovhd_liq_col1.In -> Stage_0.l.Port

/btm_liq_col1 = Stream.Stream_Material()
/btm_liq_col1.In -> Stage_45.l.Port

cd /

ovhd_liq_col1.Out
btm_liq_col1.Out

jt_n2 = Valve.Valve()
ovhd_liq_col1.Out -> jt_n2.In
jt_n2.Out.P = 1.20 bar

jt_rl = Valve.Valve()
btm_liq_col1.Out -> jt_rl.In
jt_rl.Out.P = 1.30 bar

jt_n2.Out
jt_rl.Out

fl1 = Flash.SimpleFlash()
jt_rl.Out -> fl1.In
fl1.Vap
fl1.Liq0

exp1 = Compressor.Expander()
h5.Out -> exp1.In
exp1.Efficiency = .75
exp1.Out.P = 1.30 bar
exp1.Out

ovhd_liq_col3 = Stream.Stream_Material()
btm_liq_col3 = Stream.Stream_Material()


col2 = Tower.Tower()
col2.Stage_0 + 69  # 70 stages total

# Ovhd Stage
cd col2.Stage_0
v = Tower.VapourDraw()
v.Port.P = 1.20 bar
f = Tower.Feed()
/jt_n2.Out -> f.Port
estT = Tower.Estimate('T')
estT.Value = -194 C

cd /

# Side Draw 1
cd col2.Stage_10
v = Tower.VapourDraw()
v.Port.MoleFlow = 2.4 kgmole/s
cd /

# Side Draw 2
cd col2.Stage_57
v = Tower.VapourDraw()
v.Port.MoleFlow = 0.225 kgmole/s
# v.Port.Fraction.ARGON = 0.1

estT = Tower.Estimate('T')
estT.Value = -180 C

cd /

# Stage 56
cd col2.Stage_56
f = Tower.Feed()
/btm_liq_col3.Out -> f.Port
cd /


# Stage 19
cd col2.Stage_19
f = Tower.Feed()
/exp1.Out -> f.Port
cd /

# Different stages just for testing
# Stage 27
cd col2.Stage_27
f = Tower.Feed()
/fl1.Vap -> f.Port
cd /

# Stage 28
cd col2.Stage_28
f = Tower.Feed()
/fl1.Liq0 -> f.Port
cd /


# Btm Stage
cd col2.Stage_70
l = Tower.LiquidDraw()
l.Port.P = 1.35 bar
# l.Port.MassFlow = 4.7 kg/s
l.Port.Fraction.OXYGEN = 0.99
reb = Tower.EnergyFeed(1)

estT = Tower.Estimate('T')
estT.Value = -180 C

cd /

vap_draw1_col2 = Stream.Stream_Material()
vap_draw2_col2 = Stream.Stream_Material()

# Setup ovhd and liquid streams
cd col2
/ovhd_vap_col2 = Stream.Stream_Material()
/ovhd_vap_col2.In -> Stage_0.v.Port

/vap_draw1_col2.In -> Stage_10.v.Port
/vap_draw2_col2.In -> Stage_57.v.Port

/btm_liq_col2 = Stream.Stream_Material()
/btm_liq_col2.In -> Stage_70.l.Port

# DampingFactor = 0.8
TryToSolve = 1
cd ..

ovhd_vap_col2.Out
btm_liq_col2.Out

# Provide initial estimates for recycle
vap_draw2_col2.In.P ~= 132.214 kPa
vap_draw2_col2.In.VapFrac ~= 1
vap_draw2_col2.In.Fraction ~= 0.0 0.92 0.08
vap_draw2_col2.In.MoleFlow ~= 0.225 kgmole/s
vap_draw2_col2.Out


col3 = Tower.Tower()
col3.Stage_0 + 119  # 70 stages total

# Ovhd Stage
cd col3.Stage_0
l = Tower.LiquidDraw()
l.Port.P = 1.20 bar
cond = Tower.EnergyFeed(0)
# reflux = Tower.StageSpecification('Reflux')
# reflux.Value = 33.4
l.Port.Fraction.OXYGEN = 1e-6
cd /


# Btm Stage
cd col3.Stage_120
l = Tower.LiquidDraw()
l.Port.P = 1.30 bar
f = Tower.Feed()
/vap_draw2_col2.Out -> f.Port
cd /

# Setup ovhd and liquid streams
cd col3
/ovhd_liq_col3.In -> Stage_0.l.Port
/btm_liq_col3.In -> Stage_120.l.Port
TryToSolve = 1
cd ..

# Purified Nitrogen Product
ovhd_vap_col2.Out

# Purified Oxygen Product
btm_liq_col2.Out

# Pure Argon Product
ovhd_liq_col3.Out

# Liquid recycle back to Col2
btm_liq_col3.Out

# Vapor recycle to Col3
vap_draw2_col2.Out
