# Cavett problem in Tower format
# Solves pretty quickly
units Field
$thermo = Sim21Thermo.Peng-Robinson
/ -> $thermo
thermo + NITROGEN CARBON_DIOXIDE HYDROGEN_SULFIDE
thermo + METHANE ETHANE PROPANE N-BUTANE ISOBUTANE N-PENTANE ISOPENTANE N-HEXANE N-HEPTANE N-OCTANE
thermo + N-NONANE N-DECANE N-DODECANE

col = Tower.Tower()
col.Stage_0
col.Stage_0 + 2  # four stage



cd col.Stage_0
v = Tower.VapourDraw()
v.Port.P = 814.7
v.Port.T = 100
cond = Tower.EnergyFeed(0)
cd /


cd col.Stage_1
duty1 = Tower.EnergyFeed(0)

v = Tower.VapourDraw()
v.Port.MoleFlow = 1e-8
v.Port.T = 120
v.Port.P = 284.7

f = Tower.Feed()
f.Port.T = 120
f.Port.P = 284.7
f.Port.MoleFlow = 27340.6
f.Port.Fraction = 0.0131 0.1816 0.0124 0.1096 0.0876 0.0838 0.0221 0.0563 0.0289 0.0413 0.0645 0.0953 0.0675 0.0610 0.0304 0.0444

cd /


cd col.Stage_2
duty2 = Tower.EnergyFeed(0)

v = Tower.VapourDraw()
v.Port.MoleFlow = 1e-8
v.Port.T = 96
v.Port.P = 63.7
cd /


cd col.Stage_3
l = Tower.LiquidDraw()
l.Port.T = 85.0
l.Port.P = 27.7
reb = Tower.EnergyFeed(1)

cd /


cd col
TryToSolve = 1  # start calculation - turned off by default


/overhead = Stream.Stream_Material()
/overhead.In -> Stage_0.v.Port

/bottoms = Stream.Stream_Material()
/bottoms.In -> Stage_3.l.Port


/overhead.Out
/bottoms.Out

