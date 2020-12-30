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
f.Port.MoleFlow = 3445.0
f.Port.Fraction = 358.2 4965.2 339.4 2995.5 2395.5 2291 604.1 1539.9 790.4 1129.9 1764.7 2606.7 1844.5 1669 831.7 1214.5

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

