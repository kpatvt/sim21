# Seader example problem, 10_40 (from 2nd Ed.)

units Field
$thermo = Sim21Thermo.Peng-Robinson
/ -> $thermo
thermo + METHANE ETHANE PROPANE ISOBUTANE N-BUTANE ISOPENTANE N-PENTANE N-HEXANE N-DODECANE

col = Tower.Tower()
col.Stage_0
col.Stage_0 + 14  # 16 stage


cd col.Stage_0

f = Tower.Feed()
f.Port.T = 60
f.Port.P = 230
f.Port.MoleFlow = 58
f.Port.Fraction = 0 0 0 0 0 0 0 0 1

v = Tower.VapourDraw()
v.Port.P = 230
# v.Port.T = 74
v.Port.MoleFlow = 70

estT = Tower.Estimate('T')
estT.Value = 100

cd /


cd col.Stage_8

f = Tower.Feed()
f.Port.T = 120
f.Port.P = 230
f.Port.MoleFlow = 419
f.Port.Fraction = 46 42 66 13 49 11 20 24 148

cd /

cd col.Stage_12

# duty1 = Tower.EnergyFeed(1)
# duty1.Port.Energy = 1.5e6


cd /



cd col.Stage_15

reb = Tower.EnergyFeed(1)
# reb.Port.Energy = 1.50e6

l = Tower.LiquidDraw()
l.Port.P = 230

estT = Tower.Estimate('T')
estT.Value = 300


cd ..

# DampingFactor = 0.9
# MaxOuterLoops = 100
TryToSolve = 1  # start calculation - turned off by default


/overhead = Stream.Stream_Material()
/overhead.In -> Stage_0.v.Port

/bottoms = Stream.Stream_Material()
/bottoms.In -> Stage_15.l.Port

cd ..

overhead.Out
bottoms.Out

