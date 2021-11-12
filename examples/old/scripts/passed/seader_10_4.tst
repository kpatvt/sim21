# Seader example problem, 10_4 (from 2nd Ed.)
#
units Field
$thermo = Sim21Thermo.Peng-Robinson
/ -> $thermo
thermo + METHANE ETHANE PROPANE N-BUTANE N-PENTANE N-DECANE

col = Tower.Tower()
col.Stage_0
col.Stage_0 + 4  # six stage


cd col.Stage_0

f = Tower.Feed()
f.Port.T = 90
f.Port.P = 400
f.Port.MoleFlow = 900.0
f.Port.Fraction = 0 0 0 0.05 0.78 164.17

v = Tower.VapourDraw()
v.Port.P = 400

estT = Tower.Estimate('T')
estT.Value = 100

cd /


cd col.Stage_5

f = Tower.Feed()
f.Port.T = 105
f.Port.P = 400
f.Port.MoleFlow = 800
f.Port.Fraction = 160 370 240 25 5 0

l = Tower.LiquidDraw()
l.Port.P = 400

estT = Tower.Estimate('T')
estT.Value = 100


cd ..

DampingFactor = 1.0
TryToSolve = 1  # start calculation - turned off by default


/overhead = Stream.Stream_Material()
/overhead.In -> Stage_0.v.Port

/bottoms = Stream.Stream_Material()
/bottoms.In -> Stage_5.l.Port

cd /

overhead.Out
bottoms.Out
