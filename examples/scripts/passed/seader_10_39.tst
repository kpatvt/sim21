# Seader example problem, 10_39 (from 2nd Ed.)

units Field
$thermo = Sim21Thermo.Peng-Robinson
/ -> $thermo
thermo + METHANE ETHANE PROPANE N-BUTANE N-PENTANE N-DODECANE

col = Tower.Tower()
col.Stage_0
col.Stage_0 + 6  # eight stage


cd col.Stage_0

f = Tower.Feed()
f.Port.T = 90
f.Port.P = 400
f.Port.MoleFlow = 250
f.Port.Fraction = 0 0 0 0 0 1

v = Tower.VapourDraw()
v.Port.P = 400
#v.Port.T = 120.0

cd /



cd col.Stage_3

f = Tower.Feed()
f.Port.T = 90
f.Port.P = 400
f.Port.MoleFlow = 165
f.Port.Fraction = 13 3 4 5 5 135

cd /



cd col.Stage_6

duty1 = Tower.EnergyFeed(0)
duty1.Port.Energy = 1.250e6

cd /


cd col.Stage_7

f = Tower.Feed()
f.Port.T = 90
f.Port.P = 400
f.Port.MoleFlow = 450
f.Port.Fraction = 360 40 25 15 10 0

l = Tower.LiquidDraw()
l.Port.P = 400

cd ..

# Damping factor can help with solution, not required
# DampingFactor = 0.9
TryToSolve = 1  # start calculation - turned off by default


/overhead = Stream.Stream_Material()
/overhead.In -> Stage_0.v.Port

/bottoms = Stream.Stream_Material()
/bottoms.In -> Stage_7.l.Port

cd ..

overhead.Out
bottoms.Out


