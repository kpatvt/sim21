# Seader example problem, 10_37 (from 2nd Ed.)
# Sucessfully solves
units Field
$thermo = Sim21Thermo.Peng-Robinson
/ -> $thermo
thermo + METHANE ETHANE PROPANE N-BUTANE N-PENTANE N-DECANE

col = Tower.Tower()
col.Stage_0
col.Stage_0 + 2  # four stage


cd col.Stage_0

f = Tower.Feed()
f.Port.T = 90
f.Port.P = 75
f.Port.MoleFlow = 1000
f.Port.Fraction = 0 0 0 0 0 1

v = Tower.VapourDraw()
v.Port.P = 75

cd /


cd col.Stage_3

f = Tower.Feed()
f.Port.T = 90
f.Port.P = 75
f.Port.MoleFlow = 1000
f.Port.Fraction = 286 157 240 169 148 0

l = Tower.LiquidDraw()
l.Port.P = 75

cd ..

TryToSolve = 1  # start calculation - turned off by default


/overhead = Stream.Stream_Material()
/overhead.In -> Stage_0.v.Port

/bottoms = Stream.Stream_Material()
/bottoms.In -> Stage_3.l.Port

cd ..

overhead.Out
bottoms.Out


