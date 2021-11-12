# Seader example problem, 10_36 (from 2nd Ed.)
units Field
$thermo = Sim21Thermo.Peng-Robinson
/ -> $thermo
thermo + METHANE ETHANE PROPANE N-BUTANE N-PENTANE N-DODECANE

col = Tower.Tower()
col.Stage_0
col.Stage_0 + 4  # 6 stage


cd col.Stage_0

f = Tower.Feed()
f.Port.T = 90
f.Port.P = 75
f.Port.MoleFlow = 1000
f.Port.Fraction = 0 0 0 0 0 1

v = Tower.VapourDraw()
v.Port.P = 75

cd /


cd col.Stage_5

f = Tower.Feed()
f.Port.T = 60
f.Port.P = 75
f.Port.MoleFlow = 2000
f.Port.Fraction = 0.83 0.084 0.048 0.026 0.012 0

l = Tower.LiquidDraw()
l.Port.P = 75

cd ..

# DampingFactor = 0.25
TryToSolve = 1  # start calculation - turned off by default


/overhead = Stream.Stream_Material()
/overhead.In -> Stage_0.v.Port

/bottoms = Stream.Stream_Material()
/bottoms.In -> Stage_5.l.Port

cd ..

/overhead.Out
/bottoms.Out
