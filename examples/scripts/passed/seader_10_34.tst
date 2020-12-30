# Seader example problem, 10_34 (from 2nd Ed.)
#
units Field
$thermo = Sim21Thermo.Peng-Robinson
/ -> $thermo

thermo + HYDROGEN METHANE ETHANE BENZENE TOLUENE M-XYLENE

col = Tower.Tower()
col.Stage_0
col.Stage_0 + 10  # 12 stage


cd col.Stage_0

cond = Tower.EnergyFeed(0)

v = Tower.VapourDraw()

v.Port.P = 128
# v.Port.MoleFlow = 51.0

v.Port.T = 99

reflux = Tower.StageSpecification('Reflux')
reflux.Value = 2.5

cd /


cd col.Stage_2

f = Tower.Feed()
f.Port.T = 240
f.Port.P = 275
f.Port.MoleFlow = 1748.4
f.Port.Fraction = 8.3 30.7 9.4 576.0 666.0 458.0

cd /

cd col.Stage_11

l = Tower.LiquidDraw()
l.Port.P = 132
#l.Port.Fraction.METHANE = 0.0005

reb = Tower.EnergyFeed(1)

cd ..

DampingFactor = 0.9
MaxOuterLoops = 100
TryToSolve = 1  # start calculation - turned off by default

/overhead = Stream.Stream_Material()
/overhead.In -> Stage_0.v.Port

/bottoms = Stream.Stream_Material()
/bottoms.In -> Stage_11.l.Port

cd /

overhead.Out
bottoms.Out
