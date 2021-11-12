# Perry Ex03

units Field
$thermo = Sim21Thermo.Peng-Robinson
/ -> $thermo

thermo + PROPANE ISOBUTANE N-BUTANE ISOPENTANE N-PENTANE

col = Tower.Tower()
col.Stage_0
col.Stage_0 + 9  # 11 stage


cd col.Stage_0

cond = Tower.EnergyFeed(0)

l = Tower.LiquidDraw()

l.Port.P = 120
l.Port.MoleFlow = 48.09

reflux = Tower.StageSpecification('Reflux')
reflux.Value = 3.64

cd /


cd col.Stage_6

f = Tower.Feed()
f.Port.T = 180
f.Port.P = 120
f.Port.MoleFlow = 100
f.Port.Fraction = 5 15 25 20 35

cd /


cd col.Stage_10

l = Tower.LiquidDraw()
l.Port.P = 120

reb = Tower.EnergyFeed(1)

cd ..

TryToSolve = 1  # start calculation - turned off by default

/overhead = Stream.Stream_Material()
/overhead.In -> Stage_0.l.Port

/bottoms = Stream.Stream_Material()
/bottoms.In -> Stage_10.l.Port

cd /

overhead.Out
bottoms.Out
