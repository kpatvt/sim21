# Perry Ex04

units Field
$thermo = Sim21Thermo.Peng-Robinson
/ -> $thermo

thermo + ETHANE PROPANE N-BUTANE N-PENTANE N-HEXANE

col = Tower.Tower()
col.Stage_0
col.Stage_0 + 16  # 18 stage


cd col.Stage_0

cond = Tower.EnergyFeed(0)

v = Tower.VapourDraw()

v.Port.P = 250

reflux = Tower.StageSpecification('Reflux')
reflux.Value = 6.52

cd /


cd col.Stage_13

v = Tower.VapourDraw()
v.Port.MoleFlow = 37

cd /


cd col.Stage_9

f = Tower.Feed()
f.Port.T = 213.9
f.Port.P = 260
f.Port.MoleFlow = 100
f.Port.Fraction = 0.03 0.2 0.37 0.35 0.05

cd /


cd col.Stage_17

l = Tower.LiquidDraw()
l.Port.P = 260
l.Port.MoleFlow = 40

reb = Tower.EnergyFeed(1)

cd ..

TryToSolve = 1  # start calculation - turned off by default

/overhead = Stream.Stream_Material()
/overhead.In -> Stage_0.v.Port

/bottoms = Stream.Stream_Material()
/bottoms.In -> Stage_17.l.Port

/draw1 = Stream.Stream_Material()
/draw1.In -> Stage_13.v.Port


cd /

overhead.Out
draw1.Out
bottoms.Out
