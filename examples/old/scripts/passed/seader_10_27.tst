# Seader example problem, 10_33 (from 2nd Ed.)
#
units Field
$thermo = Sim21Thermo.Peng-Robinson
/ -> $thermo

thermo + N-BUTANE N-PENTANE N-HEXANE N-OCTANE

col = Tower.Tower()
col.Stage_0
col.Stage_0 + 28  # 28 stage


cd col.Stage_0

cond = Tower.EnergyFeed(0)

l = Tower.LiquidDraw()

l.Port.P = 20
l.Port.MoleFlow = 14.08

reflux = Tower.StageSpecification('Reflux')
reflux.Value = 20

cd /


cd col.Stage_10

l = Tower.LiquidDraw()
l.Port.MoleFlow = 19.53

cd /

cd col.Stage_24

v = Tower.VapourDraw()
v.Port.MoleFlow = 24.78

cd /




cd col.Stage_14

f = Tower.Feed()
f.Port.T = 150
f.Port.P = 25
f.Port.MoleFlow = 97.8
f.Port.Fraction = 14.08 19.53 24.78 39.4

cd /


cd col.Stage_29

l = Tower.LiquidDraw()
l.Port.P = 25

reb = Tower.EnergyFeed(1)

cd ..

# DampingFactor = 0.9
# MaxOuterLoops = 100
TryToSolve = 1  # start calculation - turned off by default

/overhead = Stream.Stream_Material()
/overhead.In -> Stage_0.l.Port

/bottoms = Stream.Stream_Material()
/bottoms.In -> Stage_29.l.Port


/draw1 = Stream.Stream_Material()
/draw1.In -> Stage_10.l.Port

/draw2 = Stream.Stream_Material()
/draw2.In -> Stage_24.v.Port


cd /

overhead.Out
draw1.Out
draw2.Out
bottoms.Out
