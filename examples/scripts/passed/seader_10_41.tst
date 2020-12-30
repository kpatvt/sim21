# Seader example problem, 10_41 (from 2nd Ed.)

units Field
$thermo = Sim21Thermo.Peng-Robinson
/ -> $thermo
thermo + NITROGEN METHANE ETHANE PROPANE N-BUTANE N-PENTANE N-HEXANE

col = Tower.Tower()
col.Stage_0
col.Stage_0 + 6  # eight stage


cd col.Stage_0

f = Tower.Feed()
f.Port.T = 29.2
f.Port.P = 150
f.Port.MoleFlow = 551.59
f.Port.Fraction = 0.22 59.51 73.57 153.22 173.22 58.22 33.63

v = Tower.VapourDraw()
v.Port.P = 150

cd /


cd col.Stage_7

reb = Tower.EnergyFeed(1)

l = Tower.LiquidDraw()
l.Port.P = 150
# l.Port.MoleFlow = 99.33
l.Port.Fraction.PROPANE = 0.0001
cd ..

TryToSolve = 1  # start calculation - turned off by default


/overhead = Stream.Stream_Material()
/overhead.In -> Stage_0.v.Port

/bottoms = Stream.Stream_Material()
/bottoms.In -> Stage_7.l.Port

cd ..

overhead.Out
bottoms.Out

