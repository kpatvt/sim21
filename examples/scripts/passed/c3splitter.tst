# C3 splitter test (from old Hysim manual)
units Field
$thermo = Sim21Thermo.Peng-Robinson
/ -> $thermo
thermo + PROPANE PROPYLENE

c3splitter = Tower.Tower()
c3splitter.Stage_0 + 183  # 185 stages`

cd c3splitter.Stage_0

l = Tower.LiquidDraw()
l.Port.P = 280
l.Port.Fraction.PROPANE = 0.004

cond = Tower.EnergyFeed(0)

estT = Tower.Estimate('T')
estT.Value = 114.8

# reflux = Tower.StageSpecification('Reflux')
# reflux.Value = 16.4

cd ../Stage_137
f = Tower.Feed()

cd ../Stage_184
l = Tower.LiquidDraw()
l.Port.P = 280

reb = Tower.EnergyFeed(1)
estT = Tower.Estimate('T')
estT.Value = 129.2
l.Port.Fraction.PROPYLENE = 0.015

cd /

feed = Stream.Stream_Material()
cd feed.In
P = 280
VapFrac = 1
MoleFlow = 1322.76
Fraction = 0.4 0.6
cd /c3splitter
/feed.Out -> Stage_137.f.Port
/feed.Out

/overhead = Stream.Stream_Material()
/overhead.In -> Stage_0.l.Port

/bottoms = Stream.Stream_Material()
/bottoms.In -> Stage_184.l.Port

TryToSolve = 1  # start calculation

/overhead.Out
/bottoms.Out
