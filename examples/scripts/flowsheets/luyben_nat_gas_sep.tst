# Luyben problem - Get initial estimates with reflux ratios to eventually target composition specs

units SI

# set up thermo
$thermo = Sim21Thermo.Peng-Robinson
/ -> $thermo
thermo + ETHANE PROPANE ISOBUTANE N-BUTANE ISOPENTANE N-PENTANE

nat_gas_feed = Stream.Stream_Material()
nat_gas_feed.In.T = 378 K
nat_gas_feed.In.P = 1763.055
nat_gas_feed.In.MoleFlow = 12033
nat_gas_feed.In.Fraction = .0005 0.332 0.3583 0.1543 0.103 0.0518
nat_gas_feed.Out



col1 = Tower.Tower()
col1.Stage_0 + 48  # twenty two stages`

cd col1.Stage_24
f = Tower.Feed()
/nat_gas_feed.Out -> f.Port

cd ../Stage_0
l = Tower.LiquidDraw()
l.Port.P = 1722.525
l.Port.MoleFlow = 3993.34
# l.Port.Fraction.ISOBUTANE = 0.001


cond = Tower.EnergyFeed(0)

reflux = Tower.StageSpecification('Reflux')
reflux.Value = 3.0

# estT = Tower.Estimate('T')
# estT.Value = 322

cd ../Stage_49
l = Tower.LiquidDraw()
l.Port.P = 1773.1875
#l.Port.MoleFlow = 142.53
#l.Port.Fraction.PROPANE = 0.001

reb = Tower.EnergyFeed(1)

# estT = Tower.Estimate('T')
# estT.Value = 322

cd ..

# DampingFactor = 0.9
TryToSolve = 1  # start calculation

# First pass, solve with reflux ratio of 3

/overhead_col1 = Stream.Stream_Material()
/overhead_col1.In -> Stage_0.l.Port

/bottoms_col1 = Stream.Stream_Material()
/bottoms_col1.In -> Stage_49.l.Port

cd /

overhead_col1.Out
bottoms_col1.Out

col1.Stage_0.reflux =
col1.Stage_0.reflux = Tower.StageSpecification('Reflux')

# First pass, solve with reflux ratio of 3
col1.Stage_0.reflux.Value = 10.0


col1.TryToSolve = 0
col1.Stage_0.reflux =
col1.Stage_0.l.Port.MoleFlow =

col1.Stage_0.l.Port.Fraction.ISOBUTANE = 0.0001
col1.Stage_49.l.Port.Fraction.PROPANE = 0.0001

col1.TryToSolve = 1

overhead_col1.Out
bottoms_col1.Out



v1 = Valve.Valve()
bottoms_col1.Out -> v1.In
v1.Out.P = 739.67

v1.Out


feed_col2 = Stream.Stream_Material()
v1.Out -> feed_col2.In


col2 = Tower.Tower()
col2.Stage_0 + 29  # twenty two stages`

cd col2.Stage_16

f = Tower.Feed()
/feed_col2.Out -> f.Port

cd ../Stage_0
l = Tower.LiquidDraw()
l.Port.P = 719.0
l.Port.MoleFlow = 6000

cond = Tower.EnergyFeed(0)

reflux = Tower.StageSpecification('Reflux')
reflux.Value = 15.0

cd ../Stage_30
l = Tower.LiquidDraw()
l.Port.P = 750.0

reb = Tower.EnergyFeed(1)

cd ..

# DampingFactor = 0.9
TryToSolve = 1  # start calculation

/overhead_col2 = Stream.Stream_Material()
/overhead_col2.In -> Stage_0.l.Port

/bottoms_col2 = Stream.Stream_Material()
/bottoms_col2.In -> Stage_30.l.Port

cd /

overhead_col2.Out
bottoms_col2.Out

col2.TryToSolve = 0
col2.Stage_0.reflux =
col2.Stage_0.l.Port.MoleFlow =

col2.Stage_0.l.Port.Fraction.ISOPENTANE = 0.00001
col2.Stage_30.l.Port.Fraction.N-BUTANE = 0.001

col2.DampingFactor = 1.0
col2.TryToSolve = 1

overhead_col2.Out
bottoms_col2.Out


feed_col3 = Stream.Stream_Material()
overhead_col2.Out -> feed_col3.In



col3 = Tower.Tower()
col3.Stage_0 + 79  # twenty two stages`

cd col3.Stage_40

f = Tower.Feed()
/feed_col3.Out -> f.Port

cd ../Stage_0
l = Tower.LiquidDraw()
l.Port.P = 668.0
l.Port.MoleFlow = 3000

cond = Tower.EnergyFeed(0)

reflux = Tower.StageSpecification('Reflux')
reflux.Value = 2.0

cd ../Stage_80
l = Tower.LiquidDraw()
l.Port.P = 719.0

reb = Tower.EnergyFeed(1)

cd ..

# DampingFactor = 0.9
TryToSolve = 1  # start calculation

/overhead_col3 = Stream.Stream_Material()
/overhead_col3.In -> Stage_0.l.Port

/bottoms_col3 = Stream.Stream_Material()
/bottoms_col3.In -> Stage_80.l.Port

cd /

overhead_col3.Out
bottoms_col3.Out


col3.TryToSolve = 0
col3.Stage_0.reflux =
col3.Stage_0.l.Port.MoleFlow =

col3.Stage_0.l.Port.Fraction.N-BUTANE = 0.0001
col3.Stage_80.l.Port.Fraction.ISOBUTANE = 0.0001

# col3.DampingFactor = 1.0
col3.TryToSolve = 1

overhead_col3.Out
bottoms_col3.Out
