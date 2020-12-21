# Depeopanizer test (from old Hysim manual)
units Field
$thermo = VirtualMaterials.Peng-Robinson
/ -> $thermo
thermo + Methane Ethane PROPANE
thermo + ISOBUTANE n-BUTANE ISOPENTANE n-PENTANE n-Hexane
thermo + n-Heptane n-Octane

deprop = Tower.Tower()
deprop.Stage_0 + 18  # twenty stages`

cd deprop.Stage_0

v = Tower.VapourDraw()
v.Port.P = 200
v.Port.Fraction.ISOBUTANE = .01

cond = Tower.EnergyFeed(0)
#cond.Port.Energy = 1.667e6

#estReflux = Tower.Estimate('Reflux')
#estReflux.Value = .45
estT = Tower.Estimate('T')
estT.Value = 25

#reflux = Tower.StageSpecification('Reflux')
#reflux.Value = .5042

cd ../Stage_9
f = Tower.Feed()
f.Port.T = 50
f.Port.P = 480
f.Port.MoleFlow = 1000
f.Port.Fraction = .1702 .1473 .1132 .1166 .1066 .0963 .0829 .0694 .0558 .0417
f.Port

#estT = Tower.Estimate('T')
#estT.Value = 100

cd ../Stage_19
l = Tower.LiquidDraw()
l.Port.P = 205
l.Port.Fraction.PROPANE = .02

reb = Tower.EnergyFeed(1)
#reb.Port.Energy = 8.42e6
estT = Tower.Estimate('T')
estT.Value = 250

cd ..

/overhead = Stream.Stream_Material()
/overhead.In -> Stage_0.v.Port

/bottoms = Stream.Stream_Material()
/bottoms.In -> Stage_19.l.Port



TryToSolve = 1  # start calculation

/overhead.Out
/bottoms.Out


#Now lets do some vol flow specs
commonproperties + VolumeFlow StdLiqMolarVol
displayproperties + StdLiqMolarVol StdLiqVolumeFlow

TryToSolve = 1 
TryToRestart = 1

#Delete fractions
/deprop.LiquidDraw_19_l.Fraction = None None None None None None None None None None
/deprop.LiquidDraw_19_l.StdLiqVolumeFlow = 0.304
/overhead.Out
/bottoms.Out

/deprop.LiquidDraw_19_l.StdLiqVolumeFlow =
/deprop.VapourDraw_0_v.VolumeFlow = 2.0
/overhead.Out
/bottoms.Out

TryToRestart = 1  #Keep last solution and ramp it up
/deprop.VapourDraw_0_v.VolumeFlow = 2.8
/overhead.Out
/bottoms.Out


/deprop.VapourDraw_0_v.VolumeFlow = 2.0
cd /

#Now lets play with re-naming
/deprop.Stage_0.v.NewName = VapDist

#will not be there
/deprop.Stage_0.v

#Should be there
/deprop.Stage_0.VapDist

#Should not allow for repeated names in the same stage
/deprop.Stage_0.VapDist.NewName = cond

#Now rename to a name of a stream. It should be able to handle
#a name equal to another unit operation
/deprop.Stage_0.VapDist.NewName = overhead

/deprop


copy /deprop /bottoms /overhead
paste /

/bottoms.In
/bottomsClone.In




