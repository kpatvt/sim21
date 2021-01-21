# TEALARC Liquefaction process

units SI

# set up thermo
$thermo = Sim21Thermo.Peng-Robinson

/ -> $thermo
thermo + METHANE ETHANE PROPANE N-BUTANE NITROGEN


ng_feed = Stream.Stream_Material()
ng_feed.In.T = 30 C
ng_feed.In.P = 4000 kPa
ng_feed.In.MoleFlow = 36700 kgmole/h
ng_feed.In.Fraction = 0.897 0.055 0.018 0.001 0.029
ng_feed.Out


