from sim21.data.chemsep import casn_to_identifier
import re
import pprint

WILSON_IP_DATA = """
  67-64-1     7732-18-5    439.64     1405.49      Acetone/Water
  67-64-1       56-23-5    651.76      -12.67      Acetone/Carbon tetrachloride
  71-43-2       56-23-5   -103.41      204.82      Benzene/Carbon tetrachloride
  64-17-5     7732-18-5    382.30      955.45      Ethanol/Water
  67-56-1     7732-18-5    205.30      482.16      Methanol/Water
  75-05-8     7732-18-5    643.9541   1388.0606    Acetonitrile/Water p76 1/1a
  75-07-0     7732-18-5   -227.0345   1915.1080    Acetaldehyde/Water p86 1/1a
  75-21-8     7732-18-5   -100.6887   8741.2478    Ethyleneoxide/Water p88 1/1a
7732-18-5       64-19-7    801.0312      7.0955    Water/Acetic acid p100 1/1a
7732-18-5      123-39-7   2634.3466  -1361.2425    Water/n-Methylformamide p115 1/1a
7732-18-5       67-68-5   -479.7582  10775.2040    Water/Dimethylsulfoxide p165 1/1a
 124-40-3     7732-18-5   -135.6790    137.3556    Dimethylamine/Water p174 1/1a
  75-04-7     7732-18-5   1970.1607      8.5935    Ethylamine/Water p177 1/1a
7732-18-5      107-15-3   -230.1187   -780.4712    Water/Ethylenediamine p179 1/1a
 107-13-1     7732-18-5   1253.3617   1761.5532    Acrylonitrile/Water p182 1/1a
7732-18-5       79-10-7   1290.2479   -237.3703    Water/Acrylic acid p186 1/1a
 107-05-1     7732-18-5   1311.2013   2734.7500    3-chloro-1-propene/Water p188 1/1a
 107-12-0     7732-18-5   1331.0782   1767.8771    Propionitrile/Water p189 1/1a
 107-18-6     7732-18-5    547.5028   1131.3858    Allylalcohol/Water p208 1/1a
 123-38-6     7732-18-5   1166.4954   2119.7817    Propionicaldehyde/Water p209 1/1a
  79-20-9     7732-18-5    645.7225   1918.2320    Methylacetate/Water p214 1/1a
7732-18-5       79-09-4   1015.3651    758.4305    Water/Propionicacid p215 1/1a
7732-18-5      110-88-3   1816.6639    413.5527    Water/1,3,5-trioxane p225 1/1a
7732-18-5       68-12-2   1153.5074   -930.1771    Water/N,n-dimethylformamide p234 1/1a
  71-23-8     7732-18-5    775.4823   1351.8979    1-propanol/Water p240 1/1a
  67-63-0     7732-18-5    437.9789   1238.9966    2-propanol/Water p251 1/1a
 109-87-5     7732-18-5    737.6370   2351.7548    Dimethoxymethane/Water p252 1/1a
  75-31-0     7732-18-5   1119.3728    588.6799    Isopropylamine/Water p266 1/1a
 107-10-8     7732-18-5    324.3976    734.1443    Propylamine/Water p267 1/1a
  78-93-3     7732-18-5   6811.3433   1888.8509    2-Butanone/Water p279 1/1a
  78-84-2     7732-18-5  11499.3532   2468.8767    Isobutyraldehyde/Water p280 1/1a
 109-99-9     7732-18-5   1140.7179   1819.4033    Tetrahydrofuran/Water p286 1/1a
7732-18-5      123-91-1   1696.9831   -219.3897    Water/1,4-dioxane p303 1/1a
 141-78-6     7732-18-5   1741.9252   2377.9564    Ethylacetate/Water p304 1/1a
7732-18-5      126-33-0   1303.3407    257.4018    Water/Sulfolane p307 1/1a
7732-18-5      127-19-5   1367.6938   -947.8852    Water/N,n-dimethylacetamide p325 1/1a
7732-18-5      110-91-8   2578.6032  -1716.4844    Water/Morpholine p327 1/1a
7732-18-5       71-36-3   1565.1326   1809.8711    Water/n-butanol p336 1/1a
  78-83-1     7732-18-5  11814.8851   1643.6524    isobutanol/Water p339 1/1a
  75-65-0     7732-18-5   1197.1859   1414.6647    Tert-butanol/Water p343 1/1a
  60-29-7     7732-18-5   1351.9511   3706.1569    Diethylether/Water p344 1/1a
7732-18-5      110-63-4   1561.7932    215.7977    Water/1,4-butanediol p346 1/1a
7732-18-5      513-85-9    481.7082   1495.6890    Water/2,3-butanediol p351 1/1a
7732-18-5      111-46-6   1240.7215   -111.2383    Water/diethyleneglycol p352 1/1a
 109-73-9     7732-18-5   1445.7991   1007.3293    n-Butylamine/Water p354 1/1a
 109-89-7     7732-18-5    198.1828   1162.3436    Diethylamine/Water p355 1/1a
7732-18-5       98-01-1   1670.3195   1683.5106    Water/Furfural p361 1/1a
7732-18-5      110-86-1    966.4397   1214.8464    Water/Pyridine p374 1/1a
7732-18-5      120-94-5   1489.0589  -1202.5857    Water/n-Methylpyrrolidone p380 1/1a
7732-18-5      123-51-3   1299.8678   1955.4083    Water/3-methylbutanol p382 1/1a
7732-18-5       75-85-4   1946.2507   8884.9541    Water/2-methyl-2-butanol p384 1/1a
7732-18-5      108-95-2   1089.3663   2116.5925    Water/Phenol p399 1/1a
7732-18-5       62-53-3   1710.5187   9272.5895    Water/Aniline p401 1/1a
7732-18-5      109-06-8    804.6477   7509.1743    Water/2-methylpyridine p402 1/1a
7732-18-5      108-99-6   1268.5915   1852.9311    Water/3-methylpyridine p406 1/1a
7732-18-5      108-89-4    865.8870   9163.4352    Water/4-methylpyridine p408 1/1a
7732-18-5      100-63-0   1883.8725    341.5435    Water/Phenylhydrazine p409 1/1a
7732-18-5      108-94-1   1458.9779   4163.0033    Water/Cyclohexanone p410 1/1a
7732-18-5      141-79-7   2127.0781   1219.3478    Water/Mesityloxide p411 1/1a
7732-18-5      108-93-0   1849.5584  13413.5007    Water/Cyclohexanol p415 1/1a
7732-18-5      123-86-4   2296.5866   3215.3409    Water/n-Butylacetate p418 1/1a
7732-18-5      123-42-2   1605.1265    132.2708    Water/Diacetonealcohol p419 1/1a
 108-20-3     7732-18-5   2616.9493   3268.1641    Diisopropylether/Water p421 1/1a
7732-18-5      111-27-3   2035.9324   9281.0246    Water/1-hexanol p426 1/1a
7732-18-5      626-93-7   2210.7204   9285.8501    Water/2-hexanol p427 1/1a
7732-18-5      111-76-2   1312.9966  11433.9360    Water/2-butoxy-ethanol p433 1/1a
 121-44-8     7732-18-5  10667.6107   2207.4385    Triethylamine/Water p444 1/1a
7732-18-5      100-51-6   1700.3511  11127.7871    Water/Benzylalcohol p450 1/1a
7732-18-5      108-48-5   1105.3097   8192.2140    Water/2,6-dimethylpyridine p457 1/1a
7732-18-5      123-92-2   2622.5905   3410.8117    Water/Isopentylacetate p459 1/1a
7732-18-5       98-86-2   2225.8336   1803.2300    Water/Acetophenone p460 1/1a
7732-18-5       91-22-5    909.3783   3220.5700    Water/Quinoline p464 1/1a
7732-18-5       98-82-8   2978.1169   1684.8116    Water/Isopropylbenzene p465 1/1a
  67-56-1       56-23-5   2247.3808    134.4890    Methanol/Tetrachloromethane p18 1/2c
  67-56-1       75-25-2   1377.3042     81.6493    Methanol/Tribromomethane p19 1/2c
  75-09-2       67-56-1   -180.8154   2137.9296    Dichloromethane/Methanol p26 1/2c
  67-56-1       75-12-7    -51.6493    517.7874    Methanol/Formamide p30 1/2c
 151-67-7       67-56-1   -700.7422   2457.5998    Halothane/Methanol p33 1/2c
  75-01-4       67-56-1   -129.2581   8985.5209    VinylCloride/Methanol p36 1/2c
  67-56-1       75-05-8    504.3124    196.7513    Methanol/Acetonitrile p42 1/2c
  67-56-1      107-06-2   1546.6607     91.6875    Methanol/1,2-Dichloroethane p44 1/2c
  67-56-1       64-19-7    -93.6182    -67.5078    Methanol/AceticAcid p46 1/2c
 107-31-3       67-56-1    107.8642    898.8309    MethylFormate/Methanol p51 1/2c
  67-56-1       67-68-5    -75.9373   -575.1219    Methanol/Dimethylsulfoxide p61 1/2c
 124-40-3       67-56-1   -195.6061  -5997.6526    Dimethylamine/Methanol p64 1/2c
 123-38-6       67-56-1   -717.8108    523.3134    PropionicAldehyde/Methanol p79 1/2c
  75-56-9       67-56-1   -294.1276   1105.5422    PropyleneOxide/Methanol p80 1/2c
  79-20-9       67-56-1    -31.1923    813.1843    MethylAcetate/Methanol p86 1/2c
  67-56-1       79-09-4     -0.3127     65.5996    Methanol/Propionic Acid p87 1/2c
  67-56-1       68-12-2    546.3063   -799.3197    Methanol/n,n-Dimethylformamide p88 1/2c
  67-56-1       71-23-8    150.2057    -33.8819    Methanol/Propanol p96 1/2c
 109-87-5       67-56-1    131.9538    939.2873    Dimethoxymethane/Methanol p97 1/2c
  67-56-1      110-02-1   1466.1473   -262.0228    Methanol/Thiophene p101 1/2c
 106-99-0       67-56-1     36.1781   2193.5235    1,3-Butadiene/Methanol p103 1/2c
  67-56-1       96-33-3    810.8254    101.9879    Methanol/Methylacrylate p104 1/2c
 115-11-7       67-56-1    112.7779   2424.2111    Isobutylene/Methanol p105 1/2c
  67-56-1       78-93-3    767.5030   -217.5920    Methanol/2-Butanone p106 1/2c
  67-56-1      109-99-9    801.1134   -223.1602    Methanol/Tetrahydrofuran p111 1/2c
  67-56-1      123-91-1    650.0998     69.3923    Methanol/1,4-Dioxane p116 1/2c
  67-56-1      141-78-6   1010.9043   -185.8564    Methanol/Ethylacetate p117 1/2c
  67-56-1      126-33-0    901.6935    303.1540    Methanol/Sulfolane p120 1/2c
 106-97-8       67-56-1    382.3429   2283.8726    Butane/Methanol p126 1/2c
  67-56-1       71-36-3    198.6954     99.0138    Methanol/1-Butanol p127 1/2c
  67-56-1       78-92-2    -90.4522    599.7362    Methanol/2-Butanol p128 1/2c
  67-56-1       75-65-0    111.0867   -126.9594    Methanol/Tert-Butanol p131 1/2c
  60-29-7       67-56-1   -253.0028   1244.4807    DiethylEther/Methanol p133 1/2c
  67-56-1      110-63-4    229.8899   -248.6631    Methanol/1,4-Butanediol p138 1/2c
  67-56-1      110-86-1    528.0041   -490.0939    Methanol/Pyridine p141 1/2c
 542-92-7       67-56-1    218.7354   2387.2579    Cyclopentadiene/Methanol p144 1/2c
  78-79-5       67-56-1    -54.1512   2623.4303    Isoprene/Methanol p145 1/2c
1574-41-0       67-56-1    100.1977   2415.3970    1,3-Pentadiene(CIS)/Methanol p147 1/2c
2004-70-8       67-56-1    110.7037   2399.3197    Trans-1,3-Pentadiene/Methanol p148 1/2c
  67-56-1       80-62-6   1094.7076    -24.9618    Methanol/MethylMethacrylate p152 1/2c
 513-35-9       67-56-1    231.5055   2422.2404    2-Methyl-2-Butene/Methanol p154 1/2c
  67-56-1      563-80-4    789.4824   -135.4059    Methanol/Methylisopropyl Ketone p155 1/2c
  67-56-1       96-22-0    817.8733    -88.0385    Methanol/3-Pentanone p156 1/2c
  67-56-1      110-89-4   -543.9877   8759.3917    Methanol/Piperidine p157 1/2c
1634-04-4       67-56-1   -444.5442   1494.9731    MethylTert-ButylEther/Methanol p160 1/2c
  67-56-1      392-56-3   1995.3104    240.9048    Methanol/Hexafluorobenzene p170 1/2c
  67-56-1      108-90-7   2084.4677    171.0466    Methanol/Chlorobenzene p174 1/2c
  67-56-1       62-53-3    651.0320   -100.1009    Methanol/Aniline p192 1/2c
  67-56-1      109-06-8   -140.3975    541.0629    Methanol/2-Methylpyridine p195 1/2c
  67-56-1      108-99-6    431.3115   -676.9524    Methanol/3-Methylpyridine p196 1/2c
  67-56-1      108-89-4   -143.9626    504.3727    Methanol/4-Methylpyridine p198 1/2c
  67-56-1      110-83-8   2339.5453    527.7379    Methanol/Cyclohexene p204 1/2c
  67-56-1      110-82-7   2359.1063    743.3083    Methanol/Cyclohexane p211 1/2c
 592-41-6       67-56-1    472.7630   2045.1813    1-Hexene/Methanol p212 1/2c
  67-56-1      123-86-4   1174.8439   -326.8491    Methanol/n-ButylAcetate p217 1/2c
  67-56-1      110-54-3   2544.5566   1159.1406    Methanol/Hexane p224 1/2c
  67-56-1      121-44-8   1694.1288  -1038.9916    Methanol/Triethylamine p226 1/2c
  67-56-1      108-88-3   1794.6541    280.1956    Methanol/Toluene p236 1/2c
  67-56-1       95-48-7   1793.4221  -1585.1597    Methanol/2-Methylphenol p238 1/2c
  67-56-1      108-48-5    -97.0569    768.5834    Methanol/2,6-Dimethylpyridine p239 1/2c
  67-56-1      592-76-7   2073.2479    693.9372    Methanol/1-Heptene p241 1/2c
  67-56-1      108-87-2   2801.8090    643.4775    Methanol/Methylcyclohexane p242 1/2c
  67-56-1      142-82-5   2582.2740    835.5861    Methanol/n-Heptane p243 1/2c
  67-56-1      100-41-4   1881.0028    419.2436    Methanol/Ethylbenzene p245 1/2c
  67-56-1      108-38-3   1797.9693    501.6630    Methanol/m-xylene p246 1/2c
  67-56-1      106-42-3   1840.5073    467.4656    Methanol/p-xylene p247 1/2c
  67-56-1      111-66-0   2102.7026    735.1565    Methanol/1-Octene p248 1/2c
  67-56-1      111-65-9   2529.8138   1015.2716    Methanol/Octane p249 1/2c
  67-56-1      540-84-1   2449.6550    694.6127    Methanol/2,2,4-Trimethylpentane p250 1/2c
  67-56-1      112-53-8    679.7598    553.4018    Methanol/1-Dodecanol p252 1/2c
  67-56-1      112-80-1   1268.1406   -453.5497    Methanol/Oleic Acid p253 1/2c
  56-23-5       67-56-1     97.1460   1882.8322    Tetrachloromethane/Methanol p279 1/2c
  75-09-2       64-17-5      6.7977   1226.3325    Dichloromethane/Ethanol p284 1/2c
  64-17-5      127-18-4   1300.7306    452.2952    Ethanol/Tetrachloroethylene p285 1/2c
  64-17-5       75-05-8    338.1621    607.8083    Ethanol/Acetonitrile p289 1/2c
  64-17-5      107-06-2   1127.3485    311.5541    Ethanol/1,2-Dichloroethane p292 1/2c
  64-17-5       64-19-7   -183.5551    -38.3849    Ethanol/Acetic Acid p293 1/2c
  64-17-5      107-21-1   -129.2043   1539.4142    Ethanol/1,2-Ethanediol p297 1/2c
 124-40-3       64-17-5     54.1939  -1133.1788    Dimethylamine/Ethanol p299 1/2c
  79-20-9       64-17-5    -34.6561    393.7135    MethylAcetate/Ethanol p315 1/2c
  64-17-5       67-63-0   -258.9690    477.1468    Ethanol/2-Propanol p317 1/2c
  64-17-5       57-55-6    -82.8306    450.0867    Ethanol/1,2-Propanediol p319 1/2c
  64-17-5      110-02-1   1222.6229    208.6395    Ethanol/Thiophene p320 1/2c
 108-05-4       64-17-5    101.8376    792.0355    VinylAcetate/Ethanol p324 1/2c
  64-17-5       78-93-3    694.0825   -149.7978    Ethanol/2-Butanone p327 1/2c
 109-99-9       64-17-5   -290.6763    743.7329    Tetrahydrofuran/Ethanol p328 1/2c
  64-17-5      123-91-1    447.1418    197.9989    Ethanol/1,4-Dioxane p331 1/2c
 141-78-6       64-17-5      1.9964    671.5589    EthylAcetate/Ethanol p338 1/2c
  64-17-5      126-33-0    756.0277    899.7508    Ethanol/Sulfolane p341 1/2c
  64-17-5      110-91-8   -573.3858    993.4870    Ethanol/Morpholine p345 1/2c
  64-17-5       78-92-2    862.4408   -662.6378    Ethanol/2-Butanol p346 1/2c
  60-29-7       64-17-5   -133.4676   1027.0060    DiethylEther/Ethanol p347 1/2c
  64-17-5       78-83-1    317.4759   -181.0696    Ethanol/2-Methyl-1-Propanol p349 1/2c
 109-73-9       64-17-5   -460.4452   -152.0442    Butylamine/Ethanol p351 1/2c
 109-89-7       64-17-5   1655.6748   -575.3447    Diethylamine/Ethanol p353 1/2c
  64-17-5      110-86-1     41.4133    -41.6497    Ethanol/Pyridine p356 1/2c
  78-79-5       64-17-5    165.1218   1882.0214    Isoprene/Ethanol p358 1/2c
  64-17-5       80-62-6    799.2832    119.3206    Ethanol/MethylMethacrylate p363 1/2c
  513-35-9      64-17-5    212.8335   2127.7251    2-Methyl-2-Butene/Ethanol p365 1/2c
  64-17-5      563-80-4    952.0918   -356.5456    Ethanol/Methylisopropyl Ketone p366 1/2c
  64-17-5      109-60-4    687.4968    278.0956    Ethanol/PropylAcetate p367 1/2c
  78-78-4       64-17-5    329.3380   2013.9507    2-Methylbutane/Ethanol p373 1/2c
 109-66-0       64-17-5    267.7400   1867.3772    n-Pentane/Ethanol p376 1/2c
  64-17-5      123-51-3     39.5288    163.9351    Ethanol/3-Methyl-1-butanol p377 1/2c
  64-17-5      108-86-1   1875.2021    304.6178    Ethanol/Bromobenzene p378 1/2c
  64-17-5      108-90-7   1915.7495    129.7634    Ethanol/Chlorobezene p380 1/2c
  64-17-5       71-43-2   1399.9279    207.3433    Ethanol/Benzene p406 1/2c
  64-17-5       62-53-3    -15.7382  11303.1211    Ethanol/Aniline p410 1/2c
  64-17-5      109-06-8    -54.0076     53.7986    Ethanol/2-Methylpyridine p411 1/2c
  64-17-5      108-99-6     13.4183    -14.0113    Ethanol/3-Methylpyridine p412 1/2c
  64-17-5      110-82-7   2115.9303    469.5182    Ethanol/Cyclohexane p419 1/2c
 592-41-6       64-17-5    284.5270   1972.6835    1-Hexene/Ethanol p420 1/2c
  64-17-5      108-93-0   -111.5935    973.0365    Ethanol/Cyclohexanol p421 1/2c
  64-17-5      123-86-4    628.1683    199.2178    Ethanol/ButylAcetate p424 1/2c
 110-54-3       64-17-5    432.4113   1934.2154    n-Hexane/Ethanol p437 1/2c
  64-17-5      111-43-3   1272.7034    -70.3142    Ethanol/Dipropylether p438 1/2c
  64-17-5      121-44-8   1077.4399   -643.7960    Ethanol/Triethylamine p440 1/2c
  64-17-5      108-88-3   1556.4474    210.5219    Ethanol/Toluene p444 1/2c
  64-17-5      100-66-3     36.0389   1973.1296    Ethanol/Anisole p445 1/2c
  64-17-5      108-39-4    384.4596  -1345.6032    Ethanol/3-Methylphenol p446 1/2c
  64-17-5      108-48-5    400.3109   -467.7627    Ethanol/2,6-Dimethylpyridine p447 1/2c
  64-17-5      628-63-7    940.1564    -50.0958    Ethanol/Pentylacetate p449 1/2c
  64-17-5      142-82-5   2063.8162    459.8293    Ethanol/Heptane p459 1/2c
  64-17-5      100-41-4   1552.9319    237.8904    Ethanol/Ethylbenzene p460 1/2c
  64-17-5      106-42-3   1487.8786    317.7784    Ethanol/P-xylene p461 1/2c
  64-17-5      111-65-9   2066.1398    437.1305    Ethanol/Octane p466 1/2c
  64-17-5      540-84-1   2256.1277    346.8967    Ethanol/2,2,4-Trimethylpentane p468 1/2c
  64-17-5      544-76-3   2473.0823    566.3270    Ethanol/Hexadecane p470 1/2c
  64-17-5      112-80-1    901.1357   -267.6027    Ethanol/Oleicacid p471 1/2c
  56-23-5       71-23-8      8.1130   1922.2042    Tetrachloromethane/1-Propanol p472 1/2c
  71-23-8      127-18-4    937.3551    462.0605    1-Propanol/Tetrachloroethylene p473 1/2c
  79-01-6       71-23-8    203.5836   1045.4046    Trichloroethylene/1-Propanol p474 1/2c
 107-06-2       71-23-8    412.6782    681.1391    1,2-Dichloroethane/1-Propanol p478 1/2c
  71-23-8       64-19-7   7066.9493   -656.5328    1-Propanol/Acetic acid p479 1/2c
 124-40-3       71-23-8   -387.4789   -296.9466    Dimethylamine/1-Propanol p484 1/2c
  71-23-8       79-09-4    -42.8365   -174.6683    1-Propanol/Propionicacid p486 1/2c
  67-63-0       71-23-8   -181.7135    181.9669    2-Propanol/1-Propanol p488 1/2c
  71-23-8      109-86-4    749.4098   -316.3670    1-Propanol/2-Methoxy-ethanol p490 1/2c
 107-10-8       71-23-8   -200.8140   -439.2018    Propylamine/1-Propanol p492 1/2c
  71-23-8       79-41-4   -328.2718    219.5742    1-Propanol/Methacrylicacid p495 1/2c
  78-93-3       71-23-8     72.2001    299.3730    2-Butanone/1-Propanol p496 1/2c
 109-99-9       71-23-8   -239.2854    489.2594    Tetrahydrofuran/1-Propanol p497 1/2c
 109-69-3       71-23-8    136.2115   1510.3755    Butylchloride/1-Propanol p502 1/2c
  71-23-8       78-83-1    557.9676   -475.4616    1-Propanol/2-Methyl-1-Propanol p505 1/2c
 109-73-9       71-23-8   -331.0001   -335.7584    Butylamine/1-Propanol p510 1/2c
  71-23-8      110-86-1   -334.9027    335.2650    1-Propanol/Pyridine p513 1/2c
  71-23-8       80-62-6    416.4613    195.3664    1-Propanol/Methylmethacrylate p514 1/2c
  71-23-8      109-60-4    473.2961     -6.6597    1-Propanol/Propylacetate p517 1/2c
  71-23-8      123-51-3      7.1877     21.9942    1-Propanol/3-Methylbutanol p521 1/2c
 392-56-3       71-23-8    428.9413   1506.4333    Hexafluorobenzene/1-Propanol p526 1/2c
  71-23-8      108-90-7    301.9410    282.7762    1-Propanol/Chlorobenzene p527 1/2c
  71-43-2       71-23-8    312.1615    996.2753    Benzene/1-Propanol p537 1/2c
  71-23-8      109-06-8   -448.0010    583.3658    1-Propanol/2-Methylpyridine p538 1/2c
  71-23-8      108-99-6   -317.4484    319.0293    1-Propanol/3-Methylpyridine p539 1/2c
  71-23-8      108-89-4   -466.4362    625.2762    1-Propanol/4-Methylpyridine p540 1/2c
 110-82-7       71-23-8   -176.8307   2249.9342    Cyclohexane/1-Propanol p542 1/2c
  71-23-8      106-36-5   3067.3388  -1157.6428    1-Propanol/Propylpropionate p545 1/2c
 110-54-3       71-23-8    348.5686   1597.0679    Hexane/1-Propanol p550 1/2c
  71-23-8      142-84-7    150.4368    -11.4124    1-Propanol/Dipropylamine p551 1/2c
 121-44-8       71-23-8   -678.8111   1216.8556    Triethylamine/1-Propanol p552 1/2c
  71-23-8      107-46-0    739.0782    406.7140    1-Propanol/Hexamethyldisiloxane p555 1/2c
  71-23-8      108-88-3    837.2353    300.7495    1-Propanol/Toluene p556 1/2c
  71-23-8      108-48-5    -14.5692   -141.4123    1-Propanol/2,6-Dimethylpyridine p557 1/2c
  71-23-8      142-82-5   1773.2912    574.0796    1-Propanol/Heptane p573 1/2c
  71-23-8      100-41-4    916.6926    215.3795    1-Propanol/Ethylbenzene p574 1/2c
  71-23-8      111-65-9   1117.6745    516.4849    1-Propanol/Octane p576 1/2c
  71-23-8      124-18-5   1401.6190    280.8638    1-Propanol/Decane p577 1/2c
  71-23-8      112-30-1    175.9426    379.5002    1-Propanol/1-Decanol 579 1/2c
  67-64-1       67-66-3     28.8819   -484.3856    Acetone/Chloroform
  67-64-1       67-56-1   -161.8813    583.1054    Acetone/Methanol
  67-64-1       64-17-5    201.6284    251.5554    Acetone/Ethanol
  67-64-1       71-43-2    543.9352   -182.5230    Acetone/Benzene
  67-66-3       67-56-1   -351.1964   1760.6741    Chloroform/Methanol
  67-66-3       64-17-5   -268.7696   1270.3897    Chloroform/Ethanol
  67-66-3       71-43-2   -161.8065     49.6010    Chloroform/Benzene
  67-66-3       79-20-9   -497.9734    183.6587    Chloroform/Methylacetate
  67-56-1       71-43-2   1734.4181    183.0383    Methanol/Benzene
  64-17-5       71-43-2   1399.9279    207.3433    Ethanol/Benzene
  67-56-1       64-17-5    -65.7022    143.6658    Methanol/Ethanol
  56-23-5       71-43-2     47.5034     33.9946    Tetrachloromethane/Benzene p58 1/7
  71-43-2       75-25-2   -347.6853    650.3158    Benzene/Tribromomethane p59 1/7
  67-66-3       71-43-2   -161.8065     49.6010    Chloroform/Benzene p72 1/7
  74-88-4       71-43-2    113.3782    -12.2301    MethylIodide/Benzene p84 1/7
  71-43-2       75-52-5    194.3877    666.2584    Benzene/Nitromethane p88 1/7
  75-15-0       71-43-2    662.9720    -39.9415    CarbonDisulfide/Benzene p90 1/7
  76-13-1       71-43-2    296.5283    214.8840    1,1,2-Trichloro-1,2,2-Trifluoroethane/Benzene p111 1/7
  71-43-2      127-18-4    309.0186   -118.9592    Benzene/Tetrachloroethylene p112 1/7
  71-43-2       79-01-6   -241.2617    299.3475    Benzene/Trichloroethylene p118 1/7
  71-43-2       76-01-7    343.4482   -378.5292    Benzene/Pentachloroethane p119 1/7
  71-43-2       79-34-5    -92.9354   -226.4063    Benzene/1,1,2,2-tetrachloroethane p120 1/7
  71-55-6       71-43-2   -263.8246    317.6450    1,1,1-trichloroethane/Benzene p121 1/7
  71-43-2      106-93-4    212.1963    -16.0736    Benzene/1,2-Dibromoethane p136 1/7
  71-43-2      107-06-2    195.2920   -130.7975    Benzene/1,2-Dichloroethane p158 1/7
  75-03-6       71-43-2    185.0848    298.1172    EthylIodide/Benzene p160 1/7
  71-43-2      123-39-7    177.3275   1970.9703    Benzene/n-Methylformamide p161 1/7
  71-43-2       79-24-3   -105.3359    591.7991    Benzene/Nitroethane p163 1/7
  71-43-2       67-68-5    111.4310    966.0527    Benzene/Dimethylsulfoxide p166 1/7
  71-43-2      107-15-3    490.0693    560.0207    Benzene/Ethylenediamine p170 1/7
  71-43-2       68-12-2   -209.7030    686.8662    Benzene/n,n-dimethylformamide p184 1/7
  71-43-2      108-03-2    641.4922   -124.7180    Benzene/1-Nitropropane p185 1/7
  71-43-2       79-46-9   -281.4475    958.4486    Benzene/2-Nitropropane p186 1/7
  71-43-2      110-02-1    316.8438   -166.7836    Benzene/Thiophene p188 1/7
  71-43-2      126-33-0    571.0007    515.5025    Benzene/Sulfolane p191 1/7
 109-73-9       71-43-2    137.0632     10.9401    Butylamine/Benzene p201 1/7
  75-64-9       71-43-2    569.9558   -133.4153    Tert.Butylamine/Benzene p202 1/7
 109-89-7       71-43-2    -46.7376     57.4768    Diethylamine/Benzene p207 1/7
  71-43-2      110-86-1   -227.3437    423.2812    Benzene/Pyridine p221 1/7
  71-43-2      392-56-3   -332.9811    701.5423    Benzene/Hexafluorobenzene p226 1/7
  71-43-2      106-46-7   -408.0546   1013.3840    Benzene/P-Dichlorobenzene p238 1/7
  71-43-2      108-86-1   -541.1736   1468.8113    Benzene/Bromobenzene p241 1/7
  71-43-2      108-90-7   -291.7962    591.6755    Benzene/Chlorobenzene p251 1/7
  71-43-2      462-06-6    988.8587   -615.4317    Benzene/Fluorobenzene p252 1/7
  71-43-2       98-95-3   -248.3370   1104.5772    Benzene/Nitrobenzene p254 1/7
  71-43-2       62-53-3     -8.3413    604.6308    Benzene/Aniline p266 1/7
  71-43-2      108-91-8   -675.5901   2074.7190    Benzene/Cyclohexylamine p274 1/7
  71-43-2      121-44-8    189.7536    -63.6581    Benzene/Triethylamine p277 1/7
  71-43-2      100-47-0   -289.9707   1060.6718    Benzene/Benzonitrile p278 1/7
  71-43-2      108-88-3    377.9760   -354.9859    Benzene/Toluene p301 1/7
  71-43-2      100-60-7    293.2801   -142.8208    Benzene/n-Methylcyclohexylamine p304 1/7
  71-43-2      100-42-5   -289.0726    569.6908    Benzene/Styrene p305 1/7
  71-43-2      100-41-4   -430.7138   1063.2981    Benzene/Ethylbenzene p306 1/7
  71-43-2      108-38-3    803.5574   -570.3291    Benzene/m-Xylene p307 1/7
  71-43-2      106-42-3     -0.0399      5.1147    Benzene/p-Xylene p310 1/7
  71-43-2       91-66-7   -145.9492    602.8251    Benzene/n,n-Diethylaniline p312 1/7
  71-43-2       98-82-8   -300.1758   1320.6592    Benzene/Isopropylbenzene p322 1/7
  71-43-2      103-65-1   -349.3013    895.5382    Benzene/Propylbenzene p323 1/7
  71-43-2       92-52-4   -462.5170   9303.1133    Benzene/Biphenyl p324 1/7
  71-43-2       92-06-8    391.5215   -230.8556    Benzene/m-Terphenyl p327 1/7
  56-23-5      108-88-3   -129.1578    193.0120    Tetrachloromethane/Toluene p351 1/7
  67-66-3      108-88-3   -365.8311    552.1459    Chloroform/Toluene p352 1/7
  75-15-0      108-88-3   1143.7143   -669.0945    CarbonDisulfide/Toluene p356 1/7
  79-01-6      108-88-3   -155.6623    116.5756    Trichloroethylene/Toluenep 370 1/7
  75-05-8      108-88-3   1076.8346     -0.8703    Acetonitrile/Toluene p373 1/7
 624-83-9      108-88-3   -324.5254   1631.1669    MethylIsocyanate/Toluene p376 1/7
 107-06-2      108-88-3   -259.7091    541.9318    1,2-Dichloroethane/Toluene p384 1/7
 108-88-3       79-24-3   -132.3833    709.0904    Toluene/ Nitroethane p385 1/7
 108-88-3       67-68-5    124.4232   1331.2819    Toluene/Dimethylsulfoxide p386 1/7
 108-88-3      107-15-3    263.5715    852.4958    Toluene/Ethylenediamine p387 1/7
 107-12-0      108-88-3    794.4377   -171.5587    Propionitrile/Toluene p388 1/7
 108-88-3       68-12-2    983.7811     27.1682    Toluene/n,n-Dimethylformamide p394 1/7
 110-02-1      108-88-3    117.6916    194.9321    Thiophene/Toluene 395 1/7
 108-88-3      126-33-0   -546.4723  14731.2741    Toluene/Sulfolane p399 1/7
 109-89-7      108-88-3   -481.3134    480.7524    Diethylamine/Toluene p400 1/7
 108-88-3      110-86-1   -206.6019    408.7559    Toluene/Pyridine p407 1/7
 392-56-3      108-88-3   1637.8487   -684.8044    Hexafluorobenzene/Toluene p414 1/7
 108-88-3      108-86-1   -193.2361    189.9965    Toluene/Bromobenzene p415 1/7
 108-88-3      108-90-7    -27.8049      2.0364    Toluene/Chlorobenzene p417 1/7
 462-06-6      108-88-3   -168.2744    259.7232    Fluorobenzene/Toluene p421 1/7
 108-88-3       98-95-3   -139.1451    633.7697    Toluene/Nitrobenzene p422 1/7
 108-88-3       62-53-3   1011.4228     86.8982    Toluene/Aniline p431 1/7
 108-88-3      109-06-8    -69.7940    367.0077    Toluene/2-Methylpyridine p432 1/7
 108-88-3      108-99-6    789.4771   -237.2802    Toluene/3-Methylpyridine p433 1/7
 108-88-3      100-47-0   1026.6936   -443.9423    Toluene/Benzonitrile p435 1/7
 108-88-3      100-41-4   -258.0806    425.1046    Toluene/Ethylbenzene p443 1/7
 108-88-3      106-42-3   -120.6447    118.6929    Toluene/p-Xylene p444 1/7
 100-42-5       76-01-7    464.0601   -483.3507    Styrene/Pentachloroethane p445 1/7
 107-13-1      100-42-5    280.3027    202.4782    Acrylonitrile/Styrene p446 1/7
 100-41-4      100-42-5    712.6161   -406.5803    Ethylbenzene/Styrene p451 1/7
 100-42-5       98-83-9   -446.2287   1017.9460    Styrene/Alpha-MethylStyrene p456 1/7
 100-42-5      103-65-1   -258.1637    504.2163    Styrene/Propylbenzene p457 1/7
  56-23-5      100-41-4    152.3943   -208.9399    Tetrachloromethane/Ethylbenzene p464 1/7
  75-05-8      100-41-4    841.3154    325.7917    Acetonitrile/Ethylbenzene p465 1/7
 107-06-2      100-41-4    487.1864   -614.7516    1,2-Dichloroethane/Ethylbenzene p466 1/7
 107-13-1      100-41-4    266.8411    664.1897    Acrylonitrile/Ethylbenzene p467 1/7
 109-89-7      100-41-4   -302.4227    622.2122    Diethylamine/Ethylbenzene p468 1/7
 108-90-7      100-41-4    155.4346   -153.3937    Chlorobenzene/Ethylbenzene p469 1/7
 100-41-4       98-95-3    -92.4330    550.1965    Ethylbenzene/Nitrobenzene p470 1/7
 100-41-4       62-53-3    164.2977    495.8526    Ethylbenzene/Aniline p475 1/7
 100-41-4       98-82-8   -257.3206    362.0864    Ethylbenzene/Isopropylbenzene p476 1/7
 100-41-4      104-51-8   1129.0721   -867.9652    Ethylbenzene/Butylbenzene p477 1/7
  56-23-5      108-38-3    227.6117   -306.2953    Tetrachloromethane/m-Xylene p480 1/7
 108-38-3       68-12-2    140.2711    767.4783    m-Xylene/n,n-Dimethylformamide p481 1/7
 110-86-1      108-38-3    602.3761   -332.4513    Pyridine/m-Xylene p482 1/7
 108-38-3       62-53-3    522.6941    227.5134    m-Xylene/Aniline p483 1/7
 106-42-3      108-38-3   -203.6561    231.4823    p-Xylene/m-Xylene p485 1/7
  56-23-5       95-47-6    177.5599   -246.9008    Tetrachloromethane/o-Xylene p488 1/7
  95-47-6       76-01-7    494.9416   -515.4790    o-Xylene/Pentachloroethane p489 1/7
 107-06-2       95-47-6    -70.9046    307.0137    1,2-Dichloroethane/o-Xylene p490 1/7
 107-15-3       95-47-6    727.6080    668.5257    Ethylenediamine/o-Xylene p491 1/7
  95-47-6       68-12-2     26.7026    918.9602    o-Xylene/n,n-Dimethylformamide p493 1/7
  56-23-5      106-42-3    154.7347   -234.3053    Tetrachloromethane/p-Xylene p498 1/7
  75-05-8      106-42-3    686.8423    544.7368    Acetonitrile/p-Xylene p499 1/7
 107-06-2      106-42-3    -96.1460    380.2074    1,2-Dichloroethane/p-Xylene p500 1/7
 106-42-3       68-12-2    267.6269    649.5482    p-Xylene/n,n-Dimethylformamide p501 1/7
 392-56-3      106-42-3   -386.7882     75.2112    Hexafluorobenzene/p-Xylene p507 1/7
 108-90-7      106-42-3   -434.8331    632.3847    Chlorobenzene/p-Xylene p508 1/7
 106-42-3       62-53-3     90.7280    673.7216    p-Xylene/Aniline p509 1/7
  56-23-5       98-82-8    210.0577   -303.4551    Tetrachloromethane/Isopropylbenzene p510 1/7
 103-65-1       98-95-3    -41.8140    531.0680    Propylbenzene/Nitrobenzene p511 1/7
  95-63-6      526-73-8   -517.6616    845.1561    1,2,4-Trimethylbenzene/1,2,3-Trimethylbenzene p515 1/7
 104-51-8       98-95-3   -115.8934    645.6422    Butylbenzene/Nitrobenzene p527 1/7
  99-87-6       62-53-3    240.8443    534.5195    P-Cymene/Aniline p529 1/7
  91-57-6       90-12-0    630.0282   -442.7995    2-Methylnaphthalene/1-Methylnaphthalene p531 1/7
"""


NRTL_IP_DATA = """
67-56-1       56-23-5  378.8254  1430.7379  .2892  Methanol/Tetrachloromethane p18 1/2c
67-56-1       75-25-2  879.0968  1063.6098  .6381  Methanol/Tribromomethane p19 1/2c
67-66-3       67-56-1  1414.2712  -141.8030  .2949  Chloroform/Methanol p25 1/2c
75-09-2       67-56-1  1560.0282  441.3372  .6234  Dichloromethane/Methanol p26 1/2c
67-56-1       75-12-7  617.5847  -153.4695  .3003  Methanol/Formamide p30 1/2c
151-67-7       67-56-1  9870.3530  -6982.8569  .187e-1  Halothane/Methanol p33 1/2c
75-01-4       67-56-1  1789.7165  -34.9448  .2912  VinylChloride/Methanol p36 1/2c
67-56-1       75-05-8  343.7042  314.5879  .2981  Methanol/Acetonitrile p42 1/2c
67-56-1       107-06-2  348.6035  1020.1431  .2921  Methanol/1,2-Dichloroethane p44 1/2c
67-56-1       64-19-7  16.6465  -217.1261  .3051  Methanol/AceticAcid p46 1/2c
107-31-3       67-56-1  584.5720  298.5567  .2962  MethylFormate/Methanol p51 1/2c
67-56-1       64-17-5  -327.9991  376.2667  .3057  Methanol/Ethanol p60 1/2c
67-56-1       67-68-5  -168.3182  -497.4171  .3079  Methanol/Dimethylsulfoxide p61 1/2c
124-40-3       67-56-1  -1018.1430  -54.3882  .3134  Dimethylamine/Methanol p64 1/2c
67-64-1       67-56-1  184.2662  226.5580  .3009  Acetone/Methanol p78 1/2c
123-38-6       67-56-1  1046.6524  -865.2660  .3084  Propionic Aldehyde/Methanol p79 1/2c
75-56-9       67-56-1  924.8499  -61.1796  .2986  PropyleneOxide/Methanol p80 1/2c
79-20-9       67-56-1  381.4559  346.5360  .2965  MethylAcetate/Methanol p86 1/2c
67-56-1       79-09-4  -50.1450  -78.0859  .3056  Methanol/Propionic Acid p87 1/2c
67-56-1       68-12-2  -124.0904  .3428  9.1633  Methanol/n,n-Dimethylformamide p88 1/2c
67-56-1       71-23-8  24.9003  9.5349  .3011  Methanol/Propanol p96 1/2c
109-87-5       67-56-1  608.9115  712.0226  .7259  Dimethoxymethane/Methanol p97 1/2c
67-56-1       110-02-1  -90.1051  1217.1035  .2976  Methanol/Thiophene p101 1/2c
106-99-0       67-56-1  1353.0599  610.8292  .4670  1,3-Butadiene/Methanol p103 1/2c
67-56-1       96-33-3  676.8360  169.9831  .2958  Methanol/Methylacrylate p104 1/2c
115-11-7       67-56-1  1333.6000  556.3608  .3697  Isobutylene/Methanol p105 1/2c
67-56-1       78-93-3  307.4271  217.9098  .3003  Methanol/2-Butanone p106 1/2c
67-56-1       109-99-9  169.4153  383.1579  .3002  Methanol/Tetrahydrofuran p111 1/2c
67-56-1       123-91-1  607.4050  76.7683  .2985  Methanol/1,4-Dioxane p116 1/2c
67-56-1       141-78-6  345.5416  420.7355  .2962  Methanol/Ethylacetate p117 1/2c
67-56-1       126-33-0  1069.2756  906.5741  .7182  Methanol/Sulfolane p120 1/2c
106-97-8       67-56-1  1440.1498  1053.7716  .4647  Butane/Methanol p126 1/2c
67-56-1       71-36-3  793.8173  -486.3299  .2483  Methanol/1-Butanol p127 1/2c
67-56-1       78-92-2  -308.5610  285.4420  .3036  Methanol/2-Butanol p128 1/2c
60-29-7       67-56-1  705.9989  211.1580  .2953  DiethylEther/Methanol p133 1/2c
67-56-1       110-63-4  446.9520  -450.5858  .3152  Methanol/1,4-Butanediol p138 1/2c
67-56-1       110-86-1  -45.0888  84.1956  .3027  Methanol/Pyridine p141 1/2c
542-92-7       67-56-1  1541.9324  736.0352  .4515  Cyclopentadiene/Methanol p144 1/2c
78-79-5       67-56-1  1445.6425  543.5270  .4260  Isoprene/Methanol p145 1/2c
1574-41-0       67-56-1  1545.3339  799.1289  .4753  1,3-Pentadiene(CIS)/Methanol p147 1/2c
2004-70-8       67-56-1  1514.2761  782.1729  .4657  Trans-1,3-Pentadiene/Methanol p148 1/2c
67-56-1       80-62-6  590.2790  380.8401  .2963  Methanol/MethylMethacrylate p152 1/2c
513-35-9       67-56-1  1355.6853  660.9164  .3381  2-Methyl-2-Butene/Methanol p154 1/2c
67-56-1       563-80-4  642.3761  -6.2901  .2987  Methanol/Methylisopropyl Ketone p155 1/2c
67-56-1       110-89-4  590.8820  -1169.7242  .1387  Methanol/Piperidine p157 1/2c
1634-04-4       67-56-1  851.4954  465.8360  .8178  MethylTert-ButylEther/Methanol p160 1/2c
67-56-1       392-56-3  930.5910  1244.1303  .4701  Methanol/Hexafluorobenzene p170 1/2c
67-56-1       108-90-7  857.0852  1348.0903  .4707  Methanol/Chlorobenzene p174 1/2c
67-56-1       71-43-2  721.6136  1158.5131  .4694  Methanol/Benzene p191 1/2c
67-56-1       62-53-3  407.7440  117.2473  .3008  Methanol/Aniline p192 1/2c
67-56-1       109-06-8  226.0820  -385.6823  .3095  Methanol/2-Methylpyridine p195 1/2c
67-56-1       108-99-6  -163.4505  -86.1482  .3075  Methanol/3-Methylpyridine p196 1/2c
67-56-1       108-89-4  304.2242  -452.3483  .3053  Methanol/4-Methylpyridine p198 1/2c
67-56-1       110-83-8  1178.5792  1618.9792  .4568  Methanol/Cyclohexene p204 1/2c
67-56-1       110-82-7  1315.1631  1497.2135  .4222  Methanol/Cyclohexane p211 1/2c
592-41-6       67-56-1  1222.6032  1145.1085  .4402  1-Hexene/Methanol p212 1/2c
67-56-1       123-86-4  328.2162  453.0017  .2961  Methanol/n-ButylAcetate p217 1/2c
67-56-1       110-54-3  1619.3829  1622.2911  .4365  Methanol/Hexane p224 1/2c
67-56-1       121-44-8  -476.8503  1126.1143  .2874  Methanol/Triethylamine p226 1/2c
67-56-1       108-88-3  939.7275  1090.9297  .4643  Methanol/Toluene p237 1/2c
67-56-1       108-48-5  -273.3320  59.6250  .3051  Methanol/2,6-Dimethylpyridine p239 1/2c
67-56-1       592-76-7  1313.5497  1143.9059  .4163  Methanol/1-Heptene p241 1/2c
67-56-1       108-87-2  1444.5850  1719.4586  .4397  Methanol/Methylcyclohexane p242 1/2c
67-56-1       142-82-5  1500.2043  1519.3346  .4277  Methanol/n-Heptane p243 1/2c
67-56-1       100-41-4  1080.1231  1038.1572  .4251  Methanol/Ethylbenzene p245 1/2c
67-56-1       108-38-3  991.1609  822.1357  .2910  Methanol/M-xylene p246 1/2c
67-56-1       106-42-3  974.6545  851.1070  .2921  Methanol/P-xylene p247 1/2c
67-56-1       111-66-0  1456.3583  1147.8132  .4396  Methanol/1-Octene p248 1/2c
67-56-1       111-65-9  1681.6918  1511.4353  .4381  Methanol/Octane p249 1/2c
67-56-1       540-84-1  1447.0909  1386.4703  .4313  Methanol/2,2,4-Trimethylpentane p250 1/2c
56-23-5       67-56-1  1339.9000  488.6648  .4622  Tetrachloromethane/Methanol p279 1/2c
67-66-3       64-17-5  1438.3602  -327.5518  .3023  Chloroform/Ethanol p282 1/2c
75-09-2       64-17-5  1332.8036  -153.0761  .3057  Dichloromethane/Ethanol p284 1/2c
64-17-5       127-18-4  685.8542  807.5935  .2900  Ethanol/Tetrachloroethylene p285 1/2c
64-17-5       75-05-8  529.7267  338.1632  .2964  Ethanol/Acetonitrile p289 1/2c
64-17-5       107-06-2  333.3502  939.3870  .2926  Ethanol/1,2-Dichloroethane p292 1/2c
64-17-5       64-19-7  -34.1971  -190.7763  .3050  Ethanol/Acetic Acid p293 1/2c
64-17-5       107-21-1  1644.0484  -203.7691  .3704  Ethanol/1,2-Ethanediol p297 1/2c
124-40-3       64-17-5  -1224.5739  370.7683  .3105  Dimethylamine/Ethanol p299 1/2c
67-64-1       64-17-5  36.2965  434.8228  .2987  Acetone/Ethanol p312 1/2c
79-20-9       64-17-5  188.3139  158.0118  .3013  MethylAcetate/Ethanol p315 1/2c
64-17-5       67-63-0  690.1392  -529.3472  .3125  Ethanol/2-Propanol p317 1/2c
64-17-5       110-02-1  222.3096  1057.7115  .2918  Ethanol/Thiophene p320 1/2c
108-05-4       64-17-5  505.1637  320.7403  .2959  VinylAcetate/Ethanol p324 1/2c
64-17-5       78-93-3  64.4957  463.1931  .3010  Ethanol/2-Butanone p327 1/2c
109-99-9       64-17-5  661.3708  -200.6915  .3015  Tetrahydrofuran/Ethanol p328 1/2c
64-17-5       123-91-1  505.5637  111.8389  .2988  Ethanol/1,4-Dioxane p331 1/2c
141-78-6       64-17-5  305.6041  330.5105  .2988  EthylAcetate/Ethanol p338 1/2c
64-17-5       126-33-0  1195.1601  705.0897  .5676  Ethanol/Sulfolane p341 1/2c
64-17-5       78-92-2  -559.8205  802.5411  .2721  Ethanol/2-Butanol p346 1/2c
60-29-7       64-17-5  763.6707  71.1984  .2946  DiethylEther/Ethanol p347 1/2c
64-17-5       78-83-1  53.1671  82.0442  .3023  Ethanol/2-Methyl-1-Propanol p349 1/2c
109-73-9       64-17-5  -612.3956  -5.7834  .3062  Butylamine/Ethanol p351 1/2c
64-17-5       110-86-1  163.6655  -169.9802  .3017  Ethanol/Pyridine p356 1/2c
78-79-5       64-17-5  1402.5377  653.4866  .5056  Isoprene/Ethanol p358 1/2c
64-17-5       80-62-6  456.9676  386.5893  .2963  Ethanol/MethylMethacrylate p363 1/2c
513-35-9       64-17-5  1412.7516  674.7726  .4569  2-Methyl-2-Butene/Ethanol p365 1/2c
64-17-5       563-80-4  -54.0946  639.6806  .3009  Ethanol/Methylisopropyl Ketone p366 1/2c
64-17-5       109-60-4  760.4933  129.3970  .2950  Ethanol/PropylAcetate p367 1/2c
78-78-4       64-17-5  1610.2811  935.1426  .4960  2-Methylbutane/Ethanol p373 1/2c
109-66-0       64-17-5  1183.3812  412.7546  .2886  Pentane/Ethanol p376 1/2c
64-17-5       123-51-3  51.1705  -42.8613  .3009  Ethanol/3-Methyl-1-butanol p377 1/2c
64-17-5       108-86-1  820.8023  1349.6853  .4995  Ethanol/Bromobenzene p378 1/2c
64-17-5       108-90-7  645.7829  1383.7110  .5229  Ethanol/Chlorobezene p380 1/2c
64-17-5       71-43-2  516.1410  1065.9086  .4774  Ethanol/Benzene p406 1/2c
64-17-5       62-53-3  1823.3542  -523.0474  .3005  Ethanol/Aniline p410 1/2c
64-17-5       108-99-6  315.6078  -339.0825  .3056  Ethanol/3-Methylpyridine p412 1/2c
64-17-5       110-82-7  876.7933  1390.4162  .4485  Ethanol/Cyclohexane p419 1/2c
592-41-6       64-17-5  1399.1806  949.7239  .5011  1-Hexene/Ethanol p420 1/2c
64-17-5       108-93-0  1719.8644  -833.8389  .2920  Ethanol/Cyclohexanol p421 1/2c
64-17-5       123-86-4  841.9935  -55.3231  .2958  Ethanol/ButylAcetate p424 1/2c
110-54-3       64-17-5  1218.1065  575.2985  .2882  Hexane/Ethanol p437 1/2c
64-17-5       111-43-3  442.6124  634.1687  .2945  Ethanol/Dipropylether p438 1/2c
64-17-5       121-44-8  -248.0407  697.9004  .2996  Ethanol/Triethylamine p440 1/2c
64-17-5       108-88-3  713.5653  1147.8607  .5292  Ethanol/Toluene p444 1/2c
64-17-5       100-66-3  3458.4788  -1438.8884  .1704  Ethanol/Anisole p445 1/2c
64-17-5       108-48-5  -48.2159  -19.6212  .3016  Ethanol/2,6-Dimethylpyridine p447 1/2c
64-17-5       628-63-7  552.3897  266.1248  .2964  Ethanol/Pentylacetate p449 1/2c
64-17-5       142-82-5  1137.2650  1453.5947  .4786  Ethanol/n-Heptane p459 1/2c
64-17-5       100-41-4  801.7191  1006.8831  .4962  Ethanol/Ethylbenzene p460 1/2c
64-17-5       106-42-3  768.3633  760.0800  .2914  Ethanol/P-xylene p461 1/2c
64-17-5       111-65-9  1206.8097  1385.3721  .4717  Ethanol/Octane p466 1/2c
64-17-5       540-84-1  1091.0432  1500.6711  .4738  Ethanol/2,2,4-Trimethylpentane p468 1/2c
64-17-5       544-76-3  2359.4082  1509.2033  .4448  Ethanol/Hexadecane p470 1/2c
64-17-5       112-80-1  975.6816  -343.5446  .2988  Ethanol/Oleicacid p471 1/2c
56-23-5       71-23-8  1537.6384  537.2439  .6310  Tertachloromethane/1-Propanol p472 1/2c
71-23-8       127-18-4  608.3777  646.0412  .2913  1-Propanol/Tetrachloroethylene p473 1/2c
79-01-6       71-23-8  926.6139  196.0696  .3007  Trichloroethylene/1-Propanol p474 1/2c
107-06-2       71-23-8  636.9927  364.0596  .2956  1,2-Dichloroethane/1-Propanol p478 1/2c
71-23-8       64-19-7  256.8999  -327.5173  .3044  1-Propanol/Aceticacid p479 1/2c
124-40-3       71-23-8  81.4870  -703.3731  .2697  Dimethylamine/1-Propanol p484 1/2c
71-23-8       79-09-4  -189.7856  -32.4657  .3070  1-Propanol/Propionicacid p486 1/2c
67-63-0       71-23-8  167.2501  -175.2928  .3057  2-Propanol/1-Propanol p488 1/2c
71-23-8       109-86-4  -406.3767  830.8897  .3013  1-Propanol/2-Methoxy-ethanol p490 1/2c
107-10-8      71-23-8  -602.9687  -45.3543  .3061  Propylamine/1-Propanol p492 1/2c
71-23-8       79-41-4  276.6356  -423.9162  .3025  1-Propanol/Methacrylicacid p495 1/2c
78-93-3       71-23-8  148.8670  213.3829  .3016  2-Butanone/1-Propanol p496 1/2c
109-99-9       71-23-8  562.4611  -302.2498  .3003  Tetrahydrofuran/1-Propanol p497 1/2c
109-69-3       71-23-8  1201.9959  506.8982  .5468  Butylchloride/1-Propanol p502 1/2c
71-23-8       78-83-1  -2.8139  -13.8657  .3034  1-Propanol/2-Methyl-1-Propanol p505 1/2c
109-73-9       71-23-8  -698.9510  34.7593  .3001  Butylamine/1-Propanol p510 1/2c
71-23-8       110-86-1  374.8691  -412.2861  .3110  1-Propanol/Pyridine p513 1/2c
71-23-8       80-62-6  504.0900  125.6451  .2993  1-Propanol/Methylmethacrylate p514 1/2c
71-23-8       109-60-4  340.0210  111.7437  .3005  1-Propnaol/Propylacetate p517 1/2c
71-23-8       123-51-3  12.2207  -31.8303  .3033  1-Propanol/3-Methylbutanol p521 1/2c
392-56-3       71-23-8  922.5224  528.5894  .2937  Hexafluorobenzene/1-Propnaol p526 1/2c
71-23-8       108-90-7  456.2867  538.5114  .2946  1-Propnaol/Chlorobenzene p527 1/2c
71-43-2       71-23-8  874.2419  285.7774  .2899  Benzene/1-Propanol p537 1/2c
71-23-8       109-06-8  529.6444  -608.3163  .3054  1-Propanol/2-Methylpyridine p538 1/2c
71-23-8       108-99-6  479.7439  -540.4699  .3045  1-Propanol/3-Methylpyridine p539 1/2c
71-23-8       108-89-4  523.8291  -603.0924  .3084  1-Propanol/4-Methylpyridine p540 1/2c
110-82-7       71-23-8  1707.7883  353.2705  .5914  Cyclohexane/1-Propanol p542 1/2c
110-54-3       71-23-8  1092.1470  480.6740  .2940  Hexane/1-Propanol p550 1/2c
71-23-8       142-84-7  617.3558  -459.5845  .2892  1-Propanol/Dipropylamine p551 1/2c
121-44-8       71-23-8  991.6157  -435.2018  .3067  Triethylamine/1-Propanol p552 1/2c
71-23-8       107-46-0  1615.0711  -498.3638  .1028  1-Propanol/Hexamethyldisiloxane p555 1/2c
71-23-8       108-88-3  25.6220  922.0009  .175e-1  1-Propanol/Toluene p556 1/2c
71-23-8       108-48-5  472.9353  -545.1853  .2960  1-Propnaol/2,6-Dimethylpyridine p557 1/2c
71-23-8       142-82-5  1198.9720  1377.2975  .5193  1-Propanol/Heptane p573 1/2c
71-23-8       100-41-4  563.6173  475.5966  .2921  1-Propanol/Ethylbenzene p574 1/2c
71-23-8       111-65-9  1109.3040  334.2112  .2907  1-Propanol/Octane p576 1/2c
71-23-8       124-18-5  945.3159  520.2926  .2895  1-Propanol/Decane p577 1/2c
71-23-8       112-30-1  1068.0694  -588.3325  .2958  1-Propanol/1-Decanol 579 1/2c
67-56-1       7732-18-5  -189.0469  792.8020  .2999  Methanol/Water p61 1/1a
67-56-1       67-66-3  -105.4657  1335.3416  .2873  Methanol/Chloroform p23 1/2a
67-56-1       64-17-5  67.2902  -70.5092  .3009  Methanol/Ethanol p55 1/2a
67-56-1       67-64-1  296.2432  118.0803  .3003  Methanol/Acetone
67-56-1       71-43-2  761.7553  1094.8556  .4893  Methanol/Benzene p232 1/2a
67-56-1       110-82-7  1313.9316  1862.4639  .4410  Methanol/CycloHexane p243 1/2a
67-56-1       108-88-3  884.0230  1008.0037  .4064  Methanol/Toluene p269 1/2a
67-56-1       142-82-5  1566.4390  1598.3126  .4408  Methanol/n-Heptane
7732-18-5       67-64-1  1324.9767  814.1435  .5663  Water/Acetone 1/1a
7732-18-5       78-93-3  653.9718  1883.6007  .3607  Water/2-Butanone p277 1/1a
7732-18-5       108-95-2  2419.3354  1844.3794  .6308  Water/Phenol p394 1/1a
64-17-5       7732-18-5  -57.9601  1241.7396  .2937  Ethanol/Water
64-17-5       67-66-3  -285.3881  1289.2198  .2909  Ethanol/Chloroform p285 1/2a
64-17-5       67-64-1  375.3497  45.3706  .3006  Ethanol/Acetone p323 1/2a
64-17-5       78-93-3  437.1923  33.6363  .3022  Ethanol/2-Butanone p342 1/2a
64-17-5       71-43-2  255.3591  1047.1959  .2970  Ethanol/Benzene p405 1/2a
64-17-5       110-82-7  761.7739  1393.7993  .4376  Ethanol/CycloHexane p441 1/2a
64-17-5       108-88-3  542.4128  772.4394  .2937  Ethanol/Toluene p470 1/2a
64-17-5       142-82-5  1114.2947  1305.9242  .4758  Ethanol/n-Heptane p489 1/2a
64-17-5       106-42-3  1020.8405  889.3461  .6180  Ethanol/p-Xylene p500 1/2a
67-64-1       67-66-3  -651.1909  301.8389  .3054  Acetone/Chloroform p92 1/3+4
67-64-1       71-43-2  -396.4935  886.5703  .2971  Acetone/Benzene p195 1/3+4
67-64-1       108-95-2  -754.9547  -280.3830  .3086  Acetone/Phenol
67-64-1       110-82-7  429.8705  727.6490  .2925  Acetone/Cyclohexane p213 1/3+4
67-64-1       108-88-3  -247.9492  727.5102  .2950  Acetone/Toluene p236 1/3+4
78-93-3       71-43-2  -644.8573  898.3999  .1563  2-Butanone/Benzene p291 1/3+4
78-93-3       110-82-7  605.7381  235.4493  .2963  2-Butanone/CycloHexane p297 1/3+4
78-93-3       108-88-3  503.0737  -181.5533  .2996  2-Butanone/Toluene p308 1/3+4
78-93-3       142-82-5  681.5104  931.4616  .9809  2-Butanone/n-Heptane p311 1/3+4
71-43-2       67-66-3  -227.3671  -86.1025  .3062  Benzene/Chloroform 1/7
71-43-2       108-95-2  373.4202  318.1885  .2986  Benzene/Phenol p482 1/2d
71-43-2       108-88-3  60.1980  -51.0865  .3019  Benzene/Toluene p282 1/7
71-43-2       106-42-3  -50.2635  14.2180  .3056  Benzene/pXylene p310 1/7
108-88-3       67-66-3  -583.6169  629.2214  .2974  Toluene/Chloroform p352 1/7
108-88-3       106-42-3  226.4602  -241.7457  .2874  Toluene/pXylene p444 1/7
108-87-2       108-95-2  2587.8730  -439.4469  .2836  MethylCycloHexane/Phenol p482 1/2d
7732-18-5       79-34-5  2435.8879  102.7658  .972e-1  Water/1,1,2,2-tetrachloroethane p64 1/1a
109-87-5       7732-18-5  1608.0700  1818.2947  .4898  Dimethoxymethane/Water p252 1/1a
107-10-8       7732-18-5  -455.9152  1301.6396  .2981  Propylamine/Water p267 1/1a
78-93-3       7732-18-5  674.4614  1809.8868  .3536  2-Butanone/Water p279 1/1a
78-84-2       7732-18-5  1166.8333  1090.0262  .2862  Isobutyraldehyde/Water p280 1/1a
109-99-9       7732-18-5  915.7450  1725.0977  .4522  Tetrahydrofuran/Water p286 1/1a
7732-18-5       123-91-1  715.9592  548.8965  .2920  Water/1,4-Dioxane p303 1/1a
141-78-6       7732-18-5  1285.9880  1606.0820  .4393  Ethylacetate/Water p304 1/1a
7732-18-5       126-33-0  1160.1372  467.9008  .5573  Water/Sulfolane p307 1/1a
7732-18-5       127-19-5  75.5965  328.8977  .3009  Water/N,n-dimethylacetamide p325 1/1a
7732-18-5       110-91-8  -803.1654  1732.7268  .2954  Water/Morpholine p327 1/1a
7732-18-5       71-36-3  2633.6951  504.0381  .4447  Water/n-Butanol p336 1/1a
78-83-1       7732-18-5  639.8173  2491.0163  .4385  isobutanol/Water p339 1/1a
75-65-0       7732-18-5  471.7718  2030.8877  .5155  Tert-butanol/Water p343 1/1a
60-29-7       7732-18-5  1544.0251  2086.4776  .3792  Diethylether/Water p344 1/1a
7732-18-5       110-63-4  1310.8994  1920.1402  .5778  Water/1,4-Butanediol p346 1/1a
7732-18-5       513-85-9  2531.7402  -758.0034  .3020  Water/2,3-Butanediol p351 1/1a
7732-18-5       111-46-6  1186.7304  -99.9000  .2974  Water/Diethylenegylcol p352 1/1a
109-73-9       7732-18-5  160.3429  2104.4002  .6379  n-Butylamine/Water p354 1/1a
109-89-7       7732-18-5  -169.1652  1372.3121  .2932  Diethylamine/Water p355 1/1a
7732-18-5       98-01-1  2602.6374  436.9686  .3950  Water/Furfural p361 1/1a
7732-18-5       110-86-1  1835.0881  419.8087  .6802  Water/Pyridine p374 1/1a
7732-18-5       120-94-5  -239.6197  573.8298  .3055  Water/n-Methylpyrrolidone p380 1/1a
7732-18-5       123-51-3  3633.5330  -494.8389  .2816  Water/3-Methylbutanol p382 1/1a
7732-18-5       75-85-4  19947.2334  -15910.4563  .56e-2  Water/2-Methyl-2-butanol p384 1/1a
7732-18-5       108-95-2  2385.3714  282.6970  .4942  Water/Phenol p399 1/1a
7732-18-5       62-53-3  11965.5274  -7391.5468  .235e-1  Water/Aniline p401 1/1a
7732-18-5       109-06-8  1979.5492  197.0009  .6371  Water/2-Methylpyridine p402 1/1a
7732-18-5       108-99-6  2559.3708  418.7524  .5361  Water/3-Methylpyridine p406 1/1a
7732-18-5       108-89-4  2325.9141  162.3029  .5622  Water/4-Methylpyridine p408 1/1a
7732-18-5       100-63-0  1473.9606  -66.4169  .811e-1  Water/Phenylhydrazine p409 1/1a
7732-18-5       108-94-1  2983.8991  -171.6660  .2673  Water/Cyclohexanone p410 1/1a
7732-18-5       141-79-7  2121.4973  101.3068  .1504  Water/Mesityloxide p411 1/1a
7732-18-5       108-93-0  2232.9727  641.3504  .4399  Water/Cyclohexanol p415 1/1a
7732-18-5       123-86-4  3805.0038  918.2419  .2951  Water/n-Butylacetate p418 1/1a
7732-18-5       123-42-2  1323.2731  845.9826  .6780  Water/Diacetonealcohol p419 1/1a
108-20-3       7732-18-5  2665.1471  4202.0746  .5409  Diisopropylether/Water p421 1/1a
7732-18-5       111-27-3  2991.1845  -464.8054  .1563  Water/1-Hexanol p426 1/1a
7732-18-5       626-93-7  1880.1699  489.1746  .2938  Water/2-Hexanol p427 1/1a
7732-18-5       111-76-2  1914.0077  220.0262  .4776  Water/2-butoxy-ethanol p433 1/1a
121-44-8       7732-18-5  -5096.5280  28437.1380  .381e-1  Triethylamine/Water p444 1/1a
7732-18-5       100-51-6  4689.8409  301.3998  .3168  Water/Benzylalcohol p450 1/1a
7732-18-5       108-48-5  2222.5960  831.9908  .5706  Water/2,6-Dimethylpyridine p457 1/1a
7732-18-5       123-92-2  1874.8967  856.9565  .3734  Water/Isopentylacetate p459 1/1a
7732-18-5       98-86-2  725.1364  858.8268  0.  Water/Acetophenone p460 1/1a
7732-18-5       91-22-5  11675.1604  -3887.1802  .902e-1  Water/Quinoline p464 1/1a
7732-18-5       98-82-8  2986.1161  -84.8485  .860e-1  Water/Isopropylbenzene p465 1/1a
56-23-5       71-43-2  -4.9421  84.0212  .3055  Tetrachloromethane/Benzene p58 1/7
71-43-2       75-25-2  893.4167  -566.9011  .2551  Benzene/Tribromomethane p59 1/7
67-66-3       71-43-2  176.8791  -288.2136  .3061  Chloroform/Benzene p72 1/7
74-88-4       71-43-2  294.4424  -185.2944  .3013  MethylIodide/Benzene p84 1/7
71-43-2       75-52-5  273.5119  524.9030  .2961  Benzene/Nitromethane p88 1/7
75-15-0       71-43-2  161.2943  431.5524  .3008  CarbonDisulfide/Benzene p90 1/7
76-13-1       71-43-2  -53.1528  551.9630  .3013  1,1,2-Trichloro-1,2,2-Trifluoroethane/Benzene p111 1/7
71-43-2       127-18-4  -94.1122  288.6566  .3023  Benzene/Tetrachloroethylene p112 1/7
71-43-2       79-01-6  140.5075  -127.4605  .3064  Benzene/Trichloroethylene p118 1/7
71-43-2       76-01-7  -225.8274  197.7460  .3030  Benzene/Pentachloroethane p119 1/7
71-43-2       79-34-5  -73.7504  -250.7743  .3055  Benzene/1,1,2,2-tetrachloroethane p120 1/7
71-55-6       71-43-2  -73.4845  97.5682  .3038  1,1,1-trichloroethane/Benzene p121 1/7
71-43-2       106-93-4  -100.9240  300.0048  .3046  Benzene/1,2-Dibromoethane p136 1/7
71-43-2       107-06-2  58.8289  -39.5526  .3035  Benzene/1,2-Dichloroethane p158 1/7
75-03-6       71-43-2  394.5891  298.1172  .3004  EthylIodide/Benzene p160 1/7
71-43-2       123-39-7  1512.9737  639.2332  .5231  Benzene/n-Methylformamide p161 1/7
71-43-2       79-24-3  527.2886  -57.1531  .3004  Benzene/Nitroethane p163 1/7
71-43-2       67-68-5  810.5440  408.5646  .6691  Benzene/Dimethylsulfoxide p166 1/7
71-43-2       107-15-3  490.0693  560.0207  .2957  Benzene/Ethylenediamine p170 1/7
71-43-2       68-12-2  736.7867  -251.4046  .3074  Benzene/n,n-dimethylformamide p184 1/7
71-43-2       108-03-2  -157.3069  595.6615  .3012  Benzene/1-Nitropropane p185 1/7
71-43-2       79-46-9  1088.4773  -446.4137  .3068  Benzene/2-Nitropropane p186 1/7
71-43-2       110-02-1  -347.2708  503.5971  .3045  Benzene/Thiophene p188 1/7
109-73-9       71-43-2  65.9717  67.1231  .3024  Butylamine/Benzene p201 1/7
75-64-9       71-43-2  -344.6666  757.9930  .3067  Tert.Butylamine/Benzene p202 1/7
109-89-7       71-43-2  52.3512  -42.2029  .2813  Diethylamine/Benzene p207 1/7
71-43-2       110-86-1  541.0855  -319.8327  .2795  Benzene/Pyridine p221 1/7
71-43-2       392-56-3  1085.4557  -715.7662  .2989  Benzene/Hexafluorobenzene p226 1/7
71-43-2       106-46-7  1441.9721  -865.5699  .2830  Benzene/P-Dichlorobenzene p238 1/7
71-43-2       108-86-1  1538.3464  -819.5924  .3214  Benzene/Bromobenzene p241 1/7
71-43-2       108-90-7  700.4097  -450.6274  .3251  Benzene/Chlorobenzene p251 1/7
71-43-2       462-06-6  277.6641  -292.2391  .3040  Benzene/Fluorobenzene p252 1/7
71-43-2       98-95-3  1311.3264  -523.3212  .3110  Benzene/Nitrobenzene p254 1/7
71-43-2       62-53-3  776.8671  -178.3464  .2990  Benzene/Aniline p266 1/7
71-43-2       108-91-8  717.4228  -684.6315  .2908  Benzene/Cyclohexylamine p274 1/7
71-43-2       121-44-8  130.6061  -27.3294  .3037  Benzene/Triethylamine p277 1/7
71-43-2       100-47-0  1390.4880  -636.1853  .2851  Benzene/Benzonitrile p278 1/7
71-43-2       108-88-3  111.1157  -121.2437  .3033  Benzene/Toluene p301 1/7
71-43-2       100-60-7  52.3967  94.0417  .3020  Benzene/n-Methylcyclohexylamine p304 1/7
71-43-2       100-42-5  -643.5999  970.4264  .3110  Benzene/Styrene p305 1/7
71-43-2       100-41-4  -70.8372  57.0902  .3034  Benzene/Ethylbenzene p306 1/7
71-43-2       108-38-3  -454.1872  615.2806  .2878  Benzene/m-Xylene p307 1/7
71-43-2       106-42-3  -50.2635  14.2180  .3056  Benzene/p-Xylene p310 1/7
71-43-2       91-66-7  85.2080  104.9548  .3019  Benzene/n,n-Diethylaniline p312 1/7
71-43-2       98-82-8  1915.7178  -810.5032  .3693  Benzene/Isopropylbenzene p322 1/7
71-43-2       103-65-1  -192.1433  141.5054  .3032  Benzene/Propylbenzene p323 1/7
71-43-2       92-06-8  64.1947  78.6570  .3022  Benzene/m-Terphenyl p327 1/7
56-23-5       108-88-3  -69.6810  95.3839  .3041  Tetrachloromethane/Toluene p351 1/7
67-66-3       108-88-3  629.2214  -583.6169  .2974  Chloroform/Toluene p352 1/7
75-15-0       108-88-3  -591.6879  1052.8580  .2439  CarbonDisulfide/Toluene p356 1/7
79-01-6       108-88-3  185.3799  -250.7688  .3062  Trichloroethylene/Toluenep 370 1/7
75-05-8       108-88-3  790.7250  724.0955  .9353  Acetonitrile/Toluene p373 1/7
624-83-9       108-88-3  -167.8974  104.6027  .3029  Methylisocyanate/Toluene p376 1/7
107-06-2       108-88-3  -217.7768  251.5704  .3097  1,2-Dichloroethane/Toluene p384 1/7
108-88-3       79-24-3  537.4434  21.7626  .3011  Toluene/Nitroethane p385 1/7
108-88-3       67-68-5  1063.2839  192.0041  .2898  Toluene/Dimethylsulfoxide p386 1/7
108-88-3       107-15-3  432.7908  592.5054  .2969  Toluene/Ethylenediamine p387 1/7
107-12-0       108-88-3  -95.6685  717.0741  .3009  Propionitrile/Toluene p388 1/7
108-88-3       68-12-2  -2260.2463  3666.1775  .711e-1  Toluene/n,n-Dimethylformamide p394 1/7
110-02-1       108-88-3  510.1471  -197.5696  .3015  Thiophene/Toluene 395 1/7
108-88-3       126-33-0  5175.2573  224.8869  .4600  Toluene/Sulfolane p399 1/7
109-89-7       108-88-3  91.4853  -153.9388  5.1012  Diethylamine/Toluene p400 1/7
108-88-3       110-86-1  264.6428  -60.3423  .2992  Toluene/Pyridine p407 1/7
392-56-3       108-88-3  668.6525  -666.7128  .2414  Hexafluorobenzene/Toluene p414 1/7
108-88-3       108-86-1  -47.4722  15.0630  .3049  Toluene/Bromobenzene p415 1/7
108-88-3       108-90-7  -40.5158  15.0972  .3037  Toluene/Chlorobenzene p417 1/7
462-06-6       108-88-3  386.4643  -304.6112  .3083  Fluorobenzene/Toluene p421 1/7
108-88-3       98-95-3  806.4313  -288.9774  .2969  Toluene/Nitrobenzene p422 1/7
108-88-3       109-06-8  396.5492  -97.1224  .3028  Toluene/2-Methylpyridine p432 1/7
108-88-3       108-99-6  -490.8706  1036.9557  .2963  Toluene/3-Methylpyridine p433 1/7
108-88-3       100-47-0  -676.6725  1239.9195  .3000  Toluene/Benzonitrile p435 1/7
108-88-3       100-41-4  663.0837  -482.5109  .3005  Toluene/Ethylbenzene p443 1/7
108-88-3       106-42-3  226.4602  -241.7457  .2874  Toluene/p-Xylene p444 1/7
107-13-1       100-42-5  598.0263  -130.4323  .3023  Acrylonitrile/Styrene p446 1/7
100-41-4       100-42-5  -539.7919  813.9959  .3466  Ethylbenzene/Styrene p451 1/7
100-42-5       103-65-1  649.8687  -453.4673  .3067  Styrene/Propylbenzene p457 1/7
56-23-5       100-41-4  -172.3762  122.4657  .3034  Tetrachloromethane/Ethylbenzene p464 1/7
75-05-8       100-41-4  1102.5396  5.3234  .2980  Acetonitrile/Ethylbenzene p465 1/7
107-13-1       100-41-4  1304.6073  -338.2481  .2994  Acrylonitrile/Ethylbenzene p467 1/7
109-89-7       100-41-4  928.9662  -553.9006  .3457  Diethylamine/Ethylbenzene p468 1/7
108-90-7       100-41-4  357.7079  -307.8057  .3076  Chlorobenzene/Ethylbenzene p469 1/7
100-41-4       98-95-3  519.4154  -64.0219  .3003  Ethylbenzene/Nitrobenzene p470 1/7
100-41-4       62-53-3  243.6463  384.0030  .2989  Ethylbenzene/Aniline p475 1/7
100-41-4       98-82-8  26.2560  -27.6358  .3043  Ethylbenzene/Isopropylbenzene p476 1/7
100-41-4       104-51-8  -789.9294  957.1492  .3026  Ethylbenzene/Butylbenzene p477 1/7
56-23-5       108-38-3  -232.4578  163.8924  .3047  Tetrachloromethane/m-Xylene p480 1/7
108-38-3       68-12-2  308.9034  548.6670  .2960  m-Xylene/n,n-Dimethylformamide p481 1/7
110-86-1       108-38-3  -78.8985  351.0029  .3009  Pyridine/m-Xylene p482 1/7
108-38-3       62-53-3  -259.4169  1034.4099  .2992  m-Xylene/Aniline p483 1/7
106-42-3       108-38-3  282.0248  -254.9358  .3085  p-Xylene/m-Xylene p485 1/7
107-06-2       95-47-6  718.2538  -479.1971  .2870  1,2-Dichloroethane/o-Xylene p490 1/7
107-15-3       95-47-6  1357.7269  -110.2727  .1967  Ethylenediamine/o-Xylene p491 1/7
95-47-6       68-12-2  559.7795  332.8093  .2947  o-Xylene/n,n-Dimethylformamide p493 1/7
56-23-5       106-42-3  -192.9687  121.7193  .3044  Tetrachloromethane/p-Xylene p498 1/7
75-05-8       106-42-3  1413.0042  -210.0314  .2938  Acetonitrile/p-Xylene p499 1/7
107-06-2       106-42-3  848.1184  -557.9036  .2733  1,2-Dichloroethane/p-Xylene p500 1/7
106-42-3       68-12-2  153.1239  722.4999  .2942  p-Xylene/n,n-Dimethylformamide p501 1/7
392-56-3       106-42-3  1004.5491  -949.1003  .2906  Hexafluorobenzene/p-Xylene p507 1/7
108-90-7       106-42-3  -395.6312  359.7555  .3055  Chlorobenzene/p-Xylene p508 1/7
106-42-3       62-53-3  311.9792  408.2084  .2972  p-Xylene/Aniline p509 1/7
56-23-5       98-82-8  -106.8166  13.4903  .3033  Tetrachloromethane/Isopropylbenzene p510 1/7
103-65-1       98-95-3  329.7212  143.3943  .3007  Propylbenzene/Nitrobenzene p511 1/7
95-63-6       526-73-8  878.2759  -655.9008  .3261  1,2,4-Trimethylbenzene/1,2,3-Trimethylbenzene p515 1/7
104-51-8       98-95-3  370.6529  140.0817  .3003  Butylbenzene/Nitrobenzene p527 1/7
99-87-6       62-53-3  -118.3505  885.1196  .2950  P-Cymene/Aniline p529 1/7
91-57-6       90-12-0  -615.6730  811.3338  .3353  2-Methylnaphthalene/1-Methylnaphthalene p531 1/7
"""


def extract_ip_by_casn(source_data):
    ip_data = {}
    for line in source_data.split('\n'):
        clean_line = line.strip()
        if clean_line == '':
            continue
        data = re.split(r'\ +', clean_line, maxsplit=5)
        casn1, casn2, biga_12, biga_21, tau, *rest = data
        id1 = casn_to_identifier().get(casn1, None)
        id2 = casn_to_identifier().get(casn2, None)
        if id1 is None:
            print(casn1, 'Comp 1 was not found in the database, corresponding line', rest)
        if id2 is None:
            print(casn2, 'Comp 2 was not found in the database, corresponding line', rest)

        ip_data[(casn1, casn2)] = (id1, id2, biga_12, biga_21, tau, rest[0])
        ip_data[(casn2, casn1)] = (id2, id1, biga_21, biga_12, tau, rest[0])

    return ip_data


f = open('nrtl.txt', 'wt')
pprint.pprint(extract_ip_by_casn(NRTL_IP_DATA), f)
f.close()
