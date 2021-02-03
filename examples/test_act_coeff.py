import matplotlib.pyplot as plt
from sim21.data.chemsep import pure
from sim21.data import wilson, nrtl
import numpy as np
from sim21.provider.wilson import wilson_model_log_gamma
from sim21.provider.nrtl import nrtl_model_log_gamma

def pull_data(data_source):
    actual_data = []
    for line in data_source.split('\n'):
        clean_line = line.strip()
        if clean_line == '':
            continue
        rest = [float(i) for i in clean_line.split(' ')]
        actual_data.append(rest)

    return np.array(actual_data)


ETOH_H2O_DATA_AT_323_15_K = pull_data("""20.333 0.0874 0.4341 3.4513 1.0187
20.904 0.0967 0.4549 3.3591 1.0192
22.796 0.1411 0.5120 2.8214 1.0463
23.663 0.1756 0.5372 2.4675 1.0731
24.336 0.2065 0.5562 2.2331 1.0995
24.570 0.2253 0.5636 2.0935 1.1181
25.024 0.2552 0.5761 1.9235 1.1505
25.436 0.2856 0.5890 1.7856 1.1822
25.815 0.3133 0.6008 1.6845 1.2124
26.259 0.3535 0.6133 1.5497 1.2689
26.481 0.3773 0.6226 1.4861 1.2967
26.694 0.3999 0.6311 1.4325 1.3258
26.898 0.4258 0.6397 1.3738 1.3637
27.284 0.4691 0.6563 1.2973 1.4273
27.535 0.4987 0.6660 1.2495 1.4825
27.701 0.5218 0.6741 1.2158 1.5257
28.101 0.5692 0.6971 1.1688 1.5971
28.216 0.5907 0.7050 1.1436 1.6439
28.448 0.6242 0.7205 1.1148 1.7106
28.711 0.6697 0.7434 1.0817 1.8037
28.828 0.6868 0.7523 1.0716 1.8439
29.195 0.7586 0.7940 1.0366 2.0159
29.253 0.7811 0.8081 1.0266 2.0754
29.406 0.8299 0.8429 1.0129 2.1990
29.487 0.8454 0.8555 1.0119 2.2319
29.480 0.8559 0.8639 1.0090 2.2551
29.478 0.8638 0.8699 1.0067 2.2808
29.517 0.8801 0.8849 1.0063 2.2957
29.538 0.8911 0.8948 1.0057 2.3121
29.531 0.9031 0.9065 1.0051 2.3094
29.548 0.9528 0.9512 1.0001 2.4777
29.558 0.9344 0.9337 1.0014 2.4221
29.558 0.9480 0.9470 1.0011 2.4432
29.560 0.9136 0.9151 1.0039 2.3544
29.566 0.9263 0.9273 1.0035 2.3644""")

ETOH_H2O_DATA_AT_333_15_K = pull_data("""31.647 0.0742 0.4130 3.7957 1.0037
43.756 0.4808 0.6682 1.3001 1.3988
46.998 0.8538 0.8616 1.0113 2.2317
34.540 0.1071 0.4742 3.2893 1.0172
44.336 0.5298 0.6788 1.2140 1.5151
46.987 0.8646 0.8715 1.0099 2.2372
36.840 0.1511 0.5196 2.7208 1.0425
44.447 0.5390 0.6887 1.2136 1.5017
47.020 0.8823 0.8860 1.0067 2.2855
37.611 0.1705 0.5326 2.5221 1.0597
44.935 0.5800 0.7070 1.1700 1.5687
47.045 0.8873 0.8908 1.0070 2.2878
38.387 0.1899 0.5473 2.3738 1.0726
45.282 0.6141 0.7175 1.1299 1.6590
47.048 0.8966 0.8987 1.0054 2.3137
38.999 0.2133 0.5587 2.1909 1.0939
45.557 0.6417 0.7333 1.1115 1.6975
47.060 0.9091 0.9095 1.0037 2.3524
40.175 0.2606 0.5814 1.9209 1.1373
45.881 0.6764 0.7468 1.0813 1.7974
47.055 0.9154 0.9151 1.0029 2.3712
41.230 0.3168 0.6006 1.6741 1.2052
46.218 0.7156 0.7674 1.0577 1.8931
47.055 0.9206 0.9195 1.0020 2.3957
42.157 0.3813 0.6209 1.4693 1.2918
46.338 0.7347 0.7834 1.0542 1.8951
47.044 0.9255 0.9242 1.0015 2.4039
42.635 0.4036 0.6316 1.4276 1.3170
46.547 0.7656 0.7968 1.0335 2.0218
47.026 0.9458 0.9444 1.0010 2.4238
43.368 0.4548 0.6502 1.3260 1.3917
46.868 0.8246 0.8395 1.0176 2.1504
47.039 0.9479 0.9467 1.0015 2.4180
43.730 0.4794 0.6688 1.3043 1.3917
46.915 0.8353 0.8477 1.0153 2.1756
46.994 0.9583 0.9562 0.9997 2.4807
""")

ETOH_H2O_DATA_AT_328_15_K = pull_data("""27.774 0.1161 0.4841 3.1244 1.0261
35.215 0.5273 0.6801 1.2188 1.5091
37.325 0.8436 0.8554 1.0136 2.1901
29.017 0.1445 0.5123 2.7731 1.0470
35.507 0.5541 0.6845 1.1768 1.5908
37.352 0.8502 0.8595 1.0112 2.2235
31.391 0.2338 0.5712 2.0638 1.1119
35.516 0.5626 0.6874 1.1642 1.6073
37.378 0.8616 0.8688 1.0093 2.2493
31.767 0.2506 0.5760 1.9644 1.1375
36.055 0.6194 0.7138 1.1142 1.7173
37.393 0.8714 0.8775 1.0083 2.2614
32.475 0.2938 0.5923 1.7605 1.1866
36.530 0.6841 0.7480 1.0706 1.8465
37.395 0.8798 0.8848 1.0070 2.2757
33.048 0.3306 0.6058 1.6278 1.2318
36.779 0.7174 0.7667 1.0533 1.9244
37.416 0.8854 0.8891 1.0061 2.2992
33.350 0.3551 0.6106 1.5411 1.2745
36.812 0.7276 0.7714 1.0458 1.9581
37.415 0.8925 0.8946 1.0042 2.3297
33.611 0.3777 0.6191 1.4803 1.3022
36.973 0.7558 0.7937 1.0402 1.9804
37.421 0.9002 0.9019 1.0039 2.3363
34.012 0.4123 0.6294 1.3946 1.3576
37.172 0.7979 0.8211 1.0246 2.0872
37.425 0.9131 0.9130 1.0020 2.3803
34.343 0.4470 0.6465 1.3338 1.3898
37.223 0.8165 0.8340 1.0184 2.1363
37.437 0.9342 0.9331 1.0012 2.4189
34.519 0.4598 0.6557 1.3216 1.3929
37.307 0.8334 0.8470 1.0154 2.1741
37.412 0.9566 0.9550 1.0000 2.4662
35.017 0.5127 0.6682 1.2249 1.5096
""")


def test_2comp_vle():
    c1, c2 = pure(['ethanol', 'water'])
    # u = wilson.generate_ip([c1.casn, c2.casn])
    ip_g, ip_alpha = nrtl.generate_ip([c1.casn, c2.casn])
    for temp in [323.15, 328.15, 333.15]:
        x_comp = []
        y_comp = []
        press_total = []
        results = []

        for start_x in np.linspace(0, 1, num=50):
            x_comp.append([start_x, 1 - start_x])
            vap_press = np.array([c1.vap_press(temp), c2.vap_press(temp)])
            vol = np.array([c1.liq_vol_mole(temp), c2.liq_vol_mole(temp)])
            comp = np.array([start_x, 1 - start_x])
            # log_gamma = wilson_model_log_gamma(temp, vol, comp, u)

            log_gamma = nrtl_model_log_gamma(temp, vol, comp, ip_g, ip_alpha)

            press_partial = comp*np.exp(log_gamma)*vap_press
            press_system = np.sum(press_partial)
            press_total.append(press_system)
            y_comp.append([press_partial[0]/press_system, press_partial[1]/press_system])
            results.append((start_x, 1-start_x, press_partial[0]/press_system, press_partial[1]/press_system, press_system, vap_press[0], vap_press[1]))

        results = np.array(results)
        x1 = results[:, 0]
        x2 = results[:, 1]
        y1 = results[:, 2]
        y2 = results[:, 3]
        press = results[:, 4]
        plt.plot(x1, press/1e3)
        plt.plot(y1, press/1e3)

    plt.plot(ETOH_H2O_DATA_AT_333_15_K[:, 1], ETOH_H2O_DATA_AT_333_15_K[:, 0], 'or')
    plt.plot(ETOH_H2O_DATA_AT_333_15_K[:, 2], ETOH_H2O_DATA_AT_333_15_K[:, 0], 'og')

    plt.plot(ETOH_H2O_DATA_AT_328_15_K[:, 1], ETOH_H2O_DATA_AT_328_15_K[:, 0], 'or')
    plt.plot(ETOH_H2O_DATA_AT_328_15_K[:, 2], ETOH_H2O_DATA_AT_328_15_K[:, 0], 'og')

    plt.plot(ETOH_H2O_DATA_AT_323_15_K[:, 1], ETOH_H2O_DATA_AT_323_15_K[:, 0], 'or')
    plt.plot(ETOH_H2O_DATA_AT_323_15_K[:, 2], ETOH_H2O_DATA_AT_323_15_K[:, 0], 'og')

    plt.show()


test_2comp_vle()

MEOH_ETOH_H2O_DATA_AT_323_15_K = pull_data("""20.921 0.0140 0.0821 0.0649 0.4107 1.7759 3.5746 0.9807
30.987 0.1724 0.3455 0.2999 0.4280 0.9818 1.3014 1.4128
27.718 0.0390 0.3902 0.0774 0.5773 1.0036 1.3934 1.3546
33.564 0.1735 0.7274 0.2713 0.6483 0.9545 1.0116 2.2053
28.902 0.0392 0.5385 0.0705 0.6406 0.9476 1.1670 1.5981
30.513 0.1835 0.2343 0.3545 0.3497 1.0740 1.5447 1.2520
30.148 0.0502 0.6936 0.0848 0.7110 0.9278 1.0478 1.9438
32.966 0.1855 0.5583 0.2863 0.5355 0.9256 1.0701 1.8542
30.717 0.0579 0.7886 0.0956 0.7712 0.9237 1.0179 2.1583
33.380 0.1862 0.6220 0.2882 0.5669 0.9397 1.0292 2.0401
22.176 0.0753 0.0480 0.2984 0.2161 1.6082 3.4074 0.9920
33.602 0.2123 0.5383 0.3224 0.5072 0.9281 1.0710 1.8564
27.586 0.0897 0.2404 0.1962 0.4457 1.1010 1.7383 1.1910
30.610 0.2492 0.0619 0.5777 0.1269 1.2928 2.1288 1.0597
28.447 0.0957 0.3111 0.1885 0.4803 1.0219 1.4917 1.2830
35.878 0.2851 0.5647 0.4187 0.4788 0.9572 1.0274 1.9811
29.719 0.1043 0.4371 0.1815 0.5323 0.9425 1.2279 1.4986
33.532 0.2870 0.2486 0.4812 0.2896 1.0227 1.3221 1.3367
24.676 0.1109 0.0617 0.3591 0.2145 1.4603 2.9226 1.0270
37.666 0.3276 0.6447 0.4743 0.5051 0.9899 0.9952 2.2694
32.175 0.1126 0.8125 0.1832 0.7492 0.9527 1.0042 2.3531
36.509 0.3959 0.1881 0.5886 0.2002 0.9860 1.3127 1.4965
30.949 0.1133 0.5766 0.1853 0.5962 0.9218 1.0846 1.7632
38.298 0.4222 0.3218 0.5865 0.2759 0.9654 1.1075 1.6635
31.591 0.1198 0.6711 0.1922 0.6464 0.9226 1.0306 1.9730
42.562 0.5342 0.3998 0.6776 0.2851 0.9776 1.0205 1.9463
30.457 0.1637 0.2995 0.3012 0.4091 1.0210 1.4110 1.3277
44.984 0.6649 0.1583 0.8100 0.1127 0.9913 1.0754 1.5891
33.197 0.1700 0.6708 0.2642 0.6109 0.9384 1.0228 2.1078
""")

MEOH_ETOH_H2O_DATA_AT_328_15_K = pull_data("""23.344 0.0411 0.0221 0.2197 0.1469 1.8522 4.2003 0.9993
41.574 0.2662 0.2629 0.4370 0.3159 1.0040 1.3356 1.3803
37.529 0.0488 0.6165 0.0826 0.6707 0.9360 1.0938 1.7524
38.264 0.2758 0.0261 0.6285 0.0576 1.2850 2.2636 1.0879
34.300 0.0528 0.2876 0.1129 0.5196 1.0826 1.6644 1.2094
44.882 0.2774 0.5955 0.3985 0.5108 0.9469 1.0265 2.0313
33.662 0.0548 0.2409 0.1261 0.4896 1.1437 1.8384 1.1623
40.927 0.2874 0.1285 0.5354 0.1965 1.1220 1.6743 1.1881
35.015 0.0556 0.3401 0.1100 0.5377 1.0222 1.4861 1.2919
44.421 0.2954 0.4529 0.4292 0.4138 0.9481 1.0828 1.7550
24.334 0.0564 0.0211 0.2740 0.1302 1.7539 4.0618 0.9950
46.707 0.3363 0.5429 0.4720 0.4474 0.9620 1.0252 1.9762
38.921 0.0601 0.7876 0.0996 0.7678 0.9498 1.0152 2.1502
46.124 0.3508 0.4160 0.4956 0.3660 0.9565 1.0815 1.7338
24.944 0.0657 0.0190 0.3080 0.1124 1.7344 3.9902 0.9999
43.472 0.3610 0.1177 0.6104 0.1551 1.0805 1.5301 1.2368
36.609 0.1068 0.3221 0.2037 0.4725 1.0295 1.4402 1.3135
44.330 0.3791 0.1123 0.6321 0.1427 1.0861 1.5038 1.2414
40.194 0.1070 0.7474 0.1719 0.7068 0.9503 1.0162 2.1246
48.174 0.3856 0.4997 0.5276 0.3989 0.9667 1.0234 1.9573
39.340 0.1168 0.5707 0.1867 0.5884 0.9258 1.0854 1.7934
46.406 0.3977 0.2670 0.5712 0.2554 0.9783 1.1831 1.5188
38.130 0.1561 0.3031 0.2810 0.4174 1.0114 1.4069 1.3456
47.786 0.4067 0.3810 0.5538 0.3220 0.9545 1.0752 1.7706
41.571 0.1569 0.7036 0.2444 0.6447 0.9524 1.0176 2.0966
49.840 0.4461 0.4387 0.5879 0.3425 0.9626 1.0345 1.9088
41.080 0.1800 0.5276 0.2792 0.5268 0.9374 1.0965 1.7266
49.126 0.4464 0.3535 0.5989 0.2903 0.9662 1.0732 1.7229
39.590 0.2031 0.2859 0.3534 0.3723 1.0143 1.3800 1.3447
47.014 0.4531 0.1100 0.6838 0.1209 1.0413 1.3771 1.3290
43.320 0.2188 0.6506 0.3274 0.5730 0.9526 1.0182 2.0956
50.410 0.4861 0.3283 0.6388 0.2611 0.9706 1.0657 1.7219
38.505 0.2208 0.1430 0.4474 0.2483 1.1495 1.7914 1.1650
50.994 0.4872 0.3810 0.6337 0.2917 0.9716 1.0374 1.8290
42.714 0.2373 0.4912 0.3567 0.4672 0.9439 1.0850 1.7549
56.400 0.6877 0.1116 0.8321 0.0825 0.9974 1.1046 1.5187
39.773 0.2545 0.1347 0.4946 0.2196 1.1381 1.7360 1.1771
59.840 0.7697 0.1068 0.8770 0.0711 0.9950 1.0532 1.5919
38.320 0.2638 0.0433 0.5957 0.0925 1.2751 2.1942 1.0904
""")

MEOH_ETOH_H2O_DATA_AT_333_15_K = pull_data("""31.483 0.0342 0.0436 0.1604 0.2539 1.7848 3.9516 1.0001
50.450 0.1996 0.3759 0.3180 0.4305 0.9634 1.2307 1.4946
45.826 0.0428 0.5176 0.0734 0.6304 0.9437 1.1918 1.5447
49.758 0.2025 0.3110 0.3397 0.3892 1.0008 1.3271 1.3861
46.735 0.0477 0.5830 0.0793 0.6587 0.9325 1.1268 1.6593
52.421 0.2842 0.2394 0.4642 0.2913 1.0255 1.3575 1.3446
41.075 0.0512 0.2024 0.1234 0.4703 1.1914 2.0447 1.1178
58.782 0.3434 0.5724 0.4790 0.4627 0.9793 1.0069 2.0404
33.965 0.0535 0.0485 0.2254 0.2437 1.7277 3.6729 1.0041
57.624 0.3831 0.3091 0.5428 0.2904 0.9757 1.1485 1.5615
47.991 0.0546 0.6869 0.0877 0.7068 0.9246 1.0527 1.9108
57.020 0.3929 0.2338 0.5707 0.2370 0.9901 1.2269 1.4681
48.542 0.0583 0.7452 0.0938 0.7413 0.9366 1.0290 2.0416
55.733 0.4013 0.0937 0.6574 0.1197 1.0921 1.5128 1.2288
49.034 0.0630 0.8082 0.1031 0.7822 0.9621 1.0108 2.1904
61.204 0.4112 0.5073 0.5516 0.3951 0.9797 1.0088 2.0060
41.933 0.0710 0.1914 0.1728 0.4382 1.2278 2.0556 1.1056
59.751 0.4226 0.3367 0.5759 0.2908 0.9722 1.0934 1.6553
35.893 0.0734 0.0539 0.2714 0.2395 1.6010 3.4282 1.0058
58.663 0.4592 0.1065 0.6914 0.1179 1.0553 1.3776 1.2869
37.595 0.0938 0.0597 0.3091 0.2304 1.4933 3.1156 1.0224
63.960 0.4876 0.4325 0.6293 0.3230 0.9839 1.0094 1.9130
45.871 0.0976 0.3552 0.1773 0.5011 1.0007 1.3822 1.3479
65.489 0.5288 0.3923 0.6687 0.2860 0.9866 1.0082 1.8834
43.204 0.1029 0.1835 0.2350 0.3903 1.1864 1.9663 1.1339
61.681 0.5382 0.0530 0.7747 0.0556 1.0595 1.3704 1.2789
39.641 0.1130 0.0645 0.3522 0.2226 1.4880 2.9342 1.0243
67.166 0.5740 0.3476 0.7105 0.2470 0.9898 1.0070 1.8234
48.175 0.1340 0.4094 0.2259 0.4948 0.9743 1.2418 1.4736
64.401 0.5981 0.0517 0.8042 0.0488 1.0322 1.2854 1.3503
41.369 0.1350 0.0701 0.3866 0.2161 1.4257 2.7323 1.0334
69.374 0.6311 0.2981 0.7579 0.2054 0.9910 1.0073 1.8006
49.236 0.1353 0.4930 0.2176 0.5385 0.9495 1.1461 1.6163
67.856 0.6522 0.1290 0.8004 0.1004 0.9912 1.1143 1.5379
44.910 0.1436 0.1734 0.3130 0.3421 1.1761 1.8940 1.1334
69.793 0.6845 0.1389 0.8183 0.1021 0.9923 1.0812 1.5729
50.522 0.1530 0.5503 0.2357 0.5575 0.9327 1.0898 1.7624
66.993 0.6897 0.1355 0.8226 0.0997 0.9928 1.0853 1.5557
43.353 0.1608 0.0778 0.4234 0.2077 1.3726 2.4768 1.0496
72.935 0.7419 0.1497 0.8511 0.1010 0.9939 1.0352 1.6122
51.806 0.1689 0.6112 0.2594 0.5805 0.9529 1.0468 1.8890
75.471 0.7855 0.1584 0.8755 0.0998 0.9982 0.9989 1.6629
45.052 0.1862 0.0855 0.4493 0.2003 1.3062 2.2565 1.0830
75.470 0.8184 0.0459 0.9144 0.0309 1.0007 1.0676 1.5208
""")


def test_3comp_vle():
    all_press = MEOH_ETOH_H2O_DATA_AT_328_15_K[:, 0]
    all_x1 = MEOH_ETOH_H2O_DATA_AT_328_15_K[:, 1]
    all_x2 = MEOH_ETOH_H2O_DATA_AT_328_15_K[:, 2]
    all_x3 = 1 - (all_x1 + all_x2)
    all_y1 = MEOH_ETOH_H2O_DATA_AT_328_15_K[:, 3]
    all_y2 = MEOH_ETOH_H2O_DATA_AT_328_15_K[:, 4]
    all_y3 = 1 - (all_y1 + all_y2)
    all_gamma1 = MEOH_ETOH_H2O_DATA_AT_328_15_K[:, 5]
    all_gamma2 = MEOH_ETOH_H2O_DATA_AT_328_15_K[:, 6]
    all_gamma3 = MEOH_ETOH_H2O_DATA_AT_328_15_K[:, 7]
    temp = 328.15
    c1, c2, c3 = pure(['methanol', 'ethanol', 'water'])
    # u = wilson.generate_ip([c1.casn, c2.casn, c3.casn])
    ip_g, ip_alpha = nrtl.generate_ip([c1.casn, c2.casn, c3.casn])

    vap_press = np.array([c1.vap_press(temp), c2.vap_press(temp), c3.vap_press(temp)])
    vol = np.array([c1.liq_vol_mole(temp), c2.liq_vol_mole(temp), c3.liq_vol_mole(temp)])
    results = []
    refer = []
    for i in range(len(all_x1)):
        comp = np.array([all_x1[i], all_x2[i], all_x3[i]])
        # log_gamma = wilson_model_log_gamma(temp, vol, comp, u)

        log_gamma = nrtl_model_log_gamma(temp, vol, comp, ip_g, ip_alpha)

        gamma = np.exp(log_gamma)
        press_partial = comp * gamma * vap_press
        press_system = np.sum(press_partial)
        calc_y = press_partial/press_system
        results.append([press_system*1e-3, np.log(gamma[0]), np.log(gamma[1]), np.log(gamma[2])])
        refer.append([all_press[i], np.log(all_gamma1[i]), np.log(all_gamma2[i]), np.log(all_gamma3[i])])
        # break

    results = np.array(results)
    refer = np.array(refer)

    plt.plot(refer[:, 0], results[:, 0], 'or')
    plt.plot(refer[:, 0], refer[:, 0], '-g')
    plt.show()


test_3comp_vle()
