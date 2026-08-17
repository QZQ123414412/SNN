# ResNet20 State-LR x SNM Causal Diagnostics

- Status: **COMPLETE**
- Dataset / architecture: CIFAR-100 / ResNet20
- CSRR: disabled in every configuration
- QCFS L: 8
- Training-log checkpoint accuracy: 68.78%
- Re-evaluated ANN accuracy: 68.68%
- Pre-run A_QCFS_R0 T=32: 68.78% (gap=-0.10pp)
- Checkpoint: resnet20_L[8]_paper_bs128_noaa_seed42_lr002_resumable.pth
- Checkpoint SHA256: `1fbf69f69b56b0c258c1f4a47574aecd5c37b48e842a90e0e3ab49e8aff00dc2`
- Calibration data SHA256: `3856b5e03966e94e502b5472736e0269cd313288a07b3cbe0895d02dc80d0e18`
- Calibration: 5 x 200, alpha=0.4, ridge=0.001, per-update clip=0.25, w_under=1.0, w_over=2.5
- Time steps: [8]

## Causal Switch Matrix

| Variant | Calibration SNM | Inference SNM | State term | Post-calibration global clip |
|---|---|---|---|---|
| E_REFERENCE_STATE_LR | False | False | True | False |
| F_REFERENCE_SNM_STATE_LR | True | True | True | False |
| G_E_COEFFICIENTS_SNM_ON | False | True | True | False |
| H_F_BIAS_STATE_OFF | True | True | False | False |
| I_F_FINAL_GLOBAL_CLIP | True | True | True | True |

E/G share one unsigned calibration for each T. F/H/I share one signed calibration for each T. H disables the jointly fitted state term without refitting base/slope.

## Accuracy

| Variant | T=8 |
|---|---|
| E_REFERENCE_STATE_LR | 43.82% |
| F_REFERENCE_SNM_STATE_LR | 0.49% |
| G_E_COEFFICIENTS_SNM_ON | 49.50% |
| H_F_BIAS_STATE_OFF | 40.69% |
| I_F_FINAL_GLOBAL_CLIP | 0.44% |

## Input-driven SOPs

| Variant | T=8 |
|---|---|
| E_REFERENCE_STATE_LR | 474,373,511,028 |
| F_REFERENCE_SNM_STATE_LR | 667,998,113,300 |
| G_E_COEFFICIENTS_SNM_ON | 501,349,605,800 |
| H_F_BIAS_STATE_OFF | 795,261,991,684 |
| I_F_FINAL_GLOBAL_CLIP | 665,392,702,548 |

## Time-scale Operations

| Variant | T=8 |
|---|---|
| E_REFERENCE_STATE_LR | 0 |
| F_REFERENCE_SNM_STATE_LR | 0 |
| G_E_COEFFICIENTS_SNM_ON | 0 |
| H_F_BIAS_STATE_OFF | 0 |
| I_F_FINAL_GLOBAL_CLIP | 0 |

## Positive Spike Rate

| Variant | T=8 |
|---|---|
| E_REFERENCE_STATE_LR | 13.0374% |
| F_REFERENCE_SNM_STATE_LR | 17.3972% |
| G_E_COEFFICIENTS_SNM_ON | 13.5497% |
| H_F_BIAS_STATE_OFF | 20.3282% |
| I_F_FINAL_GLOBAL_CLIP | 17.3831% |

## Negative Spike Rate

| Variant | T=8 |
|---|---|
| E_REFERENCE_STATE_LR | 0.0000% |
| F_REFERENCE_SNM_STATE_LR | 0.7295% |
| G_E_COEFFICIENTS_SNM_ON | 0.1771% |
| H_F_BIAS_STATE_OFF | 0.3619% |
| I_F_FINAL_GLOBAL_CLIP | 0.7213% |

## Overall Spike Sparsity

| Variant | T=8 |
|---|---|
| E_REFERENCE_STATE_LR | 86.9626% |
| F_REFERENCE_SNM_STATE_LR | 81.8733% |
| G_E_COEFFICIENTS_SNM_ON | 86.2732% |
| H_F_BIAS_STATE_OFF | 79.3100% |
| I_F_FINAL_GLOBAL_CLIP | 81.8956% |

## FTBC Parameters

| Variant | T=8 |
|---|---|
| E_REFERENCE_STATE_LR | 2,064 |
| F_REFERENCE_SNM_STATE_LR | 2,064 |
| G_E_COEFFICIENTS_SNM_ON | 2,064 |
| H_F_BIAS_STATE_OFF | 1,376 |
| I_F_FINAL_GLOBAL_CLIP | 2,064 |

## FTBC Storage Bytes

| Variant | T=8 |
|---|---|
| E_REFERENCE_STATE_LR | 8,256 |
| F_REFERENCE_SNM_STATE_LR | 8,256 |
| G_E_COEFFICIENTS_SNM_ON | 8,256 |
| H_F_BIAS_STATE_OFF | 5,504 |
| I_F_FINAL_GLOBAL_CLIP | 8,256 |

## Calibration Time

| Variant | T=8 |
|---|---|
| E_REFERENCE_STATE_LR | 7.2s |
| F_REFERENCE_SNM_STATE_LR | 7.3s |
| G_E_COEFFICIENTS_SNM_ON | 7.2s |
| H_F_BIAS_STATE_OFF | 7.3s |
| I_F_FINAL_GLOBAL_CLIP | 7.3s |

## Inference Time

| Variant | T=8 |
|---|---|
| E_REFERENCE_STATE_LR | 3.0s |
| F_REFERENCE_SNM_STATE_LR | 3.3s |
| G_E_COEFFICIENTS_SNM_ON | 3.3s |
| H_F_BIAS_STATE_OFF | 3.2s |
| I_F_FINAL_GLOBAL_CLIP | 3.3s |

## Max |Coefficient| / Threshold

| Variant | T=8 |
|---|---|
| E_REFERENCE_STATE_LR | 0.4782 |
| F_REFERENCE_SNM_STATE_LR | 0.4862 |
| G_E_COEFFICIENTS_SNM_ON | 0.4782 |
| H_F_BIAS_STATE_OFF | 0.4862 |
| I_F_FINAL_GLOBAL_CLIP | 0.2500 |

## Coefficient Fraction > 0.25

| Variant | T=8 |
|---|---|
| E_REFERENCE_STATE_LR | 1.5019% |
| F_REFERENCE_SNM_STATE_LR | 4.4089% |
| G_E_COEFFICIENTS_SNM_ON | 1.5019% |
| H_F_BIAS_STATE_OFF | 5.0872% |
| I_F_FINAL_GLOBAL_CLIP | 0.0000% |

## Coefficients Changed by Final Clip

| Variant | T=8 |
|---|---|
| E_REFERENCE_STATE_LR | 0 |
| F_REFERENCE_SNM_STATE_LR | 0 |
| G_E_COEFFICIENTS_SNM_ON | 0 |
| H_F_BIAS_STATE_OFF | 0 |
| I_F_FINAL_GLOBAL_CLIP | 91 |

## Accuracy Deltas versus F Reference

| Intervention | T=8 |
|---|---|
| G_E_COEFFICIENTS_SNM_ON - F | +49.01pp |
| H_F_BIAS_STATE_OFF - F | +40.20pp |
| I_F_FINAL_GLOBAL_CLIP - F | -0.05pp |

## Per-layer Detail

### E_REFERENCE_STATE_LR, T=8

| Layer | PosRate | NegRate | Sparsity | InputSpikes | SOPs | ScaleOps |
|---|---:|---:|---:|---:|---:|---:|
| conv1.2 | 22.9514% | 0.0000% | 77.0486% | 0 | 0 | 0 |
| conv2_x.0.residual_function.2 | 18.5369% | 0.0000% | 81.4631% | 300,829,016 | 43,319,378,304 | 0 |
| conv2_x.0.act | 16.9527% | 0.0000% | 83.0473% | 242,967,457 | 34,987,313,808 | 0 |
| conv2_x.1.residual_function.2 | 4.5268% | 0.0000% | 95.4732% | 222,202,697 | 31,997,188,368 | 0 |
| conv2_x.1.act | 16.3565% | 0.0000% | 83.6435% | 59,334,114 | 8,544,112,416 | 0 |
| conv2_x.2.residual_function.2 | 6.6257% | 0.0000% | 93.3743% | 214,388,361 | 30,871,923,984 | 0 |
| conv2_x.2.act | 19.7513% | 0.0000% | 80.2487% | 86,843,769 | 12,505,502,736 | 0 |
| conv3_x.0.residual_function.2 | 10.8315% | 0.0000% | 89.1685% | 258,884,291 | 74,558,675,808 | 0 |
| conv3_x.0.act | 15.6689% | 0.0000% | 84.3311% | 329,869,324 | 28,727,986,816 | 0 |
| conv3_x.1.residual_function.2 | 4.0777% | 0.0000% | 95.9223% | 102,687,434 | 29,573,980,992 | 0 |
| conv3_x.1.act | 15.9531% | 0.0000% | 84.0469% | 26,723,304 | 7,696,311,552 | 0 |
| conv3_x.2.residual_function.2 | 3.8542% | 0.0000% | 96.1458% | 104,550,416 | 30,110,519,808 | 0 |
| conv3_x.2.act | 16.1326% | 0.0000% | 83.8674% | 25,258,807 | 7,274,536,416 | 0 |
| conv4_x.0.residual_function.2 | 7.5337% | 0.0000% | 92.4663% | 105,726,481 | 60,898,453,056 | 0 |
| conv4_x.0.act | 7.4166% | 0.0000% | 92.5834% | 130,413,008 | 20,985,934,336 | 0 |
| conv4_x.1.residual_function.2 | 3.7712% | 0.0000% | 96.2288% | 24,302,763 | 13,998,391,488 | 0 |
| conv4_x.1.act | 10.2352% | 0.0000% | 89.7648% | 12,357,574 | 7,117,962,624 | 0 |
| conv4_x.2.residual_function.2 | 4.4858% | 0.0000% | 95.5142% | 33,538,623 | 19,318,246,848 | 0 |
| conv4_x.2.act | 10.4382% | 0.0000% | 89.5618% | 14,699,143 | 8,466,706,368 | 0 |
| fc | - | - | - | 34,203,853 | 3,420,385,300 | 0 |

### F_REFERENCE_SNM_STATE_LR, T=8

| Layer | PosRate | NegRate | Sparsity | InputSpikes | SOPs | ScaleOps |
|---|---:|---:|---:|---:|---:|---:|
| conv1.2 | 30.4801% | 0.0000% | 69.5199% | 0 | 0 | 0 |
| conv2_x.0.residual_function.2 | 23.5638% | 0.7219% | 75.7143% | 399,508,624 | 57,529,241,856 | 0 |
| conv2_x.0.act | 23.3848% | 0.2169% | 76.3983% | 318,317,508 | 45,837,721,152 | 0 |
| conv2_x.1.residual_function.2 | 5.5766% | 0.9524% | 93.4710% | 309,352,228 | 44,546,720,832 | 0 |
| conv2_x.1.act | 23.6829% | 0.3819% | 75.9352% | 85,576,676 | 12,323,041,344 | 0 |
| conv2_x.2.residual_function.2 | 7.8893% | 0.8463% | 91.2644% | 315,421,769 | 45,420,734,736 | 0 |
| conv2_x.2.act | 28.0417% | 0.1718% | 71.7865% | 114,499,662 | 16,487,951,328 | 0 |
| conv3_x.0.residual_function.2 | 12.4796% | 0.5108% | 87.0096% | 369,799,682 | 106,502,308,416 | 0 |
| conv3_x.0.act | 20.1862% | 0.3639% | 79.4499% | 454,933,678 | 36,352,180,672 | 0 |
| conv3_x.1.residual_function.2 | 4.6823% | 0.9714% | 94.3463% | 134,677,065 | 38,786,994,720 | 0 |
| conv3_x.1.act | 21.2158% | 0.3461% | 78.4381% | 37,052,011 | 10,670,979,168 | 0 |
| conv3_x.2.residual_function.2 | 4.1629% | 1.0748% | 94.7623% | 141,308,091 | 40,696,730,208 | 0 |
| conv3_x.2.act | 21.9612% | 0.3038% | 77.7350% | 34,325,906 | 9,885,860,928 | 0 |
| conv4_x.0.residual_function.2 | 8.2670% | 1.2638% | 90.4692% | 145,915,849 | 84,047,529,024 | 0 |
| conv4_x.0.act | 9.3450% | 1.5136% | 89.1414% | 177,146,214 | 27,327,304,576 | 0 |
| conv4_x.1.residual_function.2 | 5.3469% | 1.6474% | 93.0057% | 35,581,528 | 20,494,960,128 | 0 |
| conv4_x.1.act | 14.4283% | 2.6590% | 82.9127% | 22,918,854 | 13,201,259,904 | 0 |
| conv4_x.2.residual_function.2 | 7.4870% | 2.8143% | 89.6987% | 55,991,725 | 32,251,233,600 | 0 |
| conv4_x.2.act | 15.5463% | 3.3514% | 81.1023% | 33,755,158 | 19,442,971,008 | 0 |
| fc | - | - | - | 61,923,897 | 6,192,389,700 | 0 |

### G_E_COEFFICIENTS_SNM_ON, T=8

| Layer | PosRate | NegRate | Sparsity | InputSpikes | SOPs | ScaleOps |
|---|---:|---:|---:|---:|---:|---:|
| conv1.2 | 22.9514% | 0.0000% | 77.0486% | 0 | 0 | 0 |
| conv2_x.0.residual_function.2 | 18.6767% | 0.4176% | 80.9057% | 300,829,016 | 43,319,378,304 | 0 |
| conv2_x.0.act | 17.3298% | 0.0599% | 82.6104% | 250,273,055 | 36,039,319,920 | 0 |
| conv2_x.1.residual_function.2 | 4.7060% | 0.3721% | 94.9219% | 227,929,546 | 32,821,854,624 | 0 |
| conv2_x.1.act | 17.6430% | 0.2520% | 82.1049% | 66,559,522 | 9,584,571,168 | 0 |
| conv2_x.2.residual_function.2 | 7.2307% | 0.1743% | 92.5950% | 234,554,401 | 33,775,833,744 | 0 |
| conv2_x.2.act | 21.1986% | 0.0875% | 78.7139% | 97,059,044 | 13,976,502,336 | 0 |
| conv3_x.0.residual_function.2 | 11.1101% | 0.1189% | 88.7710% | 279,001,129 | 80,352,325,152 | 0 |
| conv3_x.0.act | 16.1469% | 0.0684% | 83.7847% | 352,591,272 | 30,121,997,312 | 0 |
| conv3_x.1.residual_function.2 | 4.2412% | 0.2628% | 95.4961% | 106,268,330 | 30,605,279,040 | 0 |
| conv3_x.1.act | 16.6919% | 0.0941% | 83.2140% | 29,517,030 | 8,500,904,640 | 0 |
| conv3_x.2.residual_function.2 | 3.9637% | 0.2271% | 95.8092% | 110,008,831 | 31,682,543,328 | 0 |
| conv3_x.2.act | 16.8608% | 0.0782% | 83.0611% | 27,464,824 | 7,909,869,312 | 0 |
| conv4_x.0.residual_function.2 | 7.5208% | 0.1750% | 92.3041% | 111,011,085 | 63,942,384,960 | 0 |
| conv4_x.0.act | 7.7746% | 0.1572% | 92.0682% | 136,228,877 | 21,630,157,632 | 0 |
| conv4_x.1.residual_function.2 | 4.0426% | 0.1423% | 95.8151% | 25,990,961 | 14,970,793,536 | 0 |
| conv4_x.1.act | 10.8418% | 0.1858% | 88.9724% | 13,713,126 | 7,898,760,576 | 0 |
| conv4_x.2.residual_function.2 | 4.9191% | 0.2147% | 94.8662% | 36,135,182 | 20,813,864,832 | 0 |
| conv4_x.2.act | 11.2140% | 0.1189% | 88.6671% | 16,822,384 | 9,689,693,184 | 0 |
| fc | - | - | - | 37,135,722 | 3,713,572,200 | 0 |

### H_F_BIAS_STATE_OFF, T=8

| Layer | PosRate | NegRate | Sparsity | InputSpikes | SOPs | ScaleOps |
|---|---:|---:|---:|---:|---:|---:|
| conv1.2 | 30.6967% | 0.0000% | 69.3033% | 0 | 0 | 0 |
| conv2_x.0.residual_function.2 | 23.7789% | 0.7253% | 75.4958% | 402,347,528 | 57,938,044,032 | 0 |
| conv2_x.0.act | 24.7582% | 0.1547% | 75.0870% | 321,181,266 | 46,250,102,304 | 0 |
| conv2_x.1.residual_function.2 | 6.7794% | 0.6392% | 92.5814% | 326,539,100 | 47,021,630,400 | 0 |
| conv2_x.1.act | 26.7596% | 0.2246% | 73.0158% | 97,237,055 | 14,002,135,920 | 0 |
| conv2_x.2.residual_function.2 | 9.8117% | 0.4485% | 89.7398% | 353,687,178 | 50,930,953,632 | 0 |
| conv2_x.2.act | 34.3890% | 0.0632% | 65.5478% | 134,482,840 | 19,365,528,960 | 0 |
| conv3_x.0.residual_function.2 | 16.0345% | 0.2982% | 83.6673% | 451,572,258 | 130,052,810,304 | 0 |
| conv3_x.0.act | 24.2059% | 0.2856% | 75.5085% | 558,610,187 | 45,277,235,808 | 0 |
| conv3_x.1.residual_function.2 | 6.2730% | 0.5659% | 93.1611% | 160,507,391 | 46,226,128,608 | 0 |
| conv3_x.1.act | 27.8514% | 0.1546% | 71.9941% | 44,819,313 | 12,907,962,144 | 0 |
| conv3_x.2.residual_function.2 | 5.3363% | 0.5989% | 94.0648% | 183,539,643 | 52,859,417,184 | 0 |
| conv3_x.2.act | 31.3712% | 0.0964% | 68.5324% | 38,896,844 | 11,202,291,072 | 0 |
| conv4_x.0.residual_function.2 | 11.0857% | 0.6109% | 88.3035% | 206,226,123 | 118,786,246,848 | 0 |
| conv4_x.0.act | 12.7090% | 0.5237% | 86.7673% | 244,553,357 | 35,274,958,656 | 0 |
| conv4_x.1.residual_function.2 | 7.3357% | 0.5427% | 92.1216% | 43,360,775 | 24,975,806,400 | 0 |
| conv4_x.1.act | 20.8955% | 0.6320% | 78.4725% | 25,816,048 | 14,870,043,648 | 0 |
| conv4_x.2.residual_function.2 | 9.0146% | 0.8807% | 90.1047% | 70,541,454 | 40,631,877,504 | 0 |
| conv4_x.2.act | 24.0171% | 0.4339% | 75.5491% | 32,424,885 | 18,676,733,760 | 0 |
| fc | - | - | - | 80,120,845 | 8,012,084,500 | 0 |

### I_F_FINAL_GLOBAL_CLIP, T=8

| Layer | PosRate | NegRate | Sparsity | InputSpikes | SOPs | ScaleOps |
|---|---:|---:|---:|---:|---:|---:|
| conv1.2 | 30.4801% | 0.0000% | 69.5199% | 0 | 0 | 0 |
| conv2_x.0.residual_function.2 | 23.6900% | 0.7110% | 75.5990% | 399,508,624 | 57,529,241,856 | 0 |
| conv2_x.0.act | 23.4829% | 0.2193% | 76.2979% | 319,828,187 | 46,055,258,928 | 0 |
| conv2_x.1.residual_function.2 | 5.5743% | 0.9093% | 93.5164% | 310,668,682 | 44,736,290,208 | 0 |
| conv2_x.1.act | 23.6191% | 0.4015% | 75.9794% | 84,982,040 | 12,237,413,760 | 0 |
| conv2_x.2.residual_function.2 | 7.9539% | 0.8433% | 91.2027% | 314,842,322 | 45,337,294,368 | 0 |
| conv2_x.2.act | 27.8548% | 0.1799% | 71.9653% | 115,307,365 | 16,604,260,560 | 0 |
| conv3_x.0.residual_function.2 | 12.6897% | 0.4951% | 86.8152% | 367,456,335 | 105,827,424,480 | 0 |
| conv3_x.0.act | 20.1694% | 0.3624% | 79.4682% | 453,864,076 | 36,644,032,128 | 0 |
| conv3_x.1.residual_function.2 | 4.7086% | 0.9935% | 94.2979% | 134,557,279 | 38,752,496,352 | 0 |
| conv3_x.1.act | 21.1039% | 0.3720% | 78.5240% | 37,368,999 | 10,762,271,712 | 0 |
| conv3_x.2.residual_function.2 | 4.0601% | 1.0664% | 94.8735% | 140,745,016 | 40,534,564,608 | 0 |
| conv3_x.2.act | 21.6482% | 0.3346% | 78.0172% | 33,596,806 | 9,675,880,128 | 0 |
| conv4_x.0.residual_function.2 | 7.8861% | 1.2895% | 90.8244% | 144,066,590 | 82,982,355,840 | 0 |
| conv4_x.0.act | 9.3045% | 1.4053% | 89.2902% | 174,133,169 | 26,538,611,264 | 0 |
| conv4_x.1.residual_function.2 | 5.2925% | 1.6249% | 93.0826% | 35,093,816 | 20,214,038,016 | 0 |
| conv4_x.1.act | 14.4242% | 2.5494% | 83.0264% | 22,666,997 | 13,056,190,272 | 0 |
| conv4_x.2.residual_function.2 | 7.6913% | 2.7292% | 89.5795% | 55,619,223 | 32,036,672,448 | 0 |
| conv4_x.2.act | 15.6439% | 3.2784% | 81.0777% | 34,145,745 | 19,667,949,120 | 0 |
| fc | - | - | - | 62,004,565 | 6,200,456,500 | 0 |
