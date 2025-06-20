# samples obtained with input_state = (1,0,1,1) and uniformly random beam splitter (seed of 0). 2 loops.
samples_4modes_2loops_ref = {
    (1, 1, 0, 1): 0.28052,
    (2, 0, 0, 1): 0.10804,
    (1, 2, 0, 0): 0.07178,
    (0, 0, 1, 2): 0.04774,
    (1, 1, 1, 0): 0.09362,
    (1, 0, 0, 2): 0.17363,
    (0, 1, 2, 0): 0.0331,
    (0, 1, 0, 2): 0.01993,
    (2, 1, 0, 0): 0.00971,
    (0, 1, 1, 1): 0.02564,
    (1, 0, 2, 0): 0.02199,
    (0, 0, 3, 0): 0.01311,
    (0, 2, 1, 0): 0.0178,
    (0, 0, 0, 3): 0.01233,
    (0, 0, 2, 1): 0.01258,
    (1, 0, 1, 1): 0.0443,
    (2, 0, 1, 0): 0.01246,
    (0, 2, 0, 1): 0.0003,
    (0, 3, 0, 0): 0.00142,
}

samples_4modes_2loops_distinguishable_ref = {
    (0, 2, 0, 1): 0.08929,
    (0, 1, 1, 1): 0.40528,
    (1, 0, 1, 1): 0.12022,
    (0, 3, 0, 0): 0.01397,
    (0, 0, 2, 1): 0.08236,
    (1, 0, 2, 0): 0.01294,
    (0, 0, 1, 2): 0.05457,
    (0, 2, 1, 0): 0.07104,
    (1, 0, 0, 2): 0.00054,
    (0, 1, 2, 0): 0.03842,
    (0, 1, 0, 2): 0.014,
    (1, 1, 1, 0): 0.04521,
    (1, 1, 0, 1): 0.02904,
    (2, 0, 1, 0): 0.00729,
    (1, 2, 0, 0): 0.0089,
    (2, 1, 0, 0): 0.00167,
    (0, 0, 3, 0): 0.00505,
    (0, 0, 0, 3): 0.00019,
    (2, 0, 0, 1): 0.00002,
}

# samples obtained with input_state = (1,0,1,1) and Haar random matrix (seed of 0).
samples_haar_ref = {
    (0, 0, 2, 1): 0.06346,
    (0, 3, 0, 0): 0.095499,
    (3, 0, 0, 0): 0.165877,
    (0, 2, 0, 1): 0.036669,
    (0, 0, 3, 0): 0.092503,
    (1, 1, 0, 1): 0.067955,
    (1, 1, 1, 0): 0.075099,
    (2, 0, 0, 1): 0.01231,
    (1, 2, 0, 0): 0.081605,
    (0, 2, 1, 0): 0.106334,
    (0, 0, 1, 2): 0.024633,
    (1, 0, 0, 2): 0.016247,
    (2, 0, 1, 0): 0.033,
    (2, 1, 0, 0): 0.01867,
    (0, 1, 2, 0): 0.006906,
    (1, 0, 1, 1): 0.027,
    (1, 0, 2, 0): 0.034679,
    (0, 1, 1, 1): 0.017894,
    (0, 1, 0, 2): 0.01551,
    (0, 0, 0, 3): 0.00815,
}

# avg photon numbers at the outputs obtained with input_state = (1,0,1,1,1,1,1,0,0,1,0,1,1,1,0)
# and Haar random matrix (seed of 0).
samples_clifford_ref = {
    1: 0.077488,
    2: 0.075473,
    3: 0.079239,
    4: 0.050733,
    5: 0.05319,
    6: 0.072338,
    7: 0.062498,
    8: 0.07691,
    9: 0.063329,
    10: 0.062337,
    11: 0.061487,
    12: 0.070722,
    13: 0.072055,
    14: 0.050647,
    15: 0.071554,
}


# output obtained for multi-loop with 10% losses when input_state=[1,0,1,0] and 2 loops [1,1], seed of 0.
samples_multi_loop_with_loss_ref = {
    (0, 0, 0, 0): 0.06306,
    (0, 0, 0, 1): 0.02543,
    (0, 0, 0, 2): 0.00183,
    (0, 0, 1, 0): 0.05591,
    (0, 0, 1, 1): 0.01376,
    (0, 0, 2, 0): 0.01907,
    (0, 1, 0, 0): 0.09109,
    (0, 1, 0, 1): 0.0071,
    (0, 1, 1, 0): 0.03656,
    (0, 2, 0, 0): 0.00377,
    (1, 0, 0, 0): 0.20467,
    (1, 0, 0, 1): 0.05494,
    (1, 0, 1, 0): 0.06285,
    (1, 1, 0, 0): 0.27571,
    (2, 0, 0, 0): 0.08425,
}

# output distribution for multi-loop with 10% beam splitter losses when postselected=True, input_state=[1,0,1,0]
# and 2 loops [1,1], seed of 0.
samples_multi_loop_with_postselection_ref = {
    (0, 0, 0, 2): 0.0029,
    (0, 0, 1, 1): 0.0259,
    (0, 0, 2, 0): 0.03505,
    (0, 1, 0, 1): 0.0123,
    (0, 1, 1, 0): 0.0631,
    (0, 2, 0, 0): 0.00675,
    (1, 0, 0, 1): 0.09815,
    (1, 0, 1, 0): 0.1189,
    (1, 1, 0, 0): 0.48855,
    (2, 0, 0, 0): 0.1484,
}

# output distribution for singe-loop with g2=0.5, n_signal_detectors=2, detector_efficiency=0.2, postselected=True
samples_g2_pseudopnr_singleloop_ref = {(0, 2): 0.35, (1, 1): 0.279, (2, 0): 0.371}
