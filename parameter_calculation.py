time_taken_1_GPU = 0
time_taken_2_GPU = 0
time_taken_4_GPU = 0

# speedup and effiicicney for 2 GPU's
speedup_2 = time_taken_1_GPU / time_taken_2_GPU
effiiciency_2 = speedup_2 / 2
print("Speed up for 2 GPU's: ", speedup_2)
print("Efficiency for 2 GPU's: ", effiiciency_2)

# speedup and effiicicney for 4 GPU's
speedup_4 = time_taken_1_GPU / time_taken_4_GPU
effiiciency_4 = speedup_4 / 4
print("Speed up for 4 GPU's: ", speedup_4)
print("Efficiency for 4 GPU's: ", effiiciency_4)
