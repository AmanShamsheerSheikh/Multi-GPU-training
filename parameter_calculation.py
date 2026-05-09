# ran on 12-04-2026
time_taken_1_GPU = 123.43308782577515
time_taken_2_GPU = 66.92722129821777
time_taken_4_GPU = 174.81013369560242

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
