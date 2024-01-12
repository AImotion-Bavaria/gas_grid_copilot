v_estimate = 0.0
# i = 1: 0.7 * 10 + 0.3*0
# i = 2: 0.7 * 10 + 0.7 * 0.8 * 10 
# i = 3: 0.7 * 10 + 0.7 * 0.7 * 0.8 * 10
# i = 4: 0.7 * 10 + 0.7 * 0.7 * 0.8 * 10 + 0.7 * 0.7 * 0.7 * 0.8 * 0.8 * 10

import numpy as np 
gamma = 0.8

for i in range(1, 100):
    print(f"V_{i}: {v_estimate}") 
    v_estimate += np.power(0.7, i) * np.power(gamma, i-1) * 10

print(v_estimate)
