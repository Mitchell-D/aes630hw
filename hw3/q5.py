import numpy as np
from matplotlib import pyplot as plt

L = np.array([.4, .8,])

A = np.array(
        [[-2, 1, 0],
         [ 1, -2, 1],
         [ 0, 1, -1]])

S_d = -1 * np.array([0.3,0.3,0.4])
S_s = -1 * np.array([0,0,1])

sigma = 5.67e-8 # W m^-2 K^-4

T_e = 255

balance_dist_lw = (np.matmul(np.linalg.inv(A), S_d))
balance_sfc_lw = (np.matmul(np.linalg.inv(A), S_s))

balance_dist_temp = balance_dist_lw**(1/4)*T_e
balance_sfc_temp = balance_sfc_lw**(1/4)*T_e


obs_lapse = np.array([-32.5, -16.25, 0])
model_lapse = balance_sfc_temp-balance_sfc_temp[-1]

""" """
T_obs = T_s + obs_lapse
# no convective flux
T_model = T_s + model_lapse

T_adjust = T_obs - T_model
#T_adjust = T_adjust-T_adjust[0]

#'''
print(T_obs)
print(T_model)
print(T_adjust)

F_adjust = (T_s - T_adjust)**4 * sigma
H = np.array([[2,-1,0],[1,1,-1]])

print(np.matmul(H, F_adjust))
#'''

#print(f"sfc abs balance: {balance_sfc_lw}")
#print(balance_sfc_lw**(1/4))

#print(f"sfc abs temps: {balance_sfc_temp}")
#print(f"sfc abs lapse: {balance_sfc_temp-balance_sfc_temp[-1]}")

exit(0)



ylabels = ["1", "2", "S"]
yticks = range(3,0,-1)

plot = False
if plot:
    fig, ax = plt.subplots()
    ax.plot(balance_dist_lw, yticks, linestyle="dashdot", color="blue",
            label="Atmospheric shortwave absorption.")
    ax.plot(balance_sfc_lw, yticks, linestyle="solid", color="red",
            label="Surface absorption only.")
    ax.set_title("Comparison of longwave-opaque profiles with emission " +
                 f"temperature {255}K")
    ax.set_yticks(yticks, ylabels)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Layer")
    plt.grid(True)
    plt.legend()
    plt.show()

# Recalibrate to be surface temperature relative
'''
balance_sfc_lw = (balance_sfc_lw/T_e)**4
balance_sfc_lw /= balance_sfc_lw[-1]
balance_sfc_lw = balance_sfc_lw**(1/4)*T_s
'''
temp = lambda altitude: T_s

B = np.array([[-1, 0, 0],
              [1, -1, 0],
              [0, 1, -1]])

#print(np.matmul(np.linalg.inv(B), np.matmul(A, balance_dist_lw)))

#obs_temp = np.array([T_s-32.5, T_s-16.25, T_s])
#balance_sfc_lw - obs_temp
