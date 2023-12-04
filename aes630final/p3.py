import numpy as np
import matplotlib.pyplot as plt

CH = 0.0015
CE = 0.0015
inso = 412 * .9 * .92
rho = 1.2 # kg / m^3
cp = 1004 # J / kg / K
L = 2.5e6 # J / kg
Ta = 28 # C
A_Ts = np.arange(20, 36, 2) # C
A_qs = np.array([14.6, 16.4, 18.6, 21.0, 23.6, 26.6, 29.8, 33.4]) # g / kg
sigma = 5.67e-8 # W /m^2 / K^4


def p1():
    qa = 15 # g / kg
    V = 5 # m / s

    Qb = lambda T: 0.94 * sigma * (T+273.15)**4 * (0.32 - 0.045 * qa**(1/2))
    Qs = lambda T: rho*cp*CH*V*(T-Ta)
    Es = lambda q: rho*CE*V*L*1e-3*(q-qa)

    fig,ax = plt.subplots()

    A_Qb = np.array(list(map(Qb, A_Ts)))
    A_Qs = np.array(list(map(Qs, A_Ts)))
    A_Eb = np.array(list(map(Es, A_qs)))
    ax.plot(A_Ts, A_Qb, label="Radiative Loss")
    ax.plot(A_Ts, A_Qs, label="Sensible Heat")
    ax.plot(A_Ts, A_Eb, label="Moisture Flux")
    ax.plot(A_Ts, A_Qb+A_Qs+A_Eb-inso, label="Net Outgoing Flux")
    ax.set_title("Tropical Ocean Outgoing Fluxes ($V=5\\frac{m}{s}; q_a=15\\frac{g}{kg})$")
    ax.set_xlabel("Ocean Surface Temperature ($^\circ C$)")
    ax.set_ylabel("Flux Leaving Ocean ($W\,m^{-2}$)")
    ax.axhline(c="black")
    ax.legend()
    plt.grid()
    plt.show()

def p2():
    V = 7 # m / s
    Qb = lambda t: 0.94*sigma*(t[0]+273.15)**4*(0.32-0.045*(t[1]*.64)**(1/2)) # arg: (T, qa)
    Qs = lambda T: rho*cp*CH*V*(T-Ta)
    Es = lambda q: rho*CE*V*L*1e-3*(q-q*.64)

    fig,ax = plt.subplots()

    A_Qb = np.array(list(map(Qb, zip(A_Ts,A_qs))))
    A_Qs = np.array(list(map(Qs, A_Ts)))
    A_Eb = np.array(list(map(Es, A_qs)))

    ax.plot(A_Ts, A_Qb, label="Radiative Loss")
    ax.plot(A_Ts, A_Qs, label="Sensible Heat")
    ax.plot(A_Ts, A_Eb, label="Moisture Flux")
    ax.plot(A_Ts, A_Qb+A_Qs+A_Eb-inso, label="Net Outgoing Flux")
    ax.set_title("Tropical Ocean Outgoing Fluxes ($V=7\\frac{m}{s}; RH=64\%)$")
    ax.set_xlabel("Ocean Surface Temperature $^\circ C$")
    ax.set_ylabel("Flux Leaving Ocean ($W\,m^{-2}$)")
    ax.axhline(c="black")
    ax.legend()
    plt.grid()
    plt.show()

if __name__=="__main__":
    p1()
    p2()
