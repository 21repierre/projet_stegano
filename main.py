import numpy as np
from matplotlib import pyplot as plt

from newton import Newton


def q_d():
    symbols = np.array([-2, 0, 2])

    rhos = {}

    for k, s in enumerate(symbols):
        rhos[s] = abs(s)

    d_0 = sum(rhos.values()) / len(rhos)

    pi_0 = np.array([0.125, 0.1, 0.125])
    newton = Newton(symbols)
    result = newton.run(pi_0, 10 ** -12)
    print(result, newton.f0(result), newton.f0(np.array([0.125, 0.1, 0.125])))


def q_e():
    for M in range(2, 8):
        if M % 2 == 0:
            symbols = np.array([k for k in range(M)])
            pi_0 = np.array([0.1] + [0.5 / ((M - 1) * k) for k in range(1, M)])
        else:
            symbols = np.array([-k for k in range(1, M // 2 + 1)] + [k for k in range(M // 2 + 1)])
            pi_0 = np.array([0.1] + [0.5 / ((M - 1) * k) for k in range(1, M // 2 + 1)] + [0.5 / ((M - 1) * k) for k in range(1, M // 2 + 1)])
        # print(symbols)
        # print(pi_0)
        rhos = {}

        for k, s in enumerate(symbols):
            rhos[s] = abs(s)

        pi_back = np.array([x for x in pi_0])
        newton = Newton(symbols)
        result = newton.run(pi_0, 10 ** -9)
        print(f" - Result for M={M}")
        print("\t", pi_back, newton.f0(pi_back))
        print("\t", result, newton.f0(result))
        plt.plot([i for i in range(len(newton.pis))], [-newton.f0(pi) for pi in newton.pis], label=f"{M}")
    plt.legend(loc="lower right")
    plt.xlabel("Itérations")
    plt.ylabel("f0(pi^k)")
    plt.show()


q_e()
