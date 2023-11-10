import numpy as np


class Newton:

    def __init__(self, symbols: np.ndarray, poids: np.ndarray):
        """
        Initialisation de la classe Newton avec la liste des symboles
        :param symbols: liste des symboles
        """
        self.M = len(symbols)
        """self.A = np.matrix([0] * self.M)
        for k, s in enumerate(symbols):
            self.A[0, k] = abs(s)"""
        self.A = poids
        self.pis = []

    def f0(self, pi: np.ndarray) -> float:
        """
        Evalue f0 au point pi
        :param pi: le point d'evaluation
        :return: resultat de f0
        """
        tot = 0
        for k in range(self.M):
            tot += pi[k] * np.log2(pi[k])
        return tot

    def grad_f0(self, pi: np.ndarray) -> np.ndarray:
        """
        Evalue le gradient de f0 au point pi
        :param pi: le point d'evaluation
        :return: resultat du gradient
        """
        return np.array([np.log2(pi[k]) + 1 / np.log(2) for k in range(self.M)])

    def hess_f0(self, pi: np.ndarray) -> np.ndarray:
        """
        Matrice Hessienne de f0 au point pi
        :param pi: le point d'evaluation
        :return: la hessienne
        """
        return np.diag([1 / (pi[k] * np.log(2)) for k in range(self.M)])

    def run(self, pi_0: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Fait les itérations de l'algorithme de Newton
        :param pi_0: point de départ de l'algorithme
        :param epsilon: critere d'arret
        :return: le point minimum
        """

        hess = self.hess_f0(pi_0)
        # 1ere matrice des conditions KKT pour le calcul de la direction
        mat1 = np.block([
            [hess, self.A.transpose()],
            [self.A, np.zeros((1, 1))]
        ])
        # Matrice contenant d0 et v0
        calc = np.linalg.inv(mat1) * np.matrix(np.block([-self.grad_f0(pi_0), 0])).T
        d = np.matrix(calc[:-1])
        v = calc[-1][0, 0]

        pi = pi_0
        self.pis.append(np.array([x for x in pi]))

        # Critere d'arret
        while 1 / 2 * d.T * hess * d > epsilon:
            # print("Evolution of pi:", pi, pi + d.T.__array__()[0], 1 / 2 * d.T * hess * d)
            # Deplacement de pi dans la direction d
            pi += 0.1 * d.T.__array__()[0]
            self.pis.append(np.array([x for x in pi]))

            # De nouveau, on calcule d et v d'apres les conditions KKT
            hess = self.hess_f0(pi)
            mat1 = np.block([
                [hess, self.A.transpose()],
                [self.A, np.zeros((1, 1))]
            ])
            calc = np.linalg.inv(mat1) * np.matrix(np.block([-self.grad_f0(pi), 0])).T
            d = np.matrix(calc[:-1])
            v = calc[-1][0, 0]

        return pi
