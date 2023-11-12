import numpy as np


class Newton:

    def __init__(self, symbols: np.ndarray, poids: np.ndarray, d0: float):
        """
        Initialisation de la classe Newton avec la liste des symboles
        :param symbols: liste des symboles
        """
        self.M = len(symbols)
        self.A = np.block([[poids], [np.ones((1, self.M))]])
        self.pis = []
        self.b = np.matrix(np.array([d0, 1])).T

    def f0(self, pi: np.ndarray) -> float:
        """
        Evalue f0 au point pi
        :param pi: le point d'evaluation
        :return: resultat de f0
        """
        tot = 0
        for k in range(self.M):
            tot += pi[k, 0] * np.log2(pi[k, 0])
        return tot

    def grad_f0(self, pi: np.ndarray) -> np.ndarray:
        """
        Evalue le gradient de f0 au point pi
        :param pi: le point d'evaluation
        :return: resultat du gradient
        """
        return np.matrix(np.array([np.log2(pi[k, 0]) + 1 / np.log(2) for k in range(self.M)]))

    def hess_f0(self, pi: np.matrix) -> np.ndarray:
        """
        Matrice Hessienne de f0 au point pi
        :param pi: le point d'evaluation
        :return: la hessienne
        """
        return np.diag([1 / (pi[k, 0] * np.log(2)) for k in range(self.M)])

    def run(self, pi_0: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Fait les itérations de l'algorithme de Newton
        :param pi_0: point de départ de l'algorithme
        :param epsilon: critere d'arret
        :return: le point minimum
        """

        # Définition des matrices de base
        pi = np.matrix(pi_0).T
        v = np.matrix(np.array([1., 1.])).T
        hess = self.hess_f0(pi)
        
        # 1ere matrice des conditions KKT pour le calcul de la direction
        mat1 = np.block([
            [hess, self.A.transpose()],
            [self.A, np.zeros((2, 2))]
        ])
        grad = self.grad_f0(pi)
        
        # Matrice contenant d0 et v0
        calc = np.linalg.inv(mat1) * -np.matrix(np.block([grad, (self.A * pi - self.b).T])).T

        # On extrait d et v de la matrice
        d = np.matrix(calc[:-2])
        dv = calc[-2:]
        print(dv)

        # Ajout de pi pour tracer le graphe
        self.pis.append(np.matrix(pi))

        # Critere d'arret
        while 1 / 2 * d.T * hess * d > epsilon:
            # Deplacement de pi dans la direction d
            # calcul d'un pas pour eviter d'avoir une coordonnée de pi négative
            t = 1
            while True:
                pip = pi + t * d
                good = True
                for x in pip:
                    if x < 0:
                        t *= 0.9
                        good = False
                if good:
                    break

            # Mise à jour de pi et de v puis ajout de pi pour tracer le graphe
            pi += t * d
            v += t * dv
            self.pis.append(np.matrix(pi))

            # De nouveau, on calcule d et v d'apres les conditions KKT
            hess = self.hess_f0(pi)
            mat1 = np.block([
                [hess, self.A.transpose()],
                [self.A, np.zeros((2, 2))]
            ])
            calc = np.linalg.inv(mat1) * -np.matrix(np.block([self.grad_f0(pi) + (self.A.T * v).T, (self.A * pi - self.b).T])).T
            d = np.matrix(calc[:-2])
            dv = calc[-2:]

        return pi
