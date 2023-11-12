import numpy as np
from PIL import Image
from netpbmfile import imread

from newton import Newton


def make_charmap_old():
    """
    Ancienne méthode pour générer les symboles comme décrite dans le rapport
    :return:
    """
    charmap = {}

    j = -13
    for i in range(ord('a'), ord('z') + 1):
        charmap[chr(i)] = j
        j += 1
        if j == 0:
            j = 1
    charmap[' '] = 14
    symbols = np.array(list(charmap.values()))
    return charmap, symbols, len(symbols)


def make_charmap():
    """
    Nouvelle méthode pour générer les symboles basée sur les fréquences des lettres dans la langue francaise
    :return:
    """
    chars = ['e', 'a', ' ', 's', 't', 'i', 'r', 'n', 'u', 'l', 'o', 'd', 'm', 'c', 'p', 'v', 'h', 'g', 'f', 'b', 'q', 'j', 'x', 'z', 'y', 'k', 'w']
    charmap = {}

    for i in range(0, len(chars)-1, 2):
        charmap[chars[i]] = (i + 1) // 2 + 1
        charmap[chars[i + 1]] = -((i + 1) // 2) - 1
    charmap[chars[-1]] = len(chars) // 2 + 1

    symbols = np.array(list(charmap.values()))
    symbols.sort()
    return charmap, symbols, len(symbols)


def encode(original_pic: np.ndarray, message: str, symbols_gen) -> np.ndarray:
    """
    Cache un message dans une image
    :param original_pic: image d'origine sour la forme d'une matrice
    :param message: le message nétoyé a cacher
    :param symbols_gen: la fonction permettant de générer les symboles et le dictionnaire associé
    :return: la nouvelle image contenant le message caché
    """
    charmap, symbols, M = symbols_gen()

    # Génération d'un vecteur pi_0
    pi_0 = np.array([1 / M for _ in range(M)])
    print(pi_0, sum(pi_0))

    # Création du tableau des poids
    poids = np.matrix([0] * M)
    for k, s in enumerate(symbols):
        poids[0, k] = (abs(s)) ** 2 / 1 # Les caractères les moins fréquents ont un poids beaucoup plus élevé.

    # On choisit la distortion moyenne comme étant la moyenne des poids
    d0 = np.mean(poids)

    # Calcul du vecteur pi maximisant f0
    newton = Newton(symbols, poids, d0)
    pi = newton.run(pi_0, 10 ** -9)

    print(pi, sum(pi))

    # On ajoute les modifications du message dans l'image
    flat_image = original_pic.flatten()
    lastPixel = 0
    for i in range(len(message)):
        proba = pi[charmap[message[i]] + 13] if charmap[message[i]] < 0 else pi[charmap[message[i]] + 12]
        # Recherche d'un pixel dans lequel on peut insérer la modification courante
        while np.random.random() > proba or (flat_image[lastPixel] + charmap[message[i]]) > 255:
            lastPixel += 1
        flat_image[lastPixel] = (flat_image[lastPixel] + charmap[message[i]])
        lastPixel += 1
    print(f"Last pixel modified: {lastPixel}")
    destImage = flat_image.reshape(original_pic.shape)
    return destImage


def decode(modified_pic: np.ndarray, origin_pic: np.ndarray, charmap: dict[str, int]):
    """
    A partir de l'image originale et de l'image contenant le message, extrait le message caché
    :param modified_pic: l'image contenant le message caché
    :param origin_pic: l'image originale
    :param charmap: le dictionnaire caractère/modification
    :return: le message caché
    """
    # Inverse le sens du dictionnaire lettre:modification -> modification:lettre
    inv_map = {v: k for k, v in charmap.items()}
    m_flat = modified_pic.flatten()
    o_flat = origin_pic.flatten()
    mess = ""
    for i in range(len(m_flat)):
        # Si un pixel a une valeur différente entre les 2 images, alors elle vaut une modification qui correspond à une lettre
        diff = int(m_flat[i]) - int(o_flat[i])
        if diff != 0:
            mess += inv_map[diff]
    return mess

def sanitaire(message, alphabet):
    """
    Enlève les caractères du message non compris dans l'alphabet
    :param message:
    :param alphabet:
    :return:
    """
    r_mess = ""
    for x in message:
        if x in alphabet:
            r_mess += x

    return r_mess
