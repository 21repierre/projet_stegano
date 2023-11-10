import numpy as np
from PIL import Image
from netpbmfile import imread

from newton import Newton


def make_charmap_old():
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
    charmap, symbols, M = symbols_gen()
    pi_0 = np.array([0.1] + [0.5 / ((M - 1) * k) for k in range(1, M // 2 + 1)] + [0.5 / ((M - 1) * k) for k in range(1, M // 2 + 1)])

    poids = np.matrix([0] * M)
    for k, s in enumerate(symbols):
        poids[0, k] = (abs(s)) ** 2 / 1

    newton = Newton(symbols, poids)
    pi = newton.run(pi_0, 10 ** -9)
    """print(charmap)
    print(poids)
    print(symbols)"""

    flat_image = original_pic.flatten()

    lastPixel = 0
    for i in range(len(message)):
        proba = pi[charmap[message[i]] + 13] if charmap[message[i]] < 0 else pi[charmap[message[i]] + 12]
        while np.random.random() > proba or (flat_image[lastPixel] + charmap[message[i]]) > 255:
            lastPixel += 1
        flat_image[lastPixel] = (flat_image[lastPixel] + charmap[message[i]])
        lastPixel += 1
    print(f"Last pixel modified: {lastPixel}")
    destImage = flat_image.reshape(image.shape)
    return destImage


def decode(modified_pic: np.ndarray, origin_pic: np.ndarray, charmap: dict[str, int]):
    inv_map = {v: k for k, v in charmap.items()}
    m_flat = modified_pic.flatten()
    o_flat = origin_pic.flatten()
    mess = ""
    for i in range(len(m_flat)):
        diff = int(m_flat[i]) - int(o_flat[i])
        if diff != 0:
            mess += inv_map[diff]
    return mess

def sanitaire(message, alphabet):
    r_mess = ""
    for x in message:
        if x in alphabet:
            r_mess += x

    return r_mess

image = imread('source_pic/1.pgm')

message = "La stéganographie est l’art de cacher un message secret dans un contenu anodin appelé contenu original (appelé cover) pour obtenir un contenu modifié (appelé stego) en minimisant toute discri- mination possible entre le contenu original (cover) et le contenu modifié (stego). La stéganographie est souvent appliquée sur des images, en niveaux de gris ou en couleur, sur les pixels (sur l’image brute) ou sur une version compressée de cette image (par exemple son format jpeg). Nous nous intéresserons dans ce projet à appliquer la stéganographie sur des images en niveaux de gris en insérant le message dans l’image brute"
cmessage = sanitaire(message, list(make_charmap()[0].keys()))
destImage = encode(image, cmessage, make_charmap)

im_base = Image.fromarray(image)
im_dest = Image.fromarray(destImage)
im_base.show()
im_dest.show()
im_dest.save('des_pic/1.png')

testDec = decode(destImage, image, make_charmap()[0])
print(testDec)

# im.show()
