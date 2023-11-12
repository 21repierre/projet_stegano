from matplotlib import pyplot as plt

from stegano import *


def q_d():
    symbols = np.array([-1, 0, 1])
    M = len(symbols)

    rhos = {}

    for k, s in enumerate(symbols):
        rhos[s] = abs(s)

    poids = np.array([0] * M)
    for k, s in enumerate(symbols):
        poids[k] = (abs(s))

    d0 = 0.5
    pi_0 = np.array([1 / 6, 2 / 6, 3 / 6])

    newton = Newton(symbols, poids, d0)
    result = newton.run(pi_0, 10 ** -12)
    print(result, -newton.f0(result), -newton.f0(np.matrix(np.array([1 / 6, 2 / 6, 3 / 6])).T))


def q_e():
    for M in range(2, 8):
        if M % 2 == 0:
            symbols = np.array([k for k in range(M)])
        else:
            symbols = np.array([-k for k in range(1, M // 2 + 1)] + [k for k in range(M // 2 + 1)])

        pi_0 = np.array([1/M for _ in range(M)])
        rhos = {}

        for k, s in enumerate(symbols):
            rhos[s] = abs(s)
        poids = np.matrix([0] * M)
        for k, s in enumerate(symbols):
            poids[0, k] = (abs(s)) ** 2 / 1

        pi_back = np.array([x for x in pi_0])
        d0 = 0.5
        newton = Newton(symbols, poids, d0)
        result = newton.run(pi_0, 10 ** -9)
        print(f" - Result for M={M}")
        print("\t", pi_back, -newton.f0(np.matrix(pi_back).T))
        print("\t", result, -newton.f0(result))
        verifD0 = 0

        for i, pi in enumerate(newton.pis):
            if (newton.A * pi)[0,0] <= d0:
                verifD0 = i
                print(pi, -newton.f0(pi))
                break
        plt.plot([i for i in range(len(newton.pis))], [-newton.f0(pi) for pi in newton.pis], label="$M=%i$ \u2212 $k_{d_0}=%i$" % (M, verifD0))
    plt.legend(loc="upper right", ncol=2, fancybox=True)
    plt.xlabel("Itérations")
    plt.ylabel("f0(pi^k)")
    plt.show()


def q_steg():
    image = imread('source_pic/1.pgm')

    message = "La stéganographie est l’art de cacher un message secret dans un contenu anodin appelé contenu original (appelé cover) pour obtenir un contenu modifié (appelé stego) en minimisant toute discri- mination possible entre le contenu original (cover) et le contenu modifié (stego). La stéganographie est souvent appliquée sur des images, en niveaux de gris ou en couleur, sur les pixels (sur l’image brute) ou sur une version compressée de cette image (par exemple son format jpeg). Nous nous intéresserons dans ce projet à appliquer la stéganographie sur des images en niveaux de gris en insérant le message dans l’image brute"
    cmessage = sanitaire(message, list(make_charmap()[0].keys()))
    destImage = encode(image, cmessage, make_charmap)

    im_base = Image.fromarray(image)
    im_dest = Image.fromarray(destImage)
    # im_base.show()
    # im_dest.show()
    im_dest.save('des_pic/1.png')
    im_base.save('des_pic/1_or.png')

    testDec = decode(destImage, image, make_charmap()[0])
    print(testDec)


q_e()
