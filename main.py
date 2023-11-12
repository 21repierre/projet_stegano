from matplotlib import pyplot as plt

from stegano import *


def q_d():
    symbols = np.array([-1, 0, 1]) # Essai pour un dictionnaire de 3 modifications
    M = len(symbols)

    # Création du tableau des poids
    poids = np.array([0] * M)
    for k, s in enumerate(symbols):
        poids[k] = (abs(s))

    # Valeur de d0 entre 0 et 1, pi_0 dont la somme des coefficient fait 1
    d0 = 0.5
    pi_0 = np.array([1 / 6, 2 / 6, 3 / 6])

    # On effectue Newton puis on affiche le résultat pour vérifier si l'algorithme fonctionne correctement
    newton = Newton(symbols, poids, d0)
    result = newton.run(pi_0, 10 ** -12)
    print(result, -newton.f0(result), -newton.f0(np.matrix(np.array([1 / 6, 2 / 6, 3 / 6])).T))


def q_e():
    # On fait des essais pour M allant de 2 à 7
    for M in range(2, 8):
        if M % 2 == 0:
            # Si M est pair, on construit les symboles au format [0, 1, 2, ..., M-1]
            symbols = np.array([k for k in range(M)])
        else:
            # Si M est impair, on construit les symboles au format [-M//2, ..., -2, -1, 0, 1, 2, ..., M//2]
            symbols = np.array([-k for k in range(1, M // 2 + 1)] + [k for k in range(M // 2 + 1)])

        # Définition de pi_0 de sorte à ce que la somme fasse 1, puis définition des poids
        pi_0 = np.array([1/M for _ in range(M)])
        poids = np.matrix([0] * M)
        for k, s in enumerate(symbols):
            poids[0, k] = (abs(s)) ** 2 / 1

        #Backup de pi_0 pour l'affichage et le debugging et définition de d0 petit
        pi_back = np.array([x for x in pi_0])
        d0 = 0.5

        # Lancement de newton puis affichage des résultats
        newton = Newton(symbols, poids, d0)
        result = newton.run(pi_0, 10 ** -9)
        print(f" - Result for M={M}")
        print("\t", pi_back, -newton.f0(np.matrix(pi_back).T))
        print("\t", result, -newton.f0(result))

        #Calcul de l'itération à partir de laquelle les conditions sont vérifiées pour l'affichage du graphe
        verifD0 = 0
        for i, pi in enumerate(newton.pis):
            if (newton.A * pi)[0,0] <= d0:
                verifD0 = i
                print(pi, -newton.f0(pi))
                break
                
        # Affichage du graphe
        plt.plot([i for i in range(len(newton.pis))], [-newton.f0(pi) for pi in newton.pis], label="$M=%i$ \u2212 $k_{d_0}=%i$" % (M, verifD0))
        
    plt.legend(loc="upper right", ncol=2, fancybox=True)
    plt.xlabel("Itérations")
    plt.ylabel("f0(pi^k)")
    plt.show()


def q_steg():
    image = imread('source_pic/1.pgm')

    message = "La stéganographie est l’art de cacher un message secret dans un contenu anodin appelé contenu original (appelé cover) pour obtenir un contenu modifié (appelé stego) en minimisant toute discri- mination possible entre le contenu original (cover) et le contenu modifié (stego). La stéganographie est souvent appliquée sur des images, en niveaux de gris ou en couleur, sur les pixels (sur l’image brute) ou sur une version compressée de cette image (par exemple son format jpeg). Nous nous intéresserons dans ce projet à appliquer la stéganographie sur des images en niveaux de gris en insérant le message dans l’image brute"
    cmessage = sanitaire(message, list(make_charmap()[0].keys())) # Retire tous les caractères qui ne sont pas en minuscule ou un espace
    destImage = encode(image, cmessage, make_charmap) # Image de destination contenant le message codé

    # Création des images au format png
    im_base = Image.fromarray(image)
    im_dest = Image.fromarray(destImage)
    # im_base.show()
    # im_dest.show()
    im_dest.save('des_pic/1.png')
    im_base.save('des_pic/1_or.png')

    # On décode l'image codée pour vérifier qu'on retrouve effectivement le message
    testDec = decode(destImage, image, make_charmap()[0])
    print(testDec)


q_e()
