from skimage import io
import numpy as np
from skimage.color import rgb2gray
import math


def prosirivanje(tekstura, broj_parcadi_vertikalno, broj_parcadi_horizontalno, duzina_parceta):
    tekstura = tekstura.astype("float")

    preklop = duzina_parceta // 6

    duzina_teksture = tekstura.shape[0]
    sirina_teksture = tekstura.shape[1]

    opseg_duzine = duzina_teksture - duzina_parceta
    opseg_sirine = sirina_teksture - duzina_parceta

    for i in range(broj_parcadi_vertikalno):
        for j in range(broj_parcadi_horizontalno):

            # NASUMICNO
            # odredjuju koje je parce na redu
            y = i * duzina_parceta
            x = j * duzina_parceta

            if i == 0 and j == 0:
                # pravi matricu u kojoj ce se formirati rezultat
                duzinaN = broj_parcadi_vertikalno * duzina_parceta
                sirinaN = broj_parcadi_horizontalno * duzina_parceta
                rezultatN = np.zeros((duzinaN, sirinaN))
                rezultatN = np.stack((rezultatN, rezultatN, rezultatN), axis=2)

            parceN = nasumicnoParce(tekstura, duzina_parceta, opseg_duzine, opseg_sirine)
            rezultatN[y:y + duzina_parceta, x:x + duzina_parceta] = parceN

            # SA PREKLOPOM
            # odredjuju koje je parce na redu
            b = i * (duzina_parceta - preklop)
            a = j * (duzina_parceta - preklop)

            if i == 0 and j == 0:
                # pravi matricu u kojoj ce se formirati rezultat
                duzinaP = (broj_parcadi_vertikalno * duzina_parceta) - (broj_parcadi_vertikalno - 1) * preklop
                sirinaP = (broj_parcadi_horizontalno * duzina_parceta) - (broj_parcadi_horizontalno - 1) * preklop
                rezultat = np.zeros((duzinaP, sirinaP))
                rezultatP = np.stack((rezultat, rezultat, rezultat), axis=2)
                rezultatI = np.stack((rezultat, rezultat, rezultat), axis=2)

                rezultatP[b:b + duzina_parceta, a:a + duzina_parceta] = parceN
                rezultatI[b:b + duzina_parceta, a:a + duzina_parceta] = parceN
            else:
                parceP, matrica_preklopa_desno, matrica_preklopa_dole = odgovarajuceParce(tekstura, opseg_duzine,
                                                                                          opseg_sirine,
                                                                                          duzina_parceta, b, a,
                                                                                          rezultatP, preklop)
                rezultatP[b:b + duzina_parceta, a:a + duzina_parceta] = parceP
                parceI = seckanje(parceP, matrica_preklopa_desno, matrica_preklopa_dole, b, a, rezultatI)
                rezultatI[b:b + duzina_parceta, a:a + duzina_parceta] = parceI

    return rezultatN.astype('uint8'), rezultatP.astype('uint8'), rezultatI.astype('uint8')


def nasumicnoParce(tekstura, duzina_parceta, opseg_duzine, opseg_sirine):
    x = np.random.randint(opseg_duzine)
    y = np.random.randint(opseg_sirine)
    rezultat = tekstura[x:x + duzina_parceta, y:y + duzina_parceta]

    return rezultat


def odgovarajuceParce(tekstura, duzina, sirina, duzina_parceta, y, x, rezultat, preklop=0, obrazacBW=[], teksturaBW=[],
                      iteracija=1, alfa=0.1, preklopljeno=False):
    if np.sum(obrazacBW):
        trenutno_parce_obrasca = obrazacBW[y:y + duzina_parceta, x:x + duzina_parceta]
        duzina_trenutnog_parceta = trenutno_parce_obrasca.shape[0]
        sirina_trenutnog_parceta = trenutno_parce_obrasca.shape[1]
    else:
        dimenzije = rezultat[y:y + duzina_parceta, x:x + duzina_parceta]
        duzina_trenutnog_parceta, sirina_trenutnog_parceta = dimenzije.shape[0], dimenzije.shape[1]
    greske = np.zeros((duzina, sirina))

    minimum = {"vrednost": np.inf, "pozicija": None}

    for i in range(duzina):
        for j in range(sirina):
            # uzima svako moguce parce, pomerajuci se za jedan piksel
            if np.sum(obrazacBW):  # prenos teksture
                trenutno_parce_teksture = teksturaBW[i:i + duzina_trenutnog_parceta, j:j + sirina_trenutnog_parceta]
                if preklopljeno:
                    parce = tekstura[i:i + duzina_trenutnog_parceta, j:j + sirina_trenutnog_parceta]
                    greska_preklopa, desni_preklop, donji_preklop = greskaPreklopa(parce, duzina_parceta, preklop, y, x,
                                                                                   rezultat)

                    greska_neslaganja = np.sum((trenutno_parce_obrasca - trenutno_parce_teksture) ** 2)

                    if iteracija == 1:
                        prethodna_greska = 0
                    else:
                        prethodna_greska = rezultat[y + preklop:y + duzina_parceta,
                                           x + preklop:x + duzina_parceta] - parce[preklop:, preklop:]
                        prethodna_greska = np.sum(prethodna_greska ** 2)

                    greska = alfa * (greska_preklopa + prethodna_greska) + (1 - alfa) * greska_neslaganja
                    greske[i, j] = greska
                else:
                    # racuna razliku obrasca i teksture
                    greska = trenutno_parce_obrasca - trenutno_parce_teksture
                    greska = np.sum(greska ** 2)
            else:  # prosirivanje i sinteza teksture
                parce = tekstura[i:i + duzina_parceta, j:j + duzina_parceta]
                greska, desni_preklop, donji_preklop = greskaPreklopa(parce, duzina_parceta, preklop, y, x, rezultat)
                greske[i, j] = greska

            if greska < minimum["vrednost"]:
                minimum["vrednost"] = greska
                minimum["pozicija"] = (i, j)

    rez = tekstura[minimum["pozicija"][0]:minimum["pozicija"][0] + duzina_trenutnog_parceta,
          minimum["pozicija"][1]:minimum["pozicija"][1] + sirina_trenutnog_parceta]

    if np.sum(obrazacBW) and not preklopljeno:
        return rez, [], []
    else:
        return rez, desni_preklop, donji_preklop


def greskaPreklopa(parce, duzina_parceta, preklop, y, x, rezultat):
    greska = 0
    desni_preklop = np.zeros((duzina_parceta, preklop, 3))
    donji_preklop = np.zeros((preklop, duzina_parceta, 3))

    if x > 0:
        rez = rezultat[y:y + duzina_parceta, x:x + preklop]
        desni_preklop = rez - parce[:rez.shape[0], :rez.shape[1]]
        greska += np.sum(desni_preklop ** 2)

    if y > 0:
        rez = rezultat[y:y + preklop, x:x + duzina_parceta]
        donji_preklop = rez - parce[:rez.shape[0], :rez.shape[1]]
        greska += np.sum(donji_preklop ** 2)

    if x > 0 and y > 0:
        rez = rezultat[y:y + preklop, x:x + preklop]
        dupli_preklop = rez - parce[:rez.shape[0], :rez.shape[1]]
        greska -= np.sum(dupli_preklop ** 2)

    return greska, np.sum(desni_preklop ** 2, axis=2), np.sum(donji_preklop ** 2, axis=2)


def seckanje(parce, matrica_preklopa_desno, matrica_preklopa_dole, y, x, rezultat):
    parce = parce.copy()
    duzina_parceta = parce.shape[0]
    sirina_parceta = parce.shape[1]
    rez = rezultat[y:y + duzina_parceta, x:x + sirina_parceta]

    if x > 0:
        for i in range(matrica_preklopa_desno.shape[0]):
            min = matrica_preklopa_desno[i, 0]
            kolona = 0
            for j in range(1, matrica_preklopa_desno.shape[1]):
                if matrica_preklopa_desno[i, j] < min:
                    min = matrica_preklopa_desno[i, j]
                    kolona = j
            parce[i, :kolona] = rez[i, :kolona]

    if y > 0:
        for j in range(matrica_preklopa_dole.shape[1]):
            min = matrica_preklopa_dole[0, j]
            vrsta = 0
            for i in range(1, matrica_preklopa_dole.shape[0]):
                if matrica_preklopa_dole[i, j] < min:
                    min = matrica_preklopa_dole[i, j]
                    vrsta = i
            parce[:vrsta, j] = rez[:vrsta, j]

    return parce


def sinteza(tekstura, duzina_parceta):
    tekstura = tekstura.astype("float")

    preklop = duzina_parceta // 6

    duzina_teksture = tekstura.shape[0]
    sirina_teksture = tekstura.shape[1]

    duzina_rezultata = duzina_teksture * 2
    sirina_rezultata = sirina_teksture * 2

    rezultat = np.zeros((duzina_rezultata, sirina_rezultata))
    rezultat = np.stack((rezultat, rezultat, rezultat), axis=2)

    opseg_duzine = duzina_teksture - duzina_parceta
    opseg_sirine = sirina_teksture - duzina_parceta

    broj_parcadi_vertikalno = math.ceil(duzina_rezultata / (duzina_parceta - preklop))
    broj_parcadi_horizontalno = math.ceil(sirina_rezultata / (duzina_parceta - preklop))

    for i in range(broj_parcadi_vertikalno):
        for j in range(broj_parcadi_horizontalno):
            b = i * (duzina_parceta - preklop)
            a = j * (duzina_parceta - preklop)

            if i == 0 and j == 0:
                parce = nasumicnoParce(tekstura, duzina_parceta, opseg_duzine, opseg_sirine)
                rezultat[b:b + duzina_parceta, a:a + duzina_parceta] = parce
            else:
                parce, matrica_preklopa_desno, matrica_preklopa_dole = odgovarajuceParce(tekstura, opseg_duzine,
                                                                                          opseg_sirine,
                                                                                          duzina_parceta, b, a,
                                                                                          rezultat, preklop)
                parce = seckanje(parce, matrica_preklopa_desno, matrica_preklopa_dole, b, a, rezultat)
                rezultat[b:b + duzina_parceta, a:a + duzina_parceta] = parce

    return rezultat.astype('uint8')


def prenos(obrazac, tekstura, duzina_parceta, broj_iteracija):
    tekstura = tekstura.astype("float")
    obrazac = obrazac.astype("float")

    teksturaBW = rgb2gray(tekstura)
    obrazacBW = rgb2gray(obrazac)

    duzina_obrasca, sirina_obrasca = obrazac.shape[0], obrazac.shape[1]
    duzina_teksture, sirina_teksture = tekstura.shape[0], tekstura.shape[1]

    for i in range(1, broj_iteracija+1):
        if i == 1:
            rezultat = np.zeros((duzina_obrasca, sirina_obrasca, 3)).astype("float")
        else:
            duzina_parceta = duzina_parceta * 2 // 3 or 2  # dimenzija parceta se u svakoj iteraciji smanjuje za 1/3
        alfa = 0.8 * (i - 1) / (broj_iteracija - 1) + 0.1  # formula iz rada
        rezultat = prenosIteracija(tekstura, duzina_teksture, sirina_teksture, duzina_parceta, teksturaBW, obrazacBW,
                                     duzina_obrasca, sirina_obrasca, alfa, i, rezultat)

    return rezultat.astype('uint8')


def prenosIteracija(tekstura, duzina_teksture, sirina_teksture, duzina_parceta, teksturaBW, obrazacBW,
                                     duzina_obrasca, sirina_obrasca, alfa, iteracija, rezultat):
    preklop = duzina_parceta // 6 or 1

    broj_parcadi_vertikalno = math.ceil(duzina_obrasca / (duzina_parceta - preklop))
    broj_parcadi_horizontalno = math.ceil(sirina_obrasca / (duzina_parceta - preklop))

    duzina = duzina_teksture - duzina_parceta
    sirina = sirina_teksture - duzina_parceta

    for i in range(broj_parcadi_vertikalno):
        for j in range(broj_parcadi_horizontalno):
            y = i * (duzina_parceta - preklop)
            x = j * (duzina_parceta - preklop)

            if i == 0 and j == 0:
                parce, _, _ = odgovarajuceParce(tekstura, duzina, sirina, duzina_parceta, y, x, rezultat, 0,
                                                obrazacBW, teksturaBW)
            else:
                parce, matrica_preklopa_desno, matrica_preklopa_dole = odgovarajuceParce(tekstura, duzina, sirina,
                                                                                          duzina_parceta, y, x,
                                                                                          rezultat, preklop, obrazacBW,
                                                                                          teksturaBW, iteracija, alfa,
                                                                                          preklopljeno=True)
                parce = seckanje(parce, matrica_preklopa_desno, matrica_preklopa_dole, y, x, rezultat)

            rezultat[y:y + duzina_parceta, x:x + duzina_parceta] = parce

    return rezultat


# Image Quilting
tekstura = io.imread("texture1.jpg")

broj_parcadi_vertikalno = 6
broj_parcadi_horizontalno = 6
duzina_parceta = 30

nasumicna, preklopljena, iseckana = prosirivanje(tekstura, broj_parcadi_vertikalno, broj_parcadi_horizontalno,
                                                 duzina_parceta)
io.imsave("nasumicna.jpg", nasumicna)
io.imsave("preklopljena.jpg", preklopljena)
io.imsave("iseckana.jpg", iseckana)

# Texture Synthesis
tekstura = io.imread("texture4.jpg")
duzina_parceta = 30

prosirena_tekstura = sinteza(tekstura, duzina_parceta)
io.imsave("prosirena tekstura.jpg", prosirena_tekstura)

# Texture Transfer
tekstura = io.imread("texture2.bmp")
obrazac = io.imread("Bill.jpg")

duzina_parceta = 30
broj_iteracija = 3  # po radu treba da bude izmedju 3 i 5

prenos_teksture = prenos(obrazac, tekstura, duzina_parceta, broj_iteracija)
io.imsave("prenos teksture.jpg", prenos_teksture)
