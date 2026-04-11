import cv2
import numpy as np
from matplotlib import pyplot as plt


def obtenir_bruit_soustraction(chemin_image, intensite_flou=21, gain=1.0):
    """
    Obtient une approximation du bruit/hautes fréquences en soustrayant
    une version floue de l'image originale.

    Parameters
    ----------
    chemin_image : image
    
    intensite_flou : int , optional
        The default is 21.
    gain : float , optional
        The default is 1.0.

    Returns
    -------
    image : image
        
    bruit : image
        image representant le bruit et les hautes fréquences.

    """
    
    image = cv2.imread(chemin_image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Impossible de charger l'image.")
        return None

    img_floue = cv2.GaussianBlur(image, (intensite_flou, intensite_flou), 0)
    bruit = cv2.addWeighted(image, gain, img_floue, -gain, 127) 

    return image, bruit


def fourier(photo):
    """
    applique la FFTSHIFT a une image

    Parameters
    ----------
    photo : image
        

    Returns
    -------
    spectre_visuel : image
        resultat visuel de la transformee de fourier

    """
    f_transform = np.fft.fft2(photo.astype(float))   
    f_shift = np.fft.fftshift(f_transform)
    
    spectre_magnitude = np.abs(f_shift)
    spectre_visuel = 20 * np.log(spectre_magnitude + 1)
    
    return spectre_visuel

chemin = "image\image4.jpg"
originale, image_bruit = obtenir_bruit_soustraction(chemin, intensite_flou=31, gain=2.0)
image_fourier = fourier(image_bruit)

if image_bruit is not None:
    # Sauvegarder le résultat
    cv2.imwrite('image_bruit_resultat.jpg', image_fourier)
    

    # Affichage
    plt.figure(figsize=(15, 5))
    plt.imshow(originale, cmap='gray'), plt.title('Originale (Gris)')
    plt.axis('off')
    plt.imshow(image_bruit, cmap='gray'), plt.title('Bruit / Hautes Fréquences')
    plt.axis('off')
    plt.imshow(image_fourier, cmap='gray'), plt.title('Fourier')
    plt.axis('off')
    plt.show()