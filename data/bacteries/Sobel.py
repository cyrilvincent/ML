import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from scipy import ndimage


# Afficher l'image
def display_image(image, title="Image"):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


# Appliquer un filtre convolutionnel
def apply_convolution(image, kernel):
    return scipy.ndimage.convolve(image, kernel, mode='constant', cval=0.0)


# Exemple de filtre (détecteur de bord Sobel horizontal et vertical)
horizontal_sobel_kernel = np.array([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]])
vertical_sobel_kernel = np.array([[-1, -2, -1],
                                  [ 0,  0,  0],
                                  [ 1,  2,  1]]);


# Charger l'image (vous pouvez remplacer 'path_to_image.jpg' par le chemin de votre image)
image = plt.imread("bact.jpg")

# Convertir l'image en niveaux de gris si elle est en couleur
if image.ndim > 1:
    image = np.mean(image, axis=2)

# Afficher l'image originale
display_image(image, title="Image originale")

# Appliquer les convolutions
gx = apply_convolution(image, horizontal_sobel_kernel)
gy = apply_convolution(image, vertical_sobel_kernel)
g = np.sqrt(gx ** 2 + gy ** 2)

# Afficher l'image filtrée
display_image(g, title="Image filtrée avec le filtre Sobel")