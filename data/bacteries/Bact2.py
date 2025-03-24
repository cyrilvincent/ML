
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

plt.subplot(2, 3, 1)
img = (plt.imread("bact2.png")[:, :, 0] * 255).astype(np.uint8)
plt.title("Image initiale")
plt.imshow(img, cmap="gray")

plt.subplot(2, 3, 4)
plt.title("Histogramme des niveaux de gris de l'image")
plt.hist(img.ravel(), bins=256)

plt.subplot(2, 3, 2)
plt.title("Seuillage de l'image")
threshold = 20
img = np.where(img < threshold, 0, 255)
plt.imshow(img, cmap="gray")

# mask1 = img >= threshold
# mask2 = img < threshold
# img[mask1] = 0
# img[mask2] = 255
# plt.imshow(img, cmap="gray")

plt.subplot(2, 3, 3)
plt.title("Dilation suivie d'une Erosion")
img = ndi.binary_closing(img, iterations=2)   # Produit une matrice booléenne
# print(img.max(), img.dtype)

plt.imshow(img, cmap="gray")

plt.subplot(2, 3, 5)
plt.title("Labélisation de l'image")
labeled_image, label_count = ndi.label(img)
plt.imshow(labeled_image, cmap="jet")
print(label_count)

plt.subplot(2, 3, 6)
plt.title("Distribution de la taille des bactéries")
# img2 = img.copy()
# img2[img == 255] = 1       # Sum_labels somme les valeurs des pixels pour chaque label considéré
sizes = ndi.sum_labels(img, labeled_image, range(1, label_count + 1))
sizes.sort()
plt.scatter(range(label_count), sizes)

plt.tight_layout()
plt.get_current_fig_manager().window.showMaximized()
plt.show()