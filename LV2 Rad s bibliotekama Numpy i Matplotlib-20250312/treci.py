import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open('road.jpg')
image_array = np.array(image, dtype=np.float32)

# a) Posvijetliti sliku
brightened_image = np.clip(image_array * 2, 0, 255).astype(np.uint8)

# b) Druga četvrtina slike po širini
height, width, _ = image_array.shape
second_quarter = image_array[:, width // 4:width // 2, :].astype(np.uint8)

# c) Zarotirati sliku za 90 stupnjeva u smjeru kazaljke na satu
rotated_image = np.rot90(image_array, k=3).astype(np.uint8)

# d) Zrcaliti sliku
mirrored_image = np.fliplr(image_array).astype(np.uint8)




fig, axs = plt.subplots(1, 4, figsize=(20, 5))
axs[0].imshow(brightened_image)
axs[0].set_title("Posvijetljena")
axs[0].axis('off')
axs[1].imshow(second_quarter)
axs[1].set_title("Druga četvrtina")
axs[1].axis('off')
axs[2].imshow(rotated_image)
axs[2].set_title("Rotirana")
axs[2].axis('off')
axs[3].imshow(mirrored_image)
axs[3].set_title("Zrcaljena")
axs[3].axis('off')

plt.show()