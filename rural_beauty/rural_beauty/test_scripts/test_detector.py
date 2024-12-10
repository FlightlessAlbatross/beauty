from rural_beauty import sky_detector
import cv2
from matplotlib import pyplot as plt 
import os



# img_path = "data/raw/ground_level_images/scenicornot/119501_f152bdd7.jpg"
images_folder = "data/raw/ground_level_images/scenicornot"


for img_name in os.listdir(images_folder):
    img_path = f"{images_folder}/{img_name}"
    if not os.path.isfile(img_path):
        continue

    img = cv2.imread(img_path)[:,:,::-1]

    plt.figure(2)
    plt.subplot(2,1,1)
    plt.imshow(img)

    img_sky_share = sky_detector.get_sky_region_share(img)
    img_sky = sky_detector.get_sky_region_gradient(img)
    plt.figure(2)
    plt.subplot(2,1,2)
    plt.imshow(img_sky)
    plt.show()
    plt.figtext(
        0.5, 0.01,  # Centered at the bottom of the figure
        f"Sky share: {(100*img_sky_share):.2f}",
        fontsize=9,
        ha="center",  # Horizontal alignment
        va="center",  # Vertical alignment
    )

    sky_path = img_path.replace("scenicornot/", "scenicornot/sky/")
    plt.savefig(sky_path)  # Save the plot as an image
    plt.clf() # clear figure for next image. 
    del img_sky, img

print(f"images saved to {images_folder}")
