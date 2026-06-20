import os

for folder in [
    "dataset/gallery",
    "dataset/RGB_Gallery",
    "dataset/test_images"
]:
    count = len([
        f for f in os.listdir(folder)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ])

    print(folder, count)