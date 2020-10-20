import os
files = [f for f in os.listdir('.') if os.path.isfile(f)]
jpgs = [x for x in files if ".jpg" in x]

from PIL import Image
for file in jpgs:
    img = Image.open(file)
    ppm = file.replace('.jpg','.ppm')
    img.save(ppm)
    