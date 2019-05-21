#Copyright 2019 Oliver Struckmeier
#Licensed under the GNU General Public License, version 3.0. See LICENSE for details

from PIL import Image, ImageFilter

"""
This function applies a filter to a masked image and can either remove 
or add differntly colored borders around objects
"""
def modify_outline(image_path, thickness):
    image = Image.open(image_path)
    image = image.convert("RGBA")
    for t in range(thickness):
        mask = image.filter(ImageFilter.FIND_EDGES)
        mask_data = mask.getdata()
        image_data = image.getdata()
        w, h = mask_data.size
        print(w, h)

        out_data = []
        for y in range(0, h):
            for x in range(0, w):
                index = x + w * y
                pixel = (0, 0, 0, 0)
                if mask_data[index][3] > 0:
                    pixel = (255, 255, 255, 0)
                else:
                    pixel = (image_data[index][0], image_data[index][1], image_data[index][2], image_data[index][3])
                out_data.append(pixel)
        image.putdata(out_data)
    image.save(image_path)

path = "/home/oli/Workspace/LeagueAI/generate_dataset/Dataset/images

image_path = "/home/oli/Workspace/LeagueAI/generate_dataset/masked_towers/red_test/tower_4.png"
modify_outline(image_path, 1)
