import os
import shutil
root_dir = "../../image"
dirs = os.listdir(root_dir)
fp = open("./data.txt", "w")
with open("./label.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        items = line.split(" ")
        print(items)
        idx = items[0]
        age = items[1]
        sex = items[-1]
        images = os.listdir(os.path.join(root_dir, "camera1_"+str(idx)))
        for image in images:
            image_path = "./image/" + "camera1_" + str(idx) + "/" + image
            x = image_path + " " + age + " " + sex
            print(x, file=fp)
    fp.close()
