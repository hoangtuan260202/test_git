import cv2
import os



path_ = "cifar100/train"

for folder in os.listdir(path_):
    folder = os.path.join(path_, folder)
    if os.path.isdir(folder):
        for sub_folder in os.listdir(folder):
            sub_folder = os.path.join(folder, sub_folder)
            print(sub_folder)
            for img in os.listdir(sub_folder):
                img = os.path.join(sub_folder, img)
                img = cv2.imread(img)
                cv2.imshow("img", img)
                cv2.waitKey(0)



