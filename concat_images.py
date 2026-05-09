import cv2
import os


GPUS = [1,2,4]
for gpu in GPUS:
    folder_path = f'./data/plots_GPU_{gpu}'
    reads = []
    for images in os.listdir(folder_path):
        if images.startswith('gpu'):
            print(images)
            full_image_path = os.path.join(folder_path, images)
            reads.append(cv2.imread(full_image_path))
    concat = cv2.hconcat(reads)
    cv2.imwrite(f'./data/plots_GPU_{gpu}/plot_GPU_{gpu}.png',concat)
    