import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import utils

if __name__ == '__main__':
    print("Введите название директории с изображениями")
    dir_path = input()
    img_paths = []
    img_names = []

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            img_names.append(file)
            img_paths.append(os.path.join(root, file))
    answer_dir = 'answers'
    try:
        os.mkdir(answer_dir)
    except OSError:
        pass

    with open('Results.txt', 'w') as ans_file:
        for i in range(len(img_paths)):
            print('Изображение на входе ' + img_paths[i])
            img = plt.imread(img_paths[i])
            binary_img = utils.to_bin(img)
            contour = utils.get_approx_contour(binary_img)
            extensions = utils.find_extensions(contour)

            ans_img = img.copy()
            ans, ans_img, valleys, tips = utils.markup_image(contour, extensions, img=ans_img, mode=3)
            ans_img = cv.cvtColor(ans_img, cv.COLOR_RGB2BGR)

            ans_file.write(ans+'\n')
            next_str = f'!,{img_names[i]}'
            for tip in tips:
                next_str += f',T {tip[0]} {tip[1]}'
            for val in valleys:
                next_str += f',V {val[0]} {val[1]}'
            next_str += ',?\n'
            ans_file.write(next_str)

            cv.imwrite(os.path.join(answer_dir, img_names[i]), ans_img)


