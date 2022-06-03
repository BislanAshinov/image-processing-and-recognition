import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import transforms
import morphology
import os


if __name__ == '__main__':
    print("Внимание! Программа некорректно работает с изображениями с разноцветным фоном")
    print('Введите название директории с изображениями: ')
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

    for i in range(len(img_paths)):
        print('Изображение на входе ' + img_paths[i])
        img = plt.imread(img_paths[i])
        binary1_img, gray_img = transforms.to_binary_cards(img)
        cards, ext = morphology.select_cards(binary1_img)

        binary2_img = transforms.to_binary(gray_img, cards, ext)

        figures = morphology.get_figures(binary2_img, cards)
        not_smooth = []
        answer_img = img.copy()
        for f in figures:
            answer_img = cv.drawContours(answer_img, [f], contourIdx=-1,
                                         color=(255, 0, 0), thickness=2, lineType=cv.LINE_AA)
            is_sm = morphology.is_smooth(f)
            if not is_sm:
                not_smooth.append(f)
        not_smooth = morphology.approx(not_smooth)

        for figure in not_smooth:
            text = ''
            if morphology.is_convex(figure):
                text = f'P{figure.shape[0]}C'
            else:
                text = f'P{figure.shape[0]}'

            answer_img = cv.putText(answer_img, text,
                                    (int(np.mean(figure, axis=0)[0][0]) + 10, int(np.mean(figure, axis=0)[0][1]) + 10),
                                    cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
        answer_img = cv.putText(answer_img, f'{len(figures)} cards', (20, 40),
                                cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2, cv.LINE_AA)
        print('Изображение на выходе ' + os.path.join(answer_dir, img_names[i]))
        answer_img = cv.cvtColor(answer_img, cv.COLOR_RGB2BGR)
        cv.imwrite(os.path.join(answer_dir, img_names[i]), answer_img)

