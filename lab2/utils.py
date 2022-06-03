import numpy as np
import cv2 as cv


def get_thumb_extension(cnt, extensions):
    """
    Find the extension of thumb and forefinger
    :param cnt: palm contour
    :param extensions: defects that are finger extensions
    :return: (extension index, shifted list of extensions)
    """
    extensions = sorted(extensions, key=lambda x: x[0][0])
    l = len(extensions)
    offsets = []
    for i in range(l):
        if extensions[i - 1][0][1] > extensions[i][0][0]:
            offsets.append(extensions[i][0][0] - extensions[i - 1][0][1] + len(cnt))
        else:
            offsets.append(extensions[i][0][0] - extensions[i - 1][0][1])
    thumb_i = np.argmax(offsets)

    extensions1 = extensions[-thumb_i-2:] + extensions[:-thumb_i-2]
    return thumb_i, extensions1


def to_bin(img):
    """
    Binarizes the image with Otsu's method
    :param img: RGB image
    :return: binary image
    """
    dilation = cv.dilate(img, np.ones((6, 6), np.uint8), iterations=1)
    blur = cv.blur(dilation, (9, 9))
    gray = cv.cvtColor(blur, cv.COLOR_RGB2GRAY)
    otsu_threshold, bin_ex = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    bin_ex[-20:] = 0
    return bin_ex


def get_approx_contour(img):
    """
    Return approx palm contour
    :param img: binary image of palm
    :return: palm contour
    """
    contours = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]

    i_max = 0
    for i in range(1, len(contours)):
        if cv.contourArea(contours[i]) > cv.contourArea(contours[i_max]):
            i_max = i
    max_contour = contours[i_max]

    epsilon = 0.0025 * cv.arcLength(max_contour, True)
    max_contour = cv.approxPolyDP(max_contour, epsilon, True)
    return max_contour


def find_extensions(cnt, convex_hull=None):
    """
    Finds defects that are finger extensions
    :param cnt: palm contour
    :param convex_hull: convex hull of palm contour
    :return: defects that are extensions
    """
    if convex_hull is None:
        convex_hull = cv.convexHull(cnt, clockwise=True, returnPoints=False)
    defects = cv.convexityDefects(cnt, convex_hull)
    angles = []
    fars = []
    start_end_dists = []
    extensions = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i][0]
        start_end_dists.append(cv.norm(cnt[s][0], cnt[e][0]))
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

        fars.append(d)
        angles.append(angle)
        if angle <= 135 / 180 * np.pi:
            if d < 1000:
                continue
            if angle <= np.pi / 3 and d < 2800:
                continue
            extensions.append(defects[i])

    if len(extensions) < 4:
        extensions.append(defects[np.array(start_end_dists).argmin()])

    extensions_sort_by_far = sorted(extensions, key=lambda x: x[0][-1], reverse=True)[:4]
    extensions_sort_by_turn = sorted(extensions_sort_by_far, key=lambda x: x[0][0])

    return extensions_sort_by_turn


def markup_image(cnt, extensions, img=None, mode=1):
    """
    Marks up answer
    :param cnt: contour of palm
    :param extensions: defects of contour that are extensions
    :param img: image to put marks if mode is 2 or 3
    :param mode: 1 - return string, 2 - put marks on img, 3 - return string and put marks on image
    :return: (markup text, image)
    """
    thumb_i, extensions1 = get_thumb_extension(cnt, extensions)
    ans = None
    tips = []
    valleys = []
    if mode == 1 or mode == 3:
        ans = ''
        for i in range(len(extensions1)):
            if extensions1[i][0][-1] > 10000:
                ans += f'{i + 1}-'
            else:
                ans += f'{i + 1}+'
        ans += '5'

    if mode == 2 or mode == 3:
        f = extensions1[thumb_i][0][2]
        if img is None:
            raise TypeError('image to put markup is None')
        for i in range(len(extensions1)):
            if extensions1[i][0][0] < extensions1[i - 1][0][1]:
                if 0 <= extensions1[i][0][0] + 4 - extensions1[i - 1][0][1] <= 6:
                    start = tuple( (cnt[extensions1[i][0][0]][0] + cnt[extensions1[i - 1][0][1]][0]) // 2 )
                else:
                    start = tuple(cnt[extensions1[i][0][0]][0])
            else:
                if 0 <= extensions1[i][0][0] - extensions1[i - 1][0][1] <= 6:
                    start = tuple( (cnt[extensions1[i][0][0]][0] + cnt[extensions1[i - 1][0][1]][0]) // 2 )
                else:
                    start = tuple(cnt[extensions1[i][0][0]][0])

            if extensions1[i][0][1] > extensions1[(i + 1) % 4][0][0]:
                if 0 <= extensions1[(i + 1) % 4][0][0] + 4 - extensions1[i][0][1] <= 6:
                    end = tuple( (cnt[extensions1[(i + 1) % 4][0][0]][0] + cnt[extensions1[i][0][1]][0]) // 2 )
                else:
                    end = tuple(cnt[extensions1[i][0][1]][0])
            else:
                if 0 <= extensions1[(i + 1) % 4][0][0] - extensions1[i][0][1] <= 6:
                    end = tuple( (cnt[extensions1[(i + 1) % 4][0][0]][0] + cnt[extensions1[i][0][1]][0]) // 2 )
                else:
                    end = tuple(cnt[extensions1[i][0][1]][0])

            defect = tuple(cnt[extensions1[i][0][2]][0])

            valleys.append(defect)
            tips.append(start)

            cv.line(img, start, defect, (0, 255, 0), 3)
            cv.line(img, end, defect, (0, 255, 0), 3)

            put_text = '+'
            if extensions1[i][0][-1] > 10000:
                put_text = '-'
            cv.putText(img, put_text, tuple(cnt[extensions1[i][0][2]][0]), cv.FONT_HERSHEY_SIMPLEX,
                       1, (255, 0, 0), 2, cv.LINE_AA)

    return ans, img, valleys, tips


