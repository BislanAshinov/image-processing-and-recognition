import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import cv2 as cv


def binarization(img, max_value=255, mode=cv.THRESH_BINARY):
    h, b = np.histogram(img, np.arange(max_value + 2))
    s255 = np.sum(np.arange(max_value + 1) * h)
    r255 = np.sum(h)

    def f(p, hist):
        s = np.sum(np.arange(p + 1) * h[:p + 1])
        r = np.sum(h[:p + 1])
        d1 = s / r
        d2 = (s255 - s) / (r255 - r)
        return d1 + d2 - 2 * p

    p = 0
    while h[p] == 0:
        p += 1
    fp = f(p, h)
    while fp > 0:
        p += 1
        fp = f(p, h)

    return cv.threshold(img, p, max_value, mode)


def to_binary_cards(img):
    close_img = cv.morphologyEx(img, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT,(5, 5)))
    gray_closed = cv.cvtColor(close_img, cv.COLOR_RGB2GRAY)
    _, thresh1 = binarization(gray_closed)

    return thresh1, gray_closed


def to_binary(img, cards, external=None, threshold=83, dilate_size=2, g=0.8):

    # contours_for_cards, _ = cv.findContours(image=thresh1, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_NONE)
    # cards = select_cards(thresh1)
    #
    norm_gray = normalization_cards(img, cards, external)
    gamma_dilated = gamma(cv.dilate(norm_gray, np.ones((dilate_size, dilate_size))), gamma=g)
    # gamma_dilated = gamma(cv.morphologyEx(norm_gray, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (3, 3))),
    #                    gamma=g)
    _, thr = cv.threshold(gamma_dilated, threshold, 255, cv.THRESH_BINARY)

    # # contours, _ = cv.findContours(image=thr, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_NONE)

    # # img_figs = get_figures(thr, cards, approx=True)
    return thr


def normalization_cards(img, cards, external=None):
    closed_norm = img.copy()
    cards1 = cards.copy()
    if external is not None:
        cards1.append(external)
    for card in cards1:
        im_copy = np.zeros_like(img)
        cv.fillPoly(im_copy, pts=[card], color=(255, 255, 255))

        Omin, Omax = 0, 255
        Imin, Imax = np.min(closed_norm[im_copy != 0]), np.max(closed_norm[im_copy != 0])
        a = float(Omax - Omin) / (Imax - Imin)
        b = Omin - a * Imin
        closed_norm[im_copy != 0] = a * closed_norm[im_copy != 0] + b
        closed_norm = closed_norm.astype(np.uint8)

    return closed_norm


def hist_norm(img):
    Imax = np.max(img)
    Imin = np.min(img)
    Omin, Omax = 0, 255
    a = float(Omax - Omin) / (Imax - Imin)
    b = Omin - a * Imin
    out = a * img + b
    out = out.astype(np.uint8)

    return out


def gamma(img, gamma=0.4):
    fi = img / 255.0
    out = (np.power(fi, gamma) * 255).astype(np.uint8)
    return out

