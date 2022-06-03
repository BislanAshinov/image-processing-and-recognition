import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import cv2 as cv


def select_cards(img, contours=None):
    external = None
    if contours is None:
        contours, x = cv.findContours(image=img, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_NONE)

    contours_c = list(contours).copy()
    max_area, max_area2, max_i = 0, 0, 0
    areas = []
    for i in range(len(contours)):
        curr_a = cv.contourArea(contours[i])
        if curr_a > max_area:
            max_i = i
            max_area2 = max_area
            max_area = curr_a

        areas.append(curr_a)
    card_contours = []

    if max_area > max_area2 * 2:
        max_area = max_area2
        external = contours_c[max_i]
        contours_c.pop(max_i)

    for cnt in contours_c:
        if 20 * cv.contourArea(cnt) >= max_area:
            card_contours.append(cnt)

    return card_contours, external


def get_figures(bin_img, cards, approx=False):
    all_figs = []

    for i in range(len(cards)):
        card0 = cards[i]
        im_copy = np.zeros_like(bin_img)
        cv.fillPoly(im_copy, pts=[card0], color=(255, 255, 255))

        im_card0 = bin_img.copy()
        im_card0[im_copy == 0] = 0

        card0_contours, _ = cv.findContours(image=im_card0, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_NONE)

        card0_areas = []
        area_cont_dict0 = {}

        for i in range(len(card0_contours)):
            curr_a = cv.contourArea(card0_contours[i])
            card0_areas.append(curr_a)
            area_cont_dict0[curr_a] = card0_contours[i]

        sorted_areas0 = dict(sorted(area_cont_dict0.items(), key=lambda x: x[0], reverse=True))

        sort_keys = list(sorted_areas0.keys())
        fig_cnt = None
        fig_j = -1
        for i in range(len(card0_areas)):
            if sort_keys[i] > sort_keys[i + 1] * 2:
                fig_cnt = sorted_areas0[sort_keys[i + 1]]
                fig_j = i + 1
                break

        if list(sorted_areas0.keys())[fig_j] < 1300:
            im_card0 = cv.erode(im_card0, np.ones((1, 2)), iterations=2)
            card0_contours, _ = cv.findContours(image=im_card0, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_NONE)
            card0_areas = []
            area_cont_dict0 = {}

            for i in range(len(card0_contours)):
                curr_a = cv.contourArea(card0_contours[i])
                card0_areas.append(curr_a)
                area_cont_dict0[curr_a] = card0_contours[i]

            sorted_areas0 = dict(sorted(area_cont_dict0.items(), key=lambda x: x[0], reverse=True))
            sort_keys = list(sorted_areas0.keys())
            fig_cnt = sorted_areas0[sort_keys[0]]
            fig_j = 0

        fig_k = fig_j
        while list(sorted_areas0.keys())[fig_k] >= 1300:
            all_figs.append(sorted_areas0[sort_keys[fig_k]])
            fig_k += 1

    figures_im = np.zeros_like(bin_img)
    figures_im = cv.drawContours(image=figures_im, contours=all_figs, contourIdx=-1, color=(255, 0, 0), thickness=2,
                                 lineType=cv.LINE_AA)

    final_figs, _ = cv.findContours(image=figures_im, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)

    if approx:
        for i in range(len(final_figs)):
            epsilon = 0.0215 * cv.arcLength(final_figs[i], True)
            final_figs[i] = cv.approxPolyDP(final_figs[i], epsilon, True)

    return final_figs


def dists_to_center(figure):
    moments = cv.moments(figure)
    center = np.array([int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])])
    # center = np.mean(figure, axis=0)
    dists = np.zeros(figure.shape[0])

    for i in range(figure.shape[0]):
        dists[i] = np.sum((figure[i] - center) ** 2)

    dists = (dists - np.min(dists)) / (np.max(dists) - np.min(dists))
    return dists


def is_smooth(figure):
    dists = np.array(dists_to_center(figure))
    cnt = 0
    for i in range(3, len(figure) - 3):
        if dists[i] >= dists[i - 1] and dists[i] >= dists[i - 2] and dists[i] >= dists[i - 3] and\
                dists[i] >= dists[i + 1] and dists[i] >= dists[i + 2] and dists[i] >= dists[i + 3]:
            cnt += 1
        # if dists[i] <= dists[i - 1] and dists[i] <= dists[i - 2] and dists[i] <= dists[i - 3] and\
        #         dists[i] <= dists[i + 1] and dists[i] <= dists[i + 2] and dists[i] <= dists[i + 3]:
        #     cnt += 1
    # if dists[-1] <= dists[0] and dists[-1] <= dists[1] and dists[-1] <= dists[2] and\
    #         dists[-1] <= dists[-2] and dists[-1] <= dists[-3] and dists[-1] <= dists[-4]:
    #     cnt += 1
    if dists[-1] >= dists[0] and dists[-1] >= dists[1] and dists[-1] >= dists[2] and \
            dists[-1] >= dists[-2] and dists[-1] >= dists[-3] and dists[-1] >= dists[-4]:
        cnt += 1

    if cnt > 15:
        return True
    else:
        return False


def approx(figures):
    for i in range(len(figures)):
        epsilon = 0.0215 * cv.arcLength(figures[i], True)
        figures[i] = cv.approxPolyDP(figures[i], epsilon, True)

    return figures


def is_convex(figure):
    # flag = True
    # dists = dists_to_center(figure)
    # for i in range(1, len(figure) - 1):
    #     if dists[i] <= dists[i - 1] and dists[i] <= dists[i + 1]:
    #         flag = False
    # if dists[-1] <= dists[0] and dists[-1] <= dists[-2]:
    #     flag = False
    # elif dists[0] <= dists[-1] and dists[0] <= dists[1]:
    #     flag = False
    #
    # return flag
    return cv.isContourConvex(figure)
