

import os
import argparse
import json
import random
import shutil
import numpy as np
import cv2


screen_key = ['outer_rect', 'area_rect', 'inner_rect']


def read_json(json_file):
    words_result = []
    area_rect = []
    with open(json_file, 'r', encoding='utf8') as fi:
        result = json.loads(fi.read())
        # print(result)
        # area_rect = result['area_rect']['points']
        area_rect = result['outer_rect']['points']
        # if screen_key[0] in result:
        #     area_rect = result[screen_key[0]]['points']
        # elif screen_key[1] in result:
        #     area_rect = result[screen_key[0]]['points']

        for w in result['words_result']:
            words_result.append((w['points'], w['transcription']))
    return words_result, area_rect


def get_files(image_dir, json_dir):
    files = []
    for sub_dir in os.listdir(image_dir):
        if sub_dir[0] == '.':
            continue
        for filename in os.listdir(os.path.join(image_dir, sub_dir)):
            if filename[0] == '.':
                continue
            image_path = os.path.join(image_dir, sub_dir, filename)
            json_path = os.path.join(json_dir, sub_dir, os.path.splitext(filename)[0] + '.json')
            # print(image_path)
            # print(json_path)
            if os.path.isfile(image_path) and os.path.isfile(json_path):
                files.append((image_path, json_path))
    return files


def draw_words(img, words_result):
    img_show = img.copy()
    for pts, cont in words_result:
        pts = np.array(pts).reshape((-1, 2)).astype(np.int32)
        cv2.fillConvexPoly(img_show, pts, (0, 255, 0))
    img_show = cv2.addWeighted(img, 0.5, img_show, 0.5, 0)
    return img_show


def draw_area(img, area):
    img_show = img.copy()
    pts = np.array(area).reshape((-1, 2)).astype(np.int32)
    cv2.fillConvexPoly(img_show, pts, (0, 255, 0))
    img_show = cv2.addWeighted(img, 0.8, img_show, 0.2, 0)
    cv2.polylines(img_show, [pts], True, (0, 0, 255), 3)
    return img_show


def draw_screen_area_words(img, area, words_result):
    img_show = img.copy()

    for pts, cont in words_result:
        pts = np.array(pts).reshape((-1, 2)).astype(np.int32)
        cv2.fillConvexPoly(img_show, pts, (0, 255, 0))
    img_show = cv2.addWeighted(img, 0.8, img_show, 0.2, 0)

    pts = np.array(area).reshape((-1, 2)).astype(np.int32)
    cv2.polylines(img_show, [pts], True, (0, 0, 255), 3)
    # cv2.fillConvexPoly(img_show, pts, (0, 255, 0))
    return img_show


# 病历区域
def save_img_gt_files_screen(img_dir, gt_dir, files):
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
    if not os.path.isdir(gt_dir):
        os.makedirs(gt_dir)
    for i, (image_file, json_file) in enumerate(files):
        sub_dir, img_filename = os.path.split(image_file)
        sub_dir = os.path.basename(sub_dir)                 # 子文件夹名称
        name, ext = os.path.splitext(img_filename)          # 文件名、后缀名
        img_save_path = os.path.join(img_dir, sub_dir + '.' + name + ext)
        gt_save_path = os.path.join(gt_dir, sub_dir + "." + name + ".txt")
        shutil.copyfile(image_file, img_save_path)
        _, area_rect = read_json(json_file)
        print(image_file)
        print(json_file)
        print(img_save_path)
        print(gt_save_path)
        print(area_rect)
        with open(gt_save_path, 'w') as fo:
            line = ','.join([str(x) for x in area_rect]) + ',1'
            fo.write(line + '\n')


def crop_screen(image_file, words_result, area_rect):
    img = cv2.imread(image_file, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    pts1 = np.array(area_rect).reshape((-1, 2)).astype(np.float32)
    pts2 = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    print(M, M.shape)
    dst = cv2.warpPerspective(img, M, (w, h))

    words_result_new = []
    for i, (pts, trans) in enumerate(words_result):
        print(pts, trans)
        arr = [0] * 8
        for j, (x, y) in enumerate(np.array(pts).reshape((-1, 2))):
            pt_new = cv2.perspectiveTransform(np.array([x, y]).reshape((-1, 1, 2)).astype(np.float32), M)
            print(x, y, '-->', pt_new)
            arr[j * 2 + 0] = pt_new[0][0][0]
            arr[j * 2 + 1] = pt_new[0][0][1]
        words_result_new.append((arr, trans))
    return dst, words_result_new


# 裁剪病历区域，文本行
def save_img_gt_files_words(img_dir, gt_dir, files, img_dir_show):
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
    if not os.path.isdir(img_dir_show):
        os.makedirs(img_dir_show)
    if not os.path.isdir(gt_dir):
        os.makedirs(gt_dir)
    for i, (image_file, json_file) in enumerate(files):
        sub_dir, img_filename = os.path.split(image_file)
        sub_dir = os.path.basename(sub_dir)                 # 子文件夹名称
        name, ext = os.path.splitext(img_filename)          # 文件名、后缀名
        img_save_path = os.path.join(img_dir, sub_dir + '.' + name + ext)
        img_save_path_show = os.path.join(img_dir_show, sub_dir + '.' + name + ext)
        gt_save_path = os.path.join(gt_dir, sub_dir + "." + name + ".txt")

        words_result, area_rect = read_json(json_file)
        img, words_result_new = crop_screen(image_file, words_result, area_rect)
        cv2.imwrite(img_save_path, img)
        print(image_file)
        print(json_file)
        print(img_save_path)
        print(gt_save_path)
        print(area_rect)
        with open(gt_save_path, 'w', encoding='utf-8') as fo:
            for pts, trans in words_result_new:
                line = ','.join([str(x) for x in pts]) + ',' + trans
                fo.write(line + '\n')
        img_show = draw_words(img, words_result_new)
        cv2.imwrite(img_save_path_show, img_show)
        # break


def test(files):
    for i, (image_file, json_file) in enumerate(files):
        print(i, image_file, json_file)
        words_result, area_rect = read_json(json_file)
        print(words_result)
        print(area_rect)
        img = cv2.imread(image_file, cv2.IMREAD_COLOR)
        # img_show = draw_words(img, words_result)
        # img_show_screen = draw_area(img, area_rect)
        img_area_words = draw_screen_area_words(img, area_rect, words_result)

        img_screen, words_result_new = crop_screen(image_file, words_result, area_rect)
        img_crop_words = draw_words(img_screen, words_result_new)

        filename = os.path.basename(image_file)
        # cv2.imwrite(filename + 'show_words.png', img_show)
        # cv2.imwrite(filename + "show_screen.png", img_show_screen)
        cv2.imwrite(filename + '.area_words.png', img_area_words)
        cv2.imwrite(filename + '.crop_screen_words.png', img_crop_words)
        break


def main():
    in_dir = 'D:\\data_md\\liuchenxing\\20191120'
    out_dir = 'D:\\data_md\\liuchenxing\\20191120_text'
    out_dir_screen = 'D:\\data_md\\liuchenxing\\20191120_screen'
    # in_dir = 'D:\\data_md\\liuchenxing\\20191012-raw'
    # out_dir = 'D:\\data_md\\liuchenxing\\20191012-raw_text'
    # out_dir_screen = 'D:\\data_md\\liuchenxing\\20191012-raw_screen'
    image_dir = os.path.join(in_dir, "image")
    json_dir = os.path.join(in_dir, "json")
    image_dir_train_show = os.path.join(out_dir, 'train_img_show')
    image_dir_test_show = os.path.join(out_dir, 'test_img_show')
    image_dir_train = os.path.join(out_dir, 'train_img')
    image_dir_test = os.path.join(out_dir, 'test_img')
    gt_dir_train = os.path.join(out_dir, 'train_gt')
    gt_dir_test = os.path.join(out_dir, 'test_gt')
    image_dir_train_screen = os.path.join(out_dir_screen, 'train_img')
    image_dir_test_screen = os.path.join(out_dir_screen, 'test_img')
    gt_dir_train_screen = os.path.join(out_dir_screen, 'train_gt')
    gt_dir_test_screen = os.path.join(out_dir_screen, 'test_gt')

    files = get_files(image_dir, json_dir)
    test(files)
    return
    random.shuffle(files)
    test_num = int(len(files) / 100 * 10)
    train_num = len(files) - test_num
    print(len(files), train_num, test_num)

    save_img_gt_files_words(image_dir_train, gt_dir_train, files[:train_num], image_dir_train_show)
    save_img_gt_files_words(image_dir_test, gt_dir_test, files[train_num:], image_dir_test_show)
    save_img_gt_files_screen(image_dir_train_screen, gt_dir_train_screen, files[:train_num])
    save_img_gt_files_screen(image_dir_test_screen, gt_dir_test_screen, files[train_num:])


if __name__ == '__main__':
    main()


