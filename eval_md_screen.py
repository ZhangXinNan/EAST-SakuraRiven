import time
import torch
import subprocess
import os
import argparse
from model_md_screen import EAST_md_screen as EAST
# from detect import detect_dataset
import numpy as np
import cv2
import shutil
from PIL import Image, ImageDraw
from detect import resize_img, load_pil
from cv_util import cv2pil, pil2cv


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_accuracy(img_mask, quad_boxes):
    acc = 0
    num = 0
    num_right = 0

    return acc


def eval_model(model, img, quad_boxes, src_width, src_height):
    with torch.no_grad():
        score, geo = model(load_pil(img).to(device))
        print("score shape ", score.shape)
        img_score = score.squeeze(0).cpu().numpy()
        img_score = img_score[0].copy()
        img_score = img_score * 255.0 / (img_score.max() - img_score.min())
        img_score = img_score.astype(np.uint8)

        dst_width = img_score.shape[1]
        dst_height = img_score.shape[0]
        img_show = cv2.resize(pil2cv(img), (img_score.shape[1], img_score.shape[0]))
        img_gt = np.zeros((img_score.shape[0], img_score.shape[1]))
        r, g, b = cv2.split(img_show)
        print(img_show.shape, img_show.dtype)
        print(img_score.shape, img_score.dtype)
        # img_show = cv2.bitwise_and(img_show, img_score)
        g = cv2.bitwise_and(g, img_score)
        r = cv2.bitwise_and(r, img_score)
        b = cv2.bitwise_and(b, img_score)
        img_mask = cv2.merge((r, g, b))

        quad_boxes = np.array(quad_boxes)
        print(quad_boxes)
        quad_boxes[:, [1,3,5,7]] = quad_boxes[:, [1,3,5,7]] * dst_width / src_width
        quad_boxes[:, [0,2,4,6]] = quad_boxes[:, [0,2,4,6]] * dst_height / src_height
        quad_boxes = quad_boxes.reshape((-1, 4, 2)).astype(np.int32)
        print(quad_boxes)
        for quad in quad_boxes:
            cv2.fillConvexPoly(img_gt, quad, (255))
            cv2.polylines(img_mask, [quad], True, (0, 255, 255), 3)
        # cv2.polylines(img_mask, )
    return img_score, img_mask, img_gt


def get_boxes(gt_file):
    quad_boxes = []
    with open(gt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # print(lines)
        for line in lines:
            arr = line.strip().split(',')
            if len(arr) < 8:
                continue
            arr = [int(float(x)) for x in arr]
            quad_boxes.append(arr[:8])
    return quad_boxes


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir',
                        default='D:\\data_md\\liuchenxing\\20191120_screen\\test_img')
    parser.add_argument('--gt_dir', default='D:\\data_md\\liuchenxing\\20191120_screen\\test_gt')
    parser.add_argument('--score_dir', default='D:\\data_md\\liuchenxing\\20191120_screen\\test_show')
    parser.add_argument('--model_path', default='./pths_zx_md_screen512/model_epoch_100.pth')
    parser.add_argument('--length', default=512, type=int)
    return parser.parse_args()


def main(args):
    if not os.path.isdir(args.score_dir):
        os.makedirs(args.score_dir)
    # 加载模型
    model = EAST(False, length=args.length).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    t0 = time.time()
    file_num = 0
    total_acc = 0.0
    for filename in os.listdir(args.img_dir):
        if filename[0] == '.':
            continue
        img_path = os.path.join(args.img_dir, filename)
        f_name, e_name = os.path.splitext(filename)
        gt_path = os.path.join(args.gt_dir, f_name + '.txt')
        if not os.path.isfile(gt_path):
            print(gt_path, ' not exist ')
            continue
        # 读取gt
        quad_boxes = get_boxes(gt_path)
        print(gt_path, quad_boxes)

        # 读取图片
        # img = Image.open(img_path)
        img = cv2pil(cv2.imread(img_path, cv2.IMREAD_COLOR))
        src_width = img.width
        src_height = img.height
        print(img_path, img.mode, img.size)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            print(img_path, img.mode, img.size)
        img, ratio_h, ratio_w = resize_img(img)
        print('resize_img ', img.size, ratio_h, ratio_w)

        score_path = os.path.join(args.score_dir, f_name + ".score" + e_name)
        mask_path = os.path.join(args.score_dir, f_name + ".mask" + e_name)
        img_gt_path = os.path.join(args.score_dir, f_name + ".gt" + e_name)
        img_score, img_mask, img_gt = eval_model(model, img, quad_boxes, src_width, src_height)
        cv2.imwrite(mask_path, img_mask)
        cv2.imwrite(score_path, img_score)
        cv2.imwrite(img_gt_path, img_gt)
        cv2.imwrite(os.path.join(args.score_dir, filename), np.array(img))
        acc = compute_accuracy(img_score, quad_boxes)
        file_num += 1
        total_acc += acc
        print('\timg_path : ', img_path)
        print('\tacc      : ', acc)
        # break
    t1 = time.time()
    print("total time : ", t1 - t0, ' files num : ', file_num, ' total_acc : ', total_acc)


if __name__ == '__main__': 
    main(get_args())
