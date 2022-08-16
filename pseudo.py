"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""
from posixpath import dirname
from typing import Union
import argparse
import json
import os
import sys
import time
import zipfile
from collections import OrderedDict
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image
from skimage import io
from torch.autograd import Variable

import craft_utils
import file_utils
import imgproc
from craft import CRAFT


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
# added argument
parser.add_argument('--save_bbox_img', default=False, type=str2bool, help='Set True to export result image with bounding box')
parser.add_argument('--dataset_parent_dir', default='./dataset/', type=str, help='Parent dataset dir location')
parser.add_argument('--annot_parent_dir', default='./annot/', type=str, help='Parent annotation JSON file dir location')
parser.add_argument('--output_dir', default='./result/', type=str, help='JSON & Image output parent dir location')
parser.add_argument('--dataset_name', default='icdar15', type=str, help='Dataset name for finding dataset dir')


args = parser.parse_args()


""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    # boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    boxes, polys, char_boxes = craft_utils.getWordAndCharBoxes(
        img_resized,
        score_text,
        score_link,
        text_threshold,
        link_threshold,
        low_text,
        poly
    )

    # coordinate adjustment
    # boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    # polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    char_boxes = craft_utils.adjustResultCoordinates(char_boxes, ratio_w, ratio_h)
    # for k in range(len(polys)):
    #     if polys[k] is None:
    #         polys[k] = boxes[k]

    t1 = time.time() - t1

    # print(char_boxes)
    # render results (optional)
    # render_img = score_text.copy()
    # render_img = np.hstack((render_img, score_link))
    # ret_score_text = imgproc.cvt2HeatmapImg(render_img)
    ret_score_text = None

    if args.show_time:
        print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text, char_boxes


def load_image(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), -1)

    if len(im.shape) < 3:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    height, width, channel = im.shape
    if channel > 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGB)
    else:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def get_both_cropped_roi(
    im: np.ndarray,
    vertices: 'list[list[int]]',
    crop_pts: 'list[int]',
    orientation_angle: float,
    rotated_vertices: 'list[list[int]]',
    rotated_crop_pts: 'list[int]',
):
    ori_im = im.copy()
    ori_roi_im = ori_im[crop_pts[1]:crop_pts[3]+1,
                        crop_pts[0]:crop_pts[2]+1]

    vertices_np = np.array(
        vertices,
        dtype=int
    )
    rotated_vertices_np = np.array(
        rotated_vertices,
        dtype=int
    )
    rotated_crop_pts_np = np.array(
        rotated_crop_pts,
        dtype=int
    )

    image_height, image_width, _ = im.shape
    p_1, p_2, p_3, p_4 = vertices
    rp_1, rp_2, rp_3, rp_4 = rotated_vertices
    rx_1, ry_1, rx_2, ry_2 = rotated_crop_pts

    top_padding = 0
    bottom_padding = 0
    left_padding = 0
    right_padding = 0

    if ry_1 < 0:
        top_padding = -ry_1
        y_offset = np.array([0, -ry_1])
        # move points
        vertices_np += y_offset
        rotated_vertices_np += y_offset
        rotated_crop_pts_np[1] += -ry_1
        rotated_crop_pts_np[3] += -ry_1
    if ry_2 > image_height:
        bottom_padding = ry_2 - image_height

    if rx_1 < 0:
        left_padding = -rx_1
        x_offset = np.array([-rx_1, 0])
        # move points
        vertices_np += x_offset
        rotated_vertices_np += x_offset
        rotated_crop_pts_np[0] += -rx_1
        rotated_crop_pts_np[2] += -rx_1
    if rx_2 > image_width:
        right_padding = rx_2 - image_width

    p_1, p_2, p_3, p_4 = vertices_np.tolist()
    rx_1, ry_1, rx_2, ry_2 = rotated_crop_pts_np.tolist()
    # padding
    im = cv2.copyMakeBorder(im,
                            top_padding, bottom_padding, left_padding, right_padding,
                            cv2.BORDER_CONSTANT, None, value=0)
    # rotate
    rotation_mat: np.ndarray = cv2.getRotationMatrix2D(center=p_4, angle=-orientation_angle, scale=1)
    im = cv2.warpAffine(src=im, M=rotation_mat, dsize=(im.shape[1], im.shape[0]),
                        flags=cv2.INTER_CUBIC
                        )
    # crop
    im: np.ndarray = im[ry_1:ry_2+1, rx_1:rx_2+1]
    return ori_roi_im, im


def save_result(
    img_filename: Union[str, Path],
    img: np.ndarray,
    boxes: 'list[list[int]]',
    dir: Union[str, Path] = './result/',
    filename_prefix: str = '',
):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if isinstance(dir, str):
        dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)

    if isinstance(img_filename, str):
        img_filename = Path(img_filename)
    # convert to jpg extension
    if img_filename.suffix != ".jpg":
        img_filename = img_filename.with_suffix(".jpg")

    output_path = dir / f"{filename_prefix}_{str(img_filename)}"

    for box in boxes:
        cv2.polylines(
            img=img,
            pts=[box],
            isClosed=True,
            color=(0, 0, 255),
            thickness=1
        )
    cv2.imwrite(str(output_path), img)


if __name__ == '__main__':
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    # START PROCESSING
    DATASET_PARENT_DIR = Path(args.dataset_parent_dir).resolve()
    ANNOT_PARENT_DIR = Path(args.annot_parent_dir).resolve()
    OUTPUT_PARENT_DIR = Path(args.output_dir).resolve()

    DATASET_DIR: Path = (DATASET_PARENT_DIR / args.dataset_name).resolve()
    ANNOT_DIR: Path = (ANNOT_PARENT_DIR / args.dataset_name).resolve()
    OUTPUT_JSON_DIR: Path = (OUTPUT_PARENT_DIR / args.dataset_name).resolve()
    OUTPUT_IMG_DIR: Path = (OUTPUT_PARENT_DIR / f"{args.dataset_name}_img").resolve()

    OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

    # read json file
    _json_file = "/home/ting/Private-Projects/PyTorch/_Out_of_Vocabulary/tbp_digit_annot/textocr/0003.json"
    max_limit_num = 10
    file_index = 0

    for child_path in ANNOT_DIR.iterdir():
        _to_read_json_file_path = child_path
        _to_write_json_file_path = OUTPUT_JSON_DIR / _to_read_json_file_path.name

        img_count = 0
        process_count = 0
        rotated_succeed_count = 0
        ori_succeed_count = 0
        with open(_to_read_json_file_path, "r") as reader:
            digit_annot_data_list = json.load(reader)

            new_digit_annot_data_list = []  # ! OUTPUT variable
            for digit_annot_data in digit_annot_data_list:
                image_name = digit_annot_data['image_name']
                text_annot_list = digit_annot_data['text']

                img_path: Path = DATASET_PARENT_DIR / image_name
                image = load_image(img_path)
                img_count += 1

                new_text_annot_list = []  # ! OUTPUT variable
                for text_annot in text_annot_list:
                    transcription = text_annot['transcription']
                    vertices = text_annot['vertices']
                    crop_pts = text_annot['crop_pts']
                    orientation_angle = text_annot['orientation_angle']
                    rotated_vertices = text_annot['rotated_vertices']
                    rotated_crop_pts = text_annot['rotated_crop_pts']

                    ori_roi_im, im = get_both_cropped_roi(
                        im=image,
                        vertices=vertices,
                        crop_pts=crop_pts,
                        orientation_angle=orientation_angle,
                        rotated_vertices=rotated_vertices,
                        rotated_crop_pts=rotated_crop_pts,
                    )

                    # for rotated roi image
                    r_x_1, r_y_1, r_x_2, r_y_2 = rotated_crop_pts
                    height = r_y_2 - r_y_1
                    width = r_x_2 - r_x_1

                    smallest_dim = min(height, width)

                    rotated_char_boxes = []
                    if smallest_dim > 10:  # 10 is too small because training data need to be 32 * 32
                        try:
                            _, _, _, rotated_char_boxes = test_net(net,
                                                                   im,
                                                                   args.text_threshold,
                                                                   args.link_threshold,
                                                                   args.low_text,
                                                                   args.cuda,
                                                                   args.poly,
                                                                   refine_net)
                            rotated_succeed_count += 1
                        except Exception as e:
                            print(e)
                            print(f"img_path: {img_path}")
                            print(f"im shape: {im.shape}")
                            print(f"ori_roi_im shape: {ori_roi_im.shape}")
                    else:
                        print(f"img_path: {img_path}")
                        print(f"im shape: {im.shape}")
                        print(f"ori_roi_im shape: {ori_roi_im.shape}")

                    if not isinstance(rotated_char_boxes, np.ndarray):  # char_boxes is list type when no box is detected
                        rotated_char_boxes = np.array(rotated_char_boxes)
                    rotated_char_boxes = np.around(rotated_char_boxes).astype(int)

                    if args.save_bbox_img:
                        save_result(
                            img_filename=f"{img_path.stem}_{file_index:02d}.jpg",
                            img=im,
                            boxes=rotated_char_boxes,
                            dir=OUTPUT_IMG_DIR,
                            filename_prefix="rotated"
                        )

                    # for original roi image
                    p_x_1, p_y_1, p_x_2, p_y_2 = crop_pts
                    height = p_y_2 - p_y_1
                    width = p_x_2 - p_x_1

                    smallest_dim = min(height, width)

                    char_boxes = []
                    if smallest_dim > 10:  # 10 is too small because training data need to be 32 * 32
                        try:
                            _, _, _, char_boxes = test_net(net,
                                                           ori_roi_im,
                                                           args.text_threshold,
                                                           args.link_threshold,
                                                           args.low_text,
                                                           args.cuda,
                                                           args.poly,
                                                           refine_net)
                            ori_succeed_count += 1
                        except Exception as e:
                            print(e)
                            print(f"img_path: {img_path}")
                            print(f"im shape: {im.shape}")
                            print(f"ori_roi_im shape: {ori_roi_im.shape}")
                    else:
                        print(f"img_path: {img_path}")
                        print(f"im shape: {im.shape}")
                        print(f"ori_roi_im shape: {ori_roi_im.shape}")

                    if not isinstance(char_boxes, np.ndarray):  # char_boxes is list type when no box is detected
                        char_boxes = np.array(char_boxes)
                    char_boxes = np.around(char_boxes).astype(int)

                    if args.save_bbox_img:
                        save_result(
                            img_filename=f"{img_path.stem}_{file_index:02d}.jpg",
                            img=ori_roi_im,
                            boxes=char_boxes,
                            dir=OUTPUT_IMG_DIR,
                            filename_prefix="ori"
                        )

                    # ! Save to output json
                    _temp_annot_data = {
                        "char_boxes": char_boxes.tolist(),
                        "rotated_char_boxes": rotated_char_boxes.tolist(),
                    }
                    _temp_annot_data.update(text_annot)
                    new_text_annot_list.append(
                        _temp_annot_data
                    )
                    # process limiting
                    file_index += 1
                    process_count += 1
                max_limit_num -= 1

                # ! Save to output json
                _temp_digit_annot_data = digit_annot_data.copy()
                _temp_digit_annot_data['text'] = new_text_annot_list
                new_digit_annot_data_list.append(_temp_digit_annot_data)

                # if max_limit_num < 0:
                #     break

            print("==="*8)
            with open(_to_write_json_file_path, "w") as writer:
                json.dump(new_digit_annot_data_list, writer)
                print(f"Write to {_to_write_json_file_path}")

            print(f"data_length: {len(new_digit_annot_data_list)}")
            print(f"img_count: {img_count}")
            print(f"process_count: {process_count}")
            print(f"rotated_succeed_count: {rotated_succeed_count}")
            print(f"ori_succeed_count: {ori_succeed_count}")

    print("elapsed time : {}s".format(time.time() - t))
