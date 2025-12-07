import cv2
import numpy as np
from PIL import Image
from skimage import transform as tf
import torch
import torchvision.transforms as transforms
import decord
import tqdm
from src.dataset.utils import YoloFace


def align_face(image, landmark, output_shape=(112, 112)):
    points = np.asarray(landmark)
    dst_points = np.array([
        (30.2946, 51.6963),
        (65.5318, 51.5014),
        (48.0252, 71.7366),
        (33.5493, 92.3655),
        (62.7299, 92.2041)
    ], dtype=np.float32)
    dst_points[:, 0] += 8.0
    tform = tf.SimilarityTransform()
    tform.estimate(np.array(points), dst_points)
    aligned_image = tf.warp(image, tform.inverse, output_shape=output_shape)
    return aligned_image


def center_crop(img_driven, face_bbox, scale=1.0):
    h, w = img_driven.shape[:2]
    x0, y0, x1, y1 = face_bbox[:4]
    center = (int((x0 + x1) / 2), int((y0 + y1) / 2))
    crop_size = int(max(x1 - x0, y1 - y0)) // 2
    crop_size = int(crop_size * scale)
    new_x0, new_y0, new_x1, new_y1 = center[0] - crop_size, center[1] - crop_size, center[0] + crop_size, center[1] + crop_size
    pad_left, pad_top, pad_right, pad_bottom = 0, 0, 0, 0
    if new_x0 < 0:
        pad_left, new_x0 = -new_x0, 0
    if new_y0 < 0:
        pad_top, new_y0 = -new_y0, 0
    if new_x1 > w:
        pad_right, new_x1 = new_x1 - w, w
    if new_y1 > h:
        pad_bottom, new_y1 = new_y1 - h, h
    img_mtn = img_driven[new_y0:new_y1, new_x0:new_x1]
    img_mtn = cv2.copyMakeBorder(img_mtn, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return img_mtn


def process_bbox(bbox, expand_radio, height, width):
    """
    raw_vid_path:
    bbox: format: x1, y1, x2, y2
    radio: expand radio against bbox size
    height,width: source image height and width
    """

    def expand(bbox, ratio, height, width):
        
        bbox_h = bbox[3] - bbox[1]
        bbox_w = bbox[2] - bbox[0]
        
        expand_x1 = max(bbox[0] - ratio * bbox_w, 0)
        expand_y1 = max(bbox[1] - ratio * bbox_h, 0)
        expand_x2 = min(bbox[2] + ratio * bbox_w, width)
        expand_y2 = min(bbox[3] + ratio * bbox_h, height)

        return [expand_x1,expand_y1,expand_x2,expand_y2]

    def to_square(bbox_src, bbox_expend, height, width):

        h = bbox_expend[3] - bbox_expend[1]
        w = bbox_expend[2] - bbox_expend[0]
        c_h = (bbox_expend[1] + bbox_expend[3]) / 2
        c_w = (bbox_expend[0] + bbox_expend[2]) / 2

        c = min(h, w) / 2

        c_src_h = (bbox_src[1] + bbox_src[3]) / 2
        c_src_w = (bbox_src[0] + bbox_src[2]) / 2

        s_h, s_w = 0, 0
        if w < h:
            d = abs((h - w) / 2)
            s_h = min(d, abs(c_src_h-c_h))
            s_h = s_h if  c_src_h > c_h else s_h * (-1)
        else:
            d = abs((h - w) / 2)
            s_w = min(d, abs(c_src_w-c_w))
            s_w = s_w if  c_src_w > c_w else s_w * (-1)


        c_h = (bbox_expend[1] + bbox_expend[3]) / 2 + s_h
        c_w = (bbox_expend[0] + bbox_expend[2]) / 2 + s_w

        square_x1 = c_w - c
        square_y1 = c_h - c
        square_x2 = c_w + c
        square_y2 = c_h + c 

        return [round(square_x1), round(square_y1), round(square_x2), round(square_y2)]

    bbox_expend = expand(bbox, expand_radio, height=height, width=width)
    processed_bbox = to_square(bbox, bbox_expend, height=height, width=width)

    return processed_bbox


def crop_resize_img(img, bbox, image_size):
    x1, y1, x2, y2 = bbox
    img = img.crop((x1, y1, x2, y2))
    w, h = img.size
    img = img.resize((image_size, image_size))
    return img


def crop_face_motion(image, landmark, motion_transform, bbox, scale=0.45):
    face_landmark = np.asarray(landmark)
    face_x_min, face_x_max = min(face_landmark[:, 0]), max(face_landmark[:, 0])
    face_y_min, face_y_max = min(face_landmark[:, 1]), max(face_landmark[:, 1])
    box_x_min, box_y_min, box_x_max, box_y_max = bbox
    face_x_min = face_x_min - (face_x_min - box_x_min) * scale
    face_x_max = face_x_max + (box_x_max - face_x_max) * scale
    face_y_min = face_y_min - (face_y_min - box_y_min) * scale
    face_y_max = face_y_max + (box_y_max - face_y_max) * scale
    lmk_face_bbox = np.asarray([face_x_min, face_y_min, face_x_max, face_y_max])   
    motion_crop_face = center_crop(image, lmk_face_bbox)
    motion_crop_face = Image.fromarray(motion_crop_face)
    motion_crop_face_tensor = motion_transform(motion_crop_face)
    return motion_crop_face_tensor


def get_dwpose(image):
    H, W = image.shape[:2]
    dwpose_image = Image.new('RGB', (W, H), color=(0, 0, 0))
    return dwpose_image


def box_area(box):
    _, _, w, h = box
    return w * h


def preprocess(image_path, video_path, limit=100, image_size=512, area=1.25, det_path=None):
    scale = (1.0, 1.0)
    img_size = (224, 224)
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(
            img_size, scale=scale,
            ratio=(img_size[1] / img_size[0], img_size[1] / img_size[0]), antialias=True),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
            inplace=True),
    ])

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    pose_to_tensor = transforms.Compose([
        transforms.ToTensor(),
    ])
    motion_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0], [1]),
    ])
 
    align_instance = YoloFace(pt_path=det_path).detect

    imSrc_ = Image.open(image_path).convert('RGB')
    origin_src = np.array(imSrc_)
    _, _, bboxes_list = align_instance(origin_src[:,:,[2,1,0]])
    bbox = bboxes_list[0].astype(np.int32)
    
    areas = np.array([box_area(bbox) for bbox in bboxes_list])
    max_idx = np.argmax(areas)
    max_box = bboxes_list[max_idx]
    x1, y1, ww, hh = max_box
    x2, y2 = x1 + ww, y1 + hh
    crop_bbox = process_bbox([x1, y1, x2, y2], 1.0, origin_src.shape[0], origin_src.shape[1])
    imSrc = crop_resize_img(imSrc_, crop_bbox, image_size)
    
    landmarks, _, bboxes_list = align_instance(np.array(imSrc)[:,:,[2,1,0]])
    areas = np.array([box_area(bbox) for bbox in bboxes_list])
    max_idx = np.argmax(areas)
    bboxSrc = bboxes_list[max_idx]
    h, w = imSrc.size

    transformed_image = img_transform(imSrc)
    arcface_image = align_face(np.array(imSrc), landmarks[max_idx])
    pose_img = np.zeros_like(np.array(imSrc))
    h, w = imSrc_.size

    x1, y1, ww, hh = bboxSrc
    x2, y2 = x1 + ww, y1 + hh
    ww, hh = (x2-x1) * area, (y2-y1) * area
    center = [(x2+x1)//2, (y2+y1)//2]
    x1 = max(center[0] - ww//2, 0)
    y1 = max(center[1] - hh//2, 0)
    x2 = min(center[0] + ww//2, w)
    y2 = min(center[1] + hh//2, h)
    pose_img[int(y1):int(y2), int(x1):int(x2)] = 255

    cap = decord.VideoReader(video_path)
    total_frames = min(len(cap), limit)
    driven_face_images_list = []
    driven_pose_images_list = []
    driven_dwpose_images_list = []
    driven_image_tensor_list = []
    lmk_list = []
    landmark = None
    bbox_s = None
    for drive_idx in tqdm.tqdm(range(total_frames), ncols=0):
        frame = cap[drive_idx].asnumpy()
        pts_list, _, bboxes_list = align_instance(frame[:,:,[2,1,0]])
        areas = np.array([box_area(bbox) for bbox in bboxes_list])
        max_idx = np.argmax(areas)
        max_box = bboxes_list[max_idx]
        landmark = pts_list[max_idx]
        assert landmark is not None

        lmk_list.append(landmark)
        if bbox_s is None:
            x1, y1, ww, hh = max_box
            x2, y2 = x1 + ww, y1 + hh
            bbox = [x1, y1, x2, y2]
            bbox_s = process_bbox(bbox, expand_radio=1, height=frame.shape[0], width=frame.shape[1])

        driven_face_images_tensor = crop_face_motion(frame.copy(), landmark, motion_transform, bbox)
        driven_face_images_list.append(driven_face_images_tensor)
        driven_pose_image = Image.fromarray(frame)
        driven_pose_image = crop_resize_img(driven_pose_image, bbox_s, image_size)
        dwpose_image = get_dwpose(np.asarray(driven_pose_image))
        driven_dwpose_images_list.append(pose_to_tensor(dwpose_image))
        driven_pose_images_tensor = motion_transform(driven_pose_image)
        driven_pose_images_list.append(driven_pose_images_tensor)
        driven_image_tensor = to_tensor(driven_pose_image)
        driven_image_tensor_list.append(driven_image_tensor)

    sample = dict(
        img_pose=torch.stack(driven_dwpose_images_list, dim=0),
        ref_img=to_tensor(imSrc),
        transformed_images=transformed_image,
        arcface_image=arcface_image,
        motion_face_image=torch.stack(driven_face_images_list, dim=0),
        motion_pose_image=torch.stack(driven_pose_images_list, dim=0),
        driven_image=torch.stack(driven_image_tensor_list, dim=0),
        lmk_list=lmk_list,
        crop_bbox=crop_bbox,
        original_image=origin_src,
    )

    return sample
