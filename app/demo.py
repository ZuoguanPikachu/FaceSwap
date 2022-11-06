import cv2
import time
import dlib
import numpy
import torch
from PIL import Image
from gfpgan import GFPGANer
from basicsr.utils import imwrite
from faceswap.swapnet import SwapNet
from face_merge.merge import mer_face
from facenet_pytorch import MTCNN, extract_face


def get_points1(img):
    points = []
    dets_sw = detector(img, 1)[0]

    landmarks = [[p.x, p.y] for p in predictor68(img, dets_sw).parts()]
    points.extend(landmarks[:17])

    landmarks = [[p.x, p.y] for p in predictor81(img, dets_sw).parts()]
    points.append(landmarks[74])
    points.append(landmarks[73])
    points.append(landmarks[76])
    points.append(landmarks[75])

    return points


'''部分参数'''
get_points_func = get_points1
erode_size = 3
erode_iterations = 1
blur_size = 7


'''加载各模型'''
print("模型加载中...")
tic = time.time()
# 人脸提取
mt_cnn = MTCNN()
# 人脸增强
restorer = GFPGANer(
    model_path="pretrained_models/GFPGANv1.3.pth",
    upscale=8,
    arch="clean",
    channel_multiplier=2,
    bg_upsampler=None
)
# 人脸关键点
detector = dlib.get_frontal_face_detector()
predictor68 = dlib.shape_predictor('pretrained_models/shape_predictor_68_face_landmarks.dat')
predictor81 = dlib.shape_predictor('pretrained_models/shape_predictor_81_face_landmarks.dat')
# 人脸转换
swapnet = SwapNet().to(torch.device("cpu"))
swapnet.load_state_dict(torch.load("pretrained_models/FaceSwap.pth", torch.device("cpu"))["state"])
swapnet.eval()
print(f"模型加载耗时: {int((time.time()-tic)*1000)}ms")


if __name__ == "__main__":
    '''人脸提取'''
    print("人脸提取中...")
    tic = time.time()
    input_img = Image.open("mila_azul.jpg")
    boxes, _ = mt_cnn.detect(input_img)
    box = boxes[0]
    box = numpy.array(box, dtype=numpy.int32)
    width = box[2] - box[0]
    height = box[3] - box[1]

    if height > width:
        delta = (height-width)//2
        box[2] += delta
        box[0] -= delta
        img_size = height
    else:
        delta = (width-height)//2
        box[3] += delta
        box[1] -= delta
        img_size = width

    input_face = extract_face(input_img, box, image_size=img_size, save_path="input_face.jpg")
    print(f"人脸提取耗时: {int((time.time()-tic)*1000)}ms")

    '''人脸转换'''
    print("人脸转换中...")
    tic = time.time()
    input_face = input_face.resize((64, 64))
    input_face = numpy.array(input_face)

    input_face_t = torch.Tensor(input_face.transpose(2, 0, 1)/255.).float()
    input_face_t = input_face_t.unsqueeze(0)

    swaped_face = swapnet.forward(input_face_t, select="B")
    swaped_face = numpy.clip(swaped_face.detach().cpu().numpy()[0] * 255, 0, 255).astype("uint8").transpose(1, 2, 0)

    swaped_face = Image.fromarray(swaped_face)
    swaped_face.save("swaped_face.jpg")
    print(f"人脸转换耗时: {int((time.time()-tic)*1000)}ms")

    '''人脸增强'''
    print("人脸增强中...")
    tic = time.time()
    swaped_face = cv2.imread("swaped_face.jpg", cv2.IMREAD_COLOR)
    _, _, restored_img = restorer.enhance(
        swaped_face,
        has_aligned=False,
        only_center_face=True,
        paste_back=True
    )
    swaped_face = cv2.resize(restored_img, (img_size, img_size))
    imwrite(swaped_face, "swaped_face.jpg")
    print(f"人脸增强耗时: {int((time.time()-tic)*1000)}ms")

    '''人脸融合'''
    print("人脸融合中...")
    tic = time.time()
    merged_face = mer_face("input_face.jpg", "swaped_face.jpg", get_points_func, erode_size, erode_iterations, blur_size)
    cv2.imwrite("merged_face.jpg", merged_face)

    merged_face = Image.open("merged_face.jpg")
    box[2:]= box[:2]+img_size
    input_img.paste(merged_face, box=box)
    input_img.save("swaped.jpg")
    print(f"人脸融合耗时: {int((time.time()-tic)*1000)}ms")
