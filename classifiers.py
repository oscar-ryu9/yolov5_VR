import numpy as np
import torch
from tqdm import tqdm
from PIL import Image


def classify_and_detect(images):
    """

    :param np.ndarray images: N x 12288 array containing N 64x64x3 images flattened into vectors
    :return: np.ndarray, np.ndarray
    """
    N = images.shape[0]

    # pred_class: Your predicted labels for the 2 digits, shape [N, 2]
    pred_class = np.empty((N, 2), dtype=np.int32)
    # pred_bboxes: Your predicted bboxes for 2 digits, shape [N, 2, 4]
    pred_bboxes = np.empty((N, 2, 4), dtype=np.float64)

    # add your code here to fill in pred_class and pred_bboxes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt", _verbose=False, device=device)

    # predict
    images = images.reshape((N, 64, 64, 3))
    loader = torch.utils.data.DataLoader(images, batch_size=1, shuffle=False)
    for i, img in tqdm(enumerate(loader)):

        img = img.numpy()[0]
        pred = model(img).pred[0].cpu().tolist()

        if len(pred) == 0:
            num1 = [0]*6
        else:
            num1 = pred[0]

        if len(pred) > 1:
            num2 = pred[1]
        else:
            num2 = num1
        
        if num1[5] > num2[5]:
            num1, num2 = num2, num1
        nums = [num1[5], num2[5]]

        boxes = [num1[:4], num2[:4]]
        boxes2 = []

        for box in boxes:
            center_x = round((box[0] + box[2]) / 2)
            center_y = round((box[1] + box[3]) / 2)
            min_y = center_y - 14
            max_y = center_y + 14
            min_x = center_x - 14
            max_x = center_x + 14
            boxes2.append([min_y, min_x, max_y, max_x])

        pred_class[i] = nums
        pred_bboxes[i] = boxes2

    return pred_class, pred_bboxes
