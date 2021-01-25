import numpy as np
import torch
from torchvision import transforms
import cv2
import os
import glob
from faceland import FaceLanndInference
from hdface.hdface import hdface_detector
 

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def main():
    det = hdface_detector(use_cuda=False)
    checkpoint = torch.load('faceland.pth')
    plfd_backbone = FaceLanndInference().cuda()
    plfd_backbone.load_state_dict(checkpoint)
    plfd_backbone.eval()
    plfd_backbone = plfd_backbone.cuda()
    transform = transforms.Compose([transforms.ToTensor()])

    num = 0

    for img in os.listdir("images"):
        print(img)
        image = cv2.imread(os.path.join("images",img))
        if image is not None:

            height, width = image.shape[:2]
            img_det = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = det.detect_face(img_det)
            for i in range(len(result)):
                box = result[i]['box']
                cls = result[i]['cls']
                pts = result[i]['pts']
                x1, y1, x2, y2 = box
               # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0))
                w = x2 - x1 + 1
                h = y2 - y1 + 1

                size_w = int(max([w, h])*0.8)
                size_h = int(max([w, h]) * 0.8)
                cx = x1 + w//2
                cy = y1 + h//2
                x1 = cx - size_w//2
                x2 = x1 + size_w
                y1 = cy - int(size_h * 0.4)
                y2 = y1 + size_h

                left = 0
                top = 0
                bottom = 0
                right = 0
                if x1 < 0:
                    left = -x1
                if y1 < 0:
                    top = -y1
                if x2 >= width:
                    right = x2 - width
                if y2 >= height:
                    bottom = y2 - height

                x1 = max(0, int(x1))
                y1 = max(0, int(y1))

                x2 = min(width, int(x2))
                y2 = min(height, int(y2))
                cropped = image[y1:y2, x1:x2]
#                print(top, bottom, left, right)
                cropped = cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)

                cropped = cv2.resize(cropped, (112, 112))

                input = cv2.resize(cropped, (112, 112))
                input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
                input = transform(input).unsqueeze(0).cuda()
                landmarks = plfd_backbone(input)
                pre_landmark = landmarks[0]
                pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [size_w, size_h]
                cv2.rectangle(image,(x1, y1), (x2, y2),(0,255,0))
                for (x, y) in pre_landmark.astype(np.int32):
                    cv2.circle(image, (x1 - left + x, y1 - bottom + y), 2, (255, 0, 255), 2)
            cv2.imwrite("results/"+img,image)

            num +=1
        else:
            break


        # if cv2.waitKey(0) == 27:
        #     break

if __name__ == "__main__":
    main()
