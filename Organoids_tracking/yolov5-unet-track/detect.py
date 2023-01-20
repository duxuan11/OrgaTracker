# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
from email.utils import localtime
import os
import sys
#引入别人的项目，添加为根目录
from Skeleton import skeleton
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import numpy as np
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path: #模块查询列表
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  #relative，绝对路径转换成相对路径
sys.path.append(str(Path.cwd())+'\\UNET')
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, get_one_box,save_one_box
from utils.torch_utils import select_device, time_sync
from UNET.unet import Unet
from PIL import Image
import random
from skimage import morphology,io
from time import time
from draw_box_utils import draw_objs
from IOU import *
import time
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'PowderBlue', 'BurlyWood', 'Beige',
    'Bisque','BlanchedAlmond', 'LightPink', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'BlueViolet', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'Maroon', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/organoid.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.75,  # confidence threshold
        iou_thres=0.75,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    #处理预测路径
    source = str(source) #python detect. py --source data\\ images\\ bus.jpg'
    
    (filepath, tempfilename) = os.path.split(source)
    filepath = source
    print(filepath)
    save_img = not nosave and not source.endswith('.txt')  # save inference images，说明这个结果要保存下来
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS) #判断是不是文件路径，符不符合格式
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')) #是不是网络流
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file) #判断摄像头
    if is_url and is_file: 
        source = check_file(source)  # download

    # Directories 保存结果的文件夹  project = ‘runs/detect'， name='exp'
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device) #加载模型设备 
    model = DetectMultiBackend('weights/best.pt', device=device, dnn=dnn, data=ROOT / 'data/organoid.yaml', fp16=half) #加载后端 data 数据集文件，自己有数据集可以仿照这些
    unet = Unet()
    stride, names, pt = model.stride, model.names, model.pt #模型的步长 32 
    imgsz = check_img_size(imgsz, s=stride)  # check image size 默认32的倍数
    # 目标追踪
    tracker = Hungarian()
    frames = []
    first =True
    preboxes = []
    preIdDict = {}
    # Dataloader 加载待预测的图片
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt) #加载图片文件
        bs = 1  # batch_size 每次输入一张图片
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference 模型推理
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup 热身
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0] #存储中介结果信息
    for path, im, im0s, vid_cap, s in dataset: #传图片预测 dataset会执行next函数 s是图片打印信息
        t1 = time_sync()
        im = torch.from_numpy(im).to(device) #torch.Size([3,640,480])
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  #像素点归一化操作 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim torch.Size([1,3,648,480])
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference 预测 fasle不用管
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        
        pred = model(im, augment=augment, visualize=visualize)#torch.Size([1,1890,85]) 85表示 4个坐标 + 置信度 + 80个类别信息
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS 非极大值抑制      conf_thres
        classes = 0
        pred = non_max_suppression(pred, 0.4, iou_thres, classes, agnostic_nms, max_det=max_det) #max_det过滤掉1000之后的目标
        #【1，5，6】 只剩了5个目标 6 4坐标 置信度 类别
        dt[2] += time_sync() - t3


        ims1 = im0s.copy()
        im1 = im.clone()
        original_img = Image.fromarray(cv2.cvtColor(im0s.copy(), cv2.COLOR_BGR2RGB))
        #original_img.show()
        jj = 0
        det_masks = {}
        det_boxes = []
        det_conf = []
        det_cls = []

        for det in pred:
            det1 = det.clone()

            det1[:, :4] = scale_coords(im1.shape[2:], det1[:, :4], ims1.shape).round()
            #print(type(det1))

            for *xyxy, conf, cls in det1:
                box = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                xmin,ymin,xmax,ymax = int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])

                # print("xmin" + str(xmin) + ",ymin" + str(ymin) + ",xmax" + str(xmax) + ",ymax" + str(ymax))
                # print("i:" + str(j))
                
                crop,xyxy_box = get_one_box(xyxy, ims1, BGR=True)
                #crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                #crop = Image.fromarray(np.uint8(crop))
               #crop = Image.fromarray(crop.astype('uint8'))
                image = unet.detect_image(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),False,['background','organoid'])  
                # gray1 = cv2.cvtColor(np.array(image.copy()), cv2.COLOR_RGB2BGR)      
                gray = cv2.cvtColor(np.array(image.copy()), cv2.COLOR_BGR2GRAY)
                source = np.zeros((im0s.shape[0],im0s.shape[1]))
                #print("xmin:"+str(xmin)+"xmax:"+str(xmax)+"ymin:"+str(ymin)+"ymax:"+str(ymax))
                #print(gray.shape) 
                #print(source.shape) 
                # newimage = np.zeros((im0s.shape[0],im0s.shape[1]))
                # for i in range(xmin,xmax):
                #     for j in range(ymin,ymax):
                #         if gray[j-ymin,i-xmin] > 0:
                #             newimage[j][i] = 255
                # im = Image.fromarray(newimage)
                # im.show()     
                ret, thresh = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY)
                img,thresh = get_lagrest_connect_component2(thresh)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                newimage = np.zeros((im0s.shape[0],im0s.shape[1]))
                if len(contours) > 0:
                    bigger = max(contours, key=lambda item: cv2.contourArea(item))     
                    masks = np.zeros_like(gray)
                    cv2.drawContours(masks, [bigger], -1, 255, cv2.FILLED) 
                    perimeter = cv2.arcLength(contours[0],closed = True)
                    print("周长为:"+str(perimeter))
                    temp_count = 0     
                    for i in range(xmin,xmax):
                        for j in range(ymin,ymax):
                            if masks[j-ymin,i-xmin] > 0:
                                newimage[j][i] = 255
                                temp_count = temp_count + 1
                    # im = Image.fromarray(newimage)
                    # im.show()
                    box_area = (xmax-xmin)*(ymax-ymin) 
                    if temp_count*5 > box_area:
                        det_boxes.append([xmin,ymin,xmax,ymax])
                        det_conf.append(conf.item())
                        det_cls.append(cls.item())
                        det_masks[jj] = newimage.copy() 
                        jj = jj + 1  
                        print("jj="+str(jj))      
                 
        predict_mask = np.stack(det_masks.values())
        predict_boxes = np.array(det_boxes)
        predict_scores  = np.array(det_conf)
        predict_classes = np.array(det_cls)
        # print(predict_classes)
        print(predict_mask.shape)      
        print(predict_boxes.shape)
        print(predict_scores.shape)
        print(predict_classes.shape)
        # iddx = []
        # for i in range(0,len(predict_boxes)):
        #     iddx.append(True)
        # for i in range(0,len(predict_boxes)-1):
        #     for j in range(i+1,len(predict_boxes)):
        #         iou = Area_batch(predict_boxes[i],predict_boxes[j])
        #         print(iou)
        #         if iou >= 0.9:
        #             if compareArea(predict_boxes[i],predict_boxes[j]):
        #                 iddx[j] = False
        #             else:
        #                 iddx[i] = False


        # # print(iddx)
        # predict_boxes =  predict_boxes[iddx]  #  idx = [True,True,False,True]
        # predict_classes = predict_classes[iddx]
        # predict_scores = predict_scores[iddx]
        # predict_mask = predict_mask[iddx]     
               # det_masks.append(image)
                #print(im0s.shape)
                #draw_mask(im0s,box,image,j)
                
                #image.show()
                # det_contours.append(contours)
                #image.show()
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)


        boxes = predict_boxes
        IdDict = {}
        fusion_IdDict = {}
        if first is True:
            tracker.runHungarian(boxes, None, None)
            first = False
            IdDict = {i: i for i in range(0, len(boxes))}
        else:
            IdDict,fusion_IdDict = tracker.runHungarian(boxes, preboxes, preIdDict)
        preIdDict = IdDict
        print("fusion_IdDict")   
        preboxes = boxes


        category_index = {'0':"organoid",'1':"bubble"}
        plot_img = draw_objs(original_img,
                                     IdDict=IdDict,
                                     fusion_IdDict=fusion_IdDict,
                                     boxes=predict_boxes,
                                     classes=predict_classes,
                                     scores=predict_scores,
                                     masks=predict_mask,
                                     category_index=category_index,
                                     line_thickness=3,
                                     font='arial.ttf',
                                     font_size=20)
        time_name = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        
        if os.path.exists("runs/"+filepath) == False:
            os.makedirs("runs/"+filepath+"/")
        # if os.path.exists("runs/"+str(filepath)+'/'+str(time_name)+".jpg"):
        #     plot_img.save("runs/"+str(filepath)+'/'+str(time_name)+"_n.jpg")    
        # else:
        #     plot_img.save("runs/"+str(filepath)+'/'+str(time_name)+".jpg")
                # if os.path.exists("runs/"+str(filepath)) == False:
        #     os.makedirs("runs/"+str(filepath)+"/")
        if os.path.exists("runs/"+filepath+'/'+str(time_name)+".jpg"):
            x = random.randint(0,200)
            plot_img.save("runs/"+filepath+'/'+str(time_name)+str(x)+".jpg")    
        else:
            plot_img.save("runs/"+filepath+'/'+str(time_name)+".jpg")
        frames.append(np.array(plot_img))
        plot_img.close()


        # Process predictions 把所有的检测框画到原图中
        for i, det in enumerate(pred):  # per image, torch. Size([5,6])
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop 是否要裁剪下来
            annotator = Annotator(im0, line_width=line_thickness, example=str(names)) #绘图工具
            if len(det): #画框
                # Rescale boxes from img_size to im0 size 
                # 这里把你识别的坐标映射到原图坐标中
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                i = 0
                # print(type(im0))
                # print(im0[0:3,:,:])
                # for contours in det_contours:
                #      xmin,ymin,xmax,ymax = det_coord[i]
                #      print(contours)
                #      print(type(im0))
                #      print(im0.shape)
                #      cv2.drawContours(im0, contours, 0, (0, 255, 255), 3)

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string 4个person 1个bus

                # Write results 保存到txt 默认不运行
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to images
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop: #是否保存裁剪的图像
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            
            # Stream results 返回画好的图像
            im0 = annotator.result()
            #是否展示图像
            if view_img:
                if p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
        
        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
    img_to_video(frames)    

    # Print results 打印结果
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

from skimage.measure import label
def get_lagrest_connect_component2(img):
    labeled_img, num = label(img, background=0, return_num=True)
    max_label = 0
    max_num = 0
    for i in range(1, num+1):
        sub_num = np.sum(labeled_img==i)
        if sub_num > max_num:
            max_num = sub_num
            max_label = i
    if max_label > 0:
        max_area = np.sum(labeled_img==max_label)
        img[labeled_img!=max_label] = 0
    else:
        max_area = 0
    #print(img)
    return max_area, img
global Warea    
Warea = 0 
global wk
wk = 0
def draw_mask(image,boxes,masks,j,thresh: float = 0.7, alpha: float = 0.5):
    global wk
    
    # np_image = np.array(image)
    # masks = np.where(masks > 0, True, False)
    gray1 = cv2.cvtColor(np.array(masks.copy()), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(gray1, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY)
    img,thresh = get_lagrest_connect_component2(thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #ontours, thresh = cv2.threshold(bin_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.findContours(image)
    # _, contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
   
    global Warea  
    if len(contours) > 0:
         cv2.drawContours(image[boxes[1].int():boxes[3].int(), boxes[0].int():boxes[2].int(), :],contours,-1,(0,255,0),2)
         Warea += cv2.contourArea(contours[0])
         wk =wk + 1
    #print("area:"+str(Warea))
    #print(wk)
    # 找到最大的轮廓
    if len(contours) > 0:
        bigger = max(contours, key=lambda item: cv2.contourArea(item))     
        masks = np.zeros_like(masks)
        cv2.drawContours(masks, [bigger], -1, STANDARD_COLORS[j], cv2.FILLED)
    # cv2.namedWindow("mask",0)
    # cv2.imshow("mask", masks)
    # cv2.imwrite("F:\Pictures\{}.jpg".format(localtime()),masks)
    # cv2.waitKey(500)
    # for k in range(len(contours)):
    #     area.append(cv2.contourArea(contours[k]))
    # if len(contours) >0:
    #      max_idx = np.argmax(np.array(area))
    #     # cv2.fillContexPoly(mask[i], contours[max_idx], 0)
    #     # 填充最大的轮廓   
    #      print("k=:"+str(max_idx))
            
    #      masks =  cv2.drawContours(np.array(masks), contours, max_idx, 255, cv2.FILLED)
    #      cv2.namedWindow("mask",0)
    #      cv2.imshow("mask", masks)
    #      cv2.waitKey(500)
    #      del area 
    #      for k in range(len(contours)):
    #         if k != max_idx:
    #             cv2.fillPoly(masks, [contours[k]], 0)
                
    
    #细化
    #ret, threshs = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY)  
        
    # skeleton(image,np.array(masks),boxes)
    # skeleton = morphology.skeletonize(threshs)
    # #skeleton = morphology.erosion(skeleton)
    # kernel = np.ones((5, 5), np.uint8)
    # skeleton = cv2.dilate(skeleton, kernel)
    # xmin,ymin,xmax,ymax = int(boxes[0].item()),int(boxes[1].item()),boxes[2].item(),boxes[3].item()
    # height,width =skeleton.shape
    # print(boxes)
    # print((width,height))
    # polys =[]
    # for i in range(0,width):
    #     for j in range(0,height):
    #         if skeleton[j,i] == 1:
    #             image[ymin+j-1,xmin+i-1,:] = 255
    # print(polys)
    # c = (200 * random.random(), 200 * random.random(), 200 * random.random())
    # for i in range(0, len(polys)-1):
    #         cv2.line(image, (polys[i][0], polys[i][1]), (polys[i + 1][0], polys[i + 1][1]), c,2)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(skeleton,cmap=plt.cm.gray)
    # plt.savefig("D:/images/"+str(time())+".jpg")
    # plt.show()
    #im2, contour_sk, hierarchy_1 = cv2.findContours(skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#坐标修改 （元组的修改） 多行俩列
                # contemp = list(contours)
                # L = np.array(contours)
                # print(L.shape)
                # for i in range(len(contemp)):
                #         contemp[i][0] = contemp[i][0] + xmin
                #         contemp[i][1] = contemp[i][1] + ymin
                #         if (contemp[i][0] >= xmax):
                #             contemp[i][0] = contemp[i][0] - 1
                #         if (contemp[i][1] >= ymax):
                #             contemp[i][1] = contemp[i][1] - 1
                # contours = tuple(contemp)
def img_to_video(frames):
    
    # 1.每张图像大小
    height,width,channel =frames[0].shape
    print("每张图片的大小为({},{})".format(height, width))
    # 2.通过时间的方式命名 设置保存路径
    time_name = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    sav_path = 'runs/drug/'+str(time_name)+'.mp4'
    # 3.获取图片总的个数
    length = len(frames)
    # 4.设置视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式
    # 完成写入对象的创建，第一个参数是合成之后的视频的名称，第二个参数是可以使用的编码器，第三个参数是帧率即每秒钟展示多少张图片，第四个参数是图片大小信息
    videowrite = cv2.VideoWriter(sav_path, fourcc, 0.5, (width,height))  # 2是每秒的帧数，size是图片尺寸

    # 7.合成视频
    for i in range(0, length):
        videowrite.write(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
        print('第{}张图片合成成功'.format(i))
    videowrite.release() #关闭
    print('------done!!!-------')


def compareArea(bb_test,bb_gt):
    return (bb_test[2] - bb_test[0]) *(bb_test[3] - bb_test[1]) >=(bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])

def iou_batch(bb_test, bb_gt):
        """
        在两个box间计算IOU
        :param bb_test: box1 = [x1y1x2y2] 即 [左上角的x坐标，左上角的y坐标，右下角的x坐标，右下角的y坐标]
        :param bb_gt: box2 = [x1y1x2y2]
        :return: 交并比IOU
        """
        xx1 = np.maximum(bb_test[0], bb_gt[0])  # 获取交集面积四边形的 左上角的x坐标
        yy1 = np.maximum(bb_test[1], bb_gt[1])  # 获取交集面积四边形的 左上角的y坐标
        xx2 = np.minimum(bb_test[2], bb_gt[2])  # 获取交集面积四边形的 右下角的x坐标
        yy2 = np.minimum(bb_test[3], bb_gt[3])  # 获取交集面积四边形的 右下角的y坐标
        w = np.maximum(0., xx2 - xx1)  # 交集面积四边形的 右下角的x坐标 - 左上角的x坐标 = 交集面积四边形的宽
        h = np.maximum(0., yy2 - yy1)  # 交集面积四边形的 右下角的y坐标 - 左上角的y坐标 = 交集面积四边形的高
        wh = w * h  # 交集面积四边形的宽 * 交集面积四边形的高 = 交集面积
        """
        两者的交集面积，作为分子。
        两者的并集面积作为分母。
        一方box框的面积：(bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
        另外一方box框的面积：(bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) 
        """
        o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
                  + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
                  - wh)
        return o

def Area_batch(bb_test, bb_gt):
        """
        在两个box间计算IOU
        :param bb_test: box1 = [x1y1x2y2] 即 [左上角的x坐标，左上角的y坐标，右下角的x坐标，右下角的y坐标]
        :param bb_gt: box2 = [x1y1x2y2]
        :return: 交并比IOU
        """
        xx1 = np.maximum(bb_test[0], bb_gt[0])  # 获取交集面积四边形的 左上角的x坐标
        yy1 = np.maximum(bb_test[1], bb_gt[1])  # 获取交集面积四边形的 左上角的y坐标
        xx2 = np.minimum(bb_test[2], bb_gt[2])  # 获取交集面积四边形的 右下角的x坐标
        yy2 = np.minimum(bb_test[3], bb_gt[3])  # 获取交集面积四边形的 右下角的y坐标
        w = np.maximum(0., xx2 - xx1)  # 交集面积四边形的 右下角的x坐标 - 左上角的x坐标 = 交集面积四边形的宽
        h = np.maximum(0., yy2 - yy1)  # 交集面积四边形的 右下角的y坐标 - 左上角的y坐标 = 交集面积四边形的高
        wh = w * h  # 交集面积四边形的宽 * 交集面积四边形的高 = 交集面积


        areaA = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
        areaB = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
        if areaA >= areaB:
            return wh/areaB
        else:
            return wh/areaA


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images/src1', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand 大小转成650 640
    print_args(vars(opt))
    return opt


def main(opt):
    #检测包有没有安装好
    check_requirements(exclude=('tensorboard', 'thop'))
    #运行
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
