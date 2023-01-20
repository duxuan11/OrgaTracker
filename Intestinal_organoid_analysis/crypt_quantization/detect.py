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
import os
import sys
from pathlib import Path
from numpy import dtype

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path: #模块查询列表
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  #relative，绝对路径转换成相对路径

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box,get_one_box
from utils.torch_utils import select_device, time_sync
import numpy as np

@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images/src',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
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
    device1 = select_device(device=device)
    model = DetectMultiBackend('weights/org_b.pt', device=device, dnn=dnn, data=ROOT / 'data/organoid.yaml', fp16=half) #加载后端 data 数据集文件，自己有数据集可以仿照这些
    #===修改===
    model2 = DetectMultiBackend('weights/ya_b.pt', device=device1, dnn=dnn, data='data/crypt.yaml', fp16=half)
    stride, names, pt = model.stride, model.names, model.pt #模型的步长 32 
    imgsz = check_img_size(imgsz, s=stride)  # check image size 默认32的倍数
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
    model2.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup 热身
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0] #存储中介结果信息
    for path, im, im0s, vid_cap, s in dataset: #传图片预测 dataset会执行next函数 s是图片打印信息
        t1 = time_sync()
        #print("第一次输入图像im的格式")
        #print(im.shape)
        im = torch.from_numpy(im).to(device) #torch.Size([3,640,480])
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  #像素点归一化操作 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim torch.Size([1,3,648,480])
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference 预测 fasle不用管
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        #print("第二次输入图像im的格式")
        #print(im.shape)
        pred = model(im, augment=augment, visualize=visualize)#torch.Size([1,1890,85]) 85表示 4个坐标 + 置信度 + 80个类别信息
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS 非极大值抑制
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det) #max_det过滤掉1000之后的目标
        #【1，5，6】 只剩了5个目标 6 4坐标 置信度 类别
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        #for i, det in enumerate(pred):  # per image, torch. Size([5,6])
        ims1 = im0s.copy()
        im1 = im.clone()
        j = 0
        sum_crypt = 0
        no_crypt = 0
        det_allsmall =  []
        det_count = []
        # sum_w = 0
        # sum_h = 0
        sum_area = 0
        for det in pred:
            det1= det.clone()
     
            det1[:, :4] = scale_coords(im1.shape[2:], det1[:, :4], ims1.shape).round()
            #print(type(det1))
            
            for *xyxy, conf, cls in det1:
                xmin,ymin,xmax,ymax = xyxy[0],xyxy[1],xyxy[2],xyxy[3]
                #print("xmin"+str(xmin)+",ymin"+str(ymin)+",xmax"+str(xmax)+",ymax"+str(ymax))
                #print("i:"+str(j))
                j = j + 1
                crop= get_one_box(xyxy,ims1,BGR=True)
                # print(type(crop))
                # print(crop.shape)
                
                # print("crop的大小")
                # print(crop.shape)
                # print(crop)
      
                # cv2.imwrite(f, crop)  # https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
                from PIL import Image
                Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save("runs/"+str(j)+".jpg", quality=95, subsampling=0)
                # print("crop")
                # print(crop.shape)
                # print(crop)
                temp = crop_letterbox(crop.copy())[0].copy()
                # print("temp")
                # print(temp.shape)
                # print(temp)
                temp = temp.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                
                temp = np.ascontiguousarray(temp) #函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
                temp = torch.from_numpy(temp).to(device1)
                # print("temp.shape")
                # print(temp.shape)
                # print(temp)
                temp = temp.half() if model.fp16 else temp.float()  # uint8 to fp16/32
                temp /= 255  #像素点归一化操作 0 - 255 to 0.0 - 1.0
                if len(temp.shape) == 3:
                   temp = temp[None]  
                #print("img.shape")
                #print(temp.shape)
                #print(temp)        
                #LoadImages(crop,img_size=imgsz, stride=stride, auto=pt)
                #crop = check_img_size(crop, s=stride)

                pred1 = model2(temp, augment=augment, visualize=visualize)
                pred1 = non_max_suppression(pred1, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det) #max_det过滤掉1000之后的目标
                #print(pred1)
                #把小肠类器官大小记录下来
                organoid_crypt = []
                for det3 in pred1:
                    sum_crypt += len(det3)
                    if len(det3)==0:
                        no_crypt = no_crypt + 1
                    #print("sum_crypt:"+str(sum_crypt))
                    #print("no_crypt:"+str(no_crypt))
                    det_count.append(len(det3))
                    #print(det3)
                    if len(det3): #画框
                        #print(det3)
                        
                    # Rescale boxes from img_size to im0 size 
                    # 这里把你识别的坐标映射到原图坐标中
                        det3[:, :4] = scale_coords(temp.shape[2:], det3[:, :4], crop.shape).round()
                        det3[:,0] =  det3[:,0] + xmin + 1   #加三减三就不写了
                        det3[:,2] =  det3[:,2] + xmin - 1
                        det3[:,1] =  det3[:,1] + ymin + 1
                        det3[:,3] =  det3[:,3] + ymin - 1
                        #print("after,xmin"+str(det3[0,0])+",ymin"+str(det3[0,2])+",xmax"+str( det3[0,1])+",ymax"+str(det3[0,3]))  
                        det_allsmall.append(det3.clone())
                for *xyxy, conf, cls in det3:
                         w = xyxy[2].tolist() - xyxy[0].tolist()
                         h = xyxy[3].tolist() - xyxy[1].tolist()
                        #  sum_w += w
                        #  sum_h += h
                         sum_area = sum_area +w*h
        #print("crypt平均宽度为:"+str(sum_w/(sum_crypt))) 
        #print("crypt平均高度为:"+str(sum_h/(sum_crypt)))
        
        print("共有"+str(j)+"类器官")
        print("未发芽的类器官:"+str(no_crypt))
        print("隐窝总数:"+str(sum_crypt))
        print("隐窝的总面积为:"+str(sum_area))         
        # Process predictions 把所有的检测框画到原图中
        for i, det in enumerate(pred):  # per image, torch. Size([5,6])
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}:'
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop 是否要裁剪下来
            #绘制小肠的
            annotator = Annotator(im0, line_width=line_thickness, font_size=12,example=str(names)) #绘图工具
            #绘制小肠的丫的
            annotator2 = Annotator(im0, line_width=line_thickness, example=str(names)) #绘图工具
            if len(det): #画框
                # Rescale boxes from img_size to im0 size 
                # 这里把你识别的坐标映射到原图坐标中
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                #for c in det[:, -1].unique():
                    #n = (det[:, -1] == c).sum()  # detections per class
                    #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string 4个person 1个bus

                # Write results 保存到txt 默认不运行
                for dets in det_allsmall:
                    for *xxyy,conf,cls in dets:
                        annotator2.box_label(xxyy, None, color=colors(5, True))    

                index = 0
                for *xyxy, conf, cls in det: #reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    label_org = 'budding:'+str(det_count[index])
                    index = index + 1
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        #annotator.box_label(xyxy, label, color=colors(c, True))
                        annotator.box_label(xyxy, label_org, color=colors(c, True))
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

    # Print results 打印结果
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images/src', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='NMS IoU threshold')
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



# ====letterbox 2 
def crop_letterbox(im, new_shape=(160, 160), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width] #current shape [height,width],(1088,818)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1]) #缩放比例
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))#[width,height]:(480],640)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding #wh padding 168%
    if auto:  # minimum rectangle 只要32的倍数就行
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides 满足32倍数 不需要改变了
    dh /= 2

    if shape[::-1] != new_unpad:  # resize + padding [其实并没有进行padding]
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def convert(img):
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    return img

def main(opt):
    #检测包有没有安装好
    check_requirements(exclude=('tensorboard', 'thop'))
    #运行
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
