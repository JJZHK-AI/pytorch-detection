Detection
===
- [ ] 


|       | SSD | RFB | FSSD | YOLOV1 | YOLOV2 | YOLOV3 | YOLOV4 | YOLOV5 |
| ----- | --- | --- | ---- | ------ | ------ | ------ | ------ | ------ |
| train | - [x] | - []  |  [x] |   [x]  |   [x]  |   [x]  |   [x]  |   [x]  |

# 1.SSD
$300 \times 300$
|            |  vgg16  |  resnet50  |  resnet152  |  darknet19  |  mobilenetv1  |
| ---------- | :-----: | :--------: | :---------: | :---------: | :-----------: |
|   VOC-mAP  |  76.5%  |    79.2%   |    73.3%    |    74.4%    |     72.9%     |
|  VOC-Image | ![avatar](result/ssd_voc_vgg16_300.jpg) | ![avatar](result/ssd_voc_resnet50_300.jpg) | ![avatar](result/ssd_voc_resnet152_300.jpg) | ![avatar](result/ssd_voc_darknet19_300.jpg) | ![avatar](result/ssd_voc_mobilenetv1_300.jpg) |
|  COCO-mAP  |  -----  |   25.0%    |             |    20.7%    |     18.8%     |
| COCO-Image |         | ![avatar](result/ssd_coco_resnet50_300.jpg) | | ![avatar](result/ssd_coco_darknet19_300.jpg) | ![avatar](result/ssd_coco_mobilenetv1_300.jpg) |

# 2.RFB
$300 \times 300$
|            |  vgg16  |  resnet50  |  resnet152  |  darknet19  |  mobilenetv1  |
| ---------- | :-----: | :--------: | :---------: | :---------: | :-----------: |
|   VOC-mAP  |  79.0%  |            |             |    76.2%    |     73.8%     |
|  VOC-Image | ![avatar](result/rfb_voc_vgg16_300.jpg) | | | ![avatar](result/rfb_voc_darknet19_300.jpg) | ![avatar](result/rfb_voc_mobilenetv1_300.jpg) |
|  COCO-mAP  |         |            |             |    22.4%    |     19.0%     |
| COCO-Image |         | | | ![avatar](result/rfb_coco_darknet19_300.jpg) | ![avatar](result/rfb_coco_mobilenetv1_300.jpg) |

# 3.FSSD
$300 \times 300$
|            |  vgg16  |  resnet50  |  resnet152  |  darknet19  |  mobilenetv1  |
| ---------- | :-----: | :--------: | :---------: | :---------: | :-----------: |
|   VOC-mAP  |  77.9%  |    74.0%   |    74.1%    |    78.2%    |     73.5%     |
|  VOC-Image | ![avatar](result/fssd_voc_vgg16_300.jpg) | ![avatar](result/fssd_voc_resnet50_300.jpg) | ![avatar](result/fssd_voc_resnet152_300.jpg) | ![avatar](result/fssd_voc_darknet19_300.jpg) | ![avatar](result/fssd_voc_mobilenetv1_300.jpg) |
|  COCO-mAP  |         |    26.6%   |             |    25.2%    |     22.8%     |
| COCO-Image | | ![avatar](result/fssd_coco_resnet50_300.jpg) | | ![avatar](result/fssd_coco_darknet19_300.jpg) | ![avatar](result/fssd_coco_mobilenetv1_300.jpg) |

# 4.YOLO
### YOLOv1
|            |  Resnet50-7 | Resnet50-14 | resnet-7 | resnet-14 | vgg16-7 | vgg16-14 |
| ---------- | :---------: | :---------: | :------: | :-------: | :-----: | :------: |
|   VOC-mAP  |    61.1%    |     ---     |  58.4%   |           |         |          |
|  VOC-Image | ![avatar](result/yolov1_voc_resnet50_7.jpg) | - | ![avatar](result/yolov1_voc_resnet_7.jpg) | | | |
|  COCO-mAP  |             |             |          |           |         |          |
| COCO-Image |             |             |          |           |         |          |

### YOLOv2
### YOLOv3
### YOLOv4
|            |  YOLOv4 |
| ---------- | :-----: |
|   VOC-mAP  |         |
|  VOC-Image |         |
|  COCO-mAP  |         |
| COCO-Image | ![avatar](result/yolov4_voc_608.jpg) |

### YOLOv5
