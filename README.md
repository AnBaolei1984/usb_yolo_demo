#这是在比特大陆SM5\SE5上跑usb 摄像头，yolo模型的demo

1. 将工程clone到bmnnsdk2-bm1684_v2.1.0/examples/YOLOv3_object

2. cd usb_yolo_demo

3. make -f Makefile.arm 

4. ./usb_yolo.arm /dev/video0 yolov4_608_coco_int8.bmodel
