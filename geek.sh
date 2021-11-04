# python yolov4.py -dataroot=/Users/jjzhk/data/ -model=darknet -imgsize=608 -datatype=coco -net=yolov4 -phase=test -lr=0.001

python ssd.py -dataroot=/Users/jjzhk/data/ -model=ssd -imgsize=300 -datatype=voc -net=vgg16 -phase=train
# pip install git+https://gitee.com/jjzhk-tools/pyjjzhktools.git
# pip install tabulate
# conda install -c conda-forge pycocotools
# pip install alive-progress