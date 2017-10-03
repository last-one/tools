LOG="log/train-`date +'%Y-%m-%d-%H-%M-%S'`.log"
#nohup python train_scene.py --gpu 3 --train_file /home/hypan/data/place365/data/train.txt --val_file /home/hypan/data/place365/data/val.txt --config config.yml --pretrained whole_resnet50_places365.pth.tar --root /home/hypan/data/place365/data/img --num_classes 80 > $LOG 2>&1 &
python train_scene.py --gpu 3 --train_file /home/hypan/data/scene/data/train.txt --val_file /home/hypan/data/scene/data/val.txt --config config.yml --pretrained whole_resnet50_places365.pth.tar --root /home/hypan/data/scene/data/img --num_classes 80
