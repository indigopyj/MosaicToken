CUDA_VISIBLE_DEVICES=5 python3 generate_label.py /home/nas1_userB/dataset/ImageNet2012/train ../label_top5_train_nfnet  --model dm_nfnet_f6 --pretrained --img-size 576 -b 32 --crop-pct 1.0
