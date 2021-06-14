# EventMaskTrain

To train your model, run:

```
CUDA_VISIBLE_DEVICES=0,1 ./tools/train.py --cfg experiments/config/meva_kf1_train_split2_new.yaml  
```

After the training, you may find your results in $SAVE_DIR/$exp_name. By default, the results are stored at ./experiments/ckpt/exp. 

To customize the training on your datasets, you may need to modify the yaml file accordiningly. 



# Thanks

The general framework of this code is borrowed from my another repository: https://github.com/kgl-prml/Contrastive-Adaptation-Network-for-Unsupervised-Domain-Adaptation. 
