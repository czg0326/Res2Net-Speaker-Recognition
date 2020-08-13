export PATH="/home/xiaorunqiu/usr/local/anaconda3/bin:$PATH"

ini_channel=16
num_classes=5994
embedding_size=512
batch_size=64
#log_dir=log/run_CE_timepooling_${batch_size}_${ini_channel}_${embedding_size}/
log_dir=log/log_res2net_64_16_512
python train_CE.py \
--gpu-id '0,1,2,3' \
--lr 0.0001 \
--optimizer 'sgd' \
--wd 1e-3 \
--log-interval 200 \
--batch-size ${batch_size} \
--num-classes ${num_classes} \
--ini-channel ${ini_channel} \
--embedding-size ${embedding_size} \
--pool-size 10000 \
--scale 4 \
--baseWidth 7 \
--log-dir ${log_dir} \
--start-epoch=17 \
--epochs=30 \
--resume ${log_dir}/checkpoint_16.pth
#--resume none
#--resume ${log_dir}/checkpoint_6.pth
