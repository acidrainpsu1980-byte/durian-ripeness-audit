cd "$(dirname "$0")"
CNN_THREADS=3 OMP_NUM_THREADS=3 nohup python cnn_common.py outer:0 outer:3 pair:02 pair:12 pair:23 lobo:Immature > cnn_common_0.log 2>&1 &
CNN_THREADS=3 OMP_NUM_THREADS=3 nohup python cnn_common.py outer:1 outer:4 pair:03 pair:13 pair:24 lobo:Mature > cnn_common_1.log 2>&1 &
CNN_THREADS=3 OMP_NUM_THREADS=3 nohup python cnn_common.py outer:2 pair:01 pair:04 pair:14 pair:34 lobo:Overmature > cnn_common_2.log 2>&1 &
wait
