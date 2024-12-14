python3 train.py \
  --gpu_id=0 \
  --lr=1e-4 \
  --save_path=./output/ \
  --is_emb=training \
  --batch_size=256 \
  --epochs=10 \
  --model_name=FedIC \
  --alpha_rationle=0.2 \
  --infor_loss=0.1 \
  --regular=1 \
  --class_num=115 \
  --data_type=length \


