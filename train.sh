python3 \
train.py \
--model tailor --data iemocap \
--do_train --num_thread_reader=0 \
--epochs=50 --batch_size=64 \
--n_display=10 \
--lr 5e-5  \
--visual_num_hidden_layers 4 \
--bert_num_hidden_layers 6 \
--audio_num_hidden_layers 4 --aligned \
--epochs_conf=500 \
--use_kt --epochs_kt 50 \
--kt_model Dynamic-tcp