import os
import time
import random
import numpy as np
from utils import parse
from data import provider
from models import network
import tensorflow as tf
import tensorflow.keras as K

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.set_random_seed(512)
random.seed(512)


def lr_scheduler(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.001 * np.exp(0.1 * (10 - epoch))


cfg_path = "./configs/baseline_resnet50.json"
cfg = parse.process_config(cfg_path)

begin_time = time.time()
std_begin_time = time.strftime('%b-%d-%Y-%H:%M:%S', time.gmtime(begin_time))
print("begin\n{}".format(std_begin_time))

train_txt_file = os.path.join(cfg.root_dir, cfg.train_txt_file)
train_data = provider.TargetDomainData(train_txt_file, cfg.batch_size, cfg.image_size, cfg.buffer_scale,
                                       is_train=True, data_root_dir=cfg.data_root_dir).data

valid_txt_file = os.path.join(cfg.root_dir, cfg.valid_txt_file)
valid_data = provider.TargetDomainData(valid_txt_file, cfg.batch_size, cfg.image_size, cfg.buffer_scale,
                                       is_train=False, data_root_dir=cfg.data_root_dir).data

model = network.baseline_backbone(image_size=cfg.image_size, backbone_name="resnet50", training=True).build_model()

# load weight
if os.path.isfile(cfg.pretrain_weights):
    print("loading weights from {}".format(cfg.pretrain_weights))
    model.load_weights(cfg.pretrain_weights)

# callbacks
# histogram_freq: frequency in epoch
tb_callback = K.callbacks.TensorBoard(log_dir='./logs/tensorboard', histogram_freq=1, batch_size=cfg.batch_size,
                                      write_graph=False, write_grads=True)
check_callback = K.callbacks.ModelCheckpoint(filepath=cfg.save_dir, monitor='val_mae', verbose=1, save_best_only=False,
                                             save_weights_only=True, mode='min')
lr_callback = K.callbacks.LearningRateScheduler(lr_scheduler)
callbacks = [lr_callback, check_callback, tb_callback]

# train model
model.compile(optimizer="adam",
              loss="mae",
              metrics=["mae"])

model.fit(train_data, epochs=cfg.epochs, validation_data=valid_data, steps_per_epoch=cfg.train_num // cfg.batch_size,
          validation_steps=cfg.valid_num // cfg.batch_size, callbacks=callbacks)

# save model
if not os.path.isdir(cfg.save_dir):
    os.mkdir(cfg.save_dir)

save_path = os.path.join(cfg.save_dir, str(cfg.epochs) + '.h5')
model.save_weights(filepath=save_path)

end_time = time.time()
std_end_time = time.strftime('%b-%d-%Y-%H:%M:%S', time.gmtime(end_time))
time_cost = (end_time - begin_time) / 3600
print("end at {}, total {} hours".format(std_end_time, time_cost))
