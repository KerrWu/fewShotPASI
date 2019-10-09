import os
import time
import random
from utils import parse
from data import provider
from models import network
import tensorflow as tf
import tensorflow.keras as K

tf.compat.v1.set_random_seed(512)
random.seed(512)

def lr_scheduler(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.001 * tf.math.exp(0.1 * (10 - epoch))

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

model = network.baseline_backbone()

# load weight
if os.path.isfile(cfg.pretrain_weights):
    print("loading weights from {}".format(cfg.pretrain_weights))
    model.load_weights(cfg.pretrain_weights)

# callbacks
tb_callback = K.callbacks.TensorBoard(log_dir='./logs/tensorboard', histogram_freq=128, batch_size=cfg.batch_size,
                                      write_graph=False, write_grads=True, update_freq='batch')
check_callback = K.callbacks.ModelCheckpoint(filepath=cfg.save_dir, monitor='val_mae', verbose=0,
                                             save_weights_only=True)
lr_callback = K.callbacks.LearningRateScheduler(lr_scheduler)
callbacks = [lr_callback, check_callback, tb_callback]

# train model
model.compile(optimizer="adam",
              loss=["mae", "mae", "mae"],
              metrics=["mae", "mae", "mae"])

model.fit(train_data, epochs=cfg.epochs, validation_data=valid_data, validation_freq=5, callbacks=callbacks)

# save model
if not os.path.isdir(cfg.save_dir):
    os.mkdir(cfg.save_dir)

save_path = os.path.join(cfg.save_dir, cfg.epochs + '.h5')
model.save_weights(filepath=save_path)

end_time = time.time()
std_end_time = time.strftime('%b-%d-%Y-%H:%M:%S', time.gmtime(end_time))
time_cost = (end_time - begin_time) / 3600
print("end at {}, total {} hours".format(std_end_time, time_cost))
