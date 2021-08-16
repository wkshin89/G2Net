import efficientnet.tfkeras as efn
import tensorflow as tf
import tensorflow_addons as tfa
from scipy.signal import get_window
from sklearn.model_selection import KFold


def build_model(size=256, efficientnet_size=0, weights="imagenet", count=0):
    inputs = tf.keras.layers.Input(shape=(size, size, 3))
    
    efn_string= f"EfficientNetB{efficientnet_size}"
    efn_layer = getattr(efn, efn_string)(input_shape=(size, size, 3), weights=weights, include_top=False) # getattr(efn, efn_string) == efn.efn_string 같지만 활용도가 좋음

    x = efn_layer(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)

    lr_decayed_fn = tf.keras.experimental.CosineDecay(1e-3, count) # learning rate schedule
    opt = tfa.optimizers.AdamW(lr_decayed_fn, learning_rate=1e-4)
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=opt, loss=loss, metrics=["AUC"])
    return model
    
        
def get_lr_callback(batch_size=8, replicas=8):
    lr_start   = 1e-4
    lr_max     = 0.000015 * replicas * batch_size
    lr_min     = 1e-7
    lr_ramp_ep = 3
    lr_sus_ep  = 0
    lr_decay   = 0.7
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
    return lr_callback    