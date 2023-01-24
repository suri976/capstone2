import shutil

from tensorflow import keras

from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16

DIM_SIZE = 224
LAYER_NUM = 52
EPOCHS = 10

train_path = 'dataset/train'
val_path = 'dataset/valid'
test_path = 'dataset/test'

def train_vgg16(base_model, lr=0.01, inner_size=1000):
  
    base_model.trainable = False

    inputs = keras.Input(shape=(DIM_SIZE, DIM_SIZE, 3))

    base = base_model(inputs, training=False)

    vectors = keras.layers.GlobalAveragePooling2D()(base)

    inner = keras.layers.Dense(inner_size, activation='relu')(vectors)

    outputs = keras.layers.Dense(LAYER_NUM)(inner)

    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.Adam(learning_rate=lr)

    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    return model 


if __name__ == '__main__':

    try:
        shutil.rmtree(train_path+'/joker')
        shutil.rmtree(val_path+'/joker')
        shutil.rmtree(test_path+'/joker')
    except: 
        pass
    
    train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_ds = train_gen.flow_from_directory(
        train_path,
        target_size=(DIM_SIZE, DIM_SIZE),
        batch_size=32
    )

    val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    val_ds = val_gen.flow_from_directory(
        val_path,
        target_size=(DIM_SIZE, DIM_SIZE),
        batch_size=32,
        shuffle=False
    )

    vgg16_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(DIM_SIZE, DIM_SIZE, 3)
    )

    chechpoint = keras.callbacks.ModelCheckpoint(
        'vgg16_{epoch:02d}_{val_accuracy:.3f}.h5',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )


    model = train_vgg16(vgg16_model)

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[chechpoint]
    )