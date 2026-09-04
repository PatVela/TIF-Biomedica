from __future__ import division, print_function
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, add, Flatten, Dropout,MaxPooling1D, Activation, BatchNormalization, Lambda
from keras import backend as K
from keras.optimizers import Adam
from keras.saving import register_keras_serializable
import tensorflow as tf

# Suprime los mensajes de log de TensorFlow y deshabilita las optimizaciones de OneDNN.
# También establece los dispositivos visibles de la GPU a una lista vacía para no usar la GPU por defecto.
tf.config.set_visible_devices([], 'GPU')
tf.get_logger().setLevel('ERROR')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

@register_keras_serializable(package="custom")
def zeropad(x):
    """
    Función de padding personalizada para Keras.
    Esta función concatena el tensor de entrada 'x' con un tensor de ceros
    de la misma forma que 'x', duplicando la última dimensión (canales).
    Es útil para igualar las formas en las conexiones de atajo (shortcuts)
    cuando el número de filtros se duplica en el bloque principal.

    Args:
        x (tf.Tensor): El tensor de entrada.

    Returns:
        tf.Tensor: El tensor con la última dimensión duplicada con ceros.
    """
    y = tf.zeros_like(x) # Crea un tensor de ceros con la misma forma que 'x'.
    return tf.concat([x, y], axis=2) # Concatena 'x' y 'y' a lo largo del eje de características (canales).

@register_keras_serializable(package="custom")
def zeropad_output_shape(input_shape):
    """
    Función para calcular la forma de salida de la capa `zeropad`.

    Args:
        input_shape (tuple): La forma de entrada del tensor.

    Returns:
        tuple: La forma de salida esperada después de aplicar `zeropad`.
    """
    shape = list(input_shape) # Convierte la tupla de forma a una lista para poder modificarla.
    assert len(shape) == 3 # Asegura que la forma tenga 3 dimensiones (batch, timesteps, features).
    shape[2] *= 2 # Duplica el tamaño de la última dimensión (características/canales).
    return tuple(shape) # Retorna la forma como una tupla.


def ECG_model(config):
    """
    Implementación del modelo de red neuronal convolucional (CNN) para ECG,
    basado en la arquitectura ResNet propuesta en el artículo de Nature
    y referencias de GitHub.

    Args:
        config (object): Objeto de configuración que contiene parámetros
                         como filter_length, kernel_size, drop_rate e input_size.

    Returns:
        keras.Model: El modelo Keras compilado listo para entrenar o predecir.
    """
    def first_conv_block(inputs, config):
        """
        Define el primer bloque convolucional del modelo.
        Consiste en una capa convolucional, normalización por lotes, activación ReLU
        y una conexión de atajo (shortcut) con MaxPooling.

        Args:
            inputs (keras.Input): La capa de entrada del modelo.
            config (object): Objeto de configuración.

        Returns:
            keras.layers.Layer: La salida del primer bloque.
        """
        # Primera capa convolucional del bloque.
        layer = Conv1D(filters=config.filter_length, # Número de filtros.
               kernel_size=config.kernel_size, # Tamaño del kernel.
               padding='same', # Mantiene la misma dimensión de salida que la entrada.
               strides=1, # Paso de convolución de 1.
               kernel_initializer='he_normal')(inputs) # Inicializador de pesos He normal.
        layer = BatchNormalization()(layer) # Normalización por lotes.
        layer = Activation('relu')(layer) # Función de activación ReLU.

        # Conexión de atajo (shortcut) para el bloque residual.
        shortcut = MaxPooling1D(pool_size=1, # Pool_size de 1 significa que no reduce las dimensiones.
                      strides=1)(layer) # No reduce la longitud de la secuencia.

        # Segunda capa convolucional del bloque.
        layer =  Conv1D(filters=config.filter_length,
               kernel_size=config.kernel_size,
               padding='same',
               strides=1,
               kernel_initializer='he_normal')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(config.drop_rate)(layer) # Capa de Dropout para regularización.
        # Tercera capa convolucional del bloque.
        layer =  Conv1D(filters=config.filter_length,
                        kernel_size=config.kernel_size,
                        padding='same',
                        strides=1,
                        kernel_initializer='he_normal')(layer)
        return add([shortcut, layer]) # Suma la conexión de atajo con la salida de la capa convolucional (residual).

    def main_loop_blocks(layer, config):
        """
        Define los bloques principales de la red (bucle principal).
        Estos bloques son el núcleo de la arquitectura ResNet, con conexiones residuales
        y duplicación de filtros en ciertos puntos.

        Args:
            layer (keras.layers.Layer): La salida del bloque anterior.
            config (object): Objeto de configuración.

        Returns:
            keras.layers.Layer: La salida de la secuencia de bloques principales.
        """
        filter_length = config.filter_length # Número de filtros inicial.
        n_blocks = 15 # Número total de bloques principales.

        for block_index in range(n_blocks):
            # Determina el 'stride' para subsampling (reducción de dimensionalidad espacial).
            # Se reduce a la mitad cada dos bloques.
            subsample_length = 2 if block_index % 2 == 0 else 1
            # Conexión de atajo para este bloque, aplicando MaxPooling.
            shortcut = MaxPooling1D(pool_size=subsample_length)(layer)

            # Cada 4 bloques (excepto el primero), se duplica el número de filtros
            # y se ajusta la forma del 'shortcut' usando `zeropad` para que coincida.
            # El número 5 se elige en lugar del 4 original.
            if block_index % 4 == 0 and block_index > 0 :
                # Duplica el tamaño de la red (número de filtros) y ajusta las formas.
                shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)
                filter_length *= 2 # Duplica el número de filtros para las capas convolucionales siguientes.

            layer = BatchNormalization()(layer) # Normalización por lotes.
            layer = Activation('relu')(layer) # Activación ReLU.
            # Primera capa convolucional dentro del bucle.
            layer =  Conv1D(filters= filter_length,
                            kernel_size=config.kernel_size,
                            padding='same',
                            strides=subsample_length, # Puede reducir la dimensión espacial.
                            kernel_initializer='he_normal')(layer)
            layer = BatchNormalization()(layer) # Normalización por lotes.
            layer = Activation('relu')(layer) # Activación ReLU.
            layer = Dropout(config.drop_rate)(layer) # Capa de Dropout.
            # Segunda capa convolucional dentro del bucle.
            layer =  Conv1D(filters= filter_length,
                            kernel_size=config.kernel_size,
                            padding='same',
                            strides= 1, # No reduce la dimensión espacial.
                            kernel_initializer='he_normal')(layer)
            layer = add([shortcut, layer]) # Suma la conexión de atajo.
        return layer

    def output_block(layer, config):
        """
        Define el bloque de salida del modelo.
        Consiste en normalización, activación, aplanamiento de la salida
        y una capa densa con activación softmax para la clasificación.

        Args:
            layer (keras.layers.Layer): La salida del último bloque principal.
            config (object): Objeto de configuración (no se usa directamente en este bloque, pero se pasa).

        Returns:
            keras.Model: El modelo Keras compilado.
        """
        layer = BatchNormalization()(layer) # Normalización por lotes.
        layer = Activation('relu')(layer) # Activación ReLU.
        layer = Flatten()(layer) # Aplanamiento de la salida para la capa densa.
        # Capa de salida densa con activación softmax para la clasificación multiclase.
        outputs = Dense(len_classes, activation='softmax')(layer)
        
        # Crea el modelo Keras, conectando la entrada original con las salidas.
        model = Model(inputs=inputs, outputs=outputs)
        
        # Configura el optimizador Adam con una tasa de aprendizaje inicial.
        adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False)
        # Compila el modelo con el optimizador, la función de pérdida (categorical_crossentropy para one-hot encoding)
        # y la métrica a monitorear (accuracy).
        model.compile(optimizer= adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        model.summary() # Imprime un resumen de la arquitectura del modelo en la consola.
        return model

    # Definición de las clases de latidos que el modelo va a clasificar.
    # Algunas clases se excluyen si son muy pocas o no están en el conjunto de entrenamiento.
    classes = ['N','V','/','A','F','~']
    len_classes = len(classes) # Número total de clases.

    # Capa de entrada del modelo, con la forma especificada en la configuración (input_size, 1).
    inputs = Input(shape=(config.input_size, 1), name='input')
    
    # Construye el modelo secuencialmente llamando a los bloques definidos.
    layer = first_conv_block(inputs, config) # Aplica el primer bloque convolucional.
    layer = main_loop_blocks(layer, config) # Aplica los bloques principales en bucle.
    return output_block(layer, config) # Aplica el bloque de salida y compila el modelo.