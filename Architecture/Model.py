
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
from Architecture.Loss import ctcLoss
from Tools.NLP import Tokenization
from keras import layers, optimizers, Model, initializers, backend

class CustomModel():
    def __init__(self, model=None, opt=None, loss=None):
        self.model = model
        self.loss = loss
        self.opt = opt
    def fit(self):
        pass
    def build(self):
        pass
    def predict(self):
        pass
    def getConfig(self):
        pass

class DeepSpeech2(CustomModel):
    def __init__(self,
                 vocabulary: Tokenization,
                 name='DeepSpeech2', 
                 sr=16000, nfft=384, stride=160, window=256, epsilon_normalization=1e-10,
                 greedy=True, beam_width=100, top_paths=1,
                 cnn_layers=1, filters=[32], kernel_size=[[11, 41]], strides=[[2, 2]], padding=['same'], 
                 rnn_layers=5, rnn_units=128, hidden_units=256, rate_drop=0.5,
                 opt=optimizers.Adam(),
                 loss=ctcLoss):
        super().__init__(model=None, opt=opt, loss=loss)
    
        # Khởi tạo thông tin cấu trúc model
        self.name = name
        self.vocabulary = vocabulary
        self.sr = sr
        self.nfft = nfft
        self.stride = stride
        self.window = window
        self.epsilon_normalization = epsilon_normalization
        self.greedy = greedy
        self.beam_width = beam_width
        self.top_paths = top_paths
        self.input_dim = nfft // 2 + 1
        self.output_dim = vocabulary.getLenVocabulary()
        self.cnn_layer = cnn_layers
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rnn_layers = rnn_layers
        self.rnn_units = rnn_units
        self.hidden_units = hidden_units
        self.rate_drop = rate_drop
        self.opt = opt
         
    def build(self, summary=False):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            '''Model similar to DeepSpeech2.'''
            # Model's input
            input_spectrogram = layers.Input((None, self.input_dim), name='input')

            # Expand the dimension to use 2D CNN.
            x = layers.Reshape((-1, self.input_dim, 1), name='expand_dim')(input_spectrogram)

            for i in range(1, self.cnn_layer + 1):
                x = layers.Conv2D(
                    filters=self.filters[i - 1],
                    kernel_size=self.kernel_size[i - 1],
                    strides=self.strides[i - 1],
                    padding=self.padding[i - 1],
                    use_bias=False,
                    name=f'conv_{i}',
                    kernel_initializer=initializers.GlorotNormal(),
                )(x)
                x = layers.BatchNormalization(name=f'conv_{i}_bn')(x)
                x = layers.ReLU(name=f'conv_{i}_relu')(x)
                
            # Reshape the resulted volume to feed the RNNs layers
            x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)

            # RNN layers
            for i in range(1, self.rnn_layers + 1):
                recurrent = layers.GRU(
                    units=self.rnn_units,
                    activation='tanh',
                    recurrent_activation='sigmoid',
                    use_bias=True,
                    return_sequences=True,
                    reset_after=True,
                    name=f'gru_{i}',
                    kernel_initializer=initializers.GlorotNormal(),
        
                )
                x = layers.Bidirectional(
                    recurrent, name=f'bidirectional_{i}', merge_mode='concat'
                )(x)
                
                if i < self.rnn_layers:
                    x = layers.Dropout(rate=self.rate_drop)(x)
                

            # Dense layer
            x = layers.Dense(units=self.hidden_units, name='dense_1')(x)
            x = layers.ReLU(name='dense_1_relu')(x)
            x = layers.Dropout(rate=self.rate_drop)(x)
            
            # Classification layer
            output = layers.Dense(units=self.output_dim + 1, activation='softmax')(x)

            # Model
            model = Model(input_spectrogram, output, name=self.name)

            # Compile the model
            model.compile(optimizer=self.opt, loss=self.loss)
        
        # Output info model
        if summary:
            model.summary()
        self.model = model
        return self

    def fit(self, train_dataset=None, dev_dataset=None, epochs=1, callbacks=None):
        # Huấn luyện model
        self.model.fit(x=train_dataset, 
                       validation_data=dev_dataset,
                       epochs=epochs,
                       callbacks=callbacks
                       )
        return self
    
    def predict(self, audio):
        tf_audio = tf.convert_to_tensor(audio)
        tf_audio_input = self.createSpectrogram(tf_audio)
        tf_text_output = self.model.predict_on_batch(tf_audio_input)
        text_output = self.ctcDecoderInPredictions(tf_text_output)
        return text_output
    
    def getConfig(self):
        return {
            'name': self.name,
            'sr': self.sr,
            'nfft': self.nfft,
            'stride': self.stride,
            'window': self.window,
            'epsilon_normalization': self.epsilon_normalization,
            'greedy': self.greedy,
            'beam_width': self.beam_width,
            'top_paths': self.top_paths,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'cnn_layer': self.cnn_layers,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'rnn_layers': self.rnn_layers,
            'rnn_units': self.rnn_units,
            'hidden_units': self.hidden_units,
            'rate_drop': self.rate_drop,
        }
    
    def createSpectrogram(self, tf_audio):
        # Tạo spectrogram
        spectrogram = tfio.audio.spectrogram(tf_audio,
                                             nfft=self.nfft,
                                             stride=self.stride,
                                             window=self.window)
        
        # Tăng cường độ của các tín hiệu nhỏ 
        spectrogram = tf.math.pow(spectrogram, 0.5)
        
        # Chuẩn hóa
        means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
        stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
        spectrogram = (spectrogram - means) / (stddevs + self.epsilon_normalization)
        
        return spectrogram
    
    def ctcDecoderInPredictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = backend.ctc_decode(pred,
                                     input_length=input_len,
                                     greedy=self.greedy,
                                     beam_width=self.beam_width,
                                     top_paths=self.top_paths)[0][0]
        output_text = []
        for result in results:
            result = tf.strings.reduce_join(self.vocabulary.num_to_char(result)).numpy().decode('UTF-8')
            output_text.append(result)
        return ''.join(output_text)
    
    def ctcEncoder(self, text):
        output_text = tf.strings.unicode_split(text, input_encoding='UTF-8')
        output_text = self.vocabulary.char_to_num(output_text)
        return output_text
    
    def ctcDecoder(self, y):
        targets = []
        for label in y:
                label = (
                    tf.strings.reduce_join(self.vocabulary.num_to_char(label)).numpy().decode('UTF-8')
                )
                targets.append(label)
        return ''.join(targets)