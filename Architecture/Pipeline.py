import librosa
import tensorflow as tf
from audiomentations import AddGaussianSNR, TimeStretch, AddBackgroundNoise, TanhDistortion
from Architecture.Model import DeepSpeech2
from Tools.NLP import Tokenization

class PipelineDeepSpeech2(DeepSpeech2):
    def __init__(self, vocabulary:Tokenization, augmentation=False, params_noise=None, config_model=None):
        super().__init__(vocabulary=vocabulary, opt=None, loss=None, **config_model)
  
        self.augmentation = augmentation
        self.params_noise = params_noise
    
    def augmentationAudio(self, audio):
        if self.augmentation:
            transform = AddBackgroundNoise(**self.params_noise['AddBackgroundNoise'])
            audio = transform(samples=audio, sample_rate=self.sr)

            transform = AddGaussianSNR(**self.params_noise['AddGaussianSNR'])
            audio = transform(samples=audio, sample_rate=self.sr)

            transform = TanhDistortion(**self.params_noise['TanhDistortion'])
            audio = transform(samples=audio, sample_rate=self.sr)

            transform = TimeStretch(**self.params_noise['TimeStretch'])
            audio = transform(samples=audio, sample_rate=self.sr)
            
        return tf.convert_to_tensor(audio, dtype=tf.float32)
    
    def loadAudio(self, path):
        audio,_ = librosa.load(path, sr=self.sr, mono=True)
        return tf.convert_to_tensor(audio, dtype=tf.float32)
    
    def mapProcessing(self, path, lable):
        # Tải audio
        audio = tf.numpy_function(func=self.loadAudio, inp=[path], Tout=tf.float32)
        # Thêm noise
        audio =  tf.numpy_function(func=self.augmentationAudio, inp=[audio], Tout=tf.float32)

        return super().createSpectrogram(audio), super().ctcEncoder(lable)
    
    def __call__(self, dataset, batch_size=1):
        data = tf.data.Dataset.from_tensor_slices(dataset)
        data = (data.map(self.mapProcessing, num_parallel_calls=tf.data.AUTOTUNE)
                .padded_batch(batch_size, padded_shapes=(tf.TensorShape([None, None]), tf.TensorShape([None])))
                .prefetch(buffer_size=tf.data.AUTOTUNE))
        return data


