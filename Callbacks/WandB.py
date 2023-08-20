import wandb
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
from Architecture.Pipeline import PipelineDeepSpeech2
from jiwer import wer, cer
from Tools.Weights import getPathWeightsNearest

class CustomCallbacksWandB(Callback):
    def __init__(self, pipeline:PipelineDeepSpeech2, path_logs='./Checkpoint/logs/', dev_dataset=None):
        super().__init__()
        self.validation_data = dev_dataset
        self.pipeline = pipeline
        self.path_logs = path_logs
        self.__last_name_update = None
        
    def on_epoch_end(self, epoch: int, logs=None):
        
        # In thử kết quả của một ví dụ ngẫu nhiên trên tập val
        table_wandb = wandb.Table(columns=['Epoch', 'Predict', 'Target'])    
        for batch in self.validation_data.take(1):
            X, Y = batch
            if not X.shape[0] == 1:
                index = np.random.randint(low=0, high=X.shape[0] - 1)
                X, Y = X[index], Y[index]
                X = tf.expand_dims(X, axis=1)
                
            batch_predict = self.model.predict_on_batch(X)
            
            batch_predict_text = self.pipeline.ctcDecoderInPredictions(batch_predict)
            batch_target_text = self.pipeline.ctcDecoder(Y)

            table_output = []
            wandb.log({'wer': wer(batch_target_text, batch_predict_text)})
            wandb.log({'cer': cer(batch_target_text, batch_predict_text)})
            table_output.append([str(epoch+1), batch_predict_text, batch_target_text])
            
            table_wandb = wandb.Table(columns=['Epoch', 'Predict', 'Target'], data=table_output)
            wandb.log({'predict': table_wandb})
            
             # Cập nhật file weights model to cloud wandb
            path_file_update = getPathWeightsNearest(self.path_logs)
            if self.__last_name_update != path_file_update: 
                self.__last_name_update = path_file_update
                wandb.save(path_file_update)
        

            
            
        
            