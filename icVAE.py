from keras.layers import Lambda, Input, Dense,Layer
from keras.losses import mse, binary_crossentropy
from tensorflow.keras import regularizers
from keras.callbacks import Callback
from keras.utils import plot_model
from keras.datasets import mnist
from keras import backend as K
from keras.models import Model
from tensorflow import keras
import tensorflow as tf
import numpy as np
import importlib
import random
import Model
import LOSS
importlib.reload(LOSS)
importlib.reload(Model)

class icVAE:    
    
    def __init__(self,input_dim,latent_dim,high_feature_dim,num_classes,num_label,
                 temperature_c,temperature_g,
                 w_reconstruction,w_gaussian,w_like,w_higher,
                 seed,num_epochs,eps,
                 batch_size,mode):
        super(icVAE, self).__init__()


        self.eps = eps
        self.mode = mode
        self.seed = seed
        self.num_classes = num_classes
        self.num_label = num_label
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        self.w_hig  = w_higher
        self.w_gauss = w_gaussian
        self.w_like = w_like
        self.w_recon = w_reconstruction 
        
        self.temperature_c = temperature_c
        self.temperature_g = temperature_g 
        
        
        
        self.input_dim =input_dim#dictionary
        self.input_dim_rna = input_dim['rna']
        self.input_dim_atac =input_dim['atac']
        
        self.latent_dim = latent_dim
        self.latent_dim_rna = latent_dim['rna']
        self.latent_dim_atac = latent_dim['atac']
        
        self.high_feature_dim = high_feature_dim
     
        self.losses = LOSS.LossFunctions(self.batch_size,self.num_label,self.temperature_c,self.eps)
        
        tf.config.run_functions_eagerly(True)
    
    def loss(self,output):
        rna = output['rna']
        atac= output['atac']
        
        #hf_rna = output['regulon_rna']  
        #hf_atac = output['regulon_atac']  
        #hl_rna = output['cluster_rna']  
        #hl_atac = output['cluster_atac']  
        #loghl_rna = output['cluster_logrna']  
        #loghl_atac = output['cluster_logatac']   
        
        
        z_mean_c = output['z_mean_c']
        z_log_var_c=output['z_log_var_c']
        prob_c=output['prob_c']
        log_prob_c=output['log_prob_c']

        z_mean_u= output['z_mean_u']
        z_log_var_u=output['z_log_var_u']
        prob_u=output['prob_u']
        log_prob_u=output['log_prob_u']

        def losss(mean,var,prob,log_prob):#,hl,loghl
            loss_gaussian = tf.multiply(float(self.w_gauss), self.losses.kl_gaussian(mean, var,average=True)) 
            loss_categorical = tf.multiply(float(self.w_gauss),self.losses.kl_categorical(prob, log_prob, self.num_classes, average=True))
            #loss_label = tf.multiply(float(self.w_gauss),self.losses.kl_categorical(hl, loghl,self.num_classes, average=True))            
            
            loss_total = tf.math.add(loss_gaussian,loss_categorical)
            #loss_total = tf.math.add(loss_total,loss_label)
            return loss_total
        
        loss_dic_common = losss(z_mean_c,z_log_var_c,prob_c,log_prob_c)#,hl_rna,loghl_rna
        loss_dic_unique = losss(z_mean_u,z_log_var_u,prob_u,log_prob_u)#,hl_atac,loghl_atac
        
        kl_loss = tf.math.add(loss_dic_common,loss_dic_unique)
        loss2 = tf.multiply(float(self.w_gauss),kl_loss)
                 
        #higher_loss = self.losses.contrastive_learning_feature(hf_rna,hf_atac)
        #higher_loss = tf.math.add(higher_loss,self.losses.contrastive_learning_label(hl_rna,hl_atac))
        #loss3 = tf.multiply(float(self.w_hig),higher_loss)
        
        if self.mode == 'nb_zip':
            hr = output['hr'] 
            hp = output['hp'] 
            mu_rna = output['mu_rna']  
            mu_atac = output['mu_atac'] 
            pi = output['pi'] 
            
            NB_loss = self.losses.log_nb_positive(rna,hr,hp )
            ZIP_loss = self.losses.log_zip_positive(atac,mu_atac,pi)
            rna_res = self.losses.mean_square_error_positive(rna,mu_rna )
            atac_res = self.losses.mean_square_error_positive(atac,mu_atac)
            
            likelihood = -tf.math.add(NB_loss,ZIP_loss)
            reconstruction_loss = tf.math.add(rna_res ,atac_res)
            
            loss1 = tf.multiply(float(self.w_recon),reconstruction_loss)              
            loss4 = tf.multiply(float(self.w_like),likelihood)
            
            common_vae_loss = tf.math.add(loss1,loss2)
            #common_vae_loss = tf.math.add(common_vae_loss,loss3)
            common_vae_loss = tf.math.add(common_vae_loss,loss4)
            common_vae_loss =tf.keras.backend.mean(common_vae_loss)
        
        if self.mode == 'AE':
            output_rna = output['output_rna']
            output_atac = output['output_atac'] 
            
            reconstruction_loss_rna = mse(rna, output_rna)
            reconstruction_loss_atac = mse(atac, output_atac)
            reconstruction_loss = tf.math.add(reconstruction_loss_rna ,reconstruction_loss_atac)
        
            loss1 = tf.multiply(float(self.w_recon),reconstruction_loss)
            
            common_vae_loss = tf.math.add(loss1,loss2)
            common_vae_loss = tf.math.add(common_vae_loss,loss3)
            common_vae_loss =tf.keras.backend.mean(common_vae_loss)

        
        return common_vae_loss       
        
    def setup_seed(self,seed):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        #torch.backends.cudnn.deterministic = True
   
    

    def train(self,inputs):

        #dataset = tf.data.Dataset.from_tensor_slices(inputs)
        #dataset = dataset.shuffle(buffer_size=100)#咱也不知道该取多少
        #dataset = dataset.batch(batch_size=self.batch_size)
        #dataset = dataset.prefetch(buffer_size=1)
        
        self.setup_seed(self.seed)
        optimizer = tf.keras.optimizers.Adam(1e-5)#more parameters need to be tuned
        
        
        model =  Model.model(self.latent_dim,self.high_feature_dim,self.input_dim,
                        self.num_classes,self.num_label,self.temperature_g,
                        self.eps,self.mode)
        icvae,outputs = model.call()
        losss = self.loss(outputs)
        icvae.add_loss(losss)
        icvae.compile(optimizer)
       
        #iteration_threshold = 20

        # Create an instance of the custom callback
        #custom_callback = TrainPartAfterIteration(icvae,iteration_threshold)
    
        #icvae.fit(inputs,shuffle=True,epochs=self.num_epochs,batch_size=100, callbacks=[custom_callback])
        icvae.fit(inputs,shuffle=True,epochs=self.num_epochs,batch_size=100)

        return icvae


class TrainPartAfterIteration(keras.callbacks.Callback):
    def __init__(self, model,iteration_threshold):
        super(TrainPartAfterIteration, self).__init__()
        self.iteration_threshold = iteration_threshold
        self.model=model
    def on_epoch_begin(self, epoch, logs=None):
        if epoch <= self.iteration_threshold:
            # Enable training for the specific part of the model
            self.model.get_layer('contrastivelayer_feature_rna').trainable = False
            self.model.get_layer('contrastivelayer_feature_atac').trainable = False
            self.model.get_layer('contrastivelayer_label_rna').trainable = False
            self.model.get_layer('contrastivelayer_label_atac').trainable = False
            
        else:
            self.model.get_layer('contrastivelayer_feature_rna').trainable = True
            self.model.get_layer('contrastivelayer_feature_atac').trainable = True
            self.model.get_layer('contrastivelayer_label_rna').trainable = True
            self.model.get_layer('contrastivelayer_label_atac').trainable = True
           
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        