import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input,concatenate,Lambda,Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation,Dense,Reshape,Flatten
from keras.layers.convolutional import Cropping3D
from keras_contrib.layers.convolutional import Deconvolution3D
from keras.layers.convolutional import Convolution3D,Convolution2D,Conv2D,MaxPooling2D
import numpy as np
import os
import random
from keras.utils import generic_utils

from os import listdir
from os.path import isfile, join
from skimage import color,transform
from skimage import io

import time
import binvox_rw as bvx
import glob

img_loc = '/Users/aiwabdn/Downloads/ShapeNetRendering/02933112/1a1b62a38b2584874c62bee40dcdc539/rendering'
vox_loc = '/Users/aiwabdn/Downloads/ShapeNetVox32/02933112/1a1b62a38b2584874c62bee40dcdc539'

# flag for multiview encoder
multiview = True

# INPUT IMAGE DIMENSIONS
input_dim1 = 1
input_dim2 = 200
input_dim3 = 200
output_dim = 32*32*32
encoding_dim = 200

# TOTAL EPOCHS
nb_epoch = 7
n_batch_per_epoch = 5
n_screenshots = 24
BATCH_SIZE = 1
# no. of dis iteration for each gen iteration
disc_iter = 5
# GRADIENT PENALTY MULTIPLIER AS USED IN  IMPROVED WGAN
LAMBDA = 10
epoch_size = n_batch_per_epoch * BATCH_SIZE
# TRAINING DATASET DIRECTOREY
#directory='/Users/aiwabdn/Downloads/ShapeNetRendering/02933112/'
directory = img_loc

# PASS TENSORFLOW SESSION TO KERAS
sess = tf.Session()
K.set_session(sess) 

# SET LEARNING PHASE TO TRAIN
K.set_learning_phase(1)

# Keras Lambda layer to calculate row wise max of a 2D tensor
def row_max(x):
    return K.max(x, 2)

def save_model_weights(encoder_model,generator_model, discriminator_model, e):
    model_path = "../../models/3DWGAN_airplanes1/" 
    if e % 25 == 0:
        enc_weights_path = os.path.join(model_path, '%s_epoch%s.h5' % (encoder_model.name, e))
        encoder_model.save_weights(enc_weights_path, overwrite=True)
        
        gen_weights_path = os.path.join(model_path, '%s_epoch%s.h5' % (generator_model.name, e))
        generator_model.save_weights(gen_weights_path, overwrite=True)

        disc_weights_path = os.path.join(model_path, '%s_epoch%s.h5' % (discriminator_model.name, e))
        discriminator_model.save_weights(disc_weights_path, overwrite=True)

def load_image( infilename ) :
    img = color.rgb2gray(io.imread(infilename))
    img = transform.resize(img,(input_dim1,input_dim2,input_dim3)) 
    data = np.asarray(img, dtype="float32")
    return data
    
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def batch_generator(BATCH_SIZE, directory, train=True):
    img_data = [load_image(f) for f in glob.glob(img_loc+'/*.png')]
    with open(vox_loc+'/model.binvox', 'rb') as f:
        voxel = np.reshape(np.array(bvx.read_as_3d_array(f).data, dtype='float32'), (1,32,32,32))
    #_3d_data = [voxel for i in range(len(img_data))]
    #return np.asarray(img_data),np.squeeze(np.asarray(_3d_data))
    return np.array(img_data),voxel

def batch_generator1(BATCH_SIZE,directory,train=True):
    if not train:
        directory = directory[:-1]+"_test/"
    dir_list = get_immediate_subdirectories(directory)[0]
    ids = random.sample(range(0, len(dir_list)), BATCH_SIZE)
    img_data =[]
    _3d_data = []
    
    for folder_id in ids:
        image_id = random.sample(range(0, 4),1)
        folder_path = directory+dir_list[folder_id]+'/rendering/'
        onlyfiles = [f for f in listdir(folder_path) if (isfile(join(folder_path, f)))]
        
        onlyfiles= list(filter(lambda a: (a.endswith("4.png") or a.endswith("6.png") or a.endswith("7.png") or a.endswith("8.png")), onlyfiles))
        #print(onlyfiles)
        image_path = folder_path + onlyfiles[image_id[0]]
        #print(image_path)
        with open('/Users/aiwabdn/Downloads/ShapeNetVox32/02933112/'+dir_list[folder_id]+'/model.binvox', 'rb') as f:
            _3d = np.array(bvx.read_as_3d_array(f).data, dtype='float32')
        img = load_image(image_path)
        img_data.append([img])
        _3d_data.append([_3d])
        
    return np.asarray(img_data),np.squeeze(np.asarray(_3d_data),axis=1)
        

def inf_train_gen1(BATCH_SIZE,directory):
    while True:
        img,_3d = batch_generator(BATCH_SIZE,directory)
        yield img,_3d

def inf_train_gen(BATCH_SIZE, directory):
    while True:
        yield img_data,_3d_data
            
            

img_data,_3d_data = batch_generator(BATCH_SIZE, ' ')


def sample_noise(batch_size):
    return np.random.normal(0, 1, size=[batch_size,encoding_dim,1,1,1])
        
# Encoder network. Takes object image set of size n_screenshots, passes them through a VGG convolutional shared model, max-pools and encodes to z_mean and z_log_var
def multiview_encoder(object_img_set):
    input = Input(shape=(input_dim1, input_dim2, input_dim3))
    h = Conv2D(64, (3,3), activation='relu', padding='same')(input)
    h = MaxPooling2D((2,2))(h)
    h = Conv2D(128, (3,3), activation='relu', padding='same')(h)
    h = MaxPooling2D((2,2))(h)
    h = Conv2D(256, (3,3), activation='relu', padding='same')(h)
    h = Conv2D(256, (3,3), activation='relu', padding='same')(h)
    h = MaxPooling2D((2,2))(h)
    h = Conv2D(512, (3,3), activation='relu', padding='same')(h)
    h = Conv2D(512, (3,3), activation='relu', padding='same')(h)
    h = Conv2D(512, (3,3), activation='relu', padding='same')(h)
    h = Conv2D(512, (3,3), activation='relu', padding='same')(h)
    h = MaxPooling2D((2,2))(h)
    #h = Flatten()(h) #have no idea why this doesn't work in Keras2.0.3. Raised issue on fchollet github (https://github.com/fchollet/keras/issues/6369)
    h = Reshape((np.prod(list(h._keras_shape[1:])),1))(h)
    convolver = Model(inputs=input, outputs=h)
    convolved_inputs = [convolver(i) for i in object_img_set]
    conc = concatenate(convolved_inputs, axis=-1)
    max_pool = Lambda(row_max)(conc)
    inner_dense = Dense(4096, activation='relu')(max_pool)
    dropout = Dropout(0.5)(inner_dense)
    z_mean = Dense(encoding_dim, activation='relu')(dropout)
    z_log_var = Dense(encoding_dim, activation='relu')(dropout)
    return z_mean, z_log_var

def encoder(encoder_input):
    #h = Reshape((1,137,137))(encoder_input)
    h = Convolution2D(filters=64,kernel_size=11,padding='same',input_shape=(1, 200, 200),strides= 4,kernel_initializer ="he_normal")(encoder_input)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
    
    h = Convolution2D(filters=128,kernel_size=5,padding='same',strides = 2,kernel_initializer ="he_normal")(h)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
    
    h = Convolution2D(filters=256,kernel_size=5,padding='same',strides= 2,kernel_initializer ="he_normal")(h)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
    
    
    h = Convolution2D(filters=512,kernel_size=5,padding='same',strides= 2,kernel_initializer ="he_normal")(h)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
    
    
    h = Convolution2D(filters=400,kernel_size=8,padding='same',strides= 1,kernel_initializer ="he_normal")(h)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
    
    h = Flatten()(h)
    
    z_mean = Dense(200,kernel_initializer ="he_normal")(h)
    z_log_var = Dense(200,kernel_initializer ="he_normal")(h)

    return z_mean,z_log_var

def discriminator(discriminator_input):
    
    h = Reshape((1,32,32,32))(discriminator_input)
    
    h = Convolution3D(filters=64, kernel_size=4,padding='same',use_bias=False, strides = 2,kernel_initializer ="he_normal")(h)
    #h = BatchNormalization(mode=2, axis=1)(h)
    h = LeakyReLU(0.2)(h)
    
    h = Convolution3D(filters=128, kernel_size=4,padding='same',use_bias=False, strides = 2,kernel_initializer ="he_normal")(h)#(h)
    #h = BatchNormalization(mode=2, axis=1)(h)
    h = LeakyReLU(0.2)(h)
   
    h = Convolution3D(filters=256, kernel_size=4,padding='same',use_bias=False, strides = 2,kernel_initializer ="he_normal")(h)
    #h = BatchNormalization(mode=2, axis=1)(h)
    h = LeakyReLU(0.2)(h)
   
    
    h = Convolution3D(filters=512, kernel_size=4, padding='same',use_bias=False, strides = 2,kernel_initializer ="he_normal")(h)
    #h = BatchNormalization(mode=2, axis=1)(h)
    h = LeakyReLU(0.2)(h)
    
    
    h = Convolution3D(1, 3, padding="same", use_bias=False,kernel_initializer ="he_normal")(h)

    #h = Dense(1024,activation='relu') (h)
    discriminator_output = Dense(1,kernel_initializer ="he_normal")(h)
    
    return discriminator_output

def generator(generator_input):
    h = Deconvolution3D(filters=384, kernel_size=(4, 4, 4),padding="valid", output_shape=(None, 384, 4, 4, 4),use_bias=False)(generator_input)
    
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
     #h = Dropout(0.2)(h)
    
    h = Deconvolution3D(filters=192, kernel_size=(4, 4, 4),strides=(2, 2, 2),output_shape=(None, 192, 10, 10, 10) ,padding='valid',use_bias=False)(h)
    h = Cropping3D(cropping=((1, 1), (1, 1), (1, 1)))(h)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
     #h = Dropout(0.2)(h)
    
    
    h = Deconvolution3D(filters=96, kernel_size=(4, 4, 4),strides=(2, 2, 2), output_shape=(None, 96, 18, 18, 18),padding='valid',use_bias=False)(h)
    h = Cropping3D(cropping=((1, 1), (1, 1), (1, 1)))(h)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
     #h = Dropout(0.2)(h)
    
    h = Deconvolution3D(filters=48, kernel_size=(4, 4, 4),strides=(2, 2, 2), output_shape=(None, 48, 34, 34, 34),padding='valid',use_bias=False)(h)
    h = Cropping3D(cropping=((1, 1), (1, 1), (1, 1)))(h)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
     #h = Dropout(0.2)(h) 
    
    h = Deconvolution3D(filters=1, kernel_size=(3, 3, 3),padding='same',output_shape=(None, 1, 32, 32, 32),use_bias=False)(h)
    
    generator_output = Activation('tanh')(h)
    
    h= Flatten()(generator_output)
    
    return h
    
def inference(image_data,real_3d_data):
    """ Connections b/w different models"""
    z_p = tf.random_normal((BATCH_SIZE, encoding_dim,1,1,1), 0, 1) # normal dist for GAN
    eps = tf.random_normal((BATCH_SIZE, encoding_dim), 0, 1) # normal dist for VAE
    
    ### ENCODER                      
    if multiview:
        encoder_input = [Input(shape=[input_dim1,input_dim2,input_dim3]) for i in image_data]
        enc_mean,enc_sigma  = multiview_encoder(encoder_input)
    else:
        encoder_input = Input(shape = [input_dim1,input_dim2,input_dim3])
        enc_mean,enc_sigma  = encoder(encoder_input)
    e_net = Model(inputs=encoder_input, outputs=[enc_mean,enc_sigma],name="encoder")
    z_x_mean, z_x_log_sigma_sq = e_net(image_data) # get z from the input   
    
    ### GENERATOR                                 
    generator_input  = Input(shape = [200,1,1,1])
    generator_output = generator(generator_input)
    g_net = Model(inputs=generator_input, outputs=generator_output,name="generator")
    z_x = z_x_mean + K.exp(z_x_log_sigma_sq / 2) * eps # get actual values of z from mean and variance
                          
    z_x = tf.reshape(z_x,[BATCH_SIZE,200,1,1,1]) # reshape into a 200*1*1*1 array
    x_p = g_net(z_p)   # output from noise
    x_tilde = g_net(z_x)  # output 3d model of the image
    
    ### DISCRIMINATOR
    discriminator_input = Input(shape = [1,32,32,32])
    d = discriminator(discriminator_input)   
    d_net = Model(inputs=discriminator_input, outputs=d,name="discriminator")
    
    #_, l_x_tilde = discriminator(x_tilde)
    d_x =   d_net(real_3d_data) 
    d_x_p = d_net(x_p)
        
    return e_net,g_net,d_net,x_tilde,z_x_mean, z_x_log_sigma_sq, z_x, x_p, d_x, d_x_p, z_p

def loss(z_x_log_sigma_sq, z_x_mean,x_tilde, d_x, d_x_p, input_dim1, input_dim2, input_dim3,real_3d_data,x_p):
    """
    Loss functions for  KL divergence, Discrim, Generator, Lth Layer Similarity
    """
    KL_loss = (- 0.5 * K.sum(1 + z_x_log_sigma_sq - K.square(z_x_mean) - K.exp(z_x_log_sigma_sq), axis=-1))/input_dim1/input_dim2/input_dim3
    
    # Discriminator Loss
    D_loss = tf.reduce_mean(d_x_p) - tf.reduce_mean(d_x)
    # Generator Loss    
    G_loss = -tf.reduce_mean(d_x_p)
    # Reconstruction loss
    Reconstruction_loss = tf.reduce_sum(tf.square(x_tilde-tf.reshape(real_3d_data,[BATCH_SIZE,-1])))/input_dim1/input_dim2/input_dim3
                           
    alpha = tf.random_uniform(
    shape=[BATCH_SIZE,1],
    minval=0.,maxval=1.)
    
    differences = x_p-tf.reshape(real_3d_data,[BATCH_SIZE,-1])
    interpolates = tf.reshape(real_3d_data,[BATCH_SIZE,-1]) + (alpha*differences)
    gradients = tf.gradients(d_net(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
                                
    return  KL_loss, D_loss, G_loss, Reconstruction_loss,gradient_penalty
    
# define input placeholders
if multiview:
    image_data = [tf.placeholder(tf.float32, shape=[BATCH_SIZE,input_dim1,input_dim2,input_dim3]) for _ in range(n_screenshots)]
else:
    image_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE,input_dim1,input_dim2,input_dim3])
real_3d_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE,32,32,32])

# define network structure
e_net,g_net,d_net,x_tilde,z_x_mean, z_x_log_sigma_sq, z_x, x_p, d_x, d_x_p, z_p = inference(image_data,real_3d_data)

# define individual losses
KL_loss, D_loss, G_loss, Reconstruction_loss , gradient_penalty= loss(z_x_log_sigma_sq, z_x_mean,x_tilde, d_x, d_x_p, input_dim1, input_dim2, input_dim3,real_3d_data,x_p)


## add up losses for each part
L_e = KL_loss + Reconstruction_loss
L_g = Reconstruction_loss + G_loss
L_d = D_loss + LAMBDA* gradient_penalty 


# get trainable weights
E_params = e_net.trainable_weights
G_params = g_net.trainable_weights
D_params = d_net.trainable_weights




# define optimizers
enc_train_op = tf.train.AdamOptimizer(
    learning_rate=1e-4, 
    beta1=0.5,
    beta2=0.9
).minimize(L_e, var_list=E_params)

gen_train_op = tf.train.AdamOptimizer(
    learning_rate=1e-4, 
    beta1=0.5,
    beta2=0.9
).minimize(L_g, var_list=G_params)

disc_train_op = tf.train.AdamOptimizer(
    learning_rate=1e-4, 
    beta1=0.5, 
    beta2=0.9
).minimize(L_d, var_list=D_params)


# define iterator
img_batch,_3d_batch = batch_generator(BATCH_SIZE ,directory)



# generates samples each x epochs
eps = tf.random_normal((BATCH_SIZE, 200), 0, 1)
z_x_sample_mean,z_x_sample_log_sigma_sq = e_net(image_data)

z_x_sample = z_x_sample_mean + K.exp(z_x_sample_log_sigma_sq / 2) * eps
z_x_sample = tf.reshape(z_x_sample,[BATCH_SIZE,200,1,1,1])

fixed_image_samples = g_net(z_x_sample)

def generate_sample(epoch):
    samples = session.run(fixed_image_samples,feed_dict= {image_data:img_batch})
    Reconstruction_loss1 = np.sum(np.square(samples-np.reshape(_3d_batch,[BATCH_SIZE,-1])))/input_dim1/input_dim2/input_dim3
    img_batch1,_3d_batch1 = batch_generator(BATCH_SIZE ,directory,train=False)

    samples = session.run(fixed_image_samples,feed_dict= {image_data:img_batch1})
    Reconstruction_loss2 = np.sum(np.square(samples-np.reshape(_3d_batch1,[BATCH_SIZE,-1])))/input_dim1/input_dim2/input_dim3
    
                                 
    print("\ntrain :"+str(Reconstruction_loss1)+" test:"+str(Reconstruction_loss2)+"for epoch "+str(epoch))
    

    if not os.path.exists('../../results_airplane1/'):
        os.makedirs('../../results_airplane1/')
    np.save('../../results_airplane1/'+str(epoch)+"_"+'.npy',samples)
    np.save('../../results_airplane1/gt_.npy',_3d_batch)
    np.save('../../results_airplane1/gt_image_.npy',img_batch)
    



with sess as session:

    session.run(tf.initialize_all_variables())
    
    #e_net.load_weights('/home/sam/Downloads/Olam-ML-master (3)/3DWGAN/models/3DWGAN_airplanes1/encoder_epoch175.h5')
    #g_net.load_weights('/home/sam/Downloads/Olam-ML-master (3)/3DWGAN/models/3DWGAN_airplanes1/generator_epoch175.h5')
    #d_net.load_weights('/home/sam/Downloads/Olam-ML-master (3)/3DWGAN/models/3DWGAN_airplanes1/discriminator_epoch175.h5')
    
    gen =  inf_train_gen(BATCH_SIZE,directory)
    gen_iterations = 0

    #################
    # Start training
    ################
    for e in range(nb_epoch):
        # kernel_initializerialize progbar and batch counter
        progbar = generic_utils.Progbar(epoch_size)
        batch_counter = 1
        start = time.time()
        list_enc_loss = []
        list_gen_loss = []
        list_dis_loss = []
        
        while batch_counter < n_batch_per_epoch:

            
            # get next batch
            _2d_data = None
            while _2d_data is None:
                try:
                   
                    print('got data')
                    _2d_data,_3d_data = img_data,_3d_data #batch_generator(BATCH_SIZE, ' ') #gen.next()
                    print('got data')
                except:
                     pass
            
            for d in range(disc_iter):
                
                print(d)
                dis_cost, _ = session.run([L_d, disc_train_op],feed_dict={real_3d_data :_3d_data})
                list_dis_loss.append(dis_cost)
                
                
                # get next batch
                _2d_data = None
                while _2d_data is None:
                    try:
                       
                        _2d_data,_3d_data = img_data,_3d_data #batch_generator(BATCH_SIZE, ' ') #gen.next()
                    except:
                         pass

            if multiview:
                input_dict = {k:v for k,v in zip(image_data, _2d_data)}
            else:
                input_dict = {image_data: _2d_data}

            enc_cost, _ = session.run([L_e, enc_train_op],feed_dict=input_dict.update({real_3d_data:_3d_data}))
            list_enc_loss.append(enc_cost)
            
            gen_cost, _ = session.run([L_g, gen_train_op],feed_dict=input_dict.update({real_3d_data:_3d_data}))
            list_gen_loss.append(gen_cost)
                        
            tf.reshape(z_x,[BATCH_SIZE,200,1,1,1])

            gen_iterations += 1
            batch_counter += 1
            progbar.add(BATCH_SIZE, values=[("Loss_E", np.mean(list_enc_loss)),("Loss_G", np.mean(list_gen_loss)),("Loss_D", np.mean(list_dis_loss))])

            # Save images for visualization ~2 times per epoch
            if batch_counter % (n_batch_per_epoch / 2) == 0:
                generate_sample(e)

        print('\nEpoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))
        
        
        # Save model weights (by default, every 5 epochs)
        #save_model_weights(e_net,g_net, d_net, e)

