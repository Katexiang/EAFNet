import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.regularizers import l2
import numpy as np
import math


def conv(numout,kernel_size=3,strides=1,kernel_regularizer=0.0005,padding='same',use_bias=False,name='conv'):
    return tf.keras.layers.Conv2D(name=name,filters=numout, kernel_size=kernel_size,strides=strides, padding=padding,use_bias=use_bias, kernel_regularizer=l2(kernel_regularizer),kernel_initializer=tf.random_normal_initializer(stddev=0.1))
def bn(name,momentum=0.9):
    return tf.keras.layers.BatchNormalization(name=name,momentum=momentum)
def ln(name,momentum=0.9):
    return tf.keras.layers.BatchNormalization(name=name,momentum=momentum)
def mish(x):
    return x*tf.math.tanh(tf.math.softplus(x))
class branch1(keras.Model):
    def __init__(self,scope: str="branch1",numout:int =16,strides:int=1,reg:float=0.0005):
        super(branch1, self).__init__(name=scope)
        self.conv1=conv(numout=numout,strides=strides, kernel_regularizer=reg,name='conv1/conv')
        self.bn1=bn(name='conv1/bn')
        self.conv2=conv(numout=numout, kernel_regularizer=reg,name='conv2/conv')
        self.bn2=bn(name='conv2/bn')
    def call(self,x,training=False):
        x = self.conv1(x)
        x = self.bn1(x,training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x,training=training)
        return x  
		
class branch2(keras.Model):
    def __init__(self,scope: str="branch2",numout:int =16,strides:int=1,reg:float=0.0005):
        super(branch2, self).__init__(name=scope)
        self.conv1=conv(numout=numout,kernel_size=1,strides=strides, kernel_regularizer=reg)
        self.bn1=bn(name='bn')
    def call(self,x,training=False):
        x = self.conv1(x)
        x = self.bn1(x,training=training)
        return x

    
class residual(keras.Model):
    def __init__(self,scope: str="block",numout:int =16,strides:int=1,reg:float=0.0005,branch=False):
        super(residual, self).__init__(name=scope)
        self.branch1=branch1(scope="branch1",numout=numout,strides=strides, reg=reg)
        self.branch2=branch2(scope='convshortcut',numout=numout,strides=strides, reg=reg)
        self.branch=branch
        self.numout=numout
    def call(self,x,training=False):
        block = self.branch1(x,training=training)
        if x.get_shape().as_list()[3] != self.numout or self.branch:
            skip = self.branch2(x,training=training)
            x = tf.nn.relu(block+skip)
            return x
        x = tf.nn.relu(x + block)
        return x
		
class attention(keras.Model):
    def __init__(self,scope: str="attention",numout:int =16,strides:int=1,reg:float=0.0005,branch=False):
        super(attention, self).__init__(name=scope)
        t=int(abs(np.log2(numout)+1)/2)
        k= t if t%2 else t+1
        self.conv1=conv(numout=1, kernel_size=k,kernel_regularizer=reg,name='atten_conv')
    def call(self,x,training=False):
        x = tf.math.reduce_mean(x, axis=[1,2], keepdims=True)
        x = tf.transpose(x,[0,3,2,1])
        x = self.conv1(x)
        x = tf.nn.sigmoid(x)
        x = tf.transpose(x,[0,3,2,1])
        return x


class resnet18(keras.Model):
    def __init__(self,scope: str="Resnet18",reg:float=0.0005):
        super(resnet18, self).__init__(name=scope)
        self.conv1 = conv(numout=64,strides=2, kernel_size=7,kernel_regularizer=reg,name='conv0')
        self.bn1 = bn(name='conv0/bn')
        self.atten_rgb_0 = attention(scope='rgb0_atten',numout=64,reg=reg)
        self.atten_Aop_0 = attention(scope='aop0_atten',numout=64,reg=reg)
		

        self.res0a = residual(scope='group0/block0',numout=64,branch=True,reg=reg)
        self.res0b = residual(scope='group0/block1',numout=64,reg=reg)
        self.atten_rgb_1 = attention(scope='rgb1_atten',numout=64,reg=reg)
        self.atten_Aop_1 = attention(scope='aop1_atten',numout=64,reg=reg)
		
        self.res1a = residual(scope='group1/block0',numout=128,strides=2,reg=reg)
        self.res1b = residual(scope='group1/block1',numout=128,reg=reg)

        self.atten_rgb_2 = attention(scope='rgb2_atten',numout=128,reg=reg)
        self.atten_Aop_2 = attention(scope='aop2_atten',numout=128,reg=reg)
		
        self.res2a = residual(scope='group2/block0',numout=256,strides=2,reg=reg)
        self.res2b = residual(scope='group2/block1',numout=256,reg=reg)
        self.atten_rgb_3 = attention(scope='rgb3_atten',numout=256,reg=reg)
        self.atten_Aop_3 = attention(scope='aop3_atten',numout=256,reg=reg)
		
		
        self.res3a = residual(scope='group3/block0',numout=512,strides=2,reg=reg)
        self.res3b = residual(scope='group3/block1',numout=512,reg=reg)
        self.atten_rgb_4 = attention(scope='rgb4_atten',numout=512,reg=reg)
        self.atten_Aop_4 = attention(scope='aop4_atten',numout=512,reg=reg)
		
        self.conv1Aop = conv(numout=64,strides=2, kernel_size=7,kernel_regularizer=reg,name='conv0_aop')
        self.bn1Aop = bn(name='conv0/bn_aop')
        self.res0aAop = residual(scope='group0/block0_aop',numout=64,branch=True,reg=reg)
        self.res0bAop = residual(scope='group0/block1_aop',numout=64,reg=reg)
        self.res1aAop = residual(scope='group1/block0_aop',numout=128,strides=2,reg=reg)
        self.res1bAop = residual(scope='group1/block1_aop',numout=128,reg=reg)
        self.res2aAop = residual(scope='group2/block0_aop',numout=256,strides=2,reg=reg)
        self.res2bAop = residual(scope='group2/block1_aop',numout=256,reg=reg)
        self.res3aAop = residual(scope='group3/block0_aop',numout=512,strides=2,reg=reg)
        self.res3bAop = residual(scope='group3/block1_aop',numout=512,reg=reg)
		
        self.res0amerge = residual(scope='group0/block0_merge',numout=64,branch=True,reg=reg)
        self.res0bmerge = residual(scope='group0/block1_merge',numout=64,reg=reg)
        self.res1amerge = residual(scope='group1/block0_merge',numout=128,strides=2,reg=reg)
        self.res1bmerge = residual(scope='group1/block1_merge',numout=128,reg=reg)
        self.res2amerge = residual(scope='group2/block0_merge',numout=256,strides=2,reg=reg)
        self.res2bmerge = residual(scope='group2/block1_merge',numout=256,reg=reg)
        self.res3amerge = residual(scope='group3/block0_merge',numout=512,strides=2,reg=reg)
        self.res3bmerge = residual(scope='group3/block1_merge',numout=512,reg=reg)
    def call(self,x,aop,training=False):
        feature=[]
        showfeature=[]
        x=self.conv1(x)
        x=self.bn1(x,training=training)
        x=tf.nn.relu(x)
        x_atten = self.atten_rgb_0(x,training=training)
        showfeature.append(x)
        showfeature.append(x_atten)


        x_aop=self.conv1Aop(aop)
        x_aop=self.bn1Aop(x_aop,training=training)
        x_aop=tf.nn.relu(x_aop)
        x_aop_atten = self.atten_Aop_0(x_aop,training=training)
        showfeature.append(x_aop)				
        showfeature.append(x_aop_atten)		
        m = x_atten*x+x_aop_atten*x_aop		
		
        x=tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool1')
        x_aop=tf.nn.max_pool(x_aop,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool1_aop')
        m=tf.nn.max_pool(m,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool1_fuse')

        x = self.res0a(x,training=training)
        x = self.res0b(x,training=training)
        x_aop = self.res0aAop(x_aop,training=training)
        x_aop = self.res0bAop(x_aop,training=training)
		
		
        x_atten = self.atten_rgb_1(x,training=training)
        x_aop_atten = self.atten_Aop_1(x_aop,training=training)
        showfeature.append(x)
        showfeature.append(x_atten)
        showfeature.append(x_aop)				
        showfeature.append(x_aop_atten)	

        m = self.res0amerge(m,training=training)
        m = self.res0bmerge(m,training=training)		
        m = m+x_atten*x+x_aop_atten*x_aop		
        feature.append(m)
		
        x=self.res1a(x,training=training)
        x=self.res1b(x,training=training)
        x_aop = self.res1aAop(x_aop,training=training)
        x_aop = self.res1bAop(x_aop,training=training)
        x_atten = self.atten_rgb_2(x,training=training)
        x_aop_atten = self.atten_Aop_2(x_aop,training=training)
        m = self.res1amerge(m,training=training)
        m = self.res1bmerge(m,training=training)		
        m = m+x_atten*x+x_aop_atten*x_aop		
        feature.append(m)		
        showfeature.append(x)
        showfeature.append(x_atten)
        showfeature.append(x_aop)				
        showfeature.append(x_aop_atten)	
		
        x=self.res2a(x,training=training)
        x=self.res2b(x,training=training)
        x_aop = self.res2aAop(x_aop,training=training)
        x_aop = self.res2bAop(x_aop,training=training)
        x_atten = self.atten_rgb_3(x,training=training)
        x_aop_atten = self.atten_Aop_3(x_aop,training=training)
        m = self.res2amerge(m,training=training)
        m = self.res2bmerge(m,training=training)		
        m = m+x_atten*x+x_aop_atten*x_aop
        feature.append(m)	
        showfeature.append(x)
        showfeature.append(x_atten)
        showfeature.append(x_aop)				
        showfeature.append(x_aop_atten)	
		
        x=self.res3a(x,training=training)
        x=self.res3b(x,training=training)
        x_aop = self.res3aAop(x_aop,training=training)
        x_aop = self.res3bAop(x_aop,training=training)
        x_atten = self.atten_rgb_4(x,training=training)
        x_aop_atten = self.atten_Aop_4(x_aop,training=training)
        m = self.res3amerge(m,training=training)
        m = self.res3bmerge(m,training=training)		
        m = m+x_atten*x+x_aop_atten*x_aop
        feature.append(m)	
        showfeature.append(x)
        showfeature.append(x_atten)
        showfeature.append(x_aop)				
        showfeature.append(x_aop_atten)	
        return feature,showfeature



class SpatialPyramidPooling(keras.Model):
    def __init__(self,scope: str="spp",reg:float=0.0005,grids=(8,4,2)):
        super(SpatialPyramidPooling, self).__init__(name=scope)
        self.bn0=bn(name='bn0')
        self.bn1=bn(name='blendbn')
        self.conv0=conv(numout=128,kernel_size=1, kernel_regularizer=reg, name='conv0')
        self.conv1=conv(numout=128,kernel_size=1, kernel_regularizer=reg, name='blendconv')
        self.bngroup=[]
        self.convgroup=[]
        self.grids=grids
        self.level=len(grids)
        for i in range(self.level):
            self.bngroup.append(bn(name='bn'+str(i+1)))
            self.convgroup.append(conv(numout=43, kernel_size=1, kernel_regularizer=reg, name='conv1'+str(i+1)))
    def call(self,x,shape=[768,768],training=False):
        levels=[]
        height = math.ceil(shape[0]/32)
        width = math.ceil(shape[1]/32)
        x=tf.nn.relu(self.bn0(x,training=training))
        x=self.conv0(x)
        levels.append(x)
        for i in range(self.level):
            h=height//self.grids[i]
            w=width//self.grids[i]
            kh=height-(self.grids[i]-1) * h
            kw=width-(self.grids[i]-1) * w
            y=tf.nn.avg_pool(x,[1,kh,kw,1],[1,h,w,1],padding='VALID')
            y=self.bngroup[i](y,training=training)
            y=tf.nn.relu(y)
            y=self.convgroup[i](y)
            y=tf.image.resize(y, [height,width])
            levels.append(y)
        x=tf.concat(levels,-1)
        x=self.bn1(x,training=training)
        x=tf.nn.relu(x)
        x=self.conv1(x)
        return x



class Upsample(keras.Model):
    def __init__(self,scope: str="up",reg:float=0.0005):
        super(Upsample, self).__init__(name=scope)
        self.bn0=bn(name='skipbn')
        self.bn1=bn(name='blendbn')
        self.conv0=conv(numout=128, kernel_size=1, kernel_regularizer=reg, name='skipconv')
        self.conv1=conv(numout=128, kernel_regularizer=reg, name='blendconv')
    def call(self,x,skip,training=False):
        skip=tf.nn.relu(self.bn0(skip,training=training))
        skip=self.conv0(skip)
        x=tf.image.resize(x, [skip.shape[1],skip.shape[2]])
        x=x+skip
        x=self.bn1(x,training=training)
        x=tf.nn.relu(x)
        x=self.conv1(x)
        return x



class swiftnet(keras.Model):
    def __init__(self,scope: str="swiftnet",reg:float=0.0005,num_class:int=19):
        super(swiftnet, self).__init__(name=scope)
        self.bn = bn(name='classbn')
        self.conv = conv(numout=num_class, kernel_regularizer=reg, name='classconv')
        self.basenet = resnet18('Resnet18',reg=reg/4)
        self.spp = SpatialPyramidPooling('spp',reg=reg,grids=(8,4,2))
        self.up1 = Upsample('up1',reg=reg)
        self.up2 = Upsample('up2',reg=reg)
        self.up3 = Upsample('up3',reg=reg)
    def call(self,x,aop,shape=[768,768],training=False):
        h=shape[0]
        w=shape[1]
        feature,showfeature=self.basenet(x,aop,training=training)
        x=self.spp(feature[-1],shape=[h,w],training=training)
        x=self.up1(x,feature[-2],training=training)
        x=self.up2(x,feature[-3],training=training)
        x=self.up3(x,feature[-4],training=training)
        x=self.bn(x,training=training)
        x=tf.nn.relu(x)
        x=self.conv(x)
        x=tf.image.resize(x, [h,w])
        x=tf.nn.softmax(x)
        return x



