# -*- coding:utf-8    
import numpy as np     
import struct 
import random   
import matplotlib.pyplot as plt     
class Data:
    def __init__(self):# 各参数初始化
        self.images_train=np.zeros((60000,28*28))
        self.images_test=np.zeros((10000,28*28)) 
        self.labels_train=np.zeros((60000,1))
        self.labels_test=np.zeros((10000,1))
        self.init_network()
        self.BATCHSIZE=1
        self.read_images_train()
        self.read_labels_train()
        self.train_data = np.append( self.images_train, self.labels_train, axis = 1 )# 融合输入数据
        self.loss_labels=np.zeros((1, 10)) #标签值

              
    def read_images_train(self):   
        binfile = open('train-images.idx3-ubyte','rb')    
        buf = binfile.read()          
        index = 0    
        magic, numImages, numRows, numColums = struct.unpack_from('>IIII',buf,index)   #读取该文件的前四个参数  分别是魔数，图片数量 图标的行数以及列数
        print (magic,' ',numImages,' ',numRows,' ',numColums,' ')  
        index += struct.calcsize('>IIII')    #向后推进参数的响应位数    
        for i in range(0,60000):     
            im = struct.unpack_from('>784B',buf,index)  #读取每一张图片 大小为28*28=784b
            index += struct.calcsize('>784B' )      
            im = np.array(im)    # 将数据转换为矩阵模式
            im = im.reshape(1,28*28)    #设置矩阵大小为28*28  
            self.images_train[ i , : ]=im          
            #fig = plt.figure(i)    
            #plt.imshow(im,cmap = 'binary') 
            #plt.show()
    def read_images_test(self):   
        binfile = open('t10k-images.idx3-ubyte','rb')    
        buf = binfile.read()          
        index = 0    
        magic, numImages, numRows, numColums = struct.unpack_from('>IIII',buf,index)   #读取该文件的前四个参数  分别是魔数，图片数量 图标的行数以及列数
        print (magic,' ',numImages,' ',numRows,' ',numColums,' ')  
        index += struct.calcsize('>IIII')    #向后推进参数的响应位数    
        for i in range(0,numImages):     
            im = struct.unpack_from('>784B',buf,index)  #读取每一张图片 大小为28*28=784b
            index += struct.calcsize('>784B' )      
            im = np.array(im)    # 将数据转换为矩阵模式
            im = im.reshape(1,28*28)    #设置矩阵大小为28*28  
            self.images_test[ i , : ]=im          
            #fig = plt.figure(i)    
            #plt.imshow(im,cmap = 'binary') 
            #plt.show()'''
    def read_labels_train(self):   
        binfile = open('train-labels.idx1-ubyte','rb')    
        buf = binfile.read()          
        index = 0    
        magic, numLabels= struct.unpack_from('>II',buf,index)    
        index += struct.calcsize('>II')     
        for i in range(0,60000):     
            labels = struct.unpack_from('>B',buf,index)  
            index += struct.calcsize('>B' )      
            self.labels_train[ i , : ]=labels          
    def read_labels_test(self):  
        binfile = open('t10k-labels.idx1-ubyte','rb')    
        buf = binfile.read()          
        index = 0    
        magic, numLabels = struct.unpack_from('>II',buf,index)   
        index += struct.calcsize('>II')     
        for i in range(0,numLabels):     
            labels = struct.unpack_from('>B',buf,index)  
            index += struct.calcsize('>B' )      
            self.labels_test[ i , : ]=labels          
    def init_network(self):
        self.kernel_W1=0.1*np.random.randn(6,3*3) #卷积核初始化
        self.kernel_W2=0.1*np.random.randn(13*13,10) #全连接层初始化
        self.kernel_W3=0.1*np.random.randn(10,10) #全连接层初始化
        
        
    def train(self):
        for i in range( 1000 ): #训练次数
            np.random.shuffle( self.train_data ) #做打乱操作顺序
            image= self.train_data[:self.BATCHSIZE,:-1] # 把除了最后一列外赋予img BATCHSIZE取1
            label = self.train_data[:self.BATCHSIZE,-1:] # 最后一列赋予 label
            #print ("Train Time: ",i)
            image_con=self.con_layers_forward(image,28,28,self.kernel_W1,3,3) #卷积层
            image_pool=self.pooling_layers_forword(image_con,26,26,2,2) #池化层
            image_FC1=self.FC_layers_forward(image_pool,self.kernel_W2) #全连接层
            image_FC2=self.FC_layers_forward(image_FC1,self.kernel_W3)
            loss=self.loss_layers(image_FC2, label)#loss 层

            image_pool_bk1=self.FC_layers_backward(image_FC2,image_FC1,self.kernel_W3,loss)# 全连接层反向传播
            image_pool_bk2=self.FC_layers_backward(image_FC1,image_pool,self.kernel_W2,image_pool_bk1,)# 全连接层反向传播
            image_con_bk=self.pool_layers_backward(image_pool) #池化层反向传播
            self.con_layers_backward(image_con_bk,image) #卷积层反向传播
            
    def con_layers_forward(self,image,imagewide,imageheight,kernel,kernelwide,kernelheight):
        con_out_layers= np.zeros((6,(imagewide-kernelwide+1)*(imageheight-kernelheight+1)))# 行数为卷积核数量   列数为图像大小
        for x in range(6): #6为卷积核数量，下同
            flag=0 #每次执行一次卷积后 加一次
            for i in range(imagewide-kernelwide+1):
                for j in range(imageheight-kernelheight+1):
                    for m in range(kernelwide):
                        for n in range(kernelheight):
                            con_out_layers[x,flag]+=image[0,int((i+m)*imagewide+(j+n))]*kernel[x,int(m*kernelheight+n)] #对应元素相乘
                    flag+=1
            #layers_show= con_out_layers[x]
            #layers_show = layers_show.reshape((imagewide-kernelwide+1),(imageheight-kernelheight+1))
            #plt.imshow(layers_show,cmap = 'binary') 
            #plt.show()
        return self.sigmoid(con_out_layers)
    def pooling_layers_forword (self,image,imagewide,imageheight,kernelwide,kernelheight):
        self.pool_bk_out_layers= np.zeros((6,imagewide*imageheight)) #存放图像中最大值位置，为反向传播做准备
        pool_out_layers= np.zeros((6,13*13)) #6 为卷积核数量
        b= np.zeros((1,kernelwide*kernelheight)) #暂时以每四小块存放原图像中数据,
        for x in range(6): 
            flag=0
            for i in range(0,imagewide,kernelwide):
                for j in range(0,imageheight,kernelwide):
                    block=0
                    for m in range(kernelwide):
                        for n in range(kernelheight):
                            b[0,block]=image[x,(i+m)*imagewide+(j+n)]
                            block+=1
                    pool_out_layers[x,flag]=np.max(b) #得到四小块中的最大值  即最大值池化
                    self.pool_bk_out_layers[x,(np.argmax(b)//kernelwide+i)*imagewide+j+np.argmax(b)%kernelwide]=image[x,(np.argmax(b)//kernelwide+i)*imagewide+j+np.argmax(b)%kernelwide]#得到 图像中最大值的位置，为反向传递做准备
                    flag+=1
            #layers_show= self.pool_out_layers[x]
            #layers_show = layers_show.reshape(13,13)
            #plt.imshow(layers_show,cmap = 'binary') 
            #plt.show()             
        return self.sigmoid(pool_out_layers) 
    def FC_layers_forward(self,image,kernel): #输入为 原图和 卷积核参数
        FC_out_layers= np.zeros((10,6))
        FC_out_layers=np.dot(image,kernel) #全连接层的点积
        return self.sigmoid(FC_out_layers)  #相当于y
    def loss_layers(self,image,label):# image =y  label =t
        for i in range(1):#循环次数与BATCHSIZE有关
            self.loss_labels[i][int(label[i])]=1
        loss=np.sum(image,axis=0) #整理出每个卷积核识别图片的概率
        for m in range(10):
            loss[m]=loss[m]/(np.sum(loss))
        loss_list=(loss-self.loss_labels)*(loss-self.loss_labels)/2
        loss_out=np.sum(loss_list)/10
        print(loss_out)#输出平均loss值
        return (loss-self.loss_labels) 
    def FC_layers_backward(self,image_out,image_origianl,kernel,loss): #输入为原正向传播的输出，输入，所要更新的卷积核参数，以及上一层传下来的loss 输出为传入下一层的loss
        dloss=loss*image_out*(1-(image_out)) #返回y*(1-y)
        dkernel=np.dot(image_origianl.T,dloss)
        dimage_loss=np.dot(dloss,kernel.T)
        kernel-=0.01*dkernel
        return dimage_loss
    def padding(self,image,shape):
        padding_out= np.zeros((6,(26+2*shape)*(26+2*shape)))
        for x in range(6):
            flag=0
            rows=0
            for i in range(0,26):
                rows+=1
                for j in range(0,26):
                    padding_out[x,shape+shape*(2*shape+26)+i*26+j+flag]=image[x,i*26+j]
                    if j==25:
                        flag=2*shape*rows
        return padding_out
    def pool_layers_backward(self,image):# 输入为池化层数据以及反向传播得来的数据 从全连接层传入池化层
        image=image*(1-image)
        for x in range (6):
            index =0
            for i in range(0,26,2):
                for j in range(0,26,2):
                    for m in range(2):
                        for n in range(2):
                            if self.pool_bk_out_layers[x,(i+m)*26+(j+n)] !=0:
                                self.pool_bk_out_layers[x,(i+m)*26+(j+n)]=image[x,index]
                                index+=1
        return self.pool_bk_out_layers
    def sigmoid(self, in_data):
        return 1 / (1 + np.exp(-in_data))
    def con_layers_backward(self,image,image_origianl):
        image=image*(1-image)
        con_padding_bk=self.padding(image,2)
        dimage_con=self.con_layers_forward(con_padding_bk,30,30,self.kernel_W1[::-1],3,3) #向下穿的输入值梯度为 卷积运算 输出图像 与卷积核的倒转180 运算
        dW1=self.con_layers_forward(image_origianl,28,28,image,26,26)
        self.kernel_W1-=0.01*dW1       
data=Data()        
data.train()

    

