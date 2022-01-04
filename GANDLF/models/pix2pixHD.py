import torch
from .modelBase_GAN import ModelBase
from . import networks
#from GANDLF.utils import send_model_to_device
from GANDLF.schedulers import global_schedulers_dict
from GANDLF.optimizers import global_optimizer_dict
import matplotlib.pyplot as plt
from torch.autograd import Variable

class pix2pixHD(ModelBase):

    def __init__(self, parameters: dict):
        """Initialize the pix2pix class.
        """
        ModelBase.__init__(self, parameters)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        #        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']

        try:
            self.label_nc=parameters["model"]["label_nc"]
        except:
            self.label_nc=0
        try:
            self.data_type=parameters["model"]["data_type"]
        except:
            self.data_type=32
        
        try:
            self.no_instance=parameters["model"]["data_type"]
        except:
            self.no_instance=True
            
        try:
            self.feat_num=parameters["model"]["feat_num"]
        except:
            self.feat_num=3
            

        self.loss_names = ['G_GAN',"G_L1", 'D_real', 'D_fake']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G', 'D']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G_HD(self.n_channels, self.n_classes, self.base_filters, self.gen_model_name, 
                                     norm=self.norm_type, gpu_ids=self.gpu_ids, ndim=self.n_dimensions)        

        #self.netG = networks.define_G(self.n_channels, self.n_classes, self.base_filters, self.gen_model_name, self.norm_type,
         #                             gpu_ids=self.gpu_ids, ndim=self.n_dimensions)
        self.netE = networks.define_G_HD(self.n_classes, self.feat_num, self.base_filters, 'encoder', norm=self.norm_type, gpu_ids=self.gpu_ids, 
                                         ndim=self.n_dimensions)  

        self.netD = networks.define_D(self.n_channels+self.n_classes, self.base_filters, self.disc_model_name,
                                           norm=self.norm_type, gpu_ids=self.gpu_ids, ndim=self.n_dimensions)

        
        self.criterionGAN = networks.GANLoss(self.loss_mode).to(self.device)
        #GAN mode will be added to parameter parser
        self.criterionL1 = torch.nn.L1Loss().to(self.device)
        self.criterionVGG = networks.VGGLoss(self.gpu_ids)
        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        parameters["model_parameters"] = self.netG.parameters()
        self.optimizer_G = global_optimizer_dict[parameters["optimizer"]["type"]](parameters)
        parameters["model_parameters"] = self.netD.parameters()
        self.optimizer_D= global_optimizer_dict[parameters["optimizer"]["type"]](parameters)

        self.optimizer_list=[self.optimizer_G,self.optimizer_D]
        self.lambda_L1= parameters["model"]["lambda"]
        print("LAMBDA:",self.lambda_L1)
        print(self.optimizer_D==self.optimizer_G)
        
        # Here 2 different optimizer can be defined if necessary.
        #for i in range(len(self.model_names)):
         #   parameters["optimizer_object"]["{}".format(self.model_names[i])] = self.optimizer_list[i]

    

    def set_input(self, image, label):
        
    

        import cv2
        #label =torch.cat((label,label,label),1)
        #image = torch.cat((image,image,image),1)
        self.real_A = image.to(self.device)
        
        
        self.real_B = label.to(self.device)
        
        
        #self.real_A=transform(self.real_A)

    def forward(self):
        input_label, inst_map, real_image, feat_map = self.encode_input(self.real_B,real_image=self.real_A)  

                   
        input_concat = input_label
        self.fake_B = self.netG.forward(input_concat)

        self.fake_B = self.netG(self.real_A)  # G(A)
        
    def return_optimizers(self):
        return self.optimizer_list

    def backward_D(self):
         # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()


    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_VGG = self.criterionVGG(self.fake_B, self.real_B) * self.lambda_L1

        # combine loss and calculate gradients
        #self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B.detach()) 
        # combine loss and calculate gradients
        self.loss_G = (self.loss_G_GAN + (self.loss_G_L1 * self.lambda_L1)+self.loss_G_VGG)/(2*self.lambda_L1+1)

        self.loss_G.backward()



    def calc_loss(self):

                     # calculate gradients for D
        self.forward()                   # compute fake images: G(A)

        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()  

    def optimize_parameters(self):
        # update G
        # update D's weights
        self.optimizer_G.step()             
        # udpate G's weights
        self.optimizer_D.step()
        



    def return_loss(self):
        self.losses=[self.loss_G_GAN, self.loss_G_L1, self.loss_D_real, self.loss_D_fake]
        #self.losses=[self.loss_G_GAN, self.loss_D_real, self.loss_D_fake]
        
        return self.losses, self.loss_names
    
    def return_generator(self):
        return self.netG
    
    def set_scheduler(self, schedulers):
        self.schedulers=schedulers
    #Not sure if it works
    
    def return_loss_names(self, mode="train"):
        if mode=="train":
            return self.loss_names
        if mode=="valid":
            self.val_loss_names=["G"]
            return self.val_loss_names
        if mode=="test":
            self.test_loss_names=["G"]
            return self.test_loss_names
        else:
            print("Unrecognized argument for mode while returning loss names!!")
    
  

            
    
    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):             
        if self.label_nc == 0:
            input_label = label_map.data.to(self.device)
        else:
            # create one-hot vector for label map 
            size = label_map.size()
            oneHot_size = (size[0], self.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().to(self.device), 1.0)
            if self.data_type == 16:
                input_label = input_label.half()

        # get edges from instance map
        if not self.no_instance:
            inst_map = inst_map.data.to(self.device)
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)         
        input_label = Variable(input_label, volatile=infer)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.to(self.device))

        # instance map for feature encoding
       # if self.use_features:
            # get precomputed feature maps
        #    if self.load_features:
         #       feat_map = Variable(feat_map.data.cuda())
         #   if self.label_feat:
          #      inst_map = label_map.cuda()

        return input_label, inst_map, real_image, feat_map

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)
    def sample_features(self, inst): 
        # read precomputed feature clusters 
        cluster_path = os.path.join(self.checkpoints_dir, self.name, self.cluster_path)        
        features_clustered = np.load(cluster_path, encoding='latin1').item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)                                      
        feat_map = self.Tensor(inst.size()[0], self.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):    
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0]) 
                                            
                idx = (inst == int(i)).nonzero()
                for k in range(self.feat_num):                                    
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
        if self.data_type==16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.to(self.device), volatile=True)
        feat_num = self.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.to(self.device))
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.label_nc):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))                        
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]            
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

        
    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        if self.data_type==16:
            return edge.half()
        else:
            return edge.float()
                                         
    def preprocess(self,img,label):
        img=img/255.0
        label=label.float()
        label=label/255.0
        return img,label