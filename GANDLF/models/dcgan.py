import torch
from .modelBase_GAN import ModelBase
from . import networks
from GANDLF.utils import send_model_to_device
from GANDLF.schedulers import global_schedulers_dict
from GANDLF.optimizers import global_optimizer_dict
import matplotlib.pyplot as plt


class dcgan(ModelBase):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    def __init__(self, parameters: dict):
        """Initialize the pix2pix class.
        """
        ModelBase.__init__(self, parameters)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        #        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.loss_names = ['G_GAN', 'D_real', 'D_fake']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G', 'D']
        # define networks (both generator and discriminator)
        print(self.gen_model_name)
        self.netG = networks.define_G(self.n_channels, self.n_classes, self.base_filters, self.gen_model_name, self.norm_type,
                                      gpu_ids=self.gpu_ids, ndim=self.n_dimensions, parameters=parameters)

        self.netD = networks.define_D(self.n_classes, self.base_filters, self.disc_model_name,
                                           norm=self.norm_type, gpu_ids=self.gpu_ids, ndim=self.n_dimensions, parameters=parameters)

        
        self.criterionGAN = networks.GANLoss(self.loss_mode).to(self.device)
        #GAN mode will be added to parameter parser
        #self.criterionVGG = networks.VGGLoss(self.gpu_ids)
        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        parameters["model_parameters"] = self.netG.parameters()
        self.optimizer_G = global_optimizer_dict[parameters["optimizer"]["type"]](parameters)
        parameters["model_parameters"] = self.netD.parameters()
        self.optimizer_D= global_optimizer_dict[parameters["optimizer"]["type"]](parameters)

        self.optimizer_list=[self.optimizer_G,self.optimizer_D]
        self.latent_dim=parameters["latent_dim"]
        
        # Here 2 different optimizer can be defined if necessary.
        #for i in range(len(self.model_names)):
         #   parameters["optimizer_object"]["{}".format(self.model_names[i])] = self.optimizer_list[i]

        

    def set_input(self, label):
        
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
       
        #label =torch.cat((label,label,label),1)
        #image = torch.cat((image,image,image),1)
        self.input = torch.zeros(label.shape[0],  self.latent_dim).normal_(0, 1)

        
        self.real_B = label.to(self.device)

        #self.real_A=transform(self.real_A)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.input) 
        print(self.fake_B.size())
        #self.fake_B = self.fake_B.reshape(self.fake_B.shape[0],self.fake_B.shape[2],self.fake_B.shape[3])# G(A)

        
    def return_optimizers(self):
        return self.optimizer_list

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        #real = torch.ones(self.real_B.shape[0], 1).to(self.device)
        #fake = torch.zeros(self.real_B.shape[0], 1).to(self.device)
        # Fake; stop backprop to the generator by detaching fake_B
        #fake_AB = torch.cat((self.fake_B, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(self.fake_B)
        self.loss_D_fake = self.criterionGAN(pred_fake.detach(), False)
        # Real
        #real_AB = torch.cat((self.real_B, self.real_B), 1)
        pred_real = self.netD(self.real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) 
        self.loss_D.backward()


    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        #fake_AB = torch.cat((self.real_B, self.real_B), 1)
        pred_fake = self.netD(self.fake_B)
        #fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        self.loss_G = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        #self.loss_G_VGG = self.criterionVGG(self.fake_B, self.real_B) * self.lambda_L1

        
        """self.loss_G_L1=0
        feat_weights = 4.0 / (4)
        D_weights = 1.0 / self.n_dimensions
        for i in range(self.n_dimensions):
            for j in range(len(pred_fake[i])-1):
                self.loss_G_L1 += D_weights * feat_weights * \
                self.criterionL1(pred_fake[i][j], pred_real[i][j].detach())* self.lambda_L1
         """
        # combine loss and calculate gradients
        #self.loss_G = self.loss_G_GAN + self.loss_G_L1
        # combine loss and calculate gradients

        self.loss_G.backward()



    def calc_loss(self):

                     # calculate gradients for D
        self.forward()                   # compute fake images: G(A)
        self.set_requires_grad(self.netD, False)
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
        self.losses=[self.loss_G, self.loss_D_real, self.loss_D_fake]
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
            
    def preprocess(self,img):
        img=img/255.0
        return img
