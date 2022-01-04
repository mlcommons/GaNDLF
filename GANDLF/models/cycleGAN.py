import torch
from .modelBase_GAN import ModelBase
from . import networks
from GANDLF.utils import send_model_to_device
from GANDLF.utils import ImagePool

from GANDLF.schedulers import global_schedulers_dict
from GANDLF.optimizers import global_optimizer_dict
import matplotlib.pyplot as plt
#from ..utils.image_pool import ImagePool
import itertools

class cycleGAN(ModelBase):

    def __init__(self, parameters: dict):
        ModelBase.__init__(self, parameters)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        #        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names_A = ['real_A', 'fake_B', 'rec_A']
        self.visual_names_B = ['real_B', 'fake_A', 'rec_B']

        self.lambda_LA= parameters["model"]["lambda_A"]
        self.lambda_LB= parameters["model"]["lambda_B"]
        self.lambda_identity= parameters["model"]["lambda_I"]

        if self.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            self.visual_names_A.append('idt_B')
            self.visual_names_B.append('idt_A')
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        # define networks (both generator and discriminator)
        self.netG_A= networks.define_G(self.n_channels, self.n_classes, self.base_filters, self.gen_model_name, self.norm_type,
                                      gpu_ids=self.gpu_ids, ndim=self.n_dimensions)
        self.netG_B = networks.define_G(self.n_classes, self.n_channels, self.base_filters, self.gen_model_name, self.norm_type,
                                      gpu_ids=self.gpu_ids, ndim=self.n_dimensions)

        self.netD_A = networks.define_D(self.n_classes, self.base_filters, self.disc_model_name,
                                           norm=self.norm_type, gpu_ids=self.gpu_ids, ndim=self.n_dimensions)
        self.netD_B = networks.define_D(self.n_channels, self.base_filters, self.disc_model_name,
                                           norm=self.norm_type, gpu_ids=self.gpu_ids, ndim=self.n_dimensions)

        
        self.criterionGAN = networks.GANLoss(self.loss_mode).to(self.device)
        #GAN mode will be added to parameter parser
        self.criterionL1 = torch.nn.L1Loss().to(self.device)
        try:
            self.pool_size=parameters["model"]["pool_size"]
        except:
            self.pool_size=50
        #initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        parameters["model_parameters"] = itertools.chain(self.netG_A.parameters(), self.netG_B.parameters())
        self.optimizer_G = global_optimizer_dict[parameters["optimizer"]["type"]](parameters)

        parameters["model_parameters"] = itertools.chain(self.netD_A.parameters(), self.netD_B.parameters())
        self.optimizer_D= global_optimizer_dict[parameters["optimizer"]["type"]](parameters)
        
        print(self.optimizer_G)
        if self.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(self.n_channels == self.n_classes)
        self.fake_A_pool = ImagePool(self.pool_size)  # create image buffer to store previously generated images
        self.fake_B_pool = ImagePool(self.pool_size)  # create image buffer to store previously generated images
            # define loss functions
        self.criterionGAN = networks.GANLoss(self.loss_mode).to(self.device)  # define GAN loss.
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        self.optimizers=[]
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)


        self.optimizer_list=[self.optimizer_G,self.optimizer_D]


        print("LAMBDA A:",self.lambda_LA)
        print("LAMBDA B:",self.lambda_LB)

        # Here 2 different optimizer can be defined if necessary.
        #for i in range(len(self.model_names)):
         #   parameters["optimizer_object"]["{}".format(self.model_names[i])] = self.optimizer_list[i]

        

    def set_input(self, image, label):

        #label =torch.cat((label,label,label),1)
        #image = torch.cat((image,image,image),1)
        self.real_A = image.to(self.device)
        
        self.real_B = label.to(self.device)
        #self.real_A=transform(self.real_A)
        print(self.real_B.size())
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
        img=(self.fake_A[0]+1)/2 * 255
        img = img.reshape(512,512)
        import cv2
        cv2.imwrite("./test2.png", img.cpu().detach().numpy())
                    
    def return_optimizers(self):
        return self.optimizer_list

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.lambda_identity
        lambda_A = self.lambda_LA
        lambda_B = self.lambda_LB
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
            
            self.idt_B.to(self.device)
            self.idt_A.to(self.device)
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()



    def calc_loss(self):

        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B


    def optimize_parameters(self):
        # update G
        # update D's weights
        self.optimizer_G.step()             
        # udpate G's weights
        self.optimizer_D.step()
        



    def return_loss(self):
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']

        self.losses=[self.loss_D_A, self.loss_G_A, self.loss_cycle_A, self.loss_idt_A, self.loss_D_B, self.loss_G_B, self.loss_cycle_B, self.loss_idt_B]
        #self.losses=[self.loss_G_GAN, self.loss_D_real, self.loss_D_fake]
        
        return self.losses, self.loss_names
    
    def return_generator(self):
        return self.netG_A
    
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
            
    def preprocess(self,img,label):
        label = label.float()
        img=(2*img/255.0)-1
        label=(2*label/255.0)-1
        return img,label
