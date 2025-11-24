import os
import csv
import torch.nn.functional as F
import sys
from util import viz
import config as c
import modules.Unet_common as common
import warnings
from torchvision import models
import torchvision.transforms as transforms
from util.vgg_loss import VGGLoss
from util.triplet_loss import TripletLoss
from util.nccloss import NCCLoss
import time
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
from args import get_args_parser
from util.utils import *
from Hinet_model.model import *
from datetime import datetime

from diffusers.models import AutoencoderKL

from torch.utils.data import DataLoader
from util.dataset_all import Dataset

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = get_args_parser()

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
outputpath = os.path.join(args.outputpath, current_time)
os.makedirs(outputpath, exist_ok=True)

# 定义日志文件路径
log_dir = os.path.join(outputpath, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"training_log_{current_time}.csv")

# 初始化日志文件
with open(log_file, "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Loss1 (Perc)", "Loss2 (VGG)", "Loss3 (Low)", "Loss4 (Latent-copy)", "Loss5 (Latent-cover)", "Total Loss","lr"])

#####################
# Model initialize: #
#####################
INN_net = Model().to(device)
init_model(INN_net)
INN_net = torch.nn.DataParallel(INN_net,device_ids=[0])
para = get_parameter_number(INN_net)
print(para)
params_trainable = (list(filter(lambda p: p.requires_grad, INN_net.parameters())))
optim1 = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim1, c.weight_step, gamma=c.gamma)
optim_init =optim1.state_dict()
dwt = common.DWT()
iwt = common.IWT()

################
# Data prepare:#
################
image_copyright = Image.open(args.copyrightpath)
image_copyright = to_rgb(image_copyright)
transform = transforms.Compose([
    transforms.Resize(c.imagesize, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(c.imagesize),
    transforms.ToTensor(),
])
image_copyright = transform(image_copyright)
image_copyright = image_copyright.unsqueeze(0)

########################
# attack model prepare:#
########################
vae = AutoencoderKL.from_pretrained(args.T2Imodel, subfolder="vae", torch_dtype=torch.float16)
vae.to("cuda")
def vae_encoder(image_tensor):
    """
    :input: image_tensor [1,3,512,512]
    :return: latent
    """
    latent = vae.encode(image_tensor.half()).latent_dist.sample() * 0.18215
    return latent


# =======================
# 遍历文件夹
for subfolder in os.listdir(args.inputpath_all):
    subfolder_path = os.path.join(args.inputpath_all, subfolder)
    inputpath = os.path.join(subfolder_path,"set_B")

    if os.path.isdir(subfolder_path):
        print(f"Processing folder: {subfolder}")

    # 在训练数据加载器中传入 inputpath
    trainloader = DataLoader(
        Dataset(inputpath=inputpath, transforms_=transform, mode="train"),  # 传入 inputpath
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        num_workers=args.workers,
        drop_last=True
    )

    # 读取网络
    if args.pretrain:
        load(c.pre_model, INN_net)
        optim1.load_state_dict(optim_init)
        print('!!!!!!!!!!!!!! Now running on pretrained model !!!!!!!!!!!!!!!')
    

    # try:
    totalTime = time.time()
    failnum = 0
    count = 0.0
    yeeee=[]
    similar_loss = F.mse_loss
    # similar_loss.to(device)
    vgg_loss = VGGLoss(3, 1, False)
    vgg_loss.to(device)

    tri_loss = TripletLoss(margin=1.0).to(device)
    ncc_loss = NCCLoss().to(device)


    for i_batch, mydata in enumerate(trainloader):

        # 切换回训练模式
        for param in INN_net.parameters():
            param.requires_grad = True  # 允许模型参数计算梯度
        INN_net.train()  # 将模型切换为训练模式
        
        # # 在每次新的图像开始时重新初始化模型
        init_model(INN_net)  # 重置权重
        INN_net = torch.nn.DataParallel(INN_net, device_ids=[0])
        optim1 = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
        weight_scheduler = torch.optim.lr_scheduler.StepLR(optim1, c.weight_step, gamma=c.gamma)

        start_time = time.time()

        # 读取数据
        image_cover = mydata[0].to(device)
        source_name = mydata[1]
        print(" ------------- Now source_name: ", source_name," - ", i_batch,' ---------------')
        # 读取版权图像
        image_copy =image_copyright.to(device)

        # 转换到频域空间
        cover_dwt = dwt(image_cover).to(device)  # channels = 12

        cover_dwt_low = cover_dwt.narrow(1, 0, c.channels_in).to(device)  # channels = 3
        copy_dwt = dwt(image_copy).to(device)  # channels =12

        copy_dwt_low = copy_dwt.narrow(1, 0, c.channels_in).to(device)  # channels =3
        # 拼接
        input_dwt = torch.cat((cover_dwt, copy_dwt), 1).to(device)  # channels = 12*2

        for i_epoch in range(c.epochs):
            #################
            #    train:   #
            #################
            output_dwt = INN_net(input_dwt).to(device)  # channels = 24
            output_adv_dwt = output_dwt.narrow(1, 0, 4 * c.channels_in).to(device)  # channels = 12
            output_adv_dwt_low = output_adv_dwt.narrow(1, 0, c.channels_in ).to(device)  # channels = 3
            output_r_dwt = output_dwt.narrow(1, 4 * c.channels_in, 4 * c.channels_in).to(device)

            # 转换到像素空间
            output_adv = iwt(output_adv_dwt).to(device)  # channels = 3
            output_r = iwt(output_r_dwt).to(device)

            #################
            #     loss:     #
            #################
            # loss1:视觉损失-对抗样本与干净样本的 相似性损失
            perc_loss_1 = similar_loss(output_adv, image_cover).to(device)

            # loss2：视觉损失-对抗样本与干净样本的VGG-16中的高级特征 相似性损失
            vgg_on_cov = vgg_loss(image_cover).to(device)
            vgg_on_adv = vgg_loss(output_adv).to(device)
            perc_loss_2 = similar_loss(vgg_on_cov, vgg_on_adv).to(device)

            # loss3：视觉损失-对抗样本与干净样本的低频分量的 相似性损失
            perc_loss_3 = similar_loss(output_adv_dwt_low.cuda(), cover_dwt_low.cuda()).to(device)

            # loss4：对抗损失-对抗样本与版权图像的潜在特征 相似性损失
            latent_copy = vae_encoder(image_copy).to(device)
            latent_adv = vae_encoder(output_adv).to(device)
            adv_loss_1 = similar_loss(latent_copy, latent_adv).to(device)

            # 三元组损失
            margin_now =( ( i_epoch // ( c.epochs / c.margin_renew_times) ) +1 ) * ( c.margin_max /c.margin_renew_times)

            latent_cover = vae_encoder(image_cover).to(device)
            adv_loss_3 = tri_loss(anchor=latent_adv,positive=latent_copy,negative=latent_cover,margin=0.5)

            total_loss = c.lamda_perc * perc_loss_1 + c.lamda_perc_vgg * perc_loss_2 + c.lamda_perc_low * perc_loss_3 \
                + c.lamda_adv_latent_copy * adv_loss_1 \
                    + c.lamda_adv_tri * adv_loss_3 \

            #################
            #   Backward:   #
            #################
            optim1.zero_grad()
            total_loss.backward()
            optim1.step()

            weight_scheduler.step()
            lr_now = optim1.param_groups[0]['lr']
            if lr_now < c.lr_min:
                for param_group in optim1.param_groups:
                    param_group['lr'] = c.lr_min

            #################
            # 记录损失信息   #
            #################
            with open(log_file, "a", newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    i_epoch,
                    perc_loss_1.item(),
                    perc_loss_2.item(),
                    perc_loss_3.item(),
                    adv_loss_1.item(),
                    adv_loss_3.item(),
                    total_loss.item(),
                    optim1.param_groups[0]['lr']
                ])
            # 打印训练信息
            if i_epoch % 100 == 0:
                print(f"Epoch [{i_epoch}/{c.epochs}] | Loss1: {perc_loss_1.item():.4f} | "
                    f"Loss2: {perc_loss_2.item():.4f} | Loss3: {perc_loss_3.item():.4f} | "
                    f"Loss4: {adv_loss_1.item():.4f} | "
                    f"Loss_tri: {adv_loss_3.item():.4f} | "
                    f"Total Loss: {total_loss.item():.4f} | lr:{optim1.param_groups[0]['lr']:.6f}"  )
                    
                print('margin_now:',margin_now)

        ##########################
        # save model for recover #
        ##########################
        MODEL_PATH = os.path.join(outputpath, "model", subfolder)
        os.makedirs(MODEL_PATH, exist_ok=True)
        torch.save({'opt': optim1.state_dict(), 'net': INN_net.state_dict()}, os.path.join(MODEL_PATH, f"{i_batch}_model.pt"))
        
        #######################################################
        # generate adv images for this group by trained model #
        #######################################################
        for param in INN_net.parameters():
            param.requires_grad = False
        INN_net.eval()
        with torch.no_grad():
            
            image_cover = mydata[0].to(device)
            source_name = mydata[1]
            image_copy =image_copyright.to(device)

            # prepare for path
            adv_dir = os.path.join(outputpath, "adv image", f"{source_name}")
            r_dir = os.path.join(outputpath, "r image", f"{source_name}")
            os.makedirs(adv_dir, exist_ok=True)
            os.makedirs(r_dir, exist_ok=True)         

            cover_dir = os.path.join(outputpath, "cover image", f"{source_name}")
            w_dir = os.path.join(outputpath, "watermark image", f"{source_name}")
            os.makedirs(cover_dir, exist_ok=True)
            os.makedirs(w_dir, exist_ok=True)   

            cover_dwt = dwt(image_cover).to(device)  # channels = 12
            copy_dwt = dwt(image_copy).to(device)  # channels =12

            input_dwt = torch.cat((cover_dwt, copy_dwt), 1).to(device)  # channels = 12*2

            output_dwt = INN_net(input_dwt).to(device)  # channels = 24
            output_adv_dwt = output_dwt.narrow(1, 0, 4 * c.channels_in).to(device)  # channels = 12
            output_r_dwt = output_dwt.narrow(1, 4 * c.channels_in, 4 * c.channels_in).to(device)

            # 转换到像素空间
            output_adv = iwt(output_adv_dwt).to(device)  # channels = 3
            output_r = iwt(output_r_dwt).to(device)

            save_image(output_adv, os.path.join(adv_dir, f"{i_batch}.png"))
            save_image(output_r, os.path.join(r_dir, f"{i_batch}.png"))

            save_image(image_cover, os.path.join(cover_dir, f"{i_batch}.png"))
            save_image(image_copy, os.path.join(w_dir, f"{i_batch}.png"))

            #################
            #   backward:   #
            #################
            # prepare for path
            cover_rev_dir= os.path.join(outputpath, "rev_ori image", f"{source_name}")
            copy_rev_dir = os.path.join(outputpath, "rev_copy image", f"{source_name}")
            os.makedirs(cover_rev_dir, exist_ok=True)
            os.makedirs(copy_rev_dir, exist_ok=True) 

            output_adv_dwt = dwt(output_adv).to(device)  # channels = 12

            output_rev = torch.cat((output_adv_dwt, output_r_dwt), 1)
            backward_img = INN_net(output_rev, rev=True).to(device) 
            cover_rev = backward_img.narrow(1, 0, 4 * c.channels_in)
            
            cover_rev = iwt(cover_rev)
            copy_rev = backward_img.narrow(1, 4 * c.channels_in, backward_img.shape[1] - 4 * c.channels_in).to(device)
            copy_rev = iwt(copy_rev)

            save_image(cover_rev, os.path.join(cover_rev_dir, f"{i_batch}.png"))
            save_image(copy_rev, os.path.join(copy_rev_dir, f"{i_batch}.png"))


    totalstop_time = time.time()
    time_cost = totalstop_time - totalTime
    print("Total cost time :" + str(time_cost))


        


    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     raise

    # finally:
    #     viz.signal_stop()


