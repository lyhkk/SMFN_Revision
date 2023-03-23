import os
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from Network.modules import VRCNN, dual_network
import argparse
from data.data_utils import TrainsetLoader, ValidationsetLoader, ycbcr2rgb, mse_weight_loss
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from train_log.model_log import print_network
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torchvision.transforms import ToPILImage, ToTensor
from metrics.psnr import psnr, calculate_ssim
from metrics.ws_ssim import ws_ssim
from metrics.psnr import ws_psnr, ws_psnr2, psnr, calculate_ssim
import random
from torch.optim.lr_scheduler import StepLR



# Training settings 建立解析对象
parser = argparse.ArgumentParser(description='PyTorch Super Res Training')
# 为parser增加属性
parser.add_argument('--num_epochs', type=int, default=32, help='epochs num')
parser.add_argument('--learning_rate', type=float, default=4e-6, help='Initial learning rate')
parser.add_argument('--patience', type=int, default=20, help='For early stopping')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--train_batchSize', type=int, default=16, help='Training batch size')
parser.add_argument('--val_batchSize', type=int, default=4, help='validation batch size')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123') 
parser.add_argument('--gpus', default=4, type=int, help='number of gpu')
parser.add_argument('--cuda', action='store_true', help='use cuda?', default=True)
parser.add_argument('--tr_dataset_hr', type=str, default='data1/VR-super-resolution/train/VR/GT') 
parser.add_argument('--tr_dataset_lr', type=str, default='data1/VR-super-resolution/train/VR/LR') 
parser.add_argument('--model_save_folder', default='model/final_model/my_final-726dual', help='Location to save checkpoint models')
parser.add_argument('--save_folder', default='model/final_model/my-726dual', help='Location to save not checkpoint models')
parser.add_argument('--train_log', type=str, default='train_log')
parser.add_argument('--lambda_L', type=float, default=0.1)
parser.add_argument('--exp_name', type=str, default='726xr-dual')
parser.add_argument('--test_model', type=str, default='vrcnn_final_epoch_272.pth', help='lr change flag')
parser.add_argument('--val_dataset_lr', type=str, default='data1/VR-super-resolution/val/VR/LR')
parser.add_argument('--val_dataset_hr', type=str, default='data1/VR-super-resolution/val/VR/GT')
parser.add_argument('--save_freq', type=int, default=2, help='Saving models after ~ epochs')

args = parser.parse_args()
gpus_list = range(0, args.gpus)

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 保存模型的位置
model_name = os.path.join(args.model_save_folder, args.test_model)

# gpu的编号
gpus_list = range(0, args.gpus)
# device_ids = [Id for Id in range(torch.cuda.device_count())]
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载训练数据集
tr_set = TrainsetLoader(args.tr_dataset_hr, args.tr_dataset_lr, args.upscale_factor, patch_size=32, n_iters=2000) 
val_set = ValidationsetLoader(args.val_dataset_hr, args.val_dataset_lr)

# 创建数据加载器
print('===> Loading datasets')
train_loader = DataLoader(tr_set, batch_size=args.train_batchSize, shuffle=False, num_workers=4)
valLoader = DataLoader(dataset=val_set, batch_size=args.val_batchSize, shuffle=False, num_workers=4)


class Net(nn.Module):
    def __init__(self, vrcnn, dual_net):
        super(Net, self).__init__()
        self.vrcnn = vrcnn
        self.dual_net = dual_net
        
    def forward(self, x):
        output = self.vrcnn(x)
        dual_out = self.dual_net(output)
        return output, dual_out

    
# 模型
print('===> Building model')
vrcnn = VRCNN(args.upscale_factor, is_training=True)
dual_net = dual_network()
model = Net(vrcnn, dual_net)

if args.cuda:
    model = torch.nn.DataParallel(model, device_ids=gpus_list, output_device=gpus_list[1])
    model = model.cuda(gpus_list[0])
print_network(model)

# 定义损失函数和优化器
criterion = mse_weight_loss()
criterion = criterion.cuda(gpus_list[0])
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
# 设置20epochs后学习率减半
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

# 创建SummaryWriter实例
writer = SummaryWriter(log_dir='./train_log')


def train():
    train_bar = tqdm(train_loader)
    
    # 训练循环
    best_psnr = 0.0
    best_ssim = 0.0
    # 迭代的初始下标为1
    # for iteration, batch in enumerate(train_loader, 1):
    for epoch in range(args.num_epochs):
        
        psnr_loss = 0.0
        running_loss = 0.0
        patience_counter = 0
        iteration = 1
        
        for data in train_bar:
            # model.train()启动batch normalization和Dropout
            model.train()
            
            # batch[0]与[1]的形式都应是tensor(batch_size,img_size)
            # input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
            batch_lr_y, label, hr_texture, lr_texture = data
            batch_lr_y, label = Variable(batch_lr_y).cuda(gpus_list[0]), Variable(label).cuda(gpus_list[0])
            # in_tr, in_val, tar_tr, tar_val = train_test_split(input, target, test_size=0.2, random_state=42)
            optimizer.zero_grad()
            
            output, out_dual = model(batch_lr_y)
            # 计算损失
            out = output.squeeze(dim=2)
            print(output.size())
            primary_loss = criterion(out, label)
            dual_loss = args.lambda_L * criterion(out_dual, batch_lr_y)
            loss = primary_loss + dual_loss
            
            writer.add_scalar('Train/LOSS', loss.item(), epoch*len(train_loader) + iteration)
            
            # psnr_value
            psnr_value = psnr(np.array(torch.squeeze(label).data.cpu())*255, np.array(torch.squeeze(output).data.cpu())*255)
            writer.add_scalar('Train/PSNR', psnr_value, epoch*len(train_loader) + iteration)
            psnr_loss += psnr_value
            '''
            ssim_value = calculate_ssim(np.array(torch.squeeze(label).data.cpu())*255, np.array(torch.squeeze(output).data.cpu())*255)
            writer.add_scalar('Train/SSIM', ssim_value, epoch*len(train_loader) + iteration)
            ssim_loss += ssim_value
            '''
            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 更新损失
            running_loss += loss.item()
            iteration += 1

        # 计算平均损失
        epoch_loss = running_loss / len(train_loader)
        psnr_loss /= len(train_loader)
        # ssim_loss /= len(train_loader)
        writer.close()
        # 更新学习率
        scheduler.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || PSNR_Value: {:.4f}".format(epoch, epoch+1, len(train_loader), epoch_loss, psnr_loss))
        
        # Validation
        print('===> Start Validation')
        model.eval()
        val_psnr, val_ssim = validate(valLoader, model)
        print('Epoch {}/{} PSNR: {:.6f} SSIM: {:.6f}'.format(epoch, args.num_epochs, val_psnr, val_ssim))
        writer.add_scalar('Validation/PSNR', val_psnr, epoch)
        writer.add_scalar('Validation/SSIM', val_ssim, epoch)

        # 保存最好的模型
        if val_psnr > best_psnr and val_ssim > best_ssim:
            best_psnr = val_psnr
            torch.save(model.state_dict(), model_name)
            print('===> Saving Model')
            patience_counter = 0
        else:
            if (epoch + 1) % args.save_freq == 0:
                save_model(model, epoch + 1)
            patience_counter += 1

        # 提前停止
        writer.close()
        if patience_counter >= args.patience:
            print("Early stopping")
            break


def validate(valLoader, model):
    with torch.no_grad():
        ave_psnr = 0
        ave_ssim = 0
        # numb = 2
        val_bar = tqdm(valLoader)
        for data in val_bar:
            model.eval()
            # dual_net.eval()
            batch_lr_y, label, SR_cb, SR_cr, idx, bicubic_restore = data
            batch_lr_y, label = Variable(batch_lr_y).cuda(gpus_list[0]), Variable(label).cuda(gpus_list[0])
            output, out_dual = model(batch_lr_y)
            
            SR_ycbcr = np.concatenate((np.array(output.squeeze(0).data.cpu()), SR_cb, SR_cr), axis=0).transpose(1, 2, 0)
            SR_rgb = ycbcr2rgb(SR_ycbcr) * 255.0
            SR_rgb = np.clip(SR_rgb, 0, 255)
            SR_rgb = ToPILImage()(SR_rgb.astype(np.uint8))
            #ToTensor() ---image(0-255)==>image(0-1), (H,W,C)==>(C,H,W)
            SR_rgb = ToTensor()(SR_rgb)

            psnr_value = ws_psnr(np.array(torch.squeeze(label).data.cpu())*255, np.array(torch.squeeze(output).data.cpu())*255)
            ave_psnr = ave_psnr + psnr_value
            ssim_value = ws_ssim(np.array(torch.squeeze(label).data.cpu())*255, np.array(torch.squeeze(output).data.cpu())*255)
            ave_ssim = ave_ssim + ssim_value
    return ave_psnr / len(valLoader), ave_ssim / len(valLoader)


def save_model(model, epoch):
    # 构造保存路径
    save_path = os.path.join(args.save_folder, 'model-{}.ckpt'.format(epoch))
    # 保存模型
    torch.save(model.state_dict(), save_path)
    print('Model saved to {}'.format(save_path))

    
if __name__ == '__main__':
    print('===> Start Training')
    train()
    print('===> Training Finished')
    writer.close()