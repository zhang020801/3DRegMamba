# python imports
import os
import glob
import warnings
import datetime       
# external imports
import torch
import numpy as np
import SimpleITK as sitk
from torch.optim import Adam
import torch.utils.data as Data
# internal imports
from utils import losses
from utils.config import args
from utils.datagenerators_atlas import Dataset
from Model.STN import SpatialTransformer
from natsort import natsorted

from Model.RegMambaV2 import RegMamba
from tqdm import tqdm

warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument."
)
warnings.filterwarnings(
    "ignore",
    message="Default grid_sample and affine_grid behavior has changed to align_corners=False since"
)

def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def make_dirs():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.result_dir, name))


def compute_label_dice(gt, pred):
    cls_lst = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
            63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
            163, 164, 165, 166]
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return np.mean(dice_lst)


def train():
    make_dirs()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{now}_epochs{args.epochs}_lr{args.lr}.txt"
    log_path = os.path.join(args.log_dir + '/LPBA40/', log_name)
    print("Logging to:", log_path)
    f = open(log_path, "w")


    f_img = sitk.ReadImage(args.atlas_file)
    input_fixed = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]
    vol_size = input_fixed.shape[2:]
    # [B, C, D, W, H]
    input_fixed_eval = torch.from_numpy(input_fixed).to(device).float()
    input_fixed = np.repeat(input_fixed, args.batch_size, axis=0)
    input_fixed = torch.from_numpy(input_fixed).to(device).float()
    fixed_label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.label_dir, "S01.delineation.structure.label.nii.gz")))[np.newaxis, np.newaxis, ...]
    fixed_label = torch.from_numpy(fixed_label).to(device).float()


    net = RegMamba().to(device)
    total_params = count_parameters(net)
    total_MB = total_params * 4 / (1024 ** 2)  
    print(f"Total parameters: {total_params:,}  ({total_MB:.2f} MB)", file=f)
    print(f"Total parameters: {total_params:,}  ({total_MB:.2f} MB)")

    iterEpoch = 1
    contTrain = True
    if contTrain:
        checkpoint = torch.load('./Checkpoint/LPBA40/V2/dsc0.7146epoch021.pth.tar')
        net.load_state_dict(checkpoint['state_dict'])
        iterEpoch = 26

    STN = SpatialTransformer(vol_size).to(device)
    STN_label = SpatialTransformer(vol_size, mode="nearest").to(device)

    net.train()
    STN.train()

    opt = Adam(net.parameters(), lr=args.lr, weight_decay=0, amsgrad=True)
    sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss


    train_files = glob.glob(os.path.join(args.train_dir, '*.nii.gz'))
    DS = Dataset(files=train_files)
    print("Number of training images: ", len(DS))
    DL = Data.DataLoader(DS, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)


    for epoch in tqdm(range(iterEpoch, args.epochs + 1), desc="Epochs"):
        net.train()
        STN.train()
        for input_moving, fig_name in tqdm(DL, desc=f"  Epoch {epoch}", leave=False):
            # [B, C, D, W, H]
            fig_name = fig_name[0]
            input_moving = input_moving.to(device).float()

            flow_m2f = net(input_fixed, input_moving)
            m2f = STN(input_fixed, flow_m2f)

            sim_loss = sim_loss_fn(m2f, input_moving)
            grad_loss = grad_loss_fn(flow_m2f)
            # zero_loss = zero_loss_fn(flow_m2f, zero)
            loss = sim_loss + args.alpha * grad_loss #  + zero_loss

            opt.zero_grad()
            loss.backward()
            opt.step()


            flow_m2f = net(input_moving, input_fixed)
            m2f = STN(input_moving, flow_m2f)

            sim_loss = sim_loss_fn(m2f, input_fixed)
            grad_loss = grad_loss_fn(flow_m2f)
            loss = sim_loss + args.alpha * grad_loss #  + zero_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

        test_file_lst = glob.glob(os.path.join(args.test_dir, "*.nii.gz"))

        net.eval()
        STN.eval()
        STN_label.eval()

        DSC = []
        for file in test_file_lst:
            fig_name = file[58:60]
            name = os.path.split(file)[1]

            input_moving = sitk.GetArrayFromImage(sitk.ReadImage(file))[np.newaxis, np.newaxis, ...]
            input_moving = torch.from_numpy(input_moving).to(device).float()

            label_file = glob.glob(os.path.join(args.label_dir, name[:3] + "*"))[0]
            input_label = sitk.GetArrayFromImage(sitk.ReadImage(label_file))

            pred_flow = net(input_fixed_eval, input_moving)
            pred_img = STN(input_fixed_eval, pred_flow)
            pred_label = STN_label(fixed_label, pred_flow)



            dice = compute_label_dice(input_label, pred_label[0, 0, ...].cpu().detach().numpy())
            print("{0}" .format(dice))
            print(f"{epoch},{name},dice,{dice:.6f}", file=f)  
            DSC.append(dice)

            del pred_flow, pred_img, pred_label, input_moving

        print("mean(DSC):", np.mean(DSC))
        print("std(DSC):", np.std(DSC))
        print(f"Epoch {epoch} Test mean(DSC): {np.mean(DSC):.6f}", file=f)
        print(f"Epoch {epoch} Test std(DSC):  {np.std(DSC):.6f}", file=f)
        save_checkpoint({
            'epoch': epoch+1,
            'state_dict': net.state_dict(),
            'optimizer': opt.state_dict(),
            }, save_dir='Checkpoint/LPBA40/V2/', filename='dsc{:.4f}epoch{:0>3d}.pth.tar'.format(np.mean(DSC), epoch+1))

    f.close()

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=4):
    model_lists = natsorted(glob.glob(save_dir+  '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(state, save_dir+filename)

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()