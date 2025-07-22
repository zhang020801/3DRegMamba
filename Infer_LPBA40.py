import os
import glob
import warnings
import csv
import torch
import numpy as np
import SimpleITK as sitk
import time

from utils import losses
from utils.config import args
from Model.STN import SpatialTransformer
from Model.RegMambaV2 import RegMamba

from utils.utils import jacobian_determinant, LPBA40_metric_val_VOI
from medpy import metric   


warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument."
)
warnings.filterwarnings(
    "ignore",
    message="Default grid_sample and affine_grid behavior has changed to align_corners=False since"
)

def make_dirs():
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

def save_nifti(arr_np, ref_img, save_path, is_vector=False):
    
    if is_vector:
        arr_np = arr_np.transpose(1, 2, 3, 0)
    img = sitk.GetImageFromArray(arr_np, isVector=is_vector)
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, save_path)

def compute_label_dice(gt, pred):
    cls_lst = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
            63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
            163, 164, 165, 166]
    # cls_lst = [182]
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return np.mean(dice_lst)


def train():
    make_dirs()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    fixed_img = sitk.ReadImage(args.atlas_file)
    input_fixed = sitk.GetArrayFromImage(fixed_img)[np.newaxis, np.newaxis, ...]
    vol_size = input_fixed.shape[2:]
    # [B, C, D, W, H]
    input_fixed = torch.from_numpy(input_fixed).to(device).float()
    fixed_label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.label_dir, "S01.delineation.structure.label.nii.gz")))[np.newaxis, np.newaxis, ...]
    fixed_label = torch.from_numpy(fixed_label).to(device).float()


    net = RegMamba().to(device)
    ckpt = torch.load('./Checkpoint/LPBA40/V2/final.pth.tar')
    net.load_state_dict(ckpt['state_dict'])
    stn_img   = SpatialTransformer(vol_size).to(device)
    stn_label = SpatialTransformer(vol_size, mode='nearest').to(device)
    net.eval(); stn_img.eval(); stn_label.eval()

    test_list = glob.glob(os.path.join(args.test_dir, '*.nii.gz'))


    VOI_lbls = [21,22,23,24,25,26,27,28,29,30,
                31,32,33,34,41,42,43,44,45,46,
                47,48,49,50,61,62,63,64,65,66,
                67,68,81,82,83,84,85,86,87,88,
                89,90,91,92,101,102,121,122,161,
                162,163,164,165,166]

    names, dscs, neg_jac_ratios, hd95s, asd_avgs, times = [], [], [], [], [], []

    all_per_label = []

    

    with torch.no_grad():
        for file in test_list:
            name = os.path.basename(file)
            print(name)
            base = name[:3] if name.endswith('.nii.gz') else os.path.splitext(name)[0]
            names.append(name)


            moving_img = sitk.GetArrayFromImage(sitk.ReadImage(file))[np.newaxis,np.newaxis,...]
            input_moving = torch.from_numpy(moving_img).to(device).float()
            moving_label = glob.glob(os.path.join(args.label_dir, name[:3] + '*'))[0]
            moving_label  = sitk.GetArrayFromImage(sitk.ReadImage(moving_label))[np.newaxis,np.newaxis,...]
            moving_label = torch.from_numpy(moving_label).to(device).float()
            

            t0 = time.time()
            flow = net(input_moving, input_fixed)               # [1,3,D,W,H]
            warped_img   = stn_img(input_moving, flow)          # [1,1,D,W,H]
            elapsed = time.time() - t0
            pre_label = stn_label(moving_label, flow)           # [1,1,D,W,H]
            times.append(elapsed)


            flow_np       = flow[0].cpu().numpy()               # (3,D,W,H)
            img_np        = warped_img[0,0].cpu().numpy()       # (D,W,H)
            pred_lbl_np   = pre_label[0,0].cpu().numpy().astype(np.int32)
            gt_lbl_np     = fixed_label[0,0].cpu().numpy().astype(np.int32)



            save_nifti(flow_np,      fixed_img,
                       os.path.join('./Result/LPBA40/V2/warpimg/', f"{base}_flow.nii.gz"),
                       is_vector=True)
            save_nifti(img_np,       fixed_img,
                       os.path.join('./Result/LPBA40/V2/warpimg/', f"{base}_warped_image.nii.gz"),
                       is_vector=False)
            save_nifti(pred_lbl_np,  fixed_img,
                       os.path.join('./Result/LPBA40/V2/warpimg/', f"{base}_warped_label.nii.gz"),
                       is_vector=False)



            flow_np = np.moveaxis(flow_np, 0, -1)
            neg_jac = jacobian_determinant(flow_np)
            neg_jac = np.mean(neg_jac < 0)
            neg_jac_ratios.append(neg_jac)
            print("neg_jac_ratios:",neg_jac)


            dsc_avg, hd95_avg, asd_avg = LPBA40_metric_val_VOI(
                pre_label.long(), fixed_label.long()
            )


            dscs.append(dsc_avg)
            hd95s.append(hd95_avg)
            asd_avgs.append(asd_avg)
            neg_jac_ratios.append(neg_jac)
            times.append(elapsed)
            

            print(f"{name}: DSC={dsc_avg:.4f}, negJacRatio={neg_jac:.4f}, "f"HD95={hd95_avg:.2f}, ASD={asd_avg:.2f}, time={elapsed:.3f}s")

            for lbl in VOI_lbls:
                p_mask = (pred_lbl_np == lbl)
                t_mask = (gt_lbl_np == lbl)
                if p_mask.sum() + t_mask.sum() == 0:
                    dsc_val = 1.0
                else:
                    dsc_val = float(metric.binary.dc(p_mask, t_mask))
                all_per_label.append({
                    'case': name,
                    'label': lbl,
                    'dsc': dsc_val
                })


    summary = {
        'DSC_mean': np.nanmean(dscs),  'DSC_std': np.nanstd(dscs),
        'negJac_mean': np.nanmean(neg_jac_ratios), 'negJac_std': np.nanstd(neg_jac_ratios),
        'HD95_mean': np.nanmean(hd95s), 'HD95_std': np.nanstd(hd95s),
        'ASD_mean': np.nanmean(asd_avgs), 'ASD_std': np.nanstd(asd_avgs),
        'time_mean': np.nanmean(times), 'time_std': np.nanstd(times),
    }
    csv_path = os.path.join('./Result/LPBA40/V2/', 'evaluation_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(['case', 'DSC', 'negJacRatio', 'HD95', 'ASD', 'time_s'])

        for row in zip(names, dscs, neg_jac_ratios, hd95s, asd_avgs, times):
            writer.writerow(row)

        writer.writerow([])  
        writer.writerow(['MEAN',
                         summary['DSC_mean'],
                         summary['negJac_mean'],
                         summary['HD95_mean'],
                         summary['ASD_mean'],
                         summary['time_mean']])
        writer.writerow(['STD',
                         summary['DSC_std'],
                         summary['negJac_std'],
                         summary['HD95_std'],
                         summary['ASD_std'],
                         summary['time_std']])

    print(f'All indicators have been saved to {csv_path}')

   
    per_label_csv = os.path.join('./Result/LPBA40/V2/', 'per_label_dsc.csv')
    with open(per_label_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['case', 'label', 'dsc'])
        for rec in all_per_label:
            writer.writerow([rec['case'], rec['label'], rec['dsc']])
    print(f'The DSC for each tag has been saved to {per_label_csv}')

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()
