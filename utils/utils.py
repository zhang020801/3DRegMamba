import numpy as np
import pystrum.pynd.ndutils as nd
import torch.nn.functional as F
import SimpleITK as sitk
from medpy import metric

def dice(array1, array2, labels):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.
    """
    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return dicem



def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.

    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims], 
              where vol_shape is of len nb_dims

    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else: # must be 2 
        
        dfdx = J[0]
        dfdy = J[1] 
        
        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]
      
      
def mnist_data_generator(x_data, batch_size=1):
    """
    generator that takes in data of size [N, H, W], and yields data for our vxm model

    Note that we need to provide numpy data for each input, and each output

    inputs:  moving_image [bs, H, W, 1], fixed_image [bs, H, W, 1]
    outputs: moved_image  [bs, H, W, 1], zeros [bs, H, W, 2]
    """
    # preliminary sizing
    vol_shape = x_data.shape[1:]  # extract data shape
    ndims = len(vol_shape)

    # prepare a zero array the size of the deformation. We'll explain this below.
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])

    while True:
        # prepare inputs
        # inputs need to be of the size [batch_size, H, W, number_features]
        #   number_features at input is 1 for us
        idx1 = np.random.randint(0, x_data.shape[0], size=1)
        moving_images = x_data[idx1]
        idx2 = np.random.randint(0, x_data.shape[0], size=1)
        fixed_images = x_data[idx2]
        inputs = [moving_images.squeeze(), fixed_images.squeeze()]

        # outputs
        # we need to prepare the "true" moved image.
        # Of course, we don't have this, but we know we want to compare
        # the resulting moved image with the fixed image.
        # we also wish to penalize the deformation field.
        outputs = [fixed_images, zero_phi]

        yield inputs, outputs


def align_img(grid, x):
    return F.grid_sample(
        x, grid=grid, mode="bilinear", padding_mode="border", align_corners=False
    )

    


def split_seg_global(seg, labels, downsize=1):
    full_classes = int(np.max(labels) + 1)
    valid_mask = np.isin(np.arange(full_classes), labels)
    shape = seg.shape[:-1]
    one_hot_seg = np.eye(full_classes)[seg.reshape(-1).astype(int)].reshape(*shape, -1)
    return one_hot_seg[:, ::downsize, ::downsize, ::downsize, valid_mask]


def minmax_norm(x, axis=None):
    """
    Min-max normalize array using a safe division.

    Arguments:
        x: Array to be normalized.
        axis: Dimensions to reduce during normalization. If None, all axes will be considered,
            treating the input as a single image. To normalize batches or features independently,
            exclude the respective dimensions.

    Returns:
        Normalized array.
    """
    x_min = np.min(x, axis=axis, keepdims=True)
    x_max = np.max(x, axis=axis, keepdims=True)
    return np.divide(x - x_min, x_max - x_min, out=np.zeros_like(x - x_min), where=x_max != x_min)    


def compute_hd95(gt_np, pred_np, spacing):

    gt_img = sitk.GetImageFromArray((gt_np>0).astype(np.uint8))
    gt_img.SetSpacing(spacing)
    pred_img = sitk.GetImageFromArray((pred_np>0).astype(np.uint8))
    pred_img.SetSpacing(spacing)


    gt_cont = sitk.LabelContour(gt_img)
    pred_cont = sitk.LabelContour(pred_img)


    dm_gt = sitk.Abs(
        sitk.SignedMaurerDistanceMap(gt_cont, squaredDistance=False, useImageSpacing=True)
    )
    dm_pred = sitk.Abs(
        sitk.SignedMaurerDistanceMap(pred_cont, squaredDistance=False, useImageSpacing=True)
    )

    dm_gt_arr   = sitk.GetArrayFromImage(dm_gt)
    dm_pred_arr = sitk.GetArrayFromImage(dm_pred)
    gt_cont_arr   = sitk.GetArrayFromImage(gt_cont)
    pred_cont_arr = sitk.GetArrayFromImage(pred_cont)

    d_pred_to_gt = dm_gt_arr[pred_cont_arr==1]
    d_gt_to_pred = dm_pred_arr[gt_cont_arr==1]
    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])

    if all_d.size == 0:
        return np.nan
    return np.percentile(all_d, 95)

def compute_jacobian_determinant(flow_np, spacing):
   
    ux, uy, uz = flow_np
    sx, sy, sz = spacing  # spacing = (x, y, z)
   
    grad_zx, grad_yx, grad_xx = np.gradient(ux, sz, sy, sx)
    grad_z_y, grad_y_y, grad_x_y = np.gradient(uy, sz, sy, sx)
    grad_z_z, grad_y_z, grad_x_z = np.gradient(uz, sz, sy, sx)

   
    a = 1 + grad_xx; b = grad_yx; c = grad_zx
    d = grad_x_y;     e = 1 + grad_y_y; f = grad_z_y
    g = grad_x_z;     h = grad_y_z;     i = 1 + grad_z_z

   
    det = ( a*(e*i - f*h)
           - b*(d*i - f*g)
           + c*(d*h - e*g) )
    return det

def OASIS_metric_val_VOI(y_pred, y_true):
    VOI_lbls =[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]
    Lists_dsc = np.zeros((len(VOI_lbls), 1))
    Lists_hd = np.zeros((len(VOI_lbls), 1))
    Lists_asd = np.zeros((len(VOI_lbls), 1))
    idx = 0
    for i in VOI_lbls:
        pred_i = pred == i
        true_i = true == i

        dsc = metric.binary.dc(pred_i, true_i)
        hd = metric.binary.hd95(pred_i, true_i)
        asd = metric.binary.asd(pred_i, true_i)
        Lists_dsc[idx] = dsc
        Lists_hd[idx] = hd
        Lists_asd[idx] = asd
        idx += 1
    return np.mean(Lists_dsc), np.mean(Lists_hd), np.mean(Lists_asd)

def LPBA40_metric_val_VOI(y_pred, y_true):
    VOI_lbls = [21,22,23,24,25,26,27,28,29,30,
               31,32,33,34,41,42,43,44,45,46,
               47,48,49,50,61,62,63,64,65,66,
               67,68,81,82,83,84,85,86,87,88,
               89,90,91,92,101,102,121,122,161,
               162,163,164,165,166]
    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]
    Lists_dsc = np.zeros((len(VOI_lbls), 1))
    Lists_hd = np.zeros((len(VOI_lbls), 1))
    Lists_asd = np.zeros((len(VOI_lbls), 1))
    idx = 0
    for i in VOI_lbls:
        pred_i = pred == i
        true_i = true == i

        dsc = metric.binary.dc(pred_i, true_i)
        hd = metric.binary.hd95(pred_i, true_i)
        asd = metric.binary.asd(pred_i, true_i)
        Lists_dsc[idx] = dsc
        Lists_hd[idx] = hd
        Lists_asd[idx] = asd
        idx += 1
    return np.mean(Lists_dsc), np.mean(Lists_hd), np.mean(Lists_asd)

def IXI_metric_val_VOI(y_pred, y_true):
    VOI_lbls = [16, 49, 10, 8, 47, 2, 41, 7, 46, 12, 51, 28, 60, 13, 52, 11, 50, 4, 43, 17, 53, 14, 15, 18, 54, 3, 42, 24, 31, 63]
    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]
    Lists_dsc = np.zeros((len(VOI_lbls), 1))
    Lists_hd = np.zeros((len(VOI_lbls), 1))
    Lists_asd = np.zeros((len(VOI_lbls), 1))
    idx = 0
    for i in VOI_lbls:
        pred_i = pred == i
        true_i = true == i

        dsc = metric.binary.dc(pred_i, true_i)
        hd = metric.binary.hd95(pred_i, true_i)
        asd = metric.binary.asd(pred_i, true_i)
        Lists_dsc[idx] = dsc
        Lists_hd[idx] = hd
        Lists_asd[idx] = asd
        idx += 1
    return np.mean(Lists_dsc), np.mean(Lists_hd), np.mean(Lists_asd)