import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        filters = [3, 3, 5, 3, 3, 5, 3, 3, 3]
        strides = [1, 1, 2, 1, 1, 2, 1, 1, 1]
        paddings = [i // 2 for i in filters]
        in_channels = [3, 8, 8, 16, 16, 16, 32, 32, 32, 32]
        self.layers = []
        for i in range(len(filters)):
            self.layers.append(nn.Conv2d(in_channels[i], in_channels[i + 1], kernel_size=filters[i], stride=strides[i],
                                         padding=paddings[i]))
            if i < len(filters) - 1:
                self.layers.append(nn.BatchNorm2d(in_channels[i + 1]))
                self.layers.append(nn.ReLU())
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        # x: [B,3,H,W]
        for i, l in enumerate(self.layers):
            x = l.forward(x)
        return x


class SimlarityRegNet(nn.Module):
    def __init__(self, G):
        super(SimlarityRegNet, self).__init__()
        filters = [3, 3, 3, 3, 3, 3]
        strides = [1, 2, 2, 2, 2, 1]
        paddings = [i // 2 for i in filters]
        in_channels = [G, 8, 16, 32, 16, 8, 1]
        self.layers = []
        for i in range(len(filters)):
            if i < 3:
                self.layers.append(
                    nn.Conv2d(in_channels[i], in_channels[i + 1], kernel_size=filters[i], stride=strides[i],
                              padding=paddings[i]))
                self.layers.append(nn.ReLU())
            elif i < len(filters) - 1:
                self.layers.append(
                    nn.ConvTranspose2d(in_channels[i], in_channels[i + 1], kernel_size=filters[i], stride=strides[i],
                                       padding=paddings[i]))
            else:
                self.layers.append(
                    nn.Conv2d(in_channels[i], in_channels[i + 1], kernel_size=filters[i], stride=strides[i],
                              padding=paddings[i]))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        # x: [B,G,D,H,W]
        # out: [B,D,H,W]
        B, G, D, H, W = x.shape

        s = x.transpose(1,2).reshape((B * D, G, H, W))
        # s: [B*D,G,H,W]
        # Conv2d+ReLU
        c0 = self.layers[1](self.layers[0](s))
        # Conv2d+ReLU
        c1 = self.layers[3](self.layers[2].forward(c0))
        # Conv2d+ReLU
        c2 = self.layers[5](self.layers[4].forward(c1))
        c3 = self.layers[6].forward(c2, output_size=c1.size())
        c4 = self.layers[7].forward(c3 + c1, output_size=c0.size())
        y = self.layers[8].forward(c4 + c0)
        # y: [B,D,1,W/4,H/4]
        return y.view(B, D, H, W)


def warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, D]
    # out: [B, C, D, H, W]
    B,C,H,W = src_fea.size()
    D = depth_values.size(1)
    # compute the warped positions with depth values
    with torch.no_grad():
        # relative transformation from reference to source view
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]
        # return [[0,..,0],[1,...1],[2,..,2],...,[H-1,...H-1]] each had h elems for y
        # It's similar with W for x
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, W, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous() #Adapt memory
        y, x = y.view(H * W), x.view(H * W) #[17168] for both
        xyz = torch.stack((x,y,torch.ones((len(x)), device=x.device))) # dim: Cx(H*W)
        NUMB_DIM_XYZ=3
        xyz_batched = xyz.unsqueeze(0).expand(B,NUMB_DIM_XYZ,H*W)
        #Do the rotation
        xyz_rotated = torch.matmul(rot.float(),xyz_batched)

        #Add the depth
        # we have: (B,number_dim_xyz,H*W)
        # We want (N, D_out, H_out, W_out, 3) at the end
        #We need to add the depth at axis 2 (before H*W)
        xyz_rotated = xyz_rotated.unsqueeze(2).expand(B,NUMB_DIM_XYZ,D,H*W)
        #xyz_rotates now: [B,3,D,H*W]
        dept_values_view_mult = depth_values.view(B,1,D,1).expand(B,NUMB_DIM_XYZ,D,H*W)
        xyz_rot_with_depth = dept_values_view_mult * xyz_rotated #[B,3,D,H*W]

        #Add the translation
        xyz_source =  xyz_rot_with_depth + trans.unsqueeze(-1).expand(B,NUMB_DIM_XYZ,1,H*W)
        xyz_source = xyz_source.float()
        #We have to adapt the indices to the limits when the result z < 0
        negative_depth_mask = xyz_source[:, 2] < 0
        xyz_source[:, 0][negative_depth_mask] = W
        xyz_source[:, 1][negative_depth_mask] = H
        xyz_source[:, 2][negative_depth_mask] = 1
        #We put z axis to 1. it is important to do it after filtering the negative values and keep only the x and y
        xy_source = xyz_source[:, 0:2, :, :] / xyz_source[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        #To use grid_sample, we need to normalize to -1 -> +1
        #shape:  xyz_source: [B, 2, D,H*W]
        x_source = xy_source[:,0,:,:]
        y_source = xy_source[:,1,:,:]
        xy_min, x_max, y_max = 0, W-1, H-1
        x_source_norm = x_source - xy_min
        x_source_norm /= (x_max/2)
        x_source_norm -= 1

        y_source_norm = y_source - xy_min
        y_source_norm /= (y_max/2)
        y_source_norm -= 1
        xy_source[:,0,:,:] = x_source_norm
        xy_source[:,1,:,:] = y_source_norm

        #final reshape. shape before [B,2,D,H*W]
        xy_source = xy_source.permute((0,2,3,1)).view(B,D*H,W,2) # now [Bat,Dep*Hei,Wid,2]
        #We want the shape: (N,H,W,2)=[Batch_size,Dept,Hei*Wid,2] Okay!

    # get warped_src_fea with bilinear interpolation (use 'grid_sample' function from pytorch)

    #We want for input: (N,C,H,W) = [Batch_size,C,Depth,Hei*Wid]
    warped_src_fea = F.grid_sample(src_fea, xy_source, mode='bilinear', align_corners=True)
    # Shape output: [B, C, Dept, H*W]

    return warped_src_fea.view(B,C,D,H,W)

def group_wise_correlation(ref_fea, warped_src_fea, G):
    # ref_fea: [B,C,H,W]
    # warped_src_fea: [B,C,D,H,W]
    # out: [B,G,D,H,W]
    B, C, D, H, W = warped_src_fea.shape

    warped_grouped_tuple = warped_src_fea.view(B, G, C // G, D, H, W)
    res2 = (ref_fea.view(B, G, C // G, 1, H, W) * warped_grouped_tuple).mean(2)
    return res2


def depth_regression(p, depth_values):
    # p: probability volume [B, D, H, W]
    # depth_values: discrete depth values [B, D]
    B, D, H, W = p.shape
    prod = p * depth_values.view(B, D, 1, 1)
    return prod.sum(1).view(B, H, W)


def mvs_loss(depth_est, depth_gt, mask):
    # depth_est: [B,1,H,W]
    # depth_gt: [B,1,H,W]
    # mask: [B,1,H,W]
    return F.l1_loss(depth_est[mask > 0.5], depth_gt[mask > 0.5], reduction='mean')
