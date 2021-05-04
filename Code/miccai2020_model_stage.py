import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#from Functions import generate_grid_unit
from Functions import generate_grid

'''
TODOs:
   - add dice loss
   - add mutual information loss
   - add mse loss 
'''


# neural network module
class Miccai2020_LDR_laplacian_unit_disp_add_lvl1(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192, 144), range_flow=0.4):
        super(Miccai2020_LDR_laplacian_unit_disp_add_lvl1, self).__init__()
        self.in_channel    = in_channel
        self.n_classes     = n_classes
        self.start_channel = start_channel
        self.range_flow    = range_flow
        self.is_train      = is_train
        self.imgshape      = imgshape

        # genearte a grid for displacement field
        #self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = generate_grid(self.imgshape,1)
        #convert to tensor
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()

        #  this returns wrapped image at sample grid points:      flow = torch.nn.functional.grid_sample(x, sample_grid, mode=self.interpolator, padding_mode="border", align_corners=True)
        self.transform = SpatialTransform_unit().cuda()

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(self.in_channel, self.start_channel * 4, bias=bias_opt)
        ## the above returns a network layer or double layer, note start_channel * 4 = out_channel
        ## if batchnorm:
        ##     layer = nn.Sequential(
        ##         nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
        ##         nn.BatchNorm3d(out_channels), nn.ReLU())
        ## else:
        ##     layer = nn.Sequential(
        ##         nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
        ##         nn.LeakyReLU(0.2),
        ##         nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        ##     )


        self.down_conv = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1, bias=bias_opt)
        self.resblock_group_lvl1 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        # upsampling : increasing the size of the image from ? to ?
        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2, padding=0, output_padding=0, bias=bias_opt)
        # downsampling : decrease the size of the image by 1/stride  from input_size to (input_size / stride)
        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
        # downsampling : decrease the size of the image from ? to ?

        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
        # if batchnorm:
        #     layer = nn.Sequential(
        #         nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
        #         nn.BatchNorm3d(out_channels),
        #         nn.Tanh())
        # else:
        #     layer = nn.Sequential(
        #         nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
        #         nn.LeakyReLU(0.2),
        #         nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
        #         nn.Softsign())

    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.Sequential(
            PreActBlock(in_channels, in_channels, bias=bias_opt),       nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),       nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),       nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),       nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),       nn.LeakyReLU(0.2)
        )
        return layer


    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,  bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(  nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias), nn.BatchNorm3d(out_channels),  nn.ReLU()       )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels,  kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
            )
        return layer

    # def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
    #             output_padding=0, bias=True):
    #     layer = nn.Sequential(
    #         nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
    #                            padding=padding, output_padding=output_padding, bias=bias),
    #         nn.ReLU())
    #     return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

   # the neural network model starts after the initialisation
    def forward(self, x, y):
        #x and y is the model input tensors
        # x is the moving image and y is the fixed image
        cat_input      = torch.cat((x, y), 1)            # put them in one tensor
        cat_input      = self.down_avg(cat_input)        # decrease the size by half       = 1/2 of the image
        cat_input_lvl1 = self.down_avg(cat_input)        # decrease the size again by half = 1/4 of the image

        down_y = cat_input_lvl1[:, 1:2, :, :, :]         # get the downsampled fixed image (not a tensor)
        #TODO:  probably this is more clear: down_y = cat_input_lvl1[:, 1, :, :, :]

        fea_e0 = self.input_encoder_lvl1(cat_input_lvl1) # apply two lyers convolution on the 1/4 images
        e0 = self.down_conv(fea_e0)                      # downsampling
        e1 = self.resblock_group_lvl1(e0)                # R_Bolck
        e2 = self.up(e1)                                 # upsampling
        output_disp_e0_v = self.output_lvl1(torch.cat([e2, fea_e0], dim=1)) * self.range_flow             #  displacement field
        warpped_inputx_lvl1_out = self.transform(x, output_disp_e0_v.permute(0, 2, 3, 4, 1), self.grid_1) #  transformed moving image

        # it seems this is not important, we still can return all output in the case of testing as well
        f_output = output_disp_e0_v
        if self.is_train is True:
           f_output =  [output_disp_e0_v, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e2]

        return f_output


class Miccai2020_LDR_laplacian_unit_disp_add_lvl2(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192, 144), range_flow=0.4, model_lvl1=None):
        super(Miccai2020_LDR_laplacian_unit_disp_add_lvl2, self).__init__()
        self.in_channel    = in_channel
        self.n_classes     = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train   = is_train
        self.imgshape   = imgshape
        self.model_lvl1 = model_lvl1

        #self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = generate_grid(self.imgshape,1)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()
        self.transform = SpatialTransform_unit().cuda()
        bias_opt = False

        self.input_encoder_lvl1  = self.input_feature_extract(self.in_channel+3, self.start_channel * 4, bias=bias_opt)
        self.down_conv           = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1, bias=bias_opt)
        self.resblock_group_lvl1 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up_tri      = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        self.up          = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2, padding=0, output_padding=0, bias=bias_opt)
        self.down_avg    = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)

    def unfreeze_modellvl1(self):
        # unFreeze model_lvl1 weight
        print("\nunfreeze model_lvl1 parameter")
        for param in self.model_lvl1.parameters():
            param.requires_grad = True

    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.Sequential(
            PreActBlock(in_channels, in_channels, bias=bias_opt),     nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),     nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),     nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),     nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),     nn.LeakyReLU(0.2)
        )
        return layer

    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,  bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    # def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
    #             output_padding=0, bias=True):
    #     layer = nn.Sequential(
    #         nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
    #                            padding=padding, output_padding=output_padding, bias=bias),
    #         nn.ReLU())
    #     return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x, y):
        # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
        lvl1_disp, _, _, lvl1_v, lvl1_embedding = self.model_lvl1(x, y)
        lvl1_disp_up = self.up_tri(lvl1_disp)

        x_down = self.down_avg(x) # moving image
        y_down = self.down_avg(y)

        warpped_x = self.transform(x_down, lvl1_disp_up.permute(0, 2, 3, 4, 1), self.grid_1)
        cat_input_lvl2 = torch.cat((warpped_x, y_down, lvl1_disp_up), 1)

        fea_e0 = self.input_encoder_lvl1(cat_input_lvl2)
        e0 = self.down_conv(fea_e0)
        e0 = e0 + lvl1_embedding
        e0 = self.resblock_group_lvl1(e0)
        e0 = self.up(e0)
        output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        compose_field_e0_lvl1 = lvl1_disp_up + output_disp_e0_v
        warpped_inputx_lvl1_out = self.transform(x, compose_field_e0_lvl1.permute(0, 2, 3, 4, 1), self.grid_1)

        if self.is_train is True:
            return compose_field_e0_lvl1, warpped_inputx_lvl1_out, y_down, output_disp_e0_v, lvl1_v, e0
        else:
            return compose_field_e0_lvl1


class Miccai2020_LDR_laplacian_unit_disp_add_lvl3(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192, 144), range_flow=0.4,
                 model_lvl2=None):
        super(Miccai2020_LDR_laplacian_unit_disp_add_lvl3, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train   = is_train

        self.imgshape   = imgshape

        self.model_lvl2 = model_lvl2

        #self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = generate_grid(self.imgshape,1)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()

        self.transform = SpatialTransform_unit().cuda()

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(self.in_channel+3, self.start_channel * 4, bias=bias_opt)
        self.down_conv = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1, bias=bias_opt)
        self.resblock_group_lvl1 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)
        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)


    def unfreeze_modellvl2(self):
        # unFreeze model_lvl1 weight
        print("\nunfreeze model_lvl2 parameter")
        for param in self.model_lvl2.parameters():
            param.requires_grad = True

    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.Sequential(
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2)
        )
        return layer

    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                              bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    # def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
    #             output_padding=0, bias=True):
    #     layer = nn.Sequential(
    #         nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
    #                            padding=padding, output_padding=output_padding, bias=bias),
    #         nn.ReLU())
    #     return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x, y):
        # compose_field_e0_lvl1, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, lvl1_v, e0
        lvl2_disp, _, _, lvl2_v, lvl1_v, lvl2_embedding = self.model_lvl2(x, y)
        lvl2_disp_up = self.up_tri(lvl2_disp)
        warpped_x = self.transform(x, lvl2_disp_up.permute(0, 2, 3, 4, 1), self.grid_1)

        cat_input = torch.cat((warpped_x, y, lvl2_disp_up), 1)

        fea_e0 = self.input_encoder_lvl1(cat_input)
        e0 = self.down_conv(fea_e0)

        e0 = e0 + lvl2_embedding
        e0 = self.resblock_group_lvl1(e0)
        e0 = self.up(e0)
        output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        compose_field_e0_lvl1 = output_disp_e0_v + lvl2_disp_up

        warpped_inputx_lvl1_out = self.transform(x, compose_field_e0_lvl1.permute(0, 2, 3, 4, 1), self.grid_1)

        if self.is_train is True:
            return compose_field_e0_lvl1, warpped_inputx_lvl1_out, y, output_disp_e0_v, lvl1_v, lvl2_v, e0
        else:
            return compose_field_e0_lvl1


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, num_group=4, stride=1, bias=False):
        super(PreActBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)

        # if stride != 1 or in_planes != self.expansion*planes:
        #     self.shortcut = nn.Sequential( nn.Conv3d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=bias)     )


    def forward(self, x):
        out = F.leaky_relu(x, negative_slope=0.2)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.leaky_relu(out, negative_slope=0.2))
        out += shortcut
        return out




# How it works:
# 1- the image enter a localisation network that outputs parameters of pre-defined transform
#    - it uses convolution network to find these parameters
# 2- using the transform and the parameters, we generate a displacement field
# 3- the image is transformed using the displacement field

class SpatialTransform_unit(nn.Module):
    def __init__(self, interpolator='bilinear'):
        super(SpatialTransform_unit, self).__init__()
        self.interpolator = interpolator
    def forward(self, x, flow, sample_grid):
        # print("x             : ", x.shape)     # moving image  1,1, 160, 192, 144 ???
        # print("flow          : ", flow.shape)  # displacement field with size /4 :  1,40, 48, 36, 3
        #print("sample_grid   : ", sample_grid.shape) # same size as flow         :  1,40, 48, 36, 3
        # a sampling grid: a set of points where the input map should be sampled to produce the transformed output
        sample_grid = sample_grid + flow
        #print("sample_grid   : ", sample_grid.shape) # same size as flow  1,1,40, 48, 36

        #Given an input x and a flow-field grid, computes the output using input values and pixel locations from grid.
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode=self.interpolator, padding_mode="border", align_corners=True)
        #print("flow          : ", flow.shape)# ??? transformed moving image at the grid points???
        return flow


# class SpatialTransformNearest_unit(nn.Module):
#     def __init__(self):
#         super(SpatialTransformNearest_unit, self).__init__()
#
#     def forward(self, x, flow, sample_grid):
#         sample_grid = sample_grid + flow
#         flow = torch.nn.functional.grid_sample(x, sample_grid, mode='nearest', padding_mode="border", align_corners=True)
#         return flow


#TODO:  not used???
#transform layer
class DiffeomorphicTransform_unit(nn.Module):
    def __init__(self, time_step=7, interpolator='bilinear'):
        super(DiffeomorphicTransform_unit, self).__init__()
        self.time_step = time_step
        self.interpolator = interpolator
    # transform derivative
    def forward(self, velocity, sample_grid):
        flow = velocity/(2.0**self.time_step)
        # print("time_step   : ", time_step)
        # print("velocity    : ", velocity.shape)
        # print("sample_grid : ", sample_grid.shape)
        # print("flow        : ", flow.shape)
        for _ in range(self.time_step):
            #TODO:  why change the axes???
            grid = sample_grid + flow.permute(0,2,3,4,1)
            #TODO: bilinear  and nearest support
            # Given  an input and a flow_field grid, computes the output using input values and pixel locations from grid.
            #what are we doing here ??? are we adding values?
            grid_sample_out = F.grid_sample(flow, grid, mode=self.interpolator, padding_mode="border", align_corners=True)
            # print("grid_sample_out        : ", grid_sample_out.shape)
            # print(ok)
            flow = flow + grid_sample_out
        return flow


def smoothloss(y_pred):
    dy = torch.abs(y_pred[:,:,1:, :, :] - y_pred[:,:, :-1, :, :])
    dx = torch.abs(y_pred[:,:,:, 1:, :] - y_pred[:,:, :, :-1, :])
    dz = torch.abs(y_pred[:,:,:, :, 1:] - y_pred[:,:, :, :, :-1])
    return (torch.mean(dx * dx)+torch.mean(dy*dy)+torch.mean(dz*dz))/3.0


def JacboianDet(y_pred, sample_grid):
    J = y_pred + sample_grid
    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:,:,:,:,0] * (dy[:,:,:,:,1] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,1])
    Jdet1 = dx[:,:,:,:,1] * (dy[:,:,:,:,0] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,0])
    Jdet2 = dx[:,:,:,:,2] * (dy[:,:,:,:,0] * dz[:,:,:,:,1] - dy[:,:,:,:,1] * dz[:,:,:,:,0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet


def neg_Jdet_loss(y_pred, sample_grid):
    neg_Jdet = -1.0 * JacboianDet(y_pred, sample_grid)
    selected_neg_Jdet = F.relu(neg_Jdet)

    return torch.mean(selected_neg_Jdet)


class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=5, eps=1e-8):
        super(NCC, self).__init__()
        self.win = win
        self.eps = eps
        self.w_temp = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.w_temp

        # set window size
        if self.win is None:
            self.win = [5] * ndims
        else:
            self.win = [self.w_temp] * ndims

        weight_win_size = self.w_temp
        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)


class multi_resolution_NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def  __init__(self, win=None, eps=1e-5, scale=3):
        super(multi_resolution_NCC, self).__init__()
        self.num_scale = scale
        self.similarity_metric = []

        for i in range(scale):
            self.similarity_metric.append(NCC(win=win - (i*2)))

    def forward(self, I, J):
        total_NCC = []

        for i in range(self.num_scale):
            current_NCC = self.similarity_metric[i](I, J)
            total_NCC.append(current_NCC/(2**i))

            I = nn.functional.avg_pool3d(I, kernel_size=3, stride=2, padding=1, count_include_pad=False)
            J = nn.functional.avg_pool3d(J, kernel_size=3, stride=2, padding=1, count_include_pad=False)

        return sum(total_NCC)

# these losses are from voxelmorph
class mseLoss:
    """
    Mean squared error loss.
    """
    def  __init__(self,y_true=[], y_pred=[]):
         pass

    def forward(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

def mse_loss(gt, pred):
    y_true_f = gt.view(-1)
    y_pred_f = pred.view(-1)
    diff = y_true_f-y_pred_f
    mse = torch.mul(diff,diff).mean()
    return mse

"""
N-D dice for segmentation
Note: check number of classes before process
"""
# def diceLoss( y_true, y_pred):
#     ndims = len(list(y_pred.size())) - 2
#     print("ndims : ", ndims)
#     vol_axes = list(range(2, ndims+2))
#     top = 2 * (y_true * y_pred).sum(dim=vol_axes)
#     bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
#     dice = torch.mean(top / bottom)
#     return -dice

# def diceLoss(a, b):
#     a = a.ravel()
#     b = b.ravel()
#     at = torch.tensor(a)
#     bt = torch.tensor(b)
#     c = torch.add(at,bt)
#     c[c<2] = 0.0
#     iaDice = 1.0 - sum(c)
#     # print("c = ", c.item())
#     # s= 0.0
#     # for i  in range (len(a)):
#     #     if a[i]==b[i]:
#     #        s+= 1;
#     # iaDice = 1.0 - ( s/len(a) ) # similar = 0.0
#     return iaDice

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    #def forward(self, inputs, targets, smooth=1):
    def getDiceLoss(inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1.0 - dice

