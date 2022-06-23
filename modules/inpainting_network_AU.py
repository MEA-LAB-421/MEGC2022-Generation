import torch
from torch import nn
import torch.nn.functional as F
#from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
# from modules.dense_motion import DenseMotionNetwork


class InpaintingNetwork(nn.Module):
    """
    Inpaint the missing regions and reconstruct the Driving image.
    """

    def __init__(self, num_channels, block_expansion, max_features, num_down_blocks, multi_mask=True, num_cond_channels=17, **kwargs):
        super(InpaintingNetwork, self).__init__()

        self.num_down_blocks = num_down_blocks
        self.multi_mask = multi_mask
        self.first = SameBlock2d(
            num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        # Construct a sequential of 3 down_sample blocks
        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(
                in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)
        # Construct a sequential of 3 up_sample blocks(hourglass shape)
        up_blocks = []
        in_features = [max_features + num_cond_channels, max_features + num_cond_channels, max_features // 2 + num_cond_channels]
        out_features = [max_features // 2,
                        max_features // 4, max_features // 8]
        for i in range(num_down_blocks):
            up_blocks.append(
                UpBlock2d(in_features[i], out_features[i], kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)
        # Construct 2 residual connection blocks
        resblock = []
        for i in range(num_down_blocks):
            resblock.append(ResBlock2d(
                in_features[i], kernel_size=(3, 3), padding=(1, 1)))
            resblock.append(ResBlock2d(
                in_features[i], kernel_size=(3, 3), padding=(1, 1)))
        self.resblock = nn.ModuleList(resblock)

        self.final = nn.Conv2d(block_expansion, num_channels,
                               kernel_size=(7, 7), padding=(3, 3))
        self.num_channels = num_channels
        self.num_cond_channels = num_cond_channels


    def deform_input(self, inp, deformation):
        # deform img according to deformation info. (Optical flow)
        # B, H, W, 2 (grid for sampling)
        _, h_old, w_old, _ = deformation.shape
        # B, C, H, W (input for sampling)
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(
                h, w), mode='bilinear', align_corners=True)
            deformation = deformation.permute(0, 2, 3, 1)
        # Given an input and a flow-field grid, computes the output using input values and pixel locations from grid.
        return F.grid_sample(inp, deformation, align_corners=True)


    def occlude_input(self, inp, occlusion_map):
        # mask img according to occlusion map
        if not self.multi_mask:
            if inp.shape[2] != occlusion_map.shape[2] or inp.shape[3] != occlusion_map.shape[3]:
                occlusion_map = F.interpolate(
                    occlusion_map, size=inp.shape[2:], mode='bilinear', align_corners=True)
        out = inp * occlusion_map
        return out
    

    def expand_AU(self, AU_cond, out):
        AU_cond = AU_cond.unsqueeze(2).unsqueeze(3)
        AU_cond = AU_cond.expand(AU_cond.size(0), AU_cond.size(1), out.size(2), out.size(3))
        out = torch.cat([out, AU_cond], 1)
        return out


    def forward(self, source_image, dense_motion, AU_cond):
        '''
            source image: shape (B, C, H, W)
            dense_motion: a dict containing occlusion mask and deformation info.(optical flow) and others.
                        dense_motion: {
                            'occlusion_map': a mask of a list of masks of different resolutions (B, H, W, 2)
                            'deformation': optical info. for deformation (B, C, H, W)
                            'deformed source': ?
                            'contribution_maps': ?
                        }
            first layer: self.first
            three downblock layers, each add an encoder map. so 4 in total (feats. in orange color in the graph)

        '''
        out = self.first(source_image)
        encoder_map = [out]     # Record "e" in the graph
        # torch.Size([2, 3, 128, 128])
        # (2, 64, 128, 128) [2, 128, 64, 64] [2, 256, 32, 32] [2, 512, 16, 16]
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
            encoder_map.append(out)

        output_dict = {}
        output_dict['contribution_maps'] = dense_motion['contribution_maps']
        output_dict['deformed_source'] = dense_motion['deformed_source']

        occlusion_map = dense_motion['occlusion_map']
        output_dict['occlusion_map'] = occlusion_map

        deformation = dense_motion['deformation']   # optical flow
        out_ij = self.deform_input(out.detach(), deformation)
        out = self.deform_input(out, deformation)

        out_ij = self.occlude_input(out_ij, occlusion_map[0].detach())
        out = self.occlude_input(out, occlusion_map[0])

        warped_encoder_maps = []
        # a copy of feature after warped & masked
        warped_encoder_maps.append(out_ij)

        # Decoder part starts here
        for i in range(self.num_down_blocks):
            out = self.expand_AU(AU_cond, out)
            out = self.resblock[2 * i](out)
            out = self.resblock[2 * i + 1](out)
            out = self.up_blocks[i](out)
            encode_i = encoder_map[-(i + 2)]
            encode_ij = self.deform_input(encode_i.detach(), deformation)
            encode_i = self.deform_input(encode_i, deformation)

            occlusion_ind = 0
            if self.multi_mask:
                occlusion_ind = i + 1
            encode_ij = self.occlude_input(
                encode_ij, occlusion_map[occlusion_ind].detach())
            encode_i = self.occlude_input(
                encode_i, occlusion_map[occlusion_ind])
            warped_encoder_maps.append(encode_ij)

            if(i == self.num_down_blocks - 1):
                break
            
            out = torch.cat([out, encode_i], 1)
        # out: (2, 64, 128, 128)
        deformed_source = self.deform_input(source_image, deformation)
        output_dict["deformed"] = deformed_source
        output_dict["warped_encoder_maps"] = warped_encoder_maps

        occlusion_last = occlusion_map[-1]
        if not self.multi_mask:
            occlusion_last = F.interpolate(
                occlusion_last, size=out.shape[2:], mode='bilinear', align_corners=True)

        out = out * (1 - occlusion_last) + encode_i
        out = self.final(out)   # (2, 3, 128, 128)
        out = torch.sigmoid(out)
        out = out * (1 - occlusion_last) + deformed_source * occlusion_last
        output_dict["prediction"] = out

        return output_dict

    def get_encode(self, driver_image, occlusion_map):
        out = self.first(driver_image)
        encoder_map = []
        encoder_map.append(self.occlude_input(
            out.detach(), occlusion_map[-1].detach()))
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out.detach())
            out_mask = self.occlude_input(
                out.detach(), occlusion_map[2 - i].detach())
            encoder_map.append(out_mask.detach())

        return encoder_map


if __name__ == '__main__':
    config = "../config/3MEdataset.yaml"
    import yaml
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    inpainting = InpaintingNetwork(**config['model_params']['generator_params'],
                                   **config['model_params']['common_params']).cuda()
    # print(inpainting)
    source = torch.randn((2, 3, 128, 128)).cuda()   
    # enc_map:[(2, 64, 128, 128) [2, 128, 64, 64] [2, 256, 32, 32] [2, 512, 16, 16]]
    oc_map = []
    for i in range(4):
        dim = 16 * (2**(i))
        oc_map.append(torch.randn((2, 1, dim, dim)).cuda())
    defor = torch.randn((2, 32, 32, 2)).cuda()    # optical flow
    dense_motion = {
        'occlusion_map': oc_map,
        'deformation': defor,
        'deformed_source': torch.randn([2, 11, 3, 32, 32]).cuda(),
        'contribution_maps': torch.randn([2, 11, 32, 32]).cuda()}
    c = torch.randn((2, 17)).cuda()
    output_dict = inpainting(source, dense_motion, c)
    print(output_dict["prediction"].shape)

    # Add AU conditions c: (B, C_cond=17)
   # c = torch.randn((2, 17))
    # c = c.unsqueeze(2).unsqueeze(3)
    # c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))