import torch
import torch.nn as nn

# Convolution - BatchNorm - ReLU6 block (cbr)
class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super(CBR, self).__init__()
        # Padding to mimic 'same' padding in TensorFlow
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU6()  # ReLU6 activation as in TensorFlow

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

# Depthwise Convolution - BatchNorm - ReLU6 block (dbr)
class DBR(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1):
        super(DBR, self).__init__()
        padding = (kernel_size - 1) // 2
        # Depthwise convolution: groups=in_channels, out_channels=in_channels
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels
        )
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU6()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

# Mobile Inverted Residual Block (mobile_res_block)
class MobileResBlock(nn.Module):
    def __init__(self, in_channels, filters, expansion, stride=1):
        super(MobileResBlock, self).__init__()
        exp_channels = filters * expansion
        self.cbr = CBR(in_channels, exp_channels)
        self.dbr = DBR(exp_channels, stride=stride)
        self.pointwise = nn.Conv2d(exp_channels, filters, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(filters)  # No activation after this (linear output)
        self.use_residual = (in_channels == filters) and (stride == 1)

    def forward(self, x):
        input_x = x
        x = self.cbr(x)
        x = self.dbr(x)
        x = self.pointwise(x)
        x = self.bn(x)
        if self.use_residual:
            x = x + input_x  # Residual connection
        return x

# Upsampling Convolution - BatchNorm - ReLU block (up_cbr)
class UpCBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(UpCBR, self).__init__()
        # ConvTranspose2d to upsample by 2x, mimicking TensorFlow's Conv2DTranspose
        self.conv_trans = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()  # Regular ReLU as in the TensorFlow code

    def forward(self, x):
        x = self.conv_trans(x)
        x = self.bn(x)
        x = self.act(x)
        return x

# Atrous Spatial Pyramid Pooling (ASPP) module
class ASPP(nn.Module):
    def __init__(self, in_channels, num_filters, atrous_rates):
        super(ASPP, self).__init__()
        self.branches = nn.ModuleList()
        # 1x1 convolution branch
        self.branches.append(nn.Conv2d(in_channels, num_filters, kernel_size=1, padding=0))
        
        # Atrous (dilated) convolution branches
        for rate in atrous_rates:
            branch = nn.Sequential(
                nn.Conv2d(
                    in_channels, in_channels, kernel_size=3, padding=rate, dilation=rate,
                    groups=in_channels, bias=False
                ),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, num_filters, kernel_size=1, padding=0)
            )
            self.branches.append(branch)
        
        # Final processing after concatenation
        out_channels = num_filters * (len(atrous_rates) + 1)
        self.final_bn = nn.BatchNorm2d(out_channels)
        self.final_act = nn.ReLU()

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        x = torch.cat(branch_outputs, dim=1)  # Concatenate along channel dimension
        x = self.final_bn(x)
        x = self.final_act(x)
        return x

# Main Kokonet Model
class Kokonet(nn.Module):
    def __init__(self, input_channels=8, output_channels=1, activation='tanh'):
        super(Kokonet, self).__init__()

        # Downsampling path
        self.x_1 = CBR(input_channels, 32)
        self.x_2 = MobileResBlock(32, 128, expansion=1, stride=2)
        self.x_4_a = MobileResBlock(128, 256, expansion=6, stride=2)
        self.x_4_b = MobileResBlock(256, 256, expansion=6, stride=1)

        # ASPP module
        self.aspp = ASPP(256, 256, atrous_rates=[2, 4, 8])  # Outputs 256 * 4 = 1024 channels

        # Bottleneck
        self.bottleneck = CBR(1024, 256)

        # Upsampling path
        self.up_cbr_2 = UpCBR(256, 128)
        self.u_2_cbr = CBR(128 + 128, 64)  # After concatenation with x_2 (128 channels)
        self.up_cbr_1 = UpCBR(64, 32)
        self.u_1_cbr = CBR(32, 32)

        # Output layer
        self.output_conv = nn.Conv2d(32 + 32, output_channels, kernel_size=1, padding=0)  # After concatenation with x_1

        # Activation
        if activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()  # No activation if not 'tanh'

    def forward(self, x):
        # Downsampling
        x_1 = self.x_1(x)  # [B, 32, H, W]
        x_2 = self.x_2(x_1)  # [B, 128, H/2, W/2]
        x_4 = self.x_4_a(x_2)  # [B, 256, H/4, W/4]
        x_4 = self.x_4_b(x_4)  # [B, 256, H/4, W/4]

        # ASPP
        aspp = self.aspp(x_4)  # [B, 1024, H/4, W/4]
        bottleneck = self.bottleneck(aspp)  # [B, 256, H/4, W/4]

        # Upsampling with skip connections
        u_2 = self.up_cbr_2(bottleneck)  # [B, 128, H/2, W/2]
        u_2 = torch.cat([u_2, x_2], dim=1)  # [B, 256, H/2, W/2]
        u_2 = self.u_2_cbr(u_2)  # [B, 64, H/2, W/2]

        u_1 = self.up_cbr_1(u_2)  # [B, 32, H, W]
        u_1 = self.u_1_cbr(u_1)  # [B, 32, H, W]
        fex_out = torch.cat([u_1, x_1], dim=1)  # [B, 64, H, W]

        # Output
        output = self.output_conv(fex_out)  # [B, output_channels, H, W]
        output = self.activation(output)
        return output

# Example usage
if __name__ == "__main__":
    model = Kokonet(input_channels=8, output_channels=1, activation='tanh')
    x = torch.randn(1, 8, 512, 512)
    output = model(x)
    print(f"Output shape: {output.shape}")  # Expected: [1, 1, 512, 512]