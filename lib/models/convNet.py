import torch.nn as nn
import torch

def crop(x, y):
    x_size = x.shape[2]
    y_size = y.shape[2]
    return x[:,:,:y_size, :y_size]

# A function containing a linear layer followed by an activation
# function (ReLU) in this case
def linear(in_c, out_c):
    conv = nn.Sequential(
            nn.Linear(in_c, out_c),
            nn.ReLU(inplace=True)
            )
    return conv

# Last layer
def close(in_c, out_c):
    conv = nn.Sequential(
            nn.Linear(in_c, out_c),
            nn.Sigmoid()
            )
    return conv

# Last layer of the model
def conv(in_c, out_c):
    conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=0),
            nn.ReLU(inplace=True)
            )
    return conv

class UNet(nn.Module):
    """This is a linear neural network with linear layers, very simple,
    no skip connections just a standard model"""

    def __init__(self):
        super(UNet, self).__init__()

        self.linear1 = linear(50, 160)
        self.linear2 = linear(160, 32*32)

        self.maxPool = nn.MaxPool2d(2)

        self.conv1   = conv(1, 32)
        self.conv2   = conv(32, 64)
        self.conv3   = conv(64, 128)

        self.up_conv1 = conv(64, 32)
        self.up_conv2 = conv(16, 8)
        self.up_conv3 = conv(8, 2)

        self.trans1  = nn.ConvTranspose2d(
                in_channels  = 128,
                out_channels = 64,
                kernel_size  = 2,
                stride       = 2
                )

        self.trans2  = nn.ConvTranspose2d(
                in_channels  = 32,
                out_channels = 16,
                kernel_size  = 2,
                stride       = 2
                )

        self.out = close(128, 121)

    def forward(self, sample):
        # Linear upsampling
        x = self.linear1(sample)
        x = self.linear2(x)
        batch_size = x.shape[0]

        x = torch.reshape(x, (batch_size, 1, 32, 32))
        # for i in range(batch_size):
        #     print(torch.equal(x[i], torch.flatten(y[i])))

        x1 = self.conv1(x)  
        x = self.maxPool(x1)
        x2 = self.conv2(x)
        x = self.maxPool(x2)
        x = self.conv3(x)
        print(torch.sum(x[0] - x[1]))

        # Begin decoder
        x = self.trans1(x)
        x = self.up_conv1(x)

        x = self.trans2(x)
        x = self.up_conv2(x)
        x = self.up_conv3(x)
        x = torch.reshape(x, (batch_size, x.shape[1]*x.shape[2]**2))
        x = self.out(x)
        return x


if __name__ == "__main__":
    import torch
    torch.manual_seed(100)
    n_features = 50
    L = UNet()
    inpt = torch.randn(2,n_features)
    out = L(inpt)
    print("Input shape:", inpt.shape)
    print("Output shape:", out.shape)
    print(torch.sum(out[0] - out[1]))
