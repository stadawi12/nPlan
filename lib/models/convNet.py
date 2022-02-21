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

        self.up_conv1 = conv(128, 64)
        self.up_conv2 = conv(64, 32)
        self.up_conv3 = conv(32, 2)

        self.trans1  = nn.ConvTranspose2d(
                in_channels  = 128,
                out_channels = 64,
                kernel_size  = 2,
                stride       = 2
                )

        self.trans2  = nn.ConvTranspose2d(
                in_channels  = 64,
                out_channels = 32,
                kernel_size  = 2,
                stride       = 2
                )

        self.out = close(128, 121)

    def forward(self, sample):
        # Linear upsampling
        x = self.linear1(sample)
        # is x[0] the same as x[1]
        x = self.linear2(x)
        batch_size = x.shape[0]

        # TODO check if reshape function does not mix examples
        # Reshape linear layer to square matrix
        x = torch.reshape(x, (batch_size, 1, 32, 32))
        # for i in range(batch_size):
        #     print(torch.equal(x[i], torch.flatten(y[i])))

        # Begin encoder applying a series of convolutions, activations
        # and max pools
        x1 = self.conv1(x)  
        x = self.maxPool(x1)
        x2 = self.conv2(x)
        x = self.maxPool(x2)
        x = self.conv3(x)

        # Begin decoder
        x = self.trans1(x)
        x2 = crop(x2, x)
        x = self.up_conv1(torch.cat([x, x2], 1))

        x = self.trans2(x)
        x1 = crop(x1, x)
        x = self.up_conv2(torch.cat([x, x1], 1))
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
