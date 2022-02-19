import torch.nn as nn

def crop_img(tensor, target):
    """
        This function should crop the tensor to the size
        of the target,  it should crop out the centre of
        the tensor rather than a corner of it.
    """
    tensor_size = tensor.shape[2]
    target_size = target.shape[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, 
            delta:tensor_size - delta,
            delta:tensor_size - delta
            ]

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
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
            )
    return conv

class Linear(nn.Module):
    """This is a linear neural network with linear layers, very simple,
    no skip connections just a standard model"""

    def __init__(self):
        super(Linear, self).__init__()

        self.linear1 = linear(50, 160)
        self.linear2 = linear(160, 64*64)

        self.maxPool = nn.MaxPool2d(2)

        self.conv1   = conv(1, 64)
        self.conv2   = conv(64, 128)
        self.conv3   = conv(128, 256)

        self.up_conv1 = conv(256, 128)
        self.up_conv2 = conv(128, 64)
        self.up_conv3 = conv(64, 1)

        self.trans1  = nn.ConvTranspose2d(
                in_channels  = 256,
                out_channels = 128,
                kernel_size  = 2,
                stride       = 2
                )

        self.trans2  = nn.ConvTranspose2d(
                in_channels  = 128,
                out_channels = 64,
                kernel_size  = 2,
                stride       = 2
                )

        self.out = close(64*64, 121)

    def forward(self, sample):
        # Linear upsampling
        x = self.linear1(sample)
        x = self.linear2(x)
        batch_size = x.shape[0]

        # Reshape linear layer to square matrix
        x = torch.reshape(x, (batch_size, 1, 64, 64))

        # Begin encoder applying a series of convolutions, activations
        # and max pools
        x1 = self.conv1(x)
        x = self.maxPool(x1)
        x2 = self.conv2(x)
        x = self.maxPool(x2)
        x = self.conv3(x)

        # Begin decoder
        x = self.trans1(x)
        x2 = crop_img(x2, x)
        x = self.up_conv1(torch.cat([x, x2], 1))

        x = self.trans2(x)
        x1 = crop_img(x1, x)
        x = self.up_conv2(torch.cat([x, x1], 1))

        x = self.up_conv3(x)
        x = torch.reshape(x, (batch_size, 64*64))
        x = self.out(x)
        return x


if __name__ == "__main__":
    import torch
    n_features = 50
    L = Linear()
    inpt = torch.randn(100,n_features)
    out = L(inpt)
    print("Input shape:", inpt.shape)
    print("Output shape:", out.shape)
