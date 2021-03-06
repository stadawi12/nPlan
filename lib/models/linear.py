import torch.nn as nn

# A function containing a linear layer followed by an activation
# function (ReLU) in this case
def linear(in_c, out_c):
    conv = nn.Sequential(
            nn.Linear(in_c, out_c),
            nn.ReLU(inplace=True)
            )
    return conv

# Last layer of the model
def close(in_c, out_c):
    conv = nn.Sequential(
            nn.Linear(in_c, out_c),
            nn.Sigmoid()
            )
    return conv

class Linear(nn.Module):
    """This is a linear neural network with linear layers, very simple,
    no skip connections just a standard model"""

    def __init__(self):
        super(Linear, self).__init__()

        self.linear1 = linear(50, 160)
        self.linear2 = linear(160, 160)
        self.linear3 = linear(160, 160)
        self.linear4 = linear(160, 160)
        self.linear5 = close(160, 121)


    def forward(self, sample):
        x = self.linear1(sample)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        return x


if __name__ == "__main__":
    import torch
    n_features = 50
    L = Linear()
    inpt = torch.randn(2,n_features)
    out = L(inpt)
    print("Input shape:", inpt.shape)
    print("Output shape:", out.shape)
