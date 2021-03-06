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

class LinResBN(nn.Module):
    """This is a linear neural network with linear layers, very simple,
    no skip connections just a standard model"""

    def __init__(self, norm=False):
        super(LinResBN, self).__init__()

        self.norm = norm

        self.linear1 = nn.Linear(50, 160)
        self.linear2 = nn.Linear(160, 160)
        self.linear3 = nn.Linear(160, 160)
        self.linear4 = nn.Linear(160, 160)
        self.linear5 = nn.Linear(160, 121)
        self.relu = nn.ReLU(inplace=True)
        self.sigm = nn.Sigmoid()

        self.bn1   = nn.BatchNorm1d(50)
        self.bn2   = nn.BatchNorm1d(160)


    def forward(self, sample):

        sample = self.bn1(sample)

        x = self.linear1(sample)
        x = self.relu(x)

        x = self.bn2(x)

        identity = x

        x = self.linear2(x)
        x = self.relu(x + identity)

        x = self.bn2(x)

        identity = x

        x = self.linear3(x)
        x = self.relu(x + identity)

        identity = x

        x = self.linear4(x)
        x = self.relu(x + identity)

        x = self.linear5(x)
        x = self.sigm(x)
        return x


if __name__ == "__main__":
    import torch
    n_features = 50
    L = LinResBN(norm=True)
    inpt = torch.randn(2,n_features)
    out = L(inpt)
    print("Input shape:", inpt.shape)
    print("Output shape:", out.shape)
