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

    def __init__(self):
        super(Linear, self).__init__()

        self.linear1 = linear(50, 160)
        self.linear2 = linear(160, 160)
        self.linear2 = linear(160, 320)
        self.linear3 = linear(320, 160)
        self.linear4 = close(160, 121)


    def forward(self, sample):
        x = self.linear1(sample)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return x


if __name__ == "__main__":
    import torch
    n_features = 50
    L = Linear()
    inpt = torch.randn(45000,n_features)
    out = L(inpt)
    # print(out,"\n", out.shape)
