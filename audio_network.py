import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf=64, img_channels=3):
        super().__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.bias = True

        # input is Z, going into a convolution
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*4, kernel_size=4, stride=1, padding=0, bias=self.bias),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # state size. (ngf*4) x 4 x 4
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # state size. (ngf*2) x 8 x 8
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # state size. (ngf) x 16 x 16
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(ngf, img_channels, kernel_size=4, stride=2, padding=1, bias=self.bias),
            nn.Tanh()
        )

    def forward(self, input, reverse=True):
        fc1 = input.view(input.size(0), input.size(1), 1, 1)
        tconv1 = self.tconv1(fc1)
        tconv2 = self.tconv2(tconv1)
        tconv3 = self.tconv3(tconv2)
        output = self.tconv4(tconv3)
        if reverse:
            output = grad_reverse(output)
        return output



class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return (grad_output * -1)

def grad_reverse(x):
    return GradReverse.apply(x)
    
    

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)





class Discriminator(nn.Module):
    def __init__(self, ngpu, nz, ndf=64, img_channels=3, p_drop=0.0):
        super().__init__()
        self.ngpu = ngpu
        self.ndf = ndf
        self.bias = True


        # input is (3) x 32 x 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(img_channels, ndf, kernel_size=4, stride=2, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # state size. (64) x 16 x 16
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # state size. (ndf*2) x 8 x 8
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, inplace=True),
        )


        # # state size. (ndf*4) x 4 x 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf * 4, nz, kernel_size=4,stride=2, padding=0, bias=self.bias),
            Flatten()
        )
        
        self.dis = nn.Sequential(
             nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=2, padding=0, bias=self.bias),
             Flatten()
        )

        # sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        fc_dis = self.sigmoid(self.dis(conv3))
        fc_enc = self.conv4(conv3)
        realfake = fc_dis.view(-1, 1).squeeze(1)
        return fc_enc, realfake
        






class OutputClassifier(nn.Module):

    def __init__(self, nz, ni=32, num_classes=10):
        super().__init__()

        self.fc_classifier = nn.Sequential(
            nn.Linear(nz, num_classes, bias=True),
        )
        self.softmax = nn.Softmax()

    def forward(self, input):
        classes = self.fc_classifier(input)
        return classes




class InputClassifier(nn.Module):

    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.fc_classifier = nn.Sequential(
            nn.Linear(input_dim, num_classes, bias=True),
        )

    def forward(self, input):
        out = input.view(input.size(0), -1)  # convert batch_size x 28 x 28 to batch_size x (28*28)
        out = self.fc_classifier(out)  # Applies out = input * A + b. A, b are parameters of nn.Linear that we want to learn
        return out

if __name__ == "__main__":
    from torchviz import make_dot
    for model in [
        # Generator(ngpu=1, nz=256, img_channels=3),
        # Discriminator(ngpu=1, nz=256, img_channels=3, p_drop=0.0),
        OutputClassifier(nz=256, num_classes=10),
        InputClassifier(input_dim=784, num_classes=10),
        # InceptionV3()
    ]:
        print(model)
        x = torch.randn(256,3,32,32)
        y = model(x)
        print(y.size())
        make_dot(y.mean(), params=dict(model.named_parameters())).render(f"model-{model.__class__.__name__}", format="png")
    print("done")