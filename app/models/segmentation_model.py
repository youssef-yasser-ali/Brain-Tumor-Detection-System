import torch
import torch.nn as nn
from torchvision.transforms import transforms
import os
from app.config import device
import torch
import torch.nn as nn

from PIL import Image


class UNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(UNet, self).__init__()
        
        # Encoding (Downsampling)
        self.encoder1 = self.conv_block(input_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoding (Upsampling)
        self.upconv6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder6 = self.conv_block(1024, 512)

        self.upconv7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder7 = self.conv_block(512, 256)

        self.upconv8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder8 = self.conv_block(256, 128)

        self.upconv9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder9 = self.conv_block(128, 64)

        self.output = nn.Conv2d(64, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoding
        c1 = self.encoder1(x)
        p1 = self.pool1(c1)

        c2 = self.encoder2(p1)
        p2 = self.pool2(c2)

        c3 = self.encoder3(p2)
        p3 = self.pool3(c3)

        c4 = self.encoder4(p3)
        p4 = self.pool4(c4)

        # Bottleneck
        bottleneck = self.bottleneck(p4)

        # Decoding
        u6 = self.upconv6(bottleneck)
        u6 = torch.cat((u6, c4), dim=1)  # Concatenate
        c6 = self.decoder6(u6)

        u7 = self.upconv7(c6)
        u7 = torch.cat((u7, c3), dim=1)  # Concatenate
        c7 = self.decoder7(u7)

        u8 = self.upconv8(c7)
        u8 = torch.cat((u8, c2), dim=1)  # Concatenate
        c8 = self.decoder8(u8)

        u9 = self.upconv9(c8)
        u9 = torch.cat((u9, c1), dim=1)  # Concatenate
        c9 = self.decoder9(u9)

        outputs = self.output(c9)
        return outputs


seg_model = UNet(input_channels=3, num_classes=1).to(device)


checkpoint = torch.load(
    os.path.join(os.path.dirname(__file__), 'pre_trained', 'seg_brain_tumor_model.pth'), 
    map_location=torch.device('cpu')
)

seg_model.load_state_dict(checkpoint['model_state_dict'])
seg_model.eval()




seg_transformtion = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),          
])




def segmentation_prediction(image: Image.Image, model: nn.Module, transform: transforms.Compose, device: torch.device):

    image_tensor = transform(image).unsqueeze(0)  

    with torch.no_grad():
        output = model(image_tensor.to(device))
    
    return output.squeeze(0).cpu()  
