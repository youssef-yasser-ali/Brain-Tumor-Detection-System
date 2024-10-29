
# Classification Model Code
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image
import os
from app.config import device  



labels_mapping = {
    0: "Meningioma",
    1: "No Tumor",
    2: "Pituitary",
    3: "Glioma"
}


class MyModel(nn.Module):
    def __init__(self,num_classes):
        super(MyModel,self).__init__()
        
        self.conv1 = nn.Conv2d(3,32,kernel_size=4,stride=1,padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=1,padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64,128,kernel_size=4,stride=1,padding=0)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128,128,kernel_size=4,stride=1,padding=0)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        self.pool2= nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.fc1 = nn.Linear(6*6*128,512)
        self.fc2 = nn.Linear(512,num_classes)
        
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(0.5)
        
        
        
        
    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model_path = os.path.join(os.path.dirname(__file__), 'pre_trained', 'class_brain_tumor')



classification_model = MyModel(num_classes=5).to(device)

classification_model.load_state_dict(torch.load(model_path, map_location=device))
classification_model.eval()


classification_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def classification_prediction(image: Image.Image, model: nn.Module, transform: transforms.Compose, device: torch.device):

    image_tensor = transform(image).unsqueeze(0)  

    with torch.no_grad():
        output = model(image_tensor.to(device))

    _, predicted_class = torch.max(output, 1)

    return predicted_class.item()