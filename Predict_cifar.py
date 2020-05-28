import torch
from model import resnet34
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# load image
img = Image.open("./test_images/test_image2.jpg")
plt.imshow(img)
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)  # [N, C, H, W]

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# create model
model = resnet34(num_classes=10)
# load model weights
model_weight_path = "./weights/resNet34.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval()
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img))
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
    print(predict_cla)
    print(classes[predict_cla])
plt.show()


