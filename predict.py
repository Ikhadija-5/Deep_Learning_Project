import torch
import warnings
from PIL import Image
from torchvision import transforms
import models 
from argparse import ArgumentParser

def image_transform(imagepath):
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    image = Image.open(imagepath)
    imagetensor = test_transforms(image)
    return imagetensor


def predict(imagepath, verbose=False):
    if not verbose:
        warnings.filterwarnings('ignore')
    model_path = './model/model_ok.pth'
    try:
        checks_if_model_is_loaded = type(model)
    except:
        model = models.resnet50()
    model.eval()
    #summary(model, input_size=(3,244,244))
    if verbose:
        print("Model Loaded..")
    image = image_transform(imagepath)
    image1 = image[None,:,:,:]
    # ps= torch.exp(model(image1))
    ps= model(image1)
    topconf, topclass = ps.topk(1, dim=1)
    # print(ps)
    if topclass.item() == 1:
        return {'class':'dog','confidence':str(topconf.item())}
    else:
        return {'class':'cat','confidence':str(topconf.item())}

parser= ArgumentParser()
parser.add_argument('-m', '--image_path', help= 'upload the image',required= True)
main_args= vars(parser.parse_args())

image_path= main_args['image_path']
# print(predict('data/Images/image_name'))
print(Image.open(image_path))
print(predict(image_path))