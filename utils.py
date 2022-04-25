import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
def plot(train_loss, val_loss):
    plt.title("Training results: Loss")
    plt.plot(val_loss,label='val_loss')
    plt.plot(train_loss, label="train_loss")
    plt.legend()
    plt.savefig("./figures/train_res.png")
    plt.show()