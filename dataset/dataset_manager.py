from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder

class DatasetManager:
    def __init__(self, dataset_path, img_size=64, transform=None, batch_size=128, shuffle=True):
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
        self.dataset = ImageFolder(root=dataset_path, transform=self.transform)
        self.dataset_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)

    def test(self):
        sample_batch = next(iter(self.dataset_loader))
        print(sample_batch[1])


if __name__ == "__main__":
    datasetManager = DatasetManager("../../data/wikiart/")
    datasetManager.test()