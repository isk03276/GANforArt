import argparse

from dataset.dataset_manager import DatasetManager
from utils.torch import optimize


def get_model_class(model:str):
    if model == "dcgan":
        from models.dcgan import DCGAN as model_class
    elif model == "can":
        from models.can import CAN as model_class
    else:
        raise NotImplementedError
    return model_class

def train(args):
    train_dataset_loader = DatasetManager(dataset_path = args.dataset_path,
                                     batch_size = args.batch_size).get_dataset_loader()
    
    model_class = get_model_class(args.model)
    model = model_class(batch_size=args.batch_size)
    
    status_str = "[{}/{} episode] generator loss : {}   |   discriminator_loss : {}"
    for ep in range(args.epoch):
        generator_losses = []
        discriminator_losses = []
        for images, _ in train_dataset_loader:
            fake_images = model.generate_fake_images()
            generator_loss = model.train_generator(fake_images)
            generator_losses.append(generator_loss.item())
            fake_images = model.generate_fake_images()
            discriminator_loss = model.train_discriminator(images, fake_images)
            discriminator_losses.append(discriminator_loss.item())
            
        print(status_str.format(ep+1, 
                                args.epoch,
                                sum(generator_losses) / len(generator_losses),
                                sum(discriminator_losses)/len(discriminator_losses)))
    
def test(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creative Adversarial Network")
    # model
    parser.add_argument("--model", default="dcgan", type=str, help="Model to train or test (ex. 'dcgan' or 'can')")
    # data
    parser.add_argument("--dataset-path", default="data/wikiart/", type=str, help="Wikiart dataest path")
    # checkpointing
    parser.add_argument("--save", action="store_true", help="Whether to save the model")
    parser.add_argument("--save-interval", type=int, default=20, help="Model save interval")
    parser.add_argument("--load-from", type=str, help="Path to load the model")
    # train
    parser.add_argument("--epoch", type=int, default=100, help="Learning epoch")
    parser.add_argument("--batch-size", type=int, default=100, help="Learning epoch")
    # test
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    
    args = parser.parse_args()
    
    if not args.test:
        train(args)
    else:
        test(args)