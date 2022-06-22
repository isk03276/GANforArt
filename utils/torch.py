import torch


def optimize(model, optimizer, loss):
    model.zero_grad()
    loss.backward()
    optimizer.step()
    
def save(model, dir_path, file_name):
    torch.save({
        'generator' : model.state_dict(),
        'discriminator' : model.state_dict(),
        'optimizer_g' : model.state_dict(),
        'optimizer_d' : model.state_dict()
    }, dir_path+file_name)

def load(model, dir_path, file_name):
    torch.load({
        'generator' : model.state_dict(),
        'discriminator' : model.state_dict(),
        'optimizer_g' : model.state_dict(),
        'optimizer_d' : model.state_dict()
    }, dir_path+file_name)    
