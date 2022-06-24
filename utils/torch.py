import torch


def optimize(model, optimizer, loss):
    model.zero_grad()
    loss.backward()
    optimizer.step()
    