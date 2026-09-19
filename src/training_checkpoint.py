"""Epoch-boundary training checkpoints, separate from inference weight files."""
import os
import random

import numpy as np
import torch


def save_checkpoint(path, model, optimizer, scheduler, epoch, dataloader):
    numpy_state = np.random.get_state()
    state = {
        'format_version': 1,
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': scheduler.state_dict() if scheduler is not None else None,
        'python_rng': random.getstate(),
        'numpy_rng': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        'generators': {
            name: obj.generator.get_state()
            for name, obj in [('loader', dataloader), ('sampler', dataloader.sampler)]
            if getattr(obj, 'generator', None) is not None
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + '.tmp'
    torch.save(state, temporary)
    os.replace(temporary, path)


def restore_checkpoint(path, model, optimizer, scheduler, epoch, dataloader):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['format_version'] != 1 or state['epoch'] != epoch:
        raise ValueError('Training checkpoint version/epoch mismatch')
    if (state['lr_scheduler'] is None) != (scheduler is None):
        raise ValueError('Learning-rate scheduler configuration mismatch')
    if len(state['cuda_rng']) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError('Resume requires the same number of visible CUDA devices')
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['lr_scheduler'])
    for name, obj in [('loader', dataloader), ('sampler', dataloader.sampler)]:
        generator = getattr(obj, 'generator', None)
        if (name in state['generators']) != (generator is not None):
            raise ValueError(f'{name} generator configuration mismatch')
        if generator is not None:
            generator.set_state(state['generators'][name])
    random.setstate(state['python_rng'])
    numpy_state = state['numpy_rng']
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng']:
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    print(f'Restored full training state from {path}')
