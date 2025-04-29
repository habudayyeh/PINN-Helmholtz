# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import utils, modules, training

from torch.utils.data import DataLoader
from torch.cuda import is_available as cuda_available

logging_root= './logs'
experiment_name='test'
batch_size = 32
lr = 2e-5
num_epochs = 2000
epochs_til_ckpt = 1000
steps_til_summary = 100
activation = 'sine'
mode = 'mlp'
velocity = 'uniform'
clip_grad = 0.0
use_lbfgs = False
checkpoint_path = None

# if we have a velocity perturbation, offset the source
if velocity!='uniform':
    source_coords = [-0.35, 0.]
else:
    source_coords = [0., 0.]

dataset = utils.SingleHelmholtzSource(sidelength=100, velocity=velocity, source_coords=source_coords)

dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)

# Define the model.
model = modules.SingleBVPNet(out_features=2, type=activation, mode=mode, final_layer_factor=1.)

if  cuda_available():
    model.cuda()

# Define the loss
loss_fn = utils.helmholtz_pml
summary_fn = utils.write_helmholtz_summary

root_path = os.path.join(logging_root, experiment_name)

training.train(model=model, train_dataloader=dataloader, epochs=num_epochs, lr=lr,
               steps_til_summary=steps_til_summary, epochs_til_checkpoint=epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, clip_grad=clip_grad,
               use_lbfgs=use_lbfgs)
