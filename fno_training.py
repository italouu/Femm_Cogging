import torch
import matplotlib.pyplot as plt
from neural_op.fno import FNO2d
import numpy as np
from data_gen.data_utils import csv_to_pt, load_tensor
from neural_op.training_utils import count_params, l2_loss
import time

############# csv to pt #############
# csv_to_pt(n_phases=2)

############# data load #############
x, y = load_tensor(file  = 'data_test.pt')
data_in_size  = list(x.shape)
data_out_size = list(y.shape)
n_data = data_in_size[0]
print(data_in_size, data_out_size)

############# model definition #############
in_channels = data_in_size[1]
out_channels = data_out_size[1]
modes1 = data_in_size[2]        # max = data_in_size[1]
modes2 = data_in_size[3]        # max = data_in_size[2]//2 + 1
data_res = data_in_size[-2:]

conv_width = 3
conv_layers = 4
lift_width = 64
lift_layers = 3
proj_width = 64
proj_layers = 3

model = FNO2d(
    in_channels = in_channels,    # number of input channels
    out_channels = out_channels,  # number of output channels
    modes1 = modes1,              # modes to keep is axis 1
    modes2 = modes2,              # modes to keep is axis 2
    conv_width = conv_width,      # number of frequency convolution blocks per layer
    conv_layers = conv_layers,    # number of frequency convolution layers
    lift_width = lift_width,      # width of lift mlp hidden layers
    lift_layers = lift_layers,    # number of lift mlp layers
    proj_width = proj_width,      # width of proj mlp hidden layers
    proj_layers = proj_layers,    # number of proj mlp layers
    data_res = data_res,          # [res_r, res_th]
    )
############# data processing #############
batch_size = 32
train_portion = 0.75
n_train = int(n_data * train_portion)
n_test  = n_data - n_train

x_train = x[:n_train, :, :, :]
y_train = y[:n_train, :, :, :]
x_test  = x[n_train:n_train+n_test, :, :, :]
y_test =  y[n_train:n_train+n_test, :, :, :]

train_set = torch.utils.data.TensorDataset(x_train, y_train)
test_set  = torch.utils.data.TensorDataset(x_test,  y_test)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=batch_size, shuffle=True)

############# training #############
training_device = torch.device("cuda")
process_device  = torch.device("cpu")
n_epochs = 1000

model = model.to(training_device)
print(count_params(model))

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
    amsgrad=False,
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=100, 
    gamma=0.5
)

for ep in range(n_epochs):
    model.train()
    t1 = time.perf_counter()
    train_l2 = 0

    for x, y in train_loader:
        x = x.to(training_device)
        y = y.to(training_device)

        optimizer.zero_grad()
        out = model(x)

        loss = l2_loss(out, y)
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(training_device)
            y = y.to(training_device)

            out = model(x)
            test_l2 += l2_loss(out, y).item()

    train_l2 /= len(train_loader)
    test_l2  /= len(test_loader)

    t2 = time.perf_counter()
    print(f"epoch: {ep}, [{t2-t1}]s, L2: train [{train_l2}], test [{test_l2}]")

############# save #############

model_path = 'neural_op/torch_models/'
model_name = 'model.pth'
torch.save(model.state_dict(), model_path + model_name)

############# eval #############
model = model.to(process_device).eval()

with torch.no_grad():
    x  = x_test[0:1, :, :, :].to(process_device) 
    y  = y_test[0].to(process_device)

    yx = y[0, :, :]
    yy = y[1, :, :]

    mag_y = torch.sqrt(yx**2 + yy**2)

    out = model(x)
    outx = out[0, 0, :, :]
    outy = out[0, 1, :, :]

    mag_out = torch.sqrt(outx**2 + outy**2)

############ plot #############

if 1:
    data_x = x[0, 0, :, :].detach().numpy()
    data_y = mag_y.detach().numpy()
    data_out = mag_out.detach().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(data_x, cmap="viridis")
    axes[0].set_title("Entrada: x[0,0,:,:]")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(data_y, cmap="viridis")
    axes[1].set_title("Magnitude alvo |y|")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(data_out, cmap="viridis")
    axes[2].set_title("Magnitude predita |out|")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()