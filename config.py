# RRDB
nf = 3
gc = 32

# hyperparameters
eps = 10/255
clamp = 2.0
channels_in = 3
lr = 1e-4
lr_min = 1e-6
epochs = 3001
weight_decay = 1e-5
init_scale = 0.01


# hyperparameters about loss

lamda_perc = 50
lamda_perc_vgg = 0.005
lamda_perc_low = 10
lamda_adv_latent_copy = 0.5
lamda_adv_tri = 0.1


margin_renew_times = 4
margin_max = 1.0

# Train:

betas = (0.5, 0.999)
weight_step = 400
gamma = 0.9
val_freq = 50

# Display and logging:
loss_display_cutoff = 2.0  # cut off the loss so the plot isn't ruined
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False
checkpoint_on_error = True

# Load:
pre_model = './pretrained/model_1-8-23-30.pt'

# others
imagesize = 512
