import ESPCN

num_epochs = 200  # the number of training epochs

# the baseline model
model = ESPCN.Net(upscale_factor=4)
model_name = "ESPCN"