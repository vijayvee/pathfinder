# pathfinder
Pathfinder experiments

There are various .sh files in the repository’s root each of which run 3 random initialized training jobs for a particular model (e.g. run_models_5ep_resnet.sh will run 5 epochs of training for the ResNet model). Model configuration is defined inside configs/ which is the configuration to run DaleRNN for 12 steps of recurrent processing with a 32-channel 9x9 filter bank).

The process of training a new model that isn’t here involves:
creating a file that contains the model class inside models/
creating a config file that specifies the model parameters inside configs/
combining the two inside models/model_builder.py

### IMPORTANT: DaleRNN is now renamed as LocRNN 