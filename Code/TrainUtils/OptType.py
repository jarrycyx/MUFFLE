from dataclasses import dataclass

from numpy import int0

    
@dataclass
class DatasetOpt:
    train: list
    test: list
    real: list

#################
@dataclass
class TrainOpt:
    Proj_name: str
    Random_seed: int
    ### Network Architecture
    Input_frame_num: int      # Numbers of Frames in a video (sliding window)
    Network_name: str
    Channel_merge: str
    ### Network Training Process
    Epoch_size: int       # Number of batches in an epoch
    Batch_size: int         # Number of video in a mini-batch
    LR: float
    Iter_num: int           # Number of Iterations (Net trained with an epoch)
    ### Simulation
    Img_size: list
    Simulation_dsample: list
    Noise_model: object
    ### Loss function
    Loss: str
    Color_loss_milestones: list
    Color_loss_ratio: list  # same length with Color_loss_milestones
    ### Dataset Definition
    Dataset: DatasetOpt
    
