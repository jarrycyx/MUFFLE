python Test.py -opt Exp/Data0511/DCMAN_Concat_Fn15_HJHS_ALL.yaml \
    -gpu 1 -dataset real \
    -load_pkl Exp/experiments_outputs/train/DCMAN_Concat_Fn15_0320_2023_0515_170206_076694/pklmodels/train_iter_100.pkl \
    -gain 10 \
    -max_n_frame 50000

python Test.py -opt Exp/Data0511/DCMAN_Concat_Ablation_NoFus_HJHS_ALL.yaml \
    -gpu 1 -dataset real \
    -load_pkl Exp/experiments_outputs/train/DCMAN_Concat_Ablation_NoFus_2024_0419_221000_316193/pklmodels/train_iter_100.pkl \
    -gain 10 \
    -max_n_frame 50000