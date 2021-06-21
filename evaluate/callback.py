import os 
def get_checkpoint_path(args = None) :
    # ckpt_folder = 'lightning_logs/{}/checkpoints/'.format(args.load_ckpt)
    # ckpt_name = os.listdir(ckpt_folder)[0]
    # ckpt_path = ckpt_folder + ckpt_name
    ckpt_path = "./lightning_logs/version_7/checkpoints/epoch=2-step=21404.ckpt"
    return ckpt_path 