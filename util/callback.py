import os 
def get_checkpoint_path(args) :
    ckpt_folder = 'lightning_logs/{}/checkpoints/'.format(args.load_ckpt)
    ckpt_name = os.listdir(ckpt_folder)[0]
    ckpt_path = ckpt_folder + ckpt_name
    return ckpt_path 
