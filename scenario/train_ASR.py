import nemo
import nemo.collections.asr as nemo_asr
from omegaconf import DictConfig
import copy
import pytorch_lightning as pl

# Load checkpoint and continue training
model_path = "../lightning_logs/version_7/checkpoints/epoch=2-step=21404.ckpt"
asr_model = nemo_asr.models.EncDecCTCModel.load_from_checkpoint(checkpoint_path=model_path)

# Load yaml to get params
try:
    from ruamel.yaml import YAML
except ModuleNotFoundError:
    from ruamel_yaml import YAML
config_path = '../configs/config_sp.yaml'

yaml = YAML(typ='safe')
with open(config_path) as f:
    params = yaml.load(f)

# Config to load data by json
train_manifest = "../json/string/train-clean-100-nemo.json" 
test_manifest = "../json/string/dev-clean-nemo.json"

params['model']['train_ds']['manifest_filepath'] = train_manifest
params['model']['validation_ds']['manifest_filepath'] = test_manifest

print(params)

# Optimizer
new_opt = copy.deepcopy(params['model']['optim'])
new_opt['lr'] = 0.001
asr_model.setup_optimization(optim_config=DictConfig(new_opt))

# Point to the data we'll use for fine-tuning as the training set
asr_model.setup_training_data(train_data_config=params['model']['train_ds'])

# Point to the new validation data for fine-tuning
asr_model.setup_validation_data(val_data_config=params['model']['validation_ds'])

# And now we can create a PyTorch Lightning trainer and call `fit` again.
trainer = pl.Trainer(gpus=[0], max_epochs=10)
trainer.fit(asr_model)