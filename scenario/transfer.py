import nemo
import nemo.collections.asr as nemo_asr
from omegaconf import DictConfig
import copy
import pytorch_lightning as pl

# Load transfer model
quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet5x5LS-En")

# Change label
quartznet.change_vocabulary(
    new_vocabulary=[' ', 'UW1', 'AW1', 'F', 'UW2', 'AO0', 'EY1', 'V', 'ER0', 'NG', 'AY1', 'P', 'UH1', 'EH0', 'ER2', 'sil', 'DH', 'S', 'EH2', 'IH2', 'AA1', 'CH', 'sp', 'AE0', 'EY0', 'TH', 'ER1', 'HH', 'OW1', 'EH1', 'OY1', 'AO2', 'AA0', 'K', 'AA2', 'AY2', 'UH0', 'IH1', 'AW2', 'B', 'T', 'W', 'M', 'IY2', 'R', 'SH', 'OY0', 'IY0', 'Y', 'AH1', 'AE1', 'D', 'AH0', 'UW0', 'OW2', 'N', 'UH2', 'AO1', 'Z', 'JH', 'IH0', 'G', 'AE2', 'AW0', 'IY1', 'OW0', 'OY2', 'AH2', 'AY0', 'ZH', 'spn', 'EY2', 'L']
)

# Load yaml to get params
try:
    from ruamel.yaml import YAML
except ModuleNotFoundError:
    from ruamel_yaml import YAML
config_path = '../configs/config.yaml'

yaml = YAML(typ='safe')
with open(config_path) as f:
    params = yaml.load(f)

# Config to load data by json
train_manifest = "../json/train-clean-100-nemo.json" 
test_manifest = "../json/dev-clean-nemo.json"

params['model']['train_ds']['manifest_filepath'] = train_manifest
params['model']['validation_ds']['manifest_filepath'] = test_manifest

print(params)

# Optimizer
new_opt = copy.deepcopy(params['model']['optim'])
new_opt['lr'] = 0.01
quartznet.setup_optimization(optim_config=DictConfig(new_opt))

# Point to the data we'll use for fine-tuning as the training set
quartznet.setup_training_data(train_data_config=params['model']['train_ds'])

# Point to the new validation data for fine-tuning
quartznet.setup_validation_data(val_data_config=params['model']['validation_ds'])

# And now we can create a PyTorch Lightning trainer and call `fit` again.
trainer = pl.Trainer(gpus=[0], max_epochs=5)
trainer.fit(quartznet)