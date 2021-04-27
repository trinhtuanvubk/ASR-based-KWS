import util 
# from util import stream \
# from util.stream  import StreamAudio
import nemo.collections.asr as nemo_asr
model_path = "./lightning_logs/version_7/checkpoints/epoch=2-step=21404.ckpt"
asr_model = nemo_asr.models.EncDecCTCModel.load_from_checkpoint(checkpoint_path = model_path)#args = arg
def demo(model = asr_model) :
    
    #streaming 
    streaming = util.stream_audio(model)
    streaming.run()