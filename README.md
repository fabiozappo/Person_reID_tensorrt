# Person_reID_tensorrt
torch2trt is a PyTorch to TensorRT converter which utilizes the TensorRT Python API. The converter is
Easy to use - Convert modules with a single function call torch2trt

Easy to extend - Write your own layer converter in Python and register it with @tensorrt_converter

However in new models often happen to encounter known unsupported method.

in this repo we selected a few lightweight network that can be converted to tensorrt:

INSERIRE TABELLA CON COLONNE "MODEL", "NUMPARAMS", "TORCH FPS", "TRT FPS", "TORCH RANK 5", "TRT RANK 5"

mobilenet_v2
resnet50
resnet18
squeezenet1_1 

TODO: in tutte le reti va cambiato il collo, diminuendo il numero dei filtri convoluzionali. Come baseline semplicemente aggiungiamo roba ma è stupido allargare la rete e poi ristringerla nell'arco di pochi layer...


It would be cool to test some new networks like efficientnet b0 and mobilenet_v3, however trt parser is not still working.