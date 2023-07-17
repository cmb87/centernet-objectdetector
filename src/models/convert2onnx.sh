#!/bin/bash
python -m tf2onnx.convert --input models/frozen.pb  --output models/model.onnx --inputs x:0 --outputs Identity:0,Identity_1:0,Identity_2:0,Identity_3:0