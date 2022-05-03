#### To convert ONNX model to IR format use a command like:
```bash
mo --input_model your_ONNX_model -o output_directory_for_IR_model --data_type FP32_or_FP16_or_FP8
```
#### In our case we used the following commands:
FP32:
```bash
mo --input_model D:\Users\amira\openvino_env\Lib\site-packages\openvino\model_zoo\models\group_project\enet_b0_8\enet_b0_8.onnx -o D:\Users\amira\openvino_env\Lib\site-packages\openvino\model_zoo\models\group_project\enet_b2_8 --data_type FP32
```
FP16:
```bash
mo --input_model D:\Users\amira\openvino_env\Lib\site-packages\openvino\model_zoo\models\group_project\enet_b0_8\enet_b0_8.onnx -o D:\Users\amira\openvino_env\Lib\site-packages\openvino\model_zoo\models\group_project\enet_b2_8 --data_type FP16
```
