# ONNXruntimAndQuantization
try onnx runtime and quantization 

## try C api in linux

download onnxruntime.so.

https://github.com/microsoft/onnxruntime/releases

```
$wget https://github.com/microsoft/onnxruntime/releases/download/v1.
```

download cpp sample
```
$wget https://raw.githubusercontent.com/microsoft/onnxruntime/master/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp

```

build and run
```
g++ C_Api_Sample.cpp -I ./onnxruntime-linux-x64-1.2.0/include/ -L ./onnxruntime-linux-x64-1.2.0
export LD_LIBRARY_PATH=:./onnxruntime-linux-x64-1.2.0/lib:./onnxruntime-linux-x64-1.2.0/lib

./a.out
```

