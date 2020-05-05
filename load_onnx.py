import onnx_pb2
import sys

f = open("./mnist-8.onnx", "rb")
raw = f.read()
f.close()


# load model
model = onnx_pb2.ModelProto()
model.ParseFromString(raw)

print(model)
