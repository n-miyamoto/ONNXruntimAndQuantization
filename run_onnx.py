import onnx_pb2
import sys
from enum import Enum


def reshape(inputs, attributes, outputs):
    print("TODO: impl reshape")
    # for i in inputs:
    #     print(i.name)


def conv(inputs, attributes, outputs):
    print("TODO: imple conv")
    # for i in inputs:
    #     print(i.name)


def add(inputs, attributes, outputs):
    print("TODO: imple add")
    #  for i in inputs:
    #      print(i.name)


def relu(inputs, attributes, output):
    print("TODO: impl relu")


def maxpool(inputs, attributes, output):
    print("TODO: impl maxpooling")


def matmul(inputs, attributes, output):
    print("TODO: impl matmal")


Operators = {
    'Reshape': reshape,
    'Conv': conv,
    'Add': add,
    'Relu': relu,
    'MaxPool': maxpool,
    'MatMul':  matmul
}


class ONNXrunner:
    def __init__(self, filepath):
        self.input_tensors = []
        self.load_onnx(filepath)
        self.schedule_execution()

    def load_onnx(self, filepath):
        f = open(filepath, "rb")
        rawdata = f.read()
        f.close()
        self.model = onnx_pb2.ModelProto()
        self.model.ParseFromString(rawdata)

        # create node name to id
        self.node_name_to_id = {}
        for i, node in enumerate(self.model.graph.node):
            self.node_name_to_id[node.name] = i

        self.input_name_to_id = {}
        for i, input in enumerate(self.model.graph.input):
            self.input_name_to_id[input.name] = i

        self.initializer_name_to_id = {}
        for i, initializer in enumerate(self.model.graph.initializer):
            self.initializer_name_to_id[initializer.name] = i

        self.output_name_to_id = {}
        for i, output in enumerate(self.model.graph.output):
            self.output_name_to_id[output.name] = i

        self.variables = {}
        for value in self.model.graph.value_info:
            self.variables[value.name] = tensor()

        self.var_ref_count = {}
        for node in self.model.graph.node:
            for input in node.input:
                if input in self.variables:
                    if input in self.var_ref_count:
                        self.var_ref_count[input] += 1
                    else:
                        self.var_ref_count[input] = 1

    def set_input(self, inputs):
        self.input_tensors = inputs
        self.input_tensor_name_to_id = {}
        for i, it in enumerate(self.input_tensors):
            self.input_tensor_name_to_id[it.name] = i

    def schedule_execution(self):
        # TODO : implement scheduling
        self.execution_queue = []
        for i, node_id in enumerate(self.model.graph.node):
            self.execution_queue.append(i)

    def run(self):
        for n in self.execution_queue:
            # set operator
            operator = Operators[self.model.graph.node[n].op_type]

            # set input tensors
            inputs = []
            for i in self.model.graph.node[n].input:
                if i in self.initializer_name_to_id:
                    init_id = self.initializer_name_to_id[i]
                    initializer = self.model.graph.initializer[init_id]
                    inputs.append(initializer)
                elif i in self.input_tensor_name_to_id:
                    id = self.input_tensor_name_to_id[i]
                    it = self.input_tensors[id]
                    inputs.append(it)
                elif i in self.variables:
                    inputs.append(self.variables[i])
                else:
                    assert False, i

            # set attributes
            attributes = self.model.graph.node[n].attribute

            # set output tensors
            outputs = self.model.graph.node[n].output

            # run kernel
            operator(inputs, attributes, outputs)

            for i in inputs:
                if i.name in self.var_ref_count:
                    self.var_ref_count[i.name] -= 1
                    if self.var_ref_count[i.name] == 0:
                        del self.variables[i.name]


class tensor:
    def __init__(self):
        self.dim = []
        self.float_data = []
        self.name = ""


def main():

    # create input tensor
    input_tensor = tensor()
    input_tensor.dims = [1, 1, 28, 28]
    input_tensor.float_data = []
    for i in range(28):
        for j in range(28):
            input_tensor.float_data.append((i*28+j)/(28*28))
    input_tensor.name = 'Input3'

    # init runner
    runner = ONNXrunner('./mnist-8.onnx')
    runner.set_input([input_tensor])

    # debug
    print(runner.output_name_to_id)
    print(runner.input_tensor_name_to_id)
    print(runner.initializer_name_to_id)
    print(runner.variables)
    print(runner.var_ref_count)

    runner.run()


if __name__ == '__main__':
    main()
