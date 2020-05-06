import onnx_pb2
import sys
from enum import Enum


def reshape(inputs, attributes, outputs):
    # prepare in & out
    in_tensor = inputs[0]
    shape = inputs[1]
    out_tensor = outputs[0]

    # set dimension
    for dim in shape.int64_data:
        out_tensor.dims.append(dim)

    # set data
    for data in in_tensor.float_data:
        out_tensor.float_data.append(data)

    # set data type
    out_tensor.data_type = "FLOAT"

    print(out_tensor)


def conv(inputs, attributes, outputs):
    print(attributes)



    in_tensor = inputs[0]
    weight = inputs[1]
    out_tensor = outputs[0]

    in_ch = in_tensor.dims[1]
    in_width = in_tensor.dims[2]
    in_height = in_tensor.dims[3]
    out_ch = weight.dims[0]
    kernel_width = weight.dims[2]
    kernel_height = weight.dims[3]

    height = in_height
    width = in_width

    out_tensor.data_type = "FLOAT"

    out_tensor.dims = [1] * 4
    out_tensor.dims[1] = out_ch
    out_tensor.dims[2] = width
    out_tensor.dims[3] = height

    # TODO : fix implementation. this is rough implementation.
    for c_out in range(out_ch):
        for h in range(height):
            for w in range(width):
                sum = 0
                for kh in range(kernel_height):
                    for kw in range(kernel_width):
                        for c_in in range(in_ch):
                            k = c_in * (kernel_height*kernel_width) + \
                                kernel_width*kh + kw
                            i = c_in * (height*width) + width * h + w
                            flag = 1
                            if h > height - kernel_height/2:
                                flag = 0
                            if w > width - kernel_width/2:
                                flag = 0
                            sum += flag * \
                                in_tensor.float_data[i] * weight.float_data[k]
                out_tensor.float_data.append(sum)


def add(inputs, attributes, outputs):
    in_tensor = inputs[0]
    bias = inputs[1]
    out_tensor = outputs[0]

    in_ch = in_tensor.dims[1]
    width = in_tensor.dims[2]
    height = in_tensor.dims[3]

    out_tensor.float_data = [0] * in_ch*width*height
    for i, d in enumerate(in_tensor.dims):
        out_tensor.dims.append(d)

    for c in range(in_ch):
        b = bias.float_data[c]
        for h in range(height):
            for w in range(width):
                i = width*height*c + h*width + w
                out_tensor.float_data[i] = b + in_tensor.float_data[i]


def relu(inputs, attributes, outputs):
    in_tensor = inputs[0]
    out_tensor = outputs[0]

    in_ch = in_tensor.dims[1]
    width = in_tensor.dims[2]
    height = in_tensor.dims[3]

    out_tensor.dims = [1] * 4
    out_tensor.dims[1] = in_ch
    out_tensor.dims[2] = width
    out_tensor.dims[3] = height
    out_tensor.float_data = [0] * in_ch*width*height
    out_tensor.data_type = "FLOAT"

    for c in range(in_ch):
        for h in range(height):
            for w in range(width):
                i = c*width*height + h*width + w
                x = in_tensor.float_data[i]
                if x > 0:
                    out_tensor.float_data[i] = x


def maxpool(inputs, attributes, outputs):
    in_tensor = inputs[0]
    out_tensor = outputs[0]

    kernel_width = attributes[0].ints[0]
    kernel_heigh = attributes[0].ints[1]

    stride_x = attributes[1].ints[0]
    stride_y = attributes[1].ints[1]

    in_ch = in_tensor.dims[1]
    width = in_tensor.dims[2]//stride_x
    height = in_tensor.dims[3]//stride_y

    out_tensor.dims = [1] * 4
    out_tensor.dims[1] = in_ch
    out_tensor.dims[2] = width
    out_tensor.dims[3] = height
    out_tensor.float_data = [0] * in_ch*width*height
    out_tensor.data_type = "FLOAT"

    for c in range(in_ch):
        for y in range(height):
            for x in range(width):
                index_out = c*height*width + y*width + x
                l = []
                for ky in range(kernel_heigh):
                    for kx in range(kernel_width):
                        index = c*height*width + (y+ky)*width + (x+ky)
                        d =in_tensor.float_data[index]
                        l.append(d)
                out_tensor.float_data[index_out] = max(l)


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
            self.variables[value.name] = Tensor()

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
            print(self.model.graph.node[n].op_type)
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
            outputs = []
            for o in self.model.graph.node[n].output:
                out = self.variables[o]
                out.name = o
                outputs.append(out)

            # run kernel
            print("-------")
            print(outputs[0])
            operator(inputs, attributes, outputs)
            print(outputs[0])

            for i in inputs:
                if i.name in self.var_ref_count:
                    self.var_ref_count[i.name] -= 1
                    if self.var_ref_count[i.name] == 0:
                        del self.variables[i.name]


class Tensor:
    def __init__(self):
        self.dims = []
        self.float_data = []
        self.name = ""
        self.data_type = ""

    def __repr__(self):
        s = ""
        for d in self.dims:
            s += "dims: " + str(d) + "\n"
        s += "data_type: " + str(self.data_type) + "\n"
        s += "name: " + self.name + "\n"
        # for d in self.float_data:
        #   s += "data: " + str(d) + "\n"

        return s


def main():

    # create input tensor
    input_tensor = Tensor()
    input_tensor.dims = [1, 1, 28, 28]
    input_tensor.float_data = []
    input_tensor.data_type = "FLOAT"
    for i in range(28):
        for j in range(28):
            input_tensor.float_data.append((i*28+j)/(28*28))
    input_tensor.name = 'Input3'

    # init runner
    runner = ONNXrunner('./mnist-8.onnx')
    runner.set_input([input_tensor])

    # debug
    # print(runner.output_name_to_id)
    # print(runner.input_tensor_name_to_id)
    # print(runner.initializer_name_to_id)
    # print(runner.variables)
    # print(runner.var_ref_count)

    runner.run()


if __name__ == '__main__':
    main()
