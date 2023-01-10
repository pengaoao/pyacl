import argparse
import sys

import onnx     # type: ignore
import pyacl
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_model', help='Input ONNX model')   #dynamic_shape=False, dynamic_dims=False
    parser.add_argument('--dynamic_shape', help='True if the model is dynamic.',
                        action='store_true')
    parser.add_argument('--perf', help='True if test the model performance.',
                    action='store_true')
    parser.add_argument('--dynamic_dims', help='True if the modle has multi static shape',
                        action='store_true')
    parser.add_argument(
        '--input_data_shape', help='The manually-set static input shape, useful when the input shape is dynamic. The value should be "input_name:dim0,dim1,...,dimN" or simply "dim0,dim1,...,dimN" when there is only one input, for example, "data:1,3,224,224" or "1,3,224,224". Note: you might want to use some visualization tools like netron to make sure what the input name and dimension ordering (NCHW or NHWC) is.', type=str, nargs='+')
    parser.add_argument(
        '--input_data', help='input data, The value should be "input_name1:xxx1.bin"  "input_name2:xxx2.bin ...", input data should be a binary data file.', type=str)
    parser.add_argument(
        '--output_data_shape', help='output_data_shape, The value should be "shape1,shape2" ', type=str)
    parser.add_argument(
        '--dims', help='dims, The value should be "dim1,dim2", no need for static dim', type=str)
    parser.add_argument(
        '--device_id', help='npu device id', type=int, default=0)
    parser.add_argument(
        '--loop', help='loop, infer times', type=int, default=10)
    parser.add_argument(
        '--batch_size', help='batch_size, model batch size', type=int, default=1)
    args = parser.parse_args()

    if args.dynamic_shape and args.input_data_shape is None:
        raise RuntimeError(
            'Please pass "--input-shape" argument for generating random input and infering. Run "python3 -m pyacl -h" for details.')
    input_shapes = dict()
    if args.input_data_shape is not None:
        for x in args.input_data_shape:
            if ':' not in x:
                input_shapes[None] = list(map(int, x.split(',')))
            else:
                pieces = x.split(':')
                # for the input name like input:0
                name, shape = ':'.join(
                    pieces[:-1]), list(map(int, pieces[-1].split(',')))
                input_shapes.update({name: shape})

    input_data_paths = dict()
    if args.input_data is not None:
        for x in args.input_data:
            pieces = x.split(':')
            name, data = ':'.join(pieces[:-1]), pieces[-1]
            input_data_paths.update({name: data})

    dynamic_shape = False
    if args.dynamic_shape:
        dynamic_shape = True

    dynamic_dims = False
    if args.dynamic_dims:
        dynamic_dims = True

    dims = []
    if args.dims:
        dims = list(map(int, args.dims.split(',')))

    output_data_shape = []
    if args.output_data_shape:
        output_data_shape = list(map(int, args.output_data_shape.split(',')))

    if args.perf:
        #dynamic_dims=False, dims=None, device_id=0, loop=10, batch_size=1, output_data_shape=None
        pyacl.performance(
            args.input_model,
            input_data_shape=input_shapes,
            input_data=input_data_paths,
            dynamic_shape=dynamic_shape,
            dynamic_dims = dynamic_dims,
            dims = dims,
            device_id = args.device_id,
            loop = args.loop,
            batch_size = args.batch_size,
            output_data_shape = output_data_shape
            )


if __name__ == '__main__':
    main()
