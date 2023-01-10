# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================


import numpy as np
import acl
import functools
import time

# error code
ACL_ERROR_NONE = 0

# memory malloc code
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEM_MALLOC_HUGE_ONLY = 1
ACL_MEM_MALLOC_NORMAL_ONLY = 2

# memory copy code
ACL_MEMCPY_HOST_TO_HOST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3

# format
ACL_DTYPE = {
    0: 'float32',
    1: 'float16',
    2: 'int8',
    3: 'int32',
    4: 'uint8',
    6: 'int16',
    7: 'uint16',
    8: 'uint32',
    9: 'int64',
    10: 'uint64',
    11: 'float64',
    12: 'bool',
}

NP_DTYPE = {
    0: np.float32,
    1: np.float16,
    2: np.int8,
    3: np.int32,
    4: np.uint8,
    6: np.int16,
    7: np.uint16,
    8: np.uint32,
    9: np.int64,
    10: np.uint64,
    11: np.float64,
    12: np.bool,
}

ACL_DTYPE_INDEX = {
    'float32': 0,
    'float16': 1,
    'int8': 2,
    'int32': 3,
    'uint8': 4,
    'int16': 6,
    'uint16': 7,
    'uint32': 8,
    'int64': 9,
    'uint64': 10,
    'float64': 11,
    'bool': 12,
}

def performance(model, input_data_shape=None, input_data=None, dynamic_shape=False, dynamic_dims=False, dims=None, device_id=0, loop=10, batch_size=1, output_data_shape=None):
    if loop == 0:
        raise RuntimeError('the loop must be > 0')
    if model is None:
        raise RuntimeError('please specify the om model path')
    if not input_data_shape and dynamic_shape:
        raise RuntimeError('please specify the input shape of the dynamic shape model')
    if dynamic_dims and len(dims)==0:
         raise RuntimeError('please specify the dims shape of the multi shape model')
    if batch_size == 0:
        raise RuntimeError('the batch_size must be > 0')
    init_acl(device_id=device_id)
    if dynamic_shape:
        input_data_shape = list(input_data_shape.values())
        input_data_shape = [functools.reduce(lambda x, y: x * y, item) for item in input_data_shape]
        model = AclNet(model_path=model, device_id=device_id, input_data_shape=input_data_shape, dynamic=dynamic_shape, output_data_shape = output_data_shape)
    else:
        model = AclNet(model_path=model, device_id=device_id)
    

    if len(input_data) > 0:
        _input_data = []
        idx=0
        for key in model.model_input_shape.keys():
            if key not in input_data.keys():
                raise RuntimeError('the model has not input named ' + key)
            now_input_data = np.fromfile(input_data[key], dtype=NP_DTYPE[model.model_input_data_type[idx]])     # static model
            now_input_data = _input_data.reshape(input_data_shape[key])
            _input_data.append(now_input_data)
            idx+=1

    else:
        _input_data = []
        idx=0
        idx2=0
        count = 0
        dims_input = {}
        real_dims=[]
        for key2 in model.model_input_shape.keys():
            if not dynamic_dims and len(dims)==0 and not dynamic_shape:
                tmp = np.zeros(model.model_input_shape[key2]).astype(ACL_DTYPE[model.model_input_data_type[idx]])     # static model
                _input_data.append(tmp)
                idx+=1
            elif dynamic_shape:
                if key2 not in input_data_shape.keys():
                    raise RuntimeError('the model has not input named ' + key2)
                tmp = np.zeros(input_data_shape[key2]).astype(ACL_DTYPE[model.model_input_data_type[idx]])
                _input_data.append(tmp)
                idx+=1
            elif dynamic_dims or model.ascend_mbatch_shape_data:
                #get dims  
                
                if key2 != "ascend_mbatch_shape_data":
                    for dim_idx in range(len(model.model_input_shape[key2])):
                        count+=1
                        if model.model_input_shape[key2][dim_idx]==-1:
                            model.model_input_shape[key2][dim_idx]=dims[idx2]
                            idx2+=1
                    real_dims.extend(model.model_input_shape[key2])

        if dynamic_dims or model.ascend_mbatch_shape_data:
            dims_input['dimCount'] = count
            dims_input['name'] = ''
            dims_input['dims'] = real_dims

    sum_t = 0
    i = 0 
    for i in range(loop+1):
        out, exe_t = model(_input_data, dims = dims_input)
        if i > 0:
            sum_t += exe_t
            print("exe_t:", exe_t)
    ave_t = sum_t/loop
    print("ave_t:", ave_t, "Throughoutput:", 1000 * batch_size / ave_t)


def check_ret(message, ret):
    if ret != ACL_ERROR_NONE:
        raise Exception("{} failed ret = {}".format(message, ret))


def check_input_type(input_type, model_input_type):
    for i in range(len(input_type)):
        if ACL_DTYPE_INDEX.get(input_type[i]) != model_input_type[i]:
            raise Exception("the input {} input data type is {}, actual need model input type is {}".format(i, input_type[i],
                            ACL_DTYPE.get(model_input_type[i])))


def init_acl(device_id, config_path=None):
    if config_path:
        ret = acl.init(config_path)
    else:
        ret = acl.init()
    check_ret('acl.init', ret)
    ret = acl.rt.set_device(device_id)
    check_ret('acl.rt.set_device', ret)
    context, ret = acl.rt.create_context(device_id)
    check_ret('acl.rt.create_context', ret)


def release_acl(device_id):
    context, ret = acl.rt.get_context()
    check_ret('acl.rt.get_context', ret)
    ret = acl.rt.destroy_context(context)
    check_ret('acl.rt.destory_context', ret)
    ret = acl.rt.reset_device(device_id)
    check_ret('acl.rt.reset_device', ret)
    ret = acl.finalize()
    check_ret('acl.finalize', ret)


class AclNet(object):
    def __init__(self, model_path, device_id, check_input=False, input_data_shape=None, output_data_shape=None, dynamic=None, out_to_in={}, out_idx=None, pin_input=None):
        self.check_input = check_input
        self.dynamic = False
        self.dynamic_input = dynamic
        self.device_id = device_id
        self.model_path = model_path
        self.out_to_in = out_to_in
        self.in_to_out = dict(zip(out_to_in.values(), out_to_in.keys())) if out_to_in else {}
        self.model_id = None
        # if self.ascend_mbatch_shape_data = True, the model is static with multi input shape
        self.ascend_mbatch_shape_data = False
        self.input_dims = {'dimCount': 0, 'name': '', 'dims': []}
        self.input_data_type = []
        self.input_data_rank = []
        self.model_input_data_type = []
        self.model_input_data_format = []
        self.model_output_data_type = []
        self.model_input_max_mem = []
        self.output_data_shape = output_data_shape
        self.input_data_shape = input_data_shape
        self.model_input_shape = {}
        self.output_shape = []
        self.buffer_method = {
            "in": acl.mdl.get_input_size_by_index,
            "out": acl.mdl.get_output_size_by_index,
            "outhost": acl.mdl.get_output_size_by_index
        }

        self.input_data = []
        self.output_data = []
        self.output_data_host = []
        self.model_desc = None
        self.load_input_dataset = None
        self.load_output_dataset = None
        self.input_size = None
        self.output_size = None
        self.exe_t = 0
        self.bytes_data_out = []
        self.bytes_data_out_ptr = []
        self._init_resource()
        self.out_idx = out_idx if out_idx is not None else list(set([i for i in range(self.output_size)]) ^ set(list(out_to_in.keys())))
        self.pin_input = pin_input

    def __call__(self, ori_data, first_step=True, end_step=True, dims=None):
        return self.forward(ori_data, first_step, end_step, dims)
  
    def release_model(self):
        ret = acl.mdl.unload(self.model_id)
        check_ret("acl.mdl.unload", ret)
        if self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None
        self._release_data_buffer()

    def __del__(self):
        pass

    def _release_data_buffer(self):
        while self.input_data:
            item = self.input_data.pop()
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)

        start = self.output_size - 1
        while self.output_data:
            item = self.output_data.pop()
            if start not in self.out_to_in.keys():
                ret = acl.rt.free(item["buffer"])
                check_ret("acl.rt.free", ret)
            start -= 1

        # while self.output_data_host:
        #     item = self.output_data_host.pop()
        #     ret = acl.rt.free_host(item["buffer"])
        #     check_ret("acl.rt.free_host", ret)

    def _init_resource(self):
        # load_model
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        check_ret("acl.mdl.load_from_file", ret)

        self.model_desc = acl.mdl.create_desc()
        self._get_model_info()

    def _get_model_info(self):
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("acl.mdl.get_desc", ret)
        self.input_size = acl.mdl.get_num_inputs(self.model_desc)
        # get the input format, data_type and get the model static or not
        for i in range(self.input_size):
            size = self.buffer_method['in'](self.model_desc, i)
            self.model_input_max_mem.append(size)
            data_type = acl.mdl.get_input_data_type(self.model_desc, i)
            self.model_input_data_type.append(data_type)
            data_format = acl.mdl.get_input_format(self.model_desc, i)
            self.model_input_data_format.append(data_format)
            dims_input, ret = acl.mdl.get_input_dims(self.model_desc, i)
            self.model_input_shape[dims_input['name']] = dims_input['dims']
            self.input_dims['dimCount'] += dims_input['dimCount']
            # check if the model has ascend_mbatch_shape_data

            if i == self.input_size - 1 and dims_input["name"] == "ascend_mbatch_shape_data":
                self.dynamic = False
                self.ascend_mbatch_shape_data = True
            elif -1 in dims_input["dims"]:
                self.dynamic = True
        if self.ascend_mbatch_shape_data:
            self.input_dims['dimCount'] -= 1
        self.output_size = acl.mdl.get_num_outputs(self.model_desc)
        for j in range(self.output_size):
            data_type = acl.mdl.get_output_data_type(self.model_desc, j)
            self.model_output_data_type.append(data_type)
            dims_output, ret = acl.mdl.get_output_dims(self.model_desc, j)
            if -1 in dims_output["dims"]:
                self.dynamic = True
        if self.dynamic_input is not None:
            self.dynamic = self.dynamic_input

        if self.output_data_shape is not None and not isinstance(self.output_data_shape, (list, tuple)):
            self.output_data_shape = [self.output_data_shape] * self.output_size
        if self.output_data_shape is None and self.dynamic:
            self.output_data_shape = [50000000] * self.output_size
        if not self.dynamic:
            self._prepare_data_buffer_in()
            self._prepare_data_buffer_out()
            self._prepare_data_buffer_host()
        else:
            input_data_mem = None
            if self.input_data_shape is None and 0 not in self.model_input_max_mem:
                input_data_mem = self.model_input_max_mem
            elif self.input_data_shape is not None:
                input_data_mem = [AclNet.gen_data_size(self.input_data_shape[i], dtype=ACL_DTYPE.get(
                    self.model_input_data_type[i])) for i in range(self.input_size)]
            elif self.input_data_shape is None:
                raise RuntimeError('the model is transformed using -1 input, please specify the input_data_shape or transform the model by using min~max range!')
            if input_data_mem is not None:
                self._prepare_data_buffer_in(input_data_mem)
                self._prepare_data_buffer_out(self.output_data_shape)
                self._prepare_data_buffer_host(self.output_data_shape)

    def _gen_data_buffer(self, size, des, data=None):
        func = self.buffer_method[des]
        for i in range(size):
            if not self.dynamic:
                temp_buffer_size = func(self.model_desc, i)
            else:
                if des == "in":
                    temp_buffer_size = data[i]
                elif des == "out":
                    temp_buffer_size = AclNet.gen_data_size(
                        data[i], dtype=ACL_DTYPE.get(self.model_output_data_type[i]))

            temp_buffer, ret = acl.rt.malloc(
                temp_buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret("acl.rt.malloc", ret)
            acl.rt.memset(temp_buffer, temp_buffer_size, 0, temp_buffer_size)
            if des == "in":
                self.input_data.append({"buffer": temp_buffer,
                                        "size": temp_buffer_size})
            elif des == "out":
                self.output_data.append({"buffer": temp_buffer,
                                         "size": temp_buffer_size})

    def _gen_dataset_output_host(self, size, des, data=None):
        func = self.buffer_method[des]
        for i in range(size):
            if not self.dynamic:
                temp_buffer_size = func(self.model_desc, i)
            else:
                temp_buffer_size = AclNet.gen_data_size(
                    data[i], ACL_DTYPE.get(self.model_output_data_type[i]))
            bytes_data = np.empty((temp_buffer_size, ), dtype=np.bool)
            bytes_data_ptr = bytes_data.__array_interface__['data'][0]
            self.bytes_data_out.append(bytes_data)
            self.bytes_data_out_ptr.append(bytes_data_ptr)

            self.output_data_host.append({"buffer": bytes_data_ptr,
                                          "size": temp_buffer_size})

    def _data_interaction(self, dataset, policy=ACL_MEMCPY_HOST_TO_DEVICE, output_shape=None, first_step=True, end_step=False):

        temp_data_buffer = self.input_data \
            if policy == ACL_MEMCPY_HOST_TO_DEVICE \
            else self.output_data
        if len(dataset) == 0 and policy == ACL_MEMCPY_DEVICE_TO_HOST:
            dataset = self.output_data_host
        for i in range(len(dataset)):
            if policy == ACL_MEMCPY_HOST_TO_DEVICE:
                if first_step:
                    if not dataset[i].flags['C_CONTIGUOUS']:
                        dataset[i] = np.ascontiguousarray(dataset[i])
                    ptr = dataset[i].__array_interface__['data'][0]
                    malloc_size = dataset[i].size * dataset[i].itemsize
                    ret = acl.rt.memcpy(
                        temp_data_buffer[i]["buffer"], malloc_size, ptr, malloc_size, policy)
                    check_ret("acl.rt.memcpy", ret)
                else:
                    if i in self.out_to_in.values():
                        self.input_data[i] = self.output_data[self.in_to_out[i]]
                    elif i in self.pin_input:
                        continue
                    else:
                        if not dataset[i].flags['C_CONTIGUOUS']:
                            dataset[i] = np.ascontiguousarray(dataset[i])
                        ptr = dataset[i].__array_interface__['data'][0]
                        malloc_size = dataset[i].size * dataset[i].itemsize
                        ret = acl.rt.memcpy(
                            temp_data_buffer[i]["buffer"], malloc_size, ptr, malloc_size, policy)
                        check_ret("acl.rt.memcpy", ret)
            else:
                if end_step or i in self.out_idx:
                    ptr = dataset[i]["buffer"]
                    malloc_size = functools.reduce(lambda x, y: x * y, output_shape[i])*np.dtype(
                        ACL_DTYPE.get(self.model_output_data_type[i])).itemsize
                    ret = acl.rt.memcpy(
                        ptr, malloc_size, temp_data_buffer[i]["buffer"], malloc_size, policy)
                    check_ret("acl.rt.memcpy", ret)

    def _gen_dataset(self, type_str="input", input_shapes=None, first_step=True, end_step=False):
        dataset = acl.mdl.create_dataset()
        temp_dataset = None
        if type_str == "in":
            self.load_input_dataset = dataset
            temp_dataset = self.input_data
        else:
            self.load_output_dataset = dataset
            temp_dataset = self.output_data

        for i, item in enumerate(temp_dataset):
            data = acl.create_data_buffer(item["buffer"], item["size"])
            if data is None:
                ret = acl.destroy_data_buffer(dataset)
                check_ret("acl.destroy_data_buffer", ret)

            _, ret = acl.mdl.add_dataset_buffer(dataset, data)
            if ret != ACL_ERROR_NONE:
                ret = acl.destroy_data_buffer(dataset)
                check_ret("acl.destroy_data_buffer", ret)

            if type_str == "in" and not self.ascend_mbatch_shape_data:
                # set dynamic dataset tensor desc
                input_shape = input_shapes[i]
                input_desc = acl.create_tensor_desc(self.model_input_data_type[i], input_shape,
                                                    self.model_input_data_format[i])
                dataset, ret = acl.mdl.set_dataset_tensor_desc(
                    dataset, input_desc, i)
                if ret != ACL_ERROR_NONE:
                    ret = acl.destroy_data_buffer(dataset)
                    check_ret("acl.destroy_data_buffer", ret)

    def _data_from_host_to_device(self, images, input_shapes, first_step, end_step):
        self._data_interaction(dataset=images, policy=ACL_MEMCPY_HOST_TO_DEVICE, output_shape=None, first_step=first_step, end_step=end_step)
        self._gen_dataset(type_str="in", input_shapes=input_shapes, first_step=first_step, end_step=end_step)
        self._gen_dataset("out")

    def _data_from_device_to_host(self, input_data, output_shape, first_step, end_step):
        res = []
        self._data_interaction(dataset=res, policy=ACL_MEMCPY_DEVICE_TO_HOST, output_shape=output_shape, first_step=first_step, end_step=end_step)


    def _get_output_shape(self):
        output_shape = []
        num = acl.mdl.get_dataset_num_buffers(self.load_output_dataset)
        for output_index in range(num):
            if self.dynamic:
                outpu_desc = acl.mdl.get_dataset_tensor_desc(
                    self.load_output_dataset, output_index)
                temp_output_shape = []
                dim_nums = acl.get_tensor_desc_num_dims(outpu_desc)
                for i in range(dim_nums):
                    dim, ret = acl.get_tensor_desc_dim_v2(outpu_desc, i)
                    temp_output_shape.append(dim)
                output_shape.append(temp_output_shape)
            else:
                dims, ret = acl.mdl.get_cur_output_dims(
                    self.model_desc, output_index)
                data_shape = dims.get("dims")
                output_shape.append(data_shape)
        self.output_shape = output_shape
        return output_shape

    def _destroy_databuffer(self):
        for dataset in [self.load_input_dataset, self.load_output_dataset]:
            if not dataset:
                continue
            num = acl.mdl.get_dataset_num_buffers(dataset)
            for i in range(num):
                data_buf = acl.mdl.get_dataset_buffer(dataset, i)
                if data_buf:
                    ret = acl.destroy_data_buffer(data_buf)
                    check_ret("acl.destroy_data_buffer", ret)
            ret = acl.mdl.destroy_dataset(dataset)
            check_ret("acl.mdl.destroy_dataset", ret)

    def _prepare_data_buffer_in(self, input_data=None):
        self._gen_data_buffer(self.input_size, des="in", data=input_data)

    def _prepare_data_buffer_out(self, input_data=None):
        self._gen_data_buffer(self.output_size, des="out", data=input_data)

    def _prepare_data_buffer_host(self, input_data=None):
        self._gen_dataset_output_host(
            self.output_size, des="outhost", data=input_data)
    
    def _update_input_dims(input_shapes):
        dims = []
        for sh in input_shapes:
            dims.extend(sh)
        self.input_dims.update({'dims': dims})

    def forward(self, input_data, first_step, end_step, dims=None):
        if not isinstance(input_data, (list, tuple)):
            input_data = [input_data]
        input_shapes = []
        if first_step:
            input_shapes = [list(data.shape) for data in input_data]
        else:
            for idx in range(self.input_size):
                if isinstance(input_data[idx], list):
                    input_shapes.append(input_data[idx])
                else:
                    input_shapes.append(list(input_data[idx].shape))
        if self.check_input:
            self.input_data_type = []
            for data in input_data:
                self.input_data_type.append(str(data.dtype))
                self.input_data_rank.append(len(data.shape))
            check_input_type(self.input_data_type, self.model_input_data_type)

        self._data_from_host_to_device(input_data, input_shapes, first_step, end_step)

        if self.ascend_mbatch_shape_data:
            if dims is not None:
                self.input_dims = dims
            else:
                self._update_input_dims(input_shapes)
            index, ret = acl.mdl.get_input_index_by_name(
                self.model_desc, 'ascend_mbatch_shape_data')
            ret = acl.mdl.set_input_dynamic_dims(
                self.model_id, self.load_input_dataset, index, self.input_dims)
            check_ret("acl.mdl.set_input_dynamic_dims", ret)
        st = time.time()
        ret = acl.mdl.execute(
            self.model_id, self.load_input_dataset, self.load_output_dataset)
        self.exe_t = time.time() - st
        check_ret("acl.mdl.execute", ret)
        # get output shape
        output_shape = self._get_output_shape()
        self._destroy_databuffer()
        self._data_from_device_to_host(
            input_data=input_data, output_shape=output_shape, first_step=first_step, end_step=end_step)
        if end_step:
            output = self.get_result_end_step(self.output_data_host, input_data,output_shape)
        else:
            output = self.get_result(self.output_data_host, input_data, output_shape)
        return output

    def get_final_result(self,):
        self._data_from_device_to_host(
            input_data=[], output_shape=self.output_shape, first_step=False, end_step=True
        )
        output = self.get_result_end_step(self.output_data_host, [], self.output_shape)
        return output

    def get_result(self, output_data, data, output_shape):
        dataset = []
        for i in range(len(output_data)):
            # fix dynamic batch size
            if i in self.out_idx:
                data_type = self.model_output_data_type[i]
                data_len = functools.reduce(lambda x, y: x * y, output_shape[i])
                ftype = np.dtype(ACL_DTYPE.get(data_type))
                data = self.bytes_data_out[i]
                np_array = np.frombuffer(
                    data, dtype=ftype, count=data_len)
                np_array = np_array.reshape(output_shape[i])
            else:
                np_array = output_shape[i]
            dataset.append(np_array)
        return dataset, self.exe_t * 1000

    def get_result_end_step(self, output_data, data, output_shape):
        dataset = []
        for i in range(len(output_data)):
            # fix dynamic batch size
            data_type = self.model_output_data_type[i]
            data_len = functools.reduce(lambda x, y: x * y, output_shape[i])
            ftype = np.dtype(ACL_DTYPE.get(data_type))
            data = self.bytes_data_out[i]
            np_array = np.frombuffer(
                data, dtype=ftype, count=data_len)
            np_array = np_array.reshape(output_shape[i])
            dataset.append(np_array)
        return dataset, self.exe_t * 1000
    @staticmethod
    def gen_data_size(size, dtype):
        dtype = np.dtype(dtype)
        return int(size * dtype.itemsize)
