# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from pyacl.acl_infer import AclNet, init_acl, release_acl
import acl
import numpy as np

def infer_dynamic(device_id):
    output_data_shape = [1000000, 1000000]#为输出shape的乘积，一般设置为最大值的乘积，例如输出shape为，[[1~100, 30], [1~500,10]], 则设置该参数为[3000, 5000]
    input_data_shape = [100000, 100000] #为输入shape的乘积，一般设置为最大值的乘积，例如输入shape为，[[1~1000, 10], [1~500,10]], 则设置该参数为[10000, 5000]
    net = AclNet(model_path="xx.om", output_data_shape=output_data_shape, input_data_shape=input_data_shape, device_id=device_id)
    input1 = np.random.random((1,3,224,224)).astype("float32")
    input2 = np.random.random((1,80)).astype("float32") 
    #prepare a shape biger than the real shape to for the memory malloc
    output_data, exe_time = net([input1, input2])
    net.release_model()
    release_acl(device_id)

def infer_static_dydims(device_id):
    input1 = np.random.random((1,3,224,224)).astype("float32")
    input2 = np.random.random((1,80)).astype("float32")
    net = AclNet(model_path="xx.om", device_id = device_id)
    output_data, exe_time = net([input1, input2])
    net.release_model()
    release_acl(device_id)


def infer_two_models(device_id):
    net1 = AclNet(model_path="xx.om", device_id = device_id)
    net2 = AclNet(model_path="xx2.om", device_id = device_id)
    input1 = np.random.random((1,3,224,224)).astype("float32")
    input2 = np.random.random((1,80)).astype("float32")
    output_data, exe_time1 = net1([input1, input2])
    #将模型1的输出给模型2作为输入
    output_data2, exe_time2 = net2(output_data)
    net1.release_model()
    net2.release_model()
    release_acl(device_id)

if __name__ == '__main__':
    device_id = 0
    init_acl(device_id)#即使是推理多个模型也只需初始化一次
    # 推理静态shape或分档模型
    infer_static_dydims(device_id)
    #推理动态shape模型
    #infer_dynamic(device_id)
    #串行多个模型
    #infer_two_models(device_id)





