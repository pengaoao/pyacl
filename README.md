###  安装

首先clone

```
git clone https://github.com/pengaoao/pyacl.git
cd pyacl
```

安装

```
pip3 install .
```

## 使用方法

#### 实际应用推理

**静态shape或分档**(分档只支持--dynamic_dims参数转出的模型)推理请查看infer_sample中的infer_static_dydims函数；**动态shape**推理请查看infer_sample中的infer_dynamic函数。

**循环推理场景**：

假设对于常见的语音decode场景，循环推理中，第3个输入每次都不变，第1,2个输入等于上次推理第2,3个输出，out0每次都输出用作判断跳出循环条件，原代码如下：

```
net = AclNet(model_path="xx.om", device_id = 0)
max_iter = 1000
for i in range(max_iter):
    out, exe_t = net([i0, i1, i2])
    i0 = out[1]
    i1 = out[2]
    if out[0].shape[0]==100: #截止条件
    	break

#后续out继续后处理
```

我们引入参数，将循环推理中间过程中H2D和D2H的次数减少，来减少端到端的推理时间，out_to_in为输出输入对应关系，out_idx为每次迭代都输出用于判断跳出循环的输出的索引，若不存在判断条件，该参数可不写，pin_input表示每次迭代输入都不变的输入的索引，若不存在，可不写，对应的优化代码为：

```
net = AclNet(model_path="xx.om", device_id = 0, out_to_in={1:0, 2:1}, out_idx=[0], pin_input=[2])
max_iter = 1000
finish_flag = False
for i in range(max_iter-1):
	if i==0:
        out = net([i0, i1, i2], fisrt_step=True, end_step=False)
        if out[0].shape[0]==100:
        	finish_flag = True
            break
    else:
    	out, exe_t = net([out[0], out[1], i2, fisrt_step=True, end_step=False)
        if out[0].shape[0]==100:
        	finish_flag = True
            break
if finish_flag:
    out, exe_t = net.get_final_result()
else:
	out, exe_t = net([out[0], out[1], i2], fisrt_step=True, end_step=True)
#后续out继续后处理    
```



 #### 纯推理性能测试

默认情况下，构造全为0的数据送入模型推理，打印结果为多次执行时间（ms）和平均执行时间（ms，不计入第一次推理）。

##### 静态shape（batch_size默认1，loop默认10）：

```bash
python3 -m pyacl resnet50_v1.om --perf --batch_size 1 --loop 10
```

##### 动态Batch（不支持，可使用动态Dims）

##### 动态HW宽高（不支持，可使用动态Dims）

##### 动态Dims

例如转模型有1,3,224,224和 16,3,640,640两个档位，channel维度3为固定值，以设置档位1,3,224,224为例。

```bash
python3 -m pyacl resnet50_v1_multi.om --perf --dims 1,224,224
```

##### 动态Shape

以ATC设置speech:[-1,-1,80];speech_lengths:[-1]，建议通过input_data_shape参数设置对应输入的内存大小，通过output_data_shape参数设置对应输出内存大小。

```bash
python3 -m pyacl asr.om --dynamic_shape --perf --input_data_shape "speech:1,100,80" "speech_lengths:1" --output_data_shape 1000000,100000
```




