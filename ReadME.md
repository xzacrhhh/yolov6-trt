# YOLOv6

# 下载模型
- ` you should download yolov6s.pt`

# 导出模型
- `cd YOLOv6`
- `bash export-yolov6.sh`

# 执行tensorRT
- `change your Makefile`
- `make run`

# 修改过的地方：
```python
# line 79 forward function in yolov6/models/effidehead.py 
# y = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
# bs, _, ny, nx = y.shape
# y = y.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
# modified into:-
y = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
bs, _, ny, nx = y.shape
bs = -1
ny = int(ny)
nx = int(nx)
y = y.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

# line 96 in yolov6/models/effidehead.py 
#  z.append(y.view(bs, -1, self.no))
# modified into：
z.append(y.view(bs, self.na * ny * nx, self.no))


# line 52 in deploy/ONNX/export_onnx.py
# torch.onnx.export(model, img, export_file, verbose=False, opset_version=12,
#                           training=torch.onnx.TrainingMode.EVAL,
#                           do_constant_folding=True,
#                           input_names=['images'],
#                           output_names=['outputs']
#                          )
# modified into:
torch.onnx.export(model, img, export_file, verbose=False, opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=['images'],
                          output_names=['outputs'],
                          dynamic_axes={'images': {0: 'batch'}, 
                                        'outputs': {0: 'batch'}  
                                        }
                         )

# Reference
- https://github.com/shouxieai/tensorRT_Pro
- https://github.com/meituan/YOLOv6