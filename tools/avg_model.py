import torch

input_model_dir = [
                   "/home/jiangxinghua/project/rlt2t/output-models/nswa/epoch_14.ckpt.dir",
                   "/home/jiangxinghua/project/rlt2t/output-models/nswa/epoch_19.ckpt.dir",
                   "/home/jiangxinghua/project/rlt2t/output-models/nswa/epoch_24.ckpt.dir",
                   "/home/jiangxinghua/project/rlt2t/output-models/nswa/epoch_29.ckpt.dir"]

output_m = {}

import os
for base_model_dir in input_model_dir:
    m = torch.load(os.path.join(base_model_dir, 'pytorch_model.bin'))
    for k, v in m.items():
        if k not in output_m:
            output_m[k] = v/len(input_model_dir)
        else:
            output_m[k] += v/len(input_model_dir)

torch.save(output_m,
           '/home/jiangxinghua/project/rlt2t/output-models/nswa/merge/pytorch_model.bin')
