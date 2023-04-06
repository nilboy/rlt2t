import torch

input_model_dir = [
                   "/root/autodl-tmp/output-models/t2t/1/epoch_8.ckpt.dir",
                   "/root/autodl-tmp/output-models/t2t/1/epoch_11.ckpt.dir",
                   "/root/autodl-tmp/output-models/t2t/1/epoch_14.ckpt.dir",
                   "/root/autodl-tmp/output-models/t2t/1/epoch_17.ckpt.dir"]

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
           '/root/autodl-tmp/output-models/t2t/merge/1/pytorch_model.bin')
