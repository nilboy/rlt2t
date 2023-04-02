import os
import time

while True:
    for name in os.listdir('/root/autodl-tmp/output-models/clm'):
        if "checkpoint-" in name:
            base_dir = os.path.join('/root/autodl-tmp/output-models/clm', name)
            if os.path.exists(os.path.join(base_dir, 'trainer_state.json')):
                op_file = os.path.join(base_dir, 'optimizer.pt')
                try:
                    os.system(f"rm {op_file}")
                    print(f'rm {op_file}')
                    os.system(f'mv {base_dir}/* /root/m-tmp/')
                except:
                    pass
    time.sleep(60 * 3)

