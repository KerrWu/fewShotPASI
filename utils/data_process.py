import os
import shutil
import numpy as np
from PIL import Image

root_dir = "/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/wz/PASI/all_data/source_data/分类皮肤数据"


def not_corrupt(path, file):
    absolute_path = os.path.join(path, file)
    try:
        img = Image.open(absolute_path)
    except IOError:
        print("IO error", absolute_path)
    try:
        img = np.array(img, dtype=np.float32)
    except:
        print('corrupt img', absolute_path)
        return False

    return True


psoriasis_path = "/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/wz/PASI/all_data/source_data/psoriasis"
other_path = "/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/wz/PASI/all_data/source_data/other"

for root, dir_name, files in os.walk(root_dir):
    for file in files:
        if os.path.splitext(file)[-1].lower() == ".jpg" and not_corrupt(root, file):
            cur_filepath = os.path.join(root, file)

            if 'psoriasis' in root:
                shutil.copyfile(cur_filepath, os.path.join(psoriasis_path, file))

            else:
                shutil.copyfile(cur_filepath, os.path.join(other_path, file))

