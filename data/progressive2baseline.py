import cv2
import os

root_dir = "/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/wz/PASI/all_data/patient"

def progressive_to_baseline(path, file):
    tmpfile = 'tmp' + file
    print(tmpfile)
    img = cv2.imread(path + file)

    cv2.imwrite(path + tmpfile, img)
    os.remove(path + file)
    os.rename(path + tmpfile, path + file)

for dir_name in os.listdir(root_dir):

    file_dir = os.path.join(root_dir, dir_name)+'/'

    for file in os.listdir(file_dir):
        #ret = os.system('identify -verbose ' + file_dir + file + ' | grep Interlace')
        ret = os.popen('file ' + file_dir + file)
        lines = ret.readlines()
        if 'progressive,' in str(lines):
            print(file_dir + file)
            progressive_to_baseline(file_dir, file)