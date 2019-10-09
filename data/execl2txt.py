# 读取新数据label
import os
import re
import cv2
import openpyxl


# 去掉空格
root_dir = "/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/wz/PASI/all_data/patient"
def remove_space(path, file):

    new_filename = file.replace(' ', '')
    img = cv2.imread(path + file)
    cv2.imwrite(path + new_filename, img)
    os.remove(path + file)

for dir_name in os.listdir(root_dir):

    file_dir = os.path.join(root_dir, dir_name)+'/'

    for file in os.listdir(file_dir):

        if ' ' in file:
            print(file)
            remove_space(file_dir, file)


#
# # 1. 读取excel文档
# wb = openpyxl.load_workbook('/Users/liuyouru/我的论文/PASI论文/pasi.xlsx')
#
# # # 返回一个workbook对象， 有点类似于文件对象;
# # print(wb, type(wb))
#
#
#
# # 2. 在工作薄中取得工作表
# # print(wb.get_sheet_names())
# # 返回一个列表， 存储excel表中所有的sheet工作表;
# print(wb.sheetnames)
#
# # 返回一个worksheet对象， 返回当前的活动表;
# # print(wb.get_active_sheet())
# # print(wb.active)
#
# # 3. 获取工作表中， 单元格的信息
# # wb.get_sheet_by_name('Sheet1')
# sheet = wb['Sheet1']
#
# # 用一个3重嵌套的字典存储所有label
# # 先构建患者字典，患者字典中key为患 者名字，value为该患者患病部位字典
# # 患病部位字典中key为患病部位，value为指标字典
# # 指标字典中key为指标名，value为分值
#
# patient_dict = {}
#
# patient_list = sheet["A"][1:]
# for elem in patient_list:
#     if isinstance(elem.value, (str)):
#         patient_dict[elem.value] = {}
#
# part_dict = {"D": "头颈部", "I": "躯干部", "N": "上肢", "S": "下肢", "X": "PASI"}
#
# for part_index in part_dict.keys():
#
#     if part_index != "X":
#
#         part_name = part_dict[part_index]
#
#         part_area_index = chr(ord(part_index) + 1)
#         part_ery_index = chr(ord(part_index) + 2)
#         part_sca_index = chr(ord(part_index) + 3)
#         part_ind_index = chr(ord(part_index) + 4)
#
#         print(part_name)
#         print(part_area_index, part_ery_index, part_sca_index, part_ind_index)
#
#         part_area = sheet[part_area_index][1:]
#         part_ery = sheet[part_ery_index][1:]
#         part_sca = sheet[part_sca_index][1:]
#         part_ind = sheet[part_ind_index][1:]
#
#         for index, area in enumerate(part_area):
#
#             patient_name = patient_list[index].value
#
#             cur_area = part_area[index].value
#             cur_ery = part_ery[index].value
#             cur_sca = part_sca[index].value
#             cur_ind = part_ind[index].value
#
#             try:
#                 if isinstance(cur_area, int):
#                     cur_area = float(cur_area)
#                 elif isinstance(cur_area, str):
#                     cur_area = float(cur_area.strip())
#
#                 if isinstance(cur_ery, int):
#                     cur_ery = float(cur_ery)
#                 elif isinstance(cur_area, str):
#                     cur_ery = float(cur_ery.strip())
#
#                 if isinstance(cur_sca, int):
#                     cur_sca = float(cur_sca)
#                 elif isinstance(cur_area, str):
#                     cur_sca = float(cur_sca.strip())
#
#                 if isinstance(cur_ind, int):
#                     cur_ind = float(cur_ind)
#                 elif isinstance(cur_area, str):
#                     cur_ind = float(cur_ind.strip())
#
#                 if cur_area != 0:
#                     patient_name = patient_list[index].value
#
#                     patient_dict[patient_name][part_name] = {}
#                     patient_dict[patient_name][part_name]["area"] = cur_area
#                     patient_dict[patient_name][part_name]["erythema"] = cur_ery
#                     patient_dict[patient_name][part_name]["scale"] = cur_sca
#                     patient_dict[patient_name][part_name]["induration"] = cur_ind
#
#             except:
#                 continue
#
#     else:
#
#         part_pasi = sheet["X"][1:]
#
#         for index, pasi in enumerate(part_pasi):
#             patient_name = patient_list[index].value
#             patient_dict[patient_name]["pasi"] = part_pasi[index].value
#             # print(patient_dict[patient_name])
#
#
# for key in patient_dict.keys():
#     if "pasi" not in patient_dict[key]:
#         print(key)
#
#
# print(len(patient_dict.keys()))
#
# file_path = "/Volumes/Seagate Backup Plus Drive/pasi/all_data/patients"
# pattern = '躯干部|躯干|头部|头颈部|头面部|上肢|下肢'
#
# all_patient = os.listdir(file_path)
# train_patient = all_patient[:1400]
# valid_patient = all_patient[1400:]
#
# train_count = 0
# valid_count = 0
# with open("/Volumes/Seagate Backup Plus Drive/pasi/all_data/train_patients.txt", 'w') as f:
#
#     for patient in train_patient:
#
#         cur_path = os.path.join(file_path, patient)
#
#         for file in os.listdir(cur_path):
#
#             train_count+=1
#
#             body_part = re.search(pattern, file, flags=0)
#
#             if body_part:
#                 body_part = body_part.group()
#                 if patient in patient_dict and body_part in patient_dict[patient]:
#                     ery = patient_dict[patient][body_part]['erythema']
#                     sca = patient_dict[patient][body_part]['scale']
#                     ind = patient_dict[patient][body_part]['induration']
#
#                     try:
#                         ery = float(ery)
#                         sca = float(sca)
#                         ind = float(ind)
#                     except:
#                         print(os.path.join(patient, file))
#                         continue
#
#                     f.write(os.path.join(patient, file) + ',')
#                     f.write(str(patient_dict[patient][body_part]['erythema']) + ',')
#                     f.write(str(patient_dict[patient][body_part]['scale']) + ',')
#                     f.write(str(patient_dict[patient][body_part]['induration']) + '\n')
#
# with open("/Volumes/Seagate Backup Plus Drive/pasi/all_data/valid_patients.txt", 'w') as f:
#     for patient in valid_patient:
#
#         cur_path = os.path.join(file_path, patient)
#
#         for file in os.listdir(cur_path):
#
#             valid_count += 1
#
#             body_part = re.search(pattern, file, flags=0)
#
#             if body_part:
#                 body_part = body_part.group()
#                 if patient in patient_dict and body_part in patient_dict[patient]:
#
#                     ery = patient_dict[patient][body_part]['erythema']
#                     sca = patient_dict[patient][body_part]['scale']
#                     ind = patient_dict[patient][body_part]['induration']
#
#                     try:
#                         ery = float(ery)
#                         sca = float(sca)
#                         ind = float(ind)
#                     except:
#                         print(os.path.join(patient, file))
#                         continue
#
#                     f.write(os.path.join(patient, file) + ',')
#                     f.write(str(patient_dict[patient][body_part]['erythema']) + ',')
#                     f.write(str(patient_dict[patient][body_part]['scale']) + ',')
#                     f.write(str(patient_dict[patient][body_part]['induration']) + '\n')
#
# print(train_count, valid_count)
#


# 查找score不全的病人
# for line in open("/Volumes/Seagate Backup Plus Drive/pasi/all_data/train_patients.txt", 'r'):
#     try:
#         items = line.split(',')
#         items = [float(elem) for elem in items[1:]]
#     except:
#         print(line)