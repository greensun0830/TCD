import re

file_list = [str(i)+'.txt' for i in range(1, 31)]

pattern1 = re.compile(r'(HR-)([\.\d]+)')
pattern2 = re.compile(r'(HR1-)([\.\d]+)')
pattern3 = re.compile(r'(NDCG-)([\.\d]+)')
hr_list = []
hr1_list = []
ndcg_list = []
average_hr = []
average_pureSVD_hr = []
average_advSVD_hr = []
average_randomSVD_hr =[]
average_coSVD_hr = []
average_hr1 = []
average_pureSVD_hr1 = []
average_advSVD_hr1 = []
average_randomSVD_hr1 =[]
average_coSVD_hr1 = []
average_ndcg = []
average_pureSVD_ndcg = []
average_advSVD_ndcg = []
average_randomSVD_ndcg =[]
average_coSVD_ndcg = []
random_hr = []
random_pureSVD_hr = []
random_advSVD_hr = []
random_randomSVD_hr =[]
random_coSVD_hr = []
random_hr1 = []
random_pureSVD_hr1 = []
random_advSVD_hr1 = []
random_randomSVD_hr1 =[]
random_coSVD_hr1 = []
random_ndcg = []
random_pureSVD_ndcg = []
random_advSVD_ndcg = []
random_randomSVD_ndcg =[]
random_coSVD_ndcg = []
aush_hr = []
aush_pureSVD_hr = []
aush_advSVD_hr = []
aush_randomSVD_hr =[]
aush_coSVD_hr = []
aush_hr1 = []
aush_pureSVD_hr1 = []
asuh_advSVD_hr1 = []
aush_randomSVD_hr1 =[]
aush_coSVD_hr1 = []
aush_ndcg = []
aush_pureSVD_ndcg = []
aush_advSVD_ndcg = []
aush_randomSVD_ndcg =[]
aush_coSVD_ndcg = []
PGA_hr = []
PGA_pureSVD_hr = []
PGA_advSVD_hr = []
PGA_randomSVD_hr =[]
PGA_coSVD_hr = []
PGA_hr1 = []
PGA_pureSVD_hr1 = []
PGA_advSVD_hr1 = []
PGA_randomSVD_hr1 =[]
PGA_coSVD_hr1 = []
PGA_ndcg = []
PGA_pureSVD_ndcg = []
PGA_advSVD_ndcg = []
PGA_randomSVD_ndcg =[]
PGA_coSVD_ndcg = []
TNA_hr = []
TNA_pureSVD_hr = []
TNA_advSVD_hr = []
TNA_randomSVD_hr =[]
TNA_coSVD_hr = []
TNA_hr1 = []
TNA_pureSVD_hr1 = []
TNA_advSVD_hr1 = []
TNA_randomSVD_hr1 =[]
TNA_coSVD_hr1 = []
TNA_ndcg = []
TNA_pureSVD_ndcg = []
TNA_advSVD_ndcg = []
TNA_randomSVD_ndcg =[]
TNA_coSVD_ndcg = []
for file in file_list:
    f = open(file)
    lines = f.read()
    hr = pattern1.findall(lines)
    hr1 = pattern2.findall(lines)
    ndcg = pattern3.findall(lines)
    hr_list.append([float(_[1]) for _ in hr])
    hr1_list.append([float(_[1]) for _ in hr1])
    ndcg_list.append([float(_[1]) for _ in ndcg])

average_hr= [i[0] for i in hr_list] 
average_pureSVD_hr = [i[1] for i in hr_list]
average_advSVD_hr = [i[2] for i in hr_list]
average_randomSVD_hr =[i[3] for i in hr_list]
average_coSVD_hr = [i[4] for i in hr_list]
random_hr = [i[5] for i in hr_list]
random_pureSVD_hr = [i[6] for i in hr_list]
random_advSVD_hr = [i[7] for i in hr_list]
random_randomSVD_hr =[i[8] for i in hr_list]
random_coSVD_hr = [i[9] for i in hr_list]
aush_hr = [i[10] for i in hr_list]
aush_pureSVD_hr = [i[11] for i in hr_list]
aush_advSVD_hr = [i[12] for i in hr_list]
aush_randomSVD_hr =[i[13] for i in hr_list]
aush_coSVD_hr = [i[14] for i in hr_list]
PGA_hr = [i[15] for i in hr_list]
PGA_pureSVD_hr = [i[16] for i in hr_list]
PGA_advSVD_hr = [i[17] for i in hr_list]
PGA_randomSVD_hr =[i[18] for i in hr_list]
PGA_coSVD_hr = [i[19] for i in hr_list]
TNA_hr = [i[20] for i in hr_list]
TNA_pureSVD_hr = [i[21] for i in hr_list]
TNA_advSVD_hr = [i[22] for i in hr_list]
TNA_randomSVD_hr =[i[23] for i in hr_list]
TNA_coSVD_hr = [i[24] for i in hr_list]
average_hr1 = [i[0] for i in hr1_list]
average_pureSVD_hr1 = [i[1] for i in hr1_list]
average_advSVD_hr1 = [i[2] for i in hr1_list]
average_randomSVD_hr1 =[i[3] for i in hr1_list]
average_coSVD_hr1 = [i[4] for i in hr1_list]
random_hr1 = [i[5] for i in hr1_list]
random_pureSVD_hr1 = [i[6] for i in hr1_list]
random_advSVD_hr1 = [i[7] for i in hr1_list]
random_randomSVD_hr1 =[i[8] for i in hr1_list]
random_coSVD_hr1 = [i[9] for i in hr1_list]
aush_hr1 = [i[10] for i in hr1_list]
aush_pureSVD_hr1 = [i[11] for i in hr1_list]
aush_advSVD_hr1 = [i[12] for i in hr1_list]
aush_randomSVD_hr1 =[i[13] for i in hr1_list]
aush_coSVD_hr1 = [i[14] for i in hr1_list]
PGA_hr1 = [i[15] for i in hr1_list]
PGA_pureSVD_hr1 = [i[16] for i in hr1_list]
PGA_advSVD_hr1 = [i[17] for i in hr1_list]
PGA_randomSVD_hr1 =[i[18] for i in hr1_list]
PGA_coSVD_hr1 = [i[19] for i in hr1_list]
TNA_hr1 = [i[20] for i in hr1_list]
TNA_pureSVD_hr1 = [i[21] for i in hr1_list]
TNA_advSVD_hr1 = [i[22] for i in hr1_list]
TNA_randomSVD_hr1 =[i[23] for i in hr1_list]
TNA_coSVD_hr1 = [i[24] for i in hr1_list]
average_ndcg = [i[0] for i in ndcg_list]
average_pureSVD_ndcg = [i[1] for i in ndcg_list]
average_advSVD_ndcg = [i[2] for i in ndcg_list]
average_randomSVD_ndcg =[i[3] for i in ndcg_list]
average_coSVD_ndcg = [i[4] for i in ndcg_list]
random_ndcg = [i[5] for i in ndcg_list]
random_pureSVD_ndcg = [i[6] for i in ndcg_list]
random_advSVD_ndcg = [i[7] for i in ndcg_list]
random_randomSVD_ndcg =[i[8] for i in ndcg_list]
random_coSVD_ndcg = [i[9] for i in ndcg_list]
aush_ndcg = [i[10] for i in ndcg_list]
aush_pureSVD_ndcg = [i[11] for i in ndcg_list]
aush_advSVD_ndcg = [i[12] for i in ndcg_list]
aush_randomSVD_ndcg =[i[13] for i in ndcg_list]
aush_coSVD_ndcg = [i[14] for i in ndcg_list]
PGA_ndcg = [i[15] for i in ndcg_list]
PGA_pureSVD_ndcg = [i[16] for i in ndcg_list]
PGA_advSVD_ndcg = [i[17] for i in ndcg_list]
PGA_randomSVD_ndcg =[i[18] for i in ndcg_list]
PGA_coSVD_ndcg = [i[19] for i in ndcg_list]
TNA_ndcg = [i[20] for i in ndcg_list]
TNA_pureSVD_ndcg = [i[21] for i in ndcg_list]
TNA_advSVD_ndcg = [i[22] for i in ndcg_list]
TNA_randomSVD_ndcg =[i[23] for i in ndcg_list]
TNA_coSVD_ndcg = [i[24] for i in ndcg_list]


print("average_hr: ", average_hr)
print(" average_pureSVD_hr: ", average_pureSVD_hr)
print(" average_advSVD_hr: ", average_advSVD_hr)
print(" average_randomSVD_hr: ", average_randomSVD_hr)
print(" average_coSVD_hr: ", average_coSVD_hr)
print("random_hr: ", random_hr)
print(" random_pureSVD_hr: ", random_pureSVD_hr)
print(" random_advSVD_hr: ", random_advSVD_hr)
print(" random_randomSVD_hr: ", random_randomSVD_hr)
print(" random_coSVD_hr: ", random_coSVD_hr)
print("aush_hr: ", aush_hr)
print(" aush_pureSVD_hr: ", aush_pureSVD_hr)
print(" aush_advSVD_hr: ", aush_advSVD_hr)
print(" aush_randomSVD_hr: ", aush_randomSVD_hr)
print(" aush_coSVD_hr: ", aush_coSVD_hr)
print("PGA_hr: ", PGA_hr)
print(" PGA_pureSVD_hr: ", PGA_pureSVD_hr)
print(" PGA_advSVD_hr: ", PGA_advSVD_hr)
print(" PGA_randomSVD_hr: ", PGA_randomSVD_hr)
print(" PGA_coSVD_hr: ", PGA_coSVD_hr)
print("TNA_hr: ", TNA_hr)
print(" TNA_pureSVD_hr: ", TNA_pureSVD_hr)
print(" TNA_advSVD_hr: ", TNA_advSVD_hr)
print(" TNA_randomSVD_hr: ", TNA_randomSVD_hr)
print(" TNA_coSVD_hr: ", TNA_coSVD_hr)
print("average_hr1: ", average_hr1)
print(" average_pureSVD_hr1: ", average_pureSVD_hr1)
print(" average_advSVD_hr1: ", average_advSVD_hr1)
print(" average_randomSVD_hr1: ", average_randomSVD_hr1)
print(" average_coSVD_hr1: ", average_coSVD_hr1)
print("random_hr1: ", random_hr1)
print(" random_pureSVD_hr1: ", random_pureSVD_hr1)
print(" random_advSVD_hr1: ", random_advSVD_hr1)
print(" random_randomSVD_hr1: ", random_randomSVD_hr1)
print(" random_coSVD_hr1: ", random_coSVD_hr1)
print("aush_hr1: ", aush_hr1)
print(" aush_pureSVD_hr1: ", aush_pureSVD_hr1)
print(" aush_advSVD_hr1: ", aush_advSVD_hr1)
print(" aush_randomSVD_hr1: ", aush_randomSVD_hr1)
print(" aush_coSVD_hr1: ", aush_coSVD_hr1)
print("PGA_hr1: ", PGA_hr1)
print(" PGA_pureSVD_hr1: ", PGA_pureSVD_hr1)
print(" PGA_advSVD_hr1: ", PGA_advSVD_hr1)
print(" PGA_randomSVD_hr1: ", PGA_randomSVD_hr1)
print(" PGA_coSVD_hr1: ", PGA_coSVD_hr1)
print("TNA_hr1: ", TNA_hr1)
print(" TNA_pureSVD_hr1: ", TNA_pureSVD_hr1)
print(" TNA_advSVD_hr1: ", TNA_advSVD_hr1)
print(" TNA_randomSVD_hr1: ", TNA_randomSVD_hr1)
print(" TNA_coSVD_hr1: ", TNA_coSVD_hr1)
print("average_ndcg: ", average_ndcg)
print(" average_pureSVD_ndcg: ", average_pureSVD_ndcg)
print(" average_advSVD_ndcg: ", average_advSVD_ndcg)
print(" average_randomSVD_ndcg: ", average_randomSVD_ndcg)
print(" average_coSVD_ndcg: ", average_coSVD_ndcg)
print("random_ndcg: ", random_ndcg)
print(" random_pureSVD_ndcg: ", random_pureSVD_ndcg)
print(" random_advSVD_ndcg: ", random_advSVD_ndcg)
print(" random_randomSVD_ndcg: ", random_randomSVD_ndcg)
print(" random_coSVD_ndcg: ", random_coSVD_ndcg)
print("aush_ndcg: ", aush_ndcg)
print(" aush_pureSVD_ndcg: ", aush_pureSVD_ndcg)
print(" aush_advSVD_ndcg1: ", aush_advSVD_ndcg)
print(" aush_randomSVD_ndcg: ", aush_randomSVD_ndcg)
print(" aush_coSVD_ndcg: ", aush_coSVD_ndcg)
print("PGA_ndcg1: ", PGA_ndcg)
print(" PGA_pureSVD_ndcg: ", PGA_pureSVD_ndcg)
print(" PGA_advSVD_ndcg1: ", PGA_advSVD_ndcg)
print(" PGA_randomSVD_ndcg: ", PGA_randomSVD_ndcg)
print(" PGA_coSVD_ndcg: ", PGA_coSVD_ndcg)
print("TNA_ndcg: ", TNA_ndcg)
print(" TNA_pureSVD_ndcg: ", TNA_pureSVD_ndcg)
print(" TNA_advSVD_ndcg: ", TNA_advSVD_ndcg)
print(" TNA_randomSVD_ndcg: ", TNA_randomSVD_ndcg)
print(" TNA_coSVD_ndcg: ", TNA_coSVD_ndcg)
# print("HR: ", hr_list)
# print("HR1: ", hr1_list)
# print("NDCG: ", ndcg_list)
