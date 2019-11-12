from PIL import Image
import os
import os.path
import glob

def rename(rename_path, outer_path, folderlist):
    for folder in folderlist:
        if os.path.basename(folder)=='bus':
            foldnum = 0
        elif os.path.basename(folder)=='taxi':
            foldnum = 1
        elif os.path.basename(folder)=='truck':
            foldnum = 2
        elif os.path.basename(folder)=='family sedan':
            foldnum = 3
        elif os.path.basename(folder)=='minibus':
            foldnum = 4
        elif os.path.basename(folder) == 'jeep':
            foldnum = 5
        elif os.path.basename(folder)=='SUV':
            foldnum = 6
        elif os.path.basename(folder)=='heavy truck':
            foldnum = 7
        elif os.path.basename(folder)=='racing car':
            foldnum = 8
        elif os.path.basename(folder)=='fire engine':
            foldnum = 9

        inner_path = os.path.join(outer_path, folder)
        total_num_folder = len(folderlist)
        print(total_num_folder)
        filelist = os.listdir(inner_path)
        i = 0
        for item in filelist:
            total_num_file = len(filelist)
            print(total_num_file)
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(inner_path), item)
                print(src)
                dst = os.path.join(os.path.abspath(rename_path), str(foldnum) + '_' + str(i) + '.jpg')
                print(dst)
            try:
                os.rename(src, dst)
                i += 1
            except:
                continue

#训练集
rename_path1 = 'C:/Users/admin/Desktop/carclass/renametrain'
outer_path1 = 'C:/Users/admin/Desktop/carclass/train'
folderlist1 = os.listdir(r'C:/Users/admin/Desktop/carclass/train')
rename(rename_path1, outer_path1, folderlist1)
#测试集
rename_path2 = 'C:/Users/admin/Desktop/carclass/renametest'
outer_path2 = 'C:/Users/admin/Desktop/carclass/val'
folderlist2 = os.listdir(r'C:/Users/admin/Desktop/carclass/val')
rename(rename_path2, outer_path2, folderlist2)
print("test totally rename ! ! !")

#修改图片尺寸
def convertjpg(jpgfile, outdir, width=32, height=32):
    img = Image.open(jpgfile)
    img = img.convert('RGB')
    img.save(os.path.join(outdir, os.path.basename(jpgfile)))
    new_img = img.resize((width, height), Image.BILINEAR)
    new_img.save(os.path.join(outdir, os.path.basename(jpgfile)))

#训练集
for jpgfile in glob.glob('C:/Users/admin/Desktop/carclass/renametrain/*.jpg'):
    convertjpg(jpgfile, 'C:/Users/admin/Desktop/carclass/data')
print('train totally resize')

#测试集
for jpgfile in glob.glob('C:/Users/admin/Desktop/carclass/renametest/*.jpg'):
    convertjpg(jpgfile, 'C:/Users/admin/Desktop/carclass/test')
print('test totally resize')

