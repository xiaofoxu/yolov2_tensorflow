# coding=utf-8
import os
import xml.etree.ElementTree as Et
import random
import glob
import shutil

CLASSES = ['__background__']


def _get_class_names(namelist):
    with open(namelist, 'r') as f:
        lines = f.readlines()
        for line in lines:
            CLASSES.append(line.strip())


def get_bbox_txt(rootdir):
    name_list = os.path.join(rootdir, 'name-list.txt')
    _get_class_names(name_list)

    print(CLASSES)

    savePath = os.path.join(rootdir, 'train.txt')
    xmlfiles = []
    for parent, folders, filenames in os.walk(rootdir):
        if len(folders) != 0:
            for folder in folders:
                xml_files = glob.glob(rootdir + '/' + folder + '/*.xml')
                for xf in xml_files:
                    xmlfiles.append(xf)
        else:
            break
    # 随机化
    shuffled_index = list(range(len(xmlfiles)))
    random.seed(12345)
    random.shuffle(shuffled_index)
    xmlfiles = [xmlfiles[i] for i in shuffled_index]

    with open(savePath, 'w') as fid:
        for index, xmlfile in enumerate(xmlfiles):
            tree = Et.parse(xmlfile)
            root = tree.getroot()
            allObjects = root.findall('object')
            if not allObjects:
                print('invalidata file: %s' % xmlfile)
                shutil.move(xmlfile, r'E:\bin')
                continue
            fid.write('#\t%d\n' % index)
            filename = root.find('filename').text
            if filename == "莲雾_模糊_0275.jpg":
                print(xmlfile)
            fid.write(filename + '\n')
            allObjects = root.findall('object')
            fid.write('%d\n' % len(allObjects))
            for obj in allObjects:
                name = obj.find('name').text
                try:
                    labelIndexStr = str(CLASSES.index(name))
                except Exception as e:
                    print('%s is not in the list, %s', name, xmlfile)
                bndbox = obj.find('bndbox')
                xmin = bndbox.find('xmin').text.split('.')[0]
                ymin = bndbox.find('ymin').text.split('.')[0]
                xmax = bndbox.find('xmax').text.split('.')[0]
                ymax = bndbox.find('ymax').text.split('.')[0]
                difficult = obj.find('difficult').text
                fid.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (labelIndexStr, xmin, ymin, xmax, ymax, difficult))


def check(rootdir):
    xmlfiles = []
    for parent, folders, filenames in os.walk(rootdir):
        if len(folders) != 0:
            for folder in folders:
                xml_files = glob.glob(rootdir + '/' + folder + '/*.xml')
                for xf in xml_files:
                    xmlfiles.append(xf)
        else:
            break

    for xmlfile in xmlfiles:
        jpgfile = xmlfile.replace(".xml", ".jpg")
        if not os.path.exists(jpgfile):
            print(xmlfile)
            shutil.move(xmlfile, r'E:\bin')


def CheckColorImage():
    '''
    检查是否是三通道的图像
    :param rootdir:
    :return:
    '''
    rootdir = r'D:\object_detection\caffe-windows\examples\faster_rcnn\VOCdevkit\fruit\JPEGImages'
    images = glob.glob(rootdir + '/*.JPG')
    print(len(images))
    txt = r'D:\object_detection\caffe-windows\examples\faster_rcnn\VOCdevkit\fruit\image.txt'
    fd = open(txt, 'w')
    for img in images:
        fd.write(img + "\n")
    fd.close()

if __name__ == '__main__':
    namelist = r'E:\LabelImages_Fruits'
    CheckColorImage()
    # get_bbox_txt(namelist)
    # check(namelist)