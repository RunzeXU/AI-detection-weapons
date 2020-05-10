# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:53:43 2019

@author: 28771
"""

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

os.chdir('C:\\Users\\28771\\models\\research\\object_detection\\images2\\test')
path = 'C:\\Users\\28771\\models\\research\\object_detection\\images2\\test'

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = path
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('gun_test3.csv', index=None)
    print('Successfully converted xml to csv.')


main()