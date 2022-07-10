# -*- encoding: utf-8 -*-
"""
@File    :   createDataset.py
@Time    :   2021/12/14 14:44:26
@Author  :   ZhangRuiqi 
@Version :   1.0
@Contact :   zhangruiqi429@gmail.com
@License :   (C)Copyright 2019-2020
@Desc    :   In this script, add more data into the original path of SMPL vertices. The additional information includes normals for every vertex, and the transformation matrix from every face in world to uv coordinate. This script should be self-explained, and not include external functions.
Happy New Day, You Are Best~
"""
import os

import cv2
import numpy as np
from OpenGL.GL import *
from tqdm import tqdm


def read_obj_uv(obj_path='./data/smpl_uv_20200910/smpl_uv.obj'):
    points = []
    faces = []
    UVcoordinate = []
    UVmap = []
    normals = []
    with open(obj_path, 'r') as file:
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")

            if strs[0] == 'v':
                points.append([float(x) for x in strs[1:]])
            if strs[0] == "f":
                faces.append([int(x.split('/')[0]) for x in strs[1:]])
                UVmap.append([int(x.split('/')[1]) for x in strs[1:]])
            if strs[0] == 'vt':
                UVcoordinate.append([float(x) for x in strs[1:]])
            if strs[0] == 'vn':
                normals.append([float(x) for x in strs[1:]])
    points = np.array(points)
    normals = np.array(normals)
    faces = np.array(faces) - 1
    UVmap = np.array(UVmap) - 1
    UVcoordinate = np.array(UVcoordinate)
    return points, faces, UVmap, UVcoordinate, normals


def getTransformationMatrix(UVcoordinate, points, faces, UVmap):
    DDDcoordinate = points[faces]
    DDcoordinate = UVcoordinate[UVmap]
    tempDDDcoordinate = DDDcoordinate[..., :-1]
    transResult = []
    for index in range(len(DDDcoordinate)):
        trans = cv2.getAffineTransform(
            np.float32(tempDDDcoordinate[index]).copy(),
            np.float32(DDcoordinate[index]))
        #! use copy here to avoid discontinuous
        transMatrix = np.concatenate([trans, [[0, 0, 1]]], 0)
        transResult.append(transMatrix)
    transResult = np.array(transResult)
    return transResult


if __name__ == '__main__':

    currentPath = os.getcwd()

    print('please change the path manually')
    operationPath = './data/zju_mocap/CoreView_313'
    uvPath = './data/smpl_uv_20200910/smpl_uv.obj'

    points, faces, UVmap, UVcoordinate, normals = read_obj_uv(uvPath)

    verticesFolderPath = os.path.join(operationPath, 'new_vertices')
    verticesFileNames = os.listdir(verticesFolderPath)
    for verticesFileName in tqdm(verticesFileNames):
        verticesFilePath = os.path.join(verticesFolderPath, verticesFileName)
        vertices = np.load(verticesFilePath).astype(np.float32)

        transformationMatrix = getTransformationMatrix(UVcoordinate, vertices,
                                                       faces, UVmap)

        saveTransformationMatrixPath = os.path.join(
            operationPath, 'new_transformationMatrix', verticesFileName)

        if not os.path.exists(os.path.dirname(saveTransformationMatrixPath)):
            os.makedirs(os.path.dirname(saveTransformationMatrixPath))

        np.save(saveTransformationMatrixPath, transformationMatrix)
