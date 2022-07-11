import bpy
import numpy as np
import trimesh
from IPython import embed
from OpenGL.GL import *


def readNumpy(path):
    fileContent = np.load(path, allow_pickle=True)
    return fileContent


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


#This a function to normalize normal vector
def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def get_sampling_points(
    ray_o,
    ray_d,
    near,
    far,
    vertices,
    faces,
    N_samples,
    split,
    perturb,
    nrays,
    transformationMatrix,
    thickness=0.1,
    thickness_validscale=0.8,
):

    t_vals = np.linspace(0., 1., num=N_samples)

    z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

    if perturb > 0. and split == 'train':
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        # 所有采样点的中点

        upper = np.concatenate([mids, z_vals[..., -1:]], -1)
        lower = np.concatenate([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = np.random.rand(*z_vals.shape)
        # use * here to flat the shape as input
        z_vals = lower + (upper - lower) * t_rand

    pts = ray_o[:, None] + ray_d[:, None] * z_vals[..., None]
    pts = pts.reshape(-1, 3)

    nrPts = pts.shape[0]

    edges = []
    _points2 = [tuple(x) for x in vertices]
    _faces2 = [tuple(x) for x in faces]
    new_mesh = bpy.data.meshes.new('new_mesh')
    new_mesh.from_pydata(_points2, edges, _faces2)
    new_mesh.update()

    new_object = bpy.data.objects.new('new_object', new_mesh)
    new_collection = bpy.data.collections.new('new_collection')
    bpy.context.scene.collection.children.link(new_collection)

    new_collection.objects.link(new_object)

    _points1 = [tuple(x) for x in pts]
    tempHit = np.zeros(ray_o.shape[0] * N_samples, np.float32)
    projectIntersection_SampleSMPL = 1000 * np.ones(
        [ray_o.shape[0] * N_samples, 3], np.float32)
    tempNorm = np.zeros([ray_o.shape[0] * N_samples, 3], np.float32)
    projectIntersectionFace_SampleSMPL = -1 * np.ones(
        ray_o.shape[0] * N_samples, np.int32)

    for _index, local_pos in enumerate(_points1):
        (tempHit[_index], projectIntersection_SampleSMPL[_index],
         tempNorm[_index], projectIntersectionFace_SampleSMPL[_index]
         ) = new_object.closest_point_on_mesh(local_pos, distance=thickness)

    p2 = np.array(projectIntersection_SampleSMPL) - pts
    v = np.einsum('BN,BN ->B', p2, tempNorm)
    mesh = trimesh.Trimesh(vertices, faces)
    negIndexBool = mesh.contains(pts[np.where(v != 0)[0]])

    projectIntersectionDistance_SampleSMPL = calculateDistanceBatch(
        projectIntersection_SampleSMPL, pts)

    projectIntersectionDistance_SampleSMPL[np.where(
        v != 0)[0][negIndexBool]] *= -1

    templocations = np.concatenate([
        projectIntersection_SampleSMPL[:, :-1],
        np.ones_like(projectIntersection_SampleSMPL[:, :1])
    ], 1)

    resultValuedSampleIndex = np.where(v != 0)

    sampleUVcoordinates = np.einsum(
        'BNi,Bi ->BN', transformationMatrix[
            projectIntersectionFace_SampleSMPL[resultValuedSampleIndex]],
        templocations[resultValuedSampleIndex])

    resultLocation = -1 * np.ones([nrPts, 3]).astype(np.float32)

    resultLocation[resultValuedSampleIndex, :2] = sampleUVcoordinates[:, :2]

    resultLocation[resultValuedSampleIndex, 2:] = normalize(
        projectIntersectionDistance_SampleSMPL[resultValuedSampleIndex][:,
                                                                        None],
        -thickness, +thickness, 0, 1)

    biggerIndex = resultValuedSampleIndex[0][np.where(
        np.abs(projectIntersectionDistance_SampleSMPL[resultValuedSampleIndex])
        > thickness * thickness_validscale)[0]]

    resultLocation = resultLocation.reshape(ray_o.shape[0], N_samples, 3)

    # savePath = './resultLocation.obj'
    # neg_index = resultValuedSampleIndex[0][negIndexBool]
    # pos_index = resultValuedSampleIndex[0][(1-negIndexBool).astype(np.bool8)]
    # select_index = biggerIndex
    # select_points = pts.reshape(-1,3)[select_index]
    # with open(savePath, 'w') as saveFile:
    #     for index, point in enumerate(select_points):
    #         strs = []
    #         strs.append('v')

    #         strs.append(str(point[0]))
    #         strs.append(str(point[1]))
    #         strs.append(str(point[2]))

    #         changedLine = " ".join(strs)
    #         saveFile.writelines(changedLine)
    #         saveFile.writelines('\n')

    return resultLocation, z_vals, biggerIndex


def get_sampling_points_drawtexture(
    ray_o,
    ray_d,
    near,
    far,
    vertices,
    faces,
    N_samples,
    split,
    perturb,
    nrays,
    transformationMatrix,
    thickness=0.1,
    thickness_validscale=0.8,
):

    t_vals = np.linspace(0., 1., num=N_samples)

    z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

    if perturb > 0. and split == 'train':
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        # 所有采样点的中点

        upper = np.concatenate([mids, z_vals[..., -1:]], -1)
        lower = np.concatenate([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = np.random.rand(*z_vals.shape)
        # use * here to flat the shape as input
        z_vals = lower + (upper - lower) * t_rand

    pts = ray_o[:, None] + ray_d[:, None] * z_vals[..., None]
    pts = pts.reshape(-1, 3)
    pts[:, -1] = pts[:, -1] * 5
    resultLocation = pts.reshape(ray_o.shape[0], N_samples,
                                 3).astype(np.float32)
    biggerIndex = np.array([])
    return resultLocation, z_vals, biggerIndex


def calculateDistanceBatch(pt1, pt2):
    return np.sqrt(np.sum((pt1 - pt2)**2, -1))


def normalize(array, srcMinValue, srcMaxValue, targetMinValue, targetMaxValue):
    return (array - srcMinValue) / (srcMaxValue - srcMinValue) * (
        targetMaxValue - targetMinValue) + targetMinValue


if __name__ == "__main__":
    embed()
