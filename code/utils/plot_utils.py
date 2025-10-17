import plotly
import os
import utils.path_utils
from utils import general_utils, path_utils
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import torch
import plotly.graph_objects as go
from utils import geo_utils
from matplotlib import image
import scipy.io
import warnings




def plot_cameras_before_and_after_ba(outputs, errors, conf, phase, scan, epoch=None, bundle_adjustment=False):
    saveOutput = {}

    originalCamsNum = outputs['camsNum']
    originalPointsNum = outputs['pointsNum']
    outliersPercent = outputs['outliersPercent']

    Rs_gt = outputs['Rs_gt']
    ts_gt = outputs['ts_gt']
    Rs_pred = outputs['Rs_fixed']
    ts_pred = outputs['ts_fixed']

    pts3D = outputs['pts3D_pred_fixed'][:3, :]
    pts3D_triangulated = outputs['pts3D_triangulated_fixed'][:3, :]
    Rs_error = errors['Rs_err']
    ts_error = errors['ts_err']
    validPoints = outputs['validPoints']

    errorPerEntry = geo_utils.reprojection_error_with_points(outputs['Ps'], outputs['pts3D_pred'].T, outputs['xs'])

    plot_cameras(Rs_pred, ts_pred, None, Rs_gt, ts_gt, Rs_error, ts_error, conf, phase,
                 camsNum=[originalCamsNum, len(Rs_gt)], outliersPct=outliersPercent, scan=scan, epoch=epoch,
                 extraText="_cameras")

    plot_cameras(Rs_pred, ts_pred, pts3D, Rs_gt, ts_gt, Rs_error, ts_error, conf, phase,
                 camsNum=[originalCamsNum, len(Rs_gt)], pointsNum=[originalPointsNum, pts3D_triangulated.shape[1]],
                 outliersPct=outliersPercent, scan=scan, epoch=epoch,
                 reprojection_error=errorPerEntry, validPoints=validPoints)

    errorPerEntry = geo_utils.reprojection_error_with_points(outputs['Ps'], outputs['pts3D_triangulated'].T, outputs['xs'])

    plot_cameras(Rs_pred, ts_pred, pts3D_triangulated, Rs_gt, ts_gt, Rs_error, ts_error, conf, phase,
                 camsNum=[originalCamsNum, len(Rs_gt)], pointsNum=[originalPointsNum, pts3D_triangulated.shape[1]],
                 outliersPct=outliersPercent, scan=scan, epoch=epoch, extraText="_triangulation",
                 reprojection_error=errorPerEntry, validPoints=validPoints)

    if bundle_adjustment:
        for name in ['final']:
            name_tag = f"_{name}"

            Rs_pred = outputs[f'Rs_ba{name_tag}_fixed']
            ts_pred = outputs[f'ts_ba{name_tag}_fixed']
            pts3D = outputs[f'Xs_ba{name_tag}_fixed'][:3, :]
            Rs_error = errors[f'Rs_ba{name_tag}_err']
            ts_error = errors[f'ts_ba{name_tag}_err']
            validPoints = outputs[f'validPoints{name_tag}']

            errorPerEntry = geo_utils.reprojection_error_with_points(
                outputs[f'Ps_ba{name_tag}'], outputs[f'Xs_ba{name_tag}'].T, outputs[f'xs_ba{name_tag}'])
            errorPerPoint = np.nanmean(errorPerEntry, axis=0)
            errorPerCam = np.nanmean(errorPerEntry, axis=1)

            keep_indices = errorPerPoint < 4.0
            keep_indices_camera = errorPerCam < 5.0 * 10000000000000000

            saveOutput['Rs'] = Rs_pred
            saveOutput['ts'] = ts_pred
            saveOutput['Ks'] = outputs['Ks'][outputs[f'validCamIndices_ba{name_tag}']]
            saveOutput['pts3D'] = pts3D[:, keep_indices]
            saveOutput['namesList'] = outputs['namesList'][outputs[f'validCamIndices_ba{name_tag}']]
            saveOutput['M'] = geo_utils.xs_to_M(outputs[f'xs_ba{name_tag}'])[:, keep_indices]

            plot_cameras(Rs_pred, ts_pred, None,
                         Rs_gt[outputs[f'validCamIndices_ba{name_tag}']],
                         ts_gt[outputs[f'validCamIndices_ba{name_tag}']],
                         Rs_error, ts_error, conf, phase,
                         camsNum=[originalCamsNum, len(Rs_gt)], outliersPct=outliersPercent,
                         scan=scan + '_ba', epoch=epoch, extraText=name_tag + "_cameras")

            plot_cameras(Rs_pred[keep_indices_camera], ts_pred[keep_indices_camera], pts3D,
                         Rs_gt[outputs[f'validCamIndices_ba{name_tag}']][keep_indices_camera],
                         ts_gt[outputs[f'validCamIndices_ba{name_tag}']][keep_indices_camera],
                         Rs_error, ts_error, conf, phase,
                         camsNum=[originalCamsNum, len(Rs_gt)],
                         pointsNum=[originalPointsNum, pts3D_triangulated.shape[1]],
                         outliersPct=outliersPercent, scan=scan + '_ba', epoch=epoch, extraText=name_tag,
                         reprojection_error=errorPerEntry[keep_indices_camera, :], validPoints=validPoints)

    return saveOutput




def plot_cameras(Rs_pred=None, ts_pred=None, pts3D=None, Rs_gt=None, ts_gt=None, Rs_error=None, ts_error=None, conf=None, phase=None, camsNum=[0, 0], pointsNum=[0, 0], outliersPct=0.0, scan=None, epoch=None, extraText="", reprojection_error=None, validPoints=None):
    data = []

    camNames = [str(i) for i in range(len(ts_gt))] if ts_gt is not None else None
    pointsNames = [str(i) for i in range(len(pts3D[0]))] if pts3D is not None else None

    if reprojection_error is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            errorPerPoint = np.nanmean(reprojection_error, axis=0)
            errorPerCam = np.nanmean(reprojection_error, axis=1)
    else:
        errorPerPoint = errorPerCam = None

    if validPoints is not None:
        cam_per_pts = validPoints.sum(axis=0)  # [n_pts]
        pts_per_cam = validPoints.sum(axis=1)  # [n_cams]
    else:
        cam_per_pts = pts_per_cam = None

    ptsNum = len(pts3D[0]) if pts3D is not None else 0
    camerasNum = len(Rs_gt) if Rs_gt is not None else 0

    if ts_gt is not None:
        data.append(get_3D_quiver_trace(ts_gt, Rs_gt[:, :3, 2], color='#86CE00', name='cam_gt', texts=camNames, reprojection_error={"repr err": errorPerCam, "Rs err": Rs_error, "ts err": ts_error, "pts_per_cam": pts_per_cam}))
        data.append(get_3D_scater_trace(ts_gt.T, color='#86CE00', name='cam_gt', size=1, texts=camNames, reprojection_error={"repr err": errorPerCam, "Rs err": Rs_error, "ts err": ts_error, "pts_per_cam": pts_per_cam}))

    if ts_pred is not None:
        data.append(get_3D_quiver_trace(ts_pred, Rs_pred[:, :3, 2], color='#C4451C', name='cam_learn', texts=camNames, reprojection_error={"repr err": errorPerCam, "Rs err": Rs_error, "ts err": ts_error, "pts_per_cam": pts_per_cam}))
        data.append(get_3D_scater_trace(ts_pred.T, color='#C4451C', name='cam_learn', size=1, texts=camNames, reprojection_error={"repr err": errorPerCam, "Rs err": Rs_error, "ts err": ts_error, "pts_per_cam": pts_per_cam}))

    if pts3D is not None:
        data.append(get_3D_scater_trace(pts3D, '#3366CC', '3D points', size=0.5, texts=pointsNames, reprojection_error={"repr err": errorPerPoint, "cam_per_pts": cam_per_pts}))

    fig = go.Figure(data=data)

    title = ''
    if isinstance(reprojection_error, np.ndarray):
        title += 'Reprojection Mean = {:.5f}, '.format(np.nanmean(reprojection_error))
    if Rs_error is not None:
        title += 'Rotation Mean = {:.5f}, Translation Mean = {:.5f}, '.format(Rs_error.mean(), ts_error.mean())
        title += 'Rotation Median = {:.5f}, Translation Median = {:.5f}, '.format(np.median(Rs_error), np.median(ts_error))

    title += '<br> Cameras number = {}/{}/{}, points number = {}/{}/{}, outliers percent = {:.2f}%'.format(
        camerasNum, camsNum[1], camsNum[0], ptsNum, pointsNum[1], pointsNum[0], outliersPct * 100)

    fig.update_layout(title=title, showlegend=True)
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=4, range=[-100, 100]),
            yaxis=dict(nticks=4, range=[-100, 100]),
            zaxis=dict(nticks=4, range=[-100, 100])
        )
    )

    if phase is not None:
        path = utils.path_utils.path_to_plots(conf, phase, epoch=epoch, scan=scan, extraText=extraText)
    else:
        path = 'plots/' + scan + ".html"

    plotly.offline.plot(fig, filename=path, auto_open=False)
    return path



def get_3D_quiver_trace(points, directions, color='#bd1540', name='', cam_size=1, texts=None, reprojection_error=None):
    assert points.shape[1] == 3, "3d cone plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d cone plot input points are not correctely shaped "
    assert directions.shape[1] == 3, "3d cone plot input directions are not correctely shaped "
    assert len(directions.shape) == 2, "3d cone plot input directions are not correctely shaped "


    if reprojection_error is not None:
        for key, error_type in reprojection_error.items():
            if error_type is not None:
                errorPerCam = ["{:.2f}".format(e) for e in error_type]
                texts = [a + f"<br> {key}: " + b for a, b in zip(texts, errorPerCam)]

    trace = go.Cone(
        name=name,
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        u=directions[:, 0],
        v=directions[:, 1],
        w=directions[:, 2],
        sizemode='absolute',
        sizeref=cam_size,
        showscale=False,
        colorscale=[[0, color], [1, color]],
        anchor="tail",
        text=texts
    )

    return trace


def get_3D_scater_trace(points, color, name, size=0.5, texts=None, reprojection_error=None):
    assert points.shape[0] == 3, "3d plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d plot input points are not correctely shaped "


    if reprojection_error is not None:
        for key, error_type in reprojection_error.items():
            if error_type is not None:
                errorPerCam = ["{:.2f}".format(e) for e in error_type]
                texts = [a + f"<br> {key}: " + b for a, b in zip(texts, errorPerCam)]

    trace = go.Scatter3d(
        name=name,
        x=points[0, :],
        y=points[1, :],
        z=points[2, :],
        mode='markers',
        marker=dict(
            size=size,
            color=color,
        ),
        text=texts
    )

    return trace

def get_points_colors(images_path, image_names, xs, first_occurence=False):
    m, n, _ = xs.shape
    points_colors = np.zeros([n, 3])
    if first_occurence:
        images_indices = (geo_utils.xs_valid_points(xs)).argmax(axis=0)
        unique_images = np.unique(images_indices)
        for i, image_ind in enumerate(unique_images):
            image_name = str(image_names[image_ind][0]).split('/')[1]
            im = image.imread(os.path.join(images_path, image_name))
            # read the image to ndarray
            points_in_image = np.where(image_ind == images_indices)[0]
            for point_ind in points_in_image:
                point_2d_in_image = xs[image_ind, point_ind].astype(int)
                points_colors[point_ind] = im[point_2d_in_image[1], point_2d_in_image[0]]
    else:
        valid_points = geo_utils.xs_valid_points(xs)
        colors = np.zeros([m, n, 3])
        for image_ind in range(m):
            image_name = str(image_names[image_ind][0]).split('/')[1]
            im = image.imread(os.path.join(images_path, image_name))
            points_in_image = np.where(valid_points[image_ind])[0]
            for point_ind in points_in_image:
                point_2d_in_image = xs[image_ind, point_ind].astype(int)
                colors[image_ind, point_ind] = im[point_2d_in_image[1], point_2d_in_image[0]]
        for point_ind in range(n):
            points_colors[point_ind] = np.mean(colors[valid_points[:, point_ind], point_ind], axis=0)

    return points_colors
