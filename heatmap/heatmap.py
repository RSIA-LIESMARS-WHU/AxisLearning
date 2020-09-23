# -*- coding: utf-8 -*-
import os, sys
import math
import cv2 
import numpy as np
import matplotlib.pyplot as plt




def area_centerness(img_row, img_col,
                    roi_w, roi_h,
                    roi_l, roi_t, roi_r, roi_d):
    l = img_col - roi_l
    r = roi_r - img_col
    t = img_row - roi_t
    d = roi_d - img_row
    # centerness = 1 - 4*(roi_w/2.-min(l, r)) * (roi_h/2. - min(t, d))*1./(roi_w*roi_h)
    # centerness = min(l, r)*2./roi_w * min(t, d)*2./roi_h
    # print(l, r, t, d, img_row , roi_t)
    centerness = min(min(l, r)*1./max(l, r) , min(t, d)*1./max(t, d))#math.sqrt(centerness)
    return centerness


def endpt_centerness(row, col,c_x,c_y,long_edge,
                    pt1_x, pt1_y, pt2_x, pt2_y, pt3_x, pt3_y, pt4_x, pt4_y):
    dist1 = math.sqrt(((col - pt1_x) ** 2 + (row - pt1_y) ** 2))
    dist2 = math.sqrt(((col - pt2_x) ** 2 + (row - pt2_y) ** 2))
    dist3 = math.sqrt(((col - pt3_x) ** 2 + (row - pt3_y) ** 2))
    dist4 = math.sqrt(((col - pt4_x) ** 2 + (row - pt4_y) ** 2))
    centerness = min(dist1, dist2)*1./max(dist1, dist2) * min(dist3, dist4)*1./max(dist3, dist4)*(1-math.sqrt(((col - c_x) ** 2 + (row - c_y) ** 2))/long_edge)
    # centerness = min(l, r)*2./roi_w * min(t, d)*2./roi_h
    # print(l, r, t, d, img_row , roi_t)
    centerness = math.sqrt(centerness)
    return centerness


def liner(img_row, img_col,
                    roi_w, roi_h,
                    roi_l, roi_t, roi_r, roi_d):
    l = img_col - roi_l
    r = roi_r - img_col
    t = img_row - roi_t
    d = roi_d - img_row
    # centerness = min(l, r)*1./max(l, r) * min(t, d)*1./max(t, d)
    centerness = min(l, r)*2./roi_w * min(t, d)*2./roi_h
    # print(l, r, t, d, img_row , roi_t)
    # centerness = math.sqrt(centerness)
    return centerness


# Python实现正态分布
# 绘制正态分布概率密度函数

def normal_dist(mean, std, pix2mean_length):
    sig = math.sqrt(std)  # 标准差δ
    y_sig = np.exp(-(pix2mean_length - mean) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
    return y_sig


    # from cornernet
def _gaussian_radius(self, height, width, min_overlap=0.7):
    a1 = 1.
    b1 = (height + width)
    c1 = width * height * (1. - min_overlap) / (1. + min_overlap)
    sq1 = tf.sqrt(b1 ** 2. - 4. * a1 * c1)
    r1 = (b1 + sq1) / 2.
    a2 = 4.
    b2 = 2. * (height + width)
    c2 = (1. - min_overlap) * width * height
    sq2 = tf.sqrt(b2 ** 2. - 4. * a2 * c2)
    r2 = (b2 + sq2) / 2.
    a3 = 4. * min_overlap
    b3 = -2. * min_overlap * (height + width)
    c3 = (min_overlap - 1.) * width * height
    sq3 = tf.sqrt(b3 ** 2. - 4. * a3 * c3)
    r3 = (b3 + sq3) / 2.
    return tf.reduce_min([r1, r2, r3])

def gauss_kernel(c_x, c_y, sigma, col, row):
    # sig = math.sqrt(std)  # 标准差δ
    y_sig = np.exp(-((col - c_x) ** 2 + (row - c_y) ** 2)  / (2 * sigma**2) )
    return y_sig


def multivariate_normal(x, d, mean, covariance):
    """
    pdf of the multivariate normal distribution.
    https://www.cnblogs.com/jermmyhsu/p/8251013.html
    """
    x_m = x - mean
    # print(x_m.shape)
    # return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) * 
    return        np.exp(-x_m.T*np.linalg.inv(covariance)*x_m / 2)#)

def endpt(c_x, c_y, len_edge, col, row, pt1_x, pt1_y, pt2_x, pt2_y):
    dist1 = math.sqrt(((col - pt1_x) ** 2 + (row - pt1_y) ** 2))
    dist2 = math.sqrt(((col - pt2_x) ** 2 + (row - pt2_y) ** 2))
    return (1-math.sqrt(((col - c_x) ** 2 + (row - c_y) ** 2))/len_edge) * \
        1.*min(dist1, dist2)/max(dist1, dist2)
        



def gaussian_radius(roi_h, roi_w, min_overlap):
    height, width = roi_h, roi_w
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 - sq1) / (2 * a1)

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 - sq2) / (2 * a2)
    
    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)

    r3  = (b3 + sq3) / (2 * a3)

    return min(r1, r2, r3)


def fcos_centerness(img_row, img_col,
                    roi_w, roi_h,
                    roi_l, roi_t, roi_r, roi_d):
    l = img_col - roi_l
    r = roi_r - img_col
    t = img_row - roi_t
    d = roi_d - img_row
    # centerness
    centerness = math.sqrt(min(l, r)*1./max(l, r) * min(t, d)*1./max(t, d))

    # long
    # centerness = min(l, r)*2./roi_w;
    # return centerness*centerness

    # min centerness
    # centerness = min(min(l, r)*2./roi_w , min(t, d)*2./roi_h)
    # return centerness

    ratio = math.pow(max(roi_w*1./roi_h, roi_h*1./roi_w),1/3.)

    # if roi_w<roi_h:
    #     centerness = math.sqrt(min(l, r)*3./roi_w * min(t, d)*1./max(t, d))
    # else:
    #     centerness = math.sqrt(min(l, r)*1./max(l, r) * min(t, d)*3./roi_h)
    # centerness = min(min(l, r)*2./roi_w , min(t, d)*2./roi_h)
    # print(centerness)#math.sqrt(ratio))
    # centerness = ratio*centerness
    # print(centerness)
    return min(centerness, 1.)

    # centerness larger range
    # centerness = min(min(l, r)*1./max(l, r) , min(t, d)*1./max(t, d))#math.sqrt(centerness)
    # return math.sqrt(centerness)



# 定义长宽hw 定义注意力中心 扩散方式
if __name__ == "__main__":
    dstH = 800
    dstW = 800
    heatmap=np.zeros((dstH,dstW),dtype=np.float)

    roi_cx, roi_cy = 400, 400
    roi_w, roi_h = 92, 24
    # roi_w, roi_h = 200, 200
    # ||
    # ||
    # ||
    if roi_h>roi_w:
        pt1_x = roi_cx
        pt1_y = roi_cy - roi_h/2
        pt2_x = roi_cx
        pt2_y = pt1_y + roi_h

        pt3_x = roi_cx + roi_w/2
        pt3_y = roi_cy
        pt4_x = pt3_x - roi_w
        pt4_y = roi_cy
    else:
        pt1_x = roi_cx + roi_w/2
        pt1_y = roi_cy
        pt2_x = pt1_x - roi_w
        pt2_y = roi_cy

        pt3_x = roi_cx
        pt3_y = roi_cy - roi_h/2
        pt4_x = roi_cx
        pt4_y = pt3_y + roi_h

    # roi_w, roi_h = 200, 200
    roi_l = roi_cx - roi_w //2
    roi_r = roi_l + roi_w
    roi_t = roi_cy - roi_h//2
    roi_d = roi_t + roi_h 
    sigma = gaussian_radius(roi_h, roi_w, 0.7)

    d = 2  # number of dimensions

    # Plot of independent Normals
    bivariate_mean = np.matrix([[roi_cx], [roi_cy]])  # Mean
    bivariate_covariance = np.matrix([
        [roi_w, 0.], 
        [0., roi_h]])  # Covariance

    for i in range(dstH):
        for j in range(dstW):
            if any([j <= roi_l, j >= roi_r, i <= roi_t, i >= roi_d]):
                heatmap[i,j] =  1
            else:
                value = fcos_centerness(i, j, roi_w, roi_h, roi_l, roi_t, roi_r, roi_d)#math.sqrt(fcos_centerness(i, j, roi_w, roi_h, roi_l, roi_t, roi_r, roi_d))
                # value = endpt_centerness(i, j, roi_cx, roi_cy, max(roi_h, roi_w), pt1_x, pt1_y, pt2_x, pt2_y, pt3_x, pt3_y, pt4_x, pt4_y)
                # value = area_centerness(i, j, roi_w, roi_h, roi_l, roi_t, roi_r, roi_d)#math.sqrt(fcos_centerness(i, j, roi_w, roi_h, roi_l, roi_t, roi_r, roi_d))
                
                # heatmap[i,j] = gauss_dist(0, roi_w, math.sqrt((j-roi_cx)*(j-roi_cx) + (i-roi_cy)*(i-roi_cy)))
                # heatmap[i,j] = gauss_kernel(roi_cx, roi_cy, 0.15*min(roi_w, roi_h), j, i )
                # if math.sqrt(math.pow(i-roi_cy, 2)+ math.pow(j-roi_cx, 2)) < 10:
                #     print(heatmap[i,j])
                # heatmap[i,j] = gauss_kernel(roi_cx, roi_cy, sigma, j, i )
                # print(multivariate_normal(np.matrix([[j], [i]]), d, bivariate_mean, bivariate_covariance))
                # [[roi_w, roi_w/2],[roi_h/2, roi_h]]
                # value = multivariate_normal(np.matrix([[j], [i]]), d, bivariate_mean, [[10*roi_w, 0],[0, 5*roi_h]])
                # heatmap[i,j] = endpt(roi_cx, roi_cy, min(roi_h, roi_w), j, i, pt1_x, pt1_y, pt2_x, pt2_y)
                # value = 1./(1+math.exp(-0.5*value))
                heatmap[i,j] = value
                # if value <= 0.2:
                #      heatmap[i,j] = 0.5
                if j==400 and i==400: #or (j==430 and i==410) or  (j==430 and i==415):
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    heatmap = cv2.circle(heatmap,(j,i),5,(0,0,213),-1)
                    heatmap = cv2.putText(heatmap, '{}'.format(value), (j, i), font, 0.8, (255, 255, 255), 2)
                # print(value)

    # heatmap = heatmap / np.max(heatmap)
    colormap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    # colormap[...,0] = (1-(colormap[...,0]==128))*colormap[...,0]
    # print(colormap)

    # merge_float = np.float32(colormap) + np.float32(image)
    # result_img = np.uint8(255 * merge_float / np.max(merge_float))
 
    cv2.imwrite(os.path.join('./', "{}.jpg".format(sys.argv[1])), colormap)



# # C H W -> H W C
# grad = grad_features.data.cpu().numpy().transpose((1,2,0))

# size_upsample = raw_img.shape[:2]
# c, h, w = grad_features.shape

# # norm by channel
# grad = grad/ (np.sqrt(np.mean(np.square(grad), axis=(0,1))) + 1e-5)#归一化

# # resize_grad = cv2.resize(grad, size_upsample)
# nozero_grad = np.maximum(grad, 0)

# heatmap = nozero_grad / np.max(nozero_grad)

# #Return to BGR [0..255] from the preprocessed image
# # image = raw_img
# # image -= np.min(image)
# # image = np.minimum(image, 255)
# # print(image.shape)

# colormap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
# colormap[...,0] = (1-(colormap[...,0]==128))*colormap[...,0]
# # print(colormap)

# merge_float = np.float32(colormap) + np.float32(image)
# result_img = np.uint8(255 * merge_float / np.max(merge_float))

# if  all([box_pos is not None, box_cls is not None, box_label is not None]):
#     box_pos =  box_pos.detach().cpu().numpy()
#     box_pos[:,[0,2]] = box_pos[:,[0,2]]*ratex
#     box_pos[:,[1,3]] = box_pos[:,[1,3]]*ratey
#     box_cls =  box_cls.detach().cpu().numpy()
#     box_label =  box_label.detach().cpu().numpy()
#     result_img = draw_rotate_box_cv(result_img, box_pos, box_label, box_cls)
# cv2.imwrite(os.path.join(self.save_path, "{}_{}.jpg".format(idx+2273, ext_info)), result_img)












def _compute_one_image_loss(self, keypoints, offset, size, ground_truth, meshgrid_y, meshgrid_x,
                            stride, pshape):
    slice_index = tf.argmin(ground_truth, axis=0)[0]
    ground_truth = tf.gather(ground_truth, tf.range(0, slice_index, dtype=tf.int64))
    ngbbox_y = ground_truth[..., 0] / stride
    ngbbox_x = ground_truth[..., 1] / stride
    ngbbox_h = ground_truth[..., 2] / stride
    ngbbox_w = ground_truth[..., 3] / stride
    class_id = tf.cast(ground_truth[..., 4], dtype=tf.int32)
    ngbbox_yx = ground_truth[..., 0:2] / stride
    ngbbox_yx_round = tf.floor(ngbbox_yx)
    offset_gt = ngbbox_yx - ngbbox_yx_round
    size_gt = ground_truth[..., 2:4] / stride
    ngbbox_yx_round_int = tf.cast(ngbbox_yx_round, tf.int64)
    keypoints_loss = self._keypoints_loss(keypoints, ngbbox_yx_round_int, ngbbox_y, ngbbox_x, ngbbox_h,
                                            ngbbox_w, class_id, meshgrid_y, meshgrid_x, pshape)

    offset = tf.gather_nd(offset, ngbbox_yx_round_int)
    size = tf.gather_nd(size, ngbbox_yx_round_int)
    offset_loss = tf.reduce_mean(tf.abs(offset_gt - offset))
    size_loss = tf.reduce_mean(tf.abs(size_gt - size))
    total_loss = keypoints_loss + 0.1*size_loss + offset_loss
    return total_loss

def _keypoints_loss(self, keypoints, gbbox_yx, gbbox_y, gbbox_x, gbbox_h, gbbox_w,
                    classid, meshgrid_y, meshgrid_x, pshape):
    sigma = self._gaussian_radius(gbbox_h, gbbox_w, 0.7)
    gbbox_y = tf.reshape(gbbox_y, [-1, 1, 1])
    gbbox_x = tf.reshape(gbbox_x, [-1, 1, 1])
    sigma = tf.reshape(sigma, [-1, 1, 1])

    num_g = tf.shape(gbbox_y)[0]
    meshgrid_y = tf.expand_dims(meshgrid_y, 0)
    meshgrid_y = tf.tile(meshgrid_y, [num_g, 1, 1])
    meshgrid_x = tf.expand_dims(meshgrid_x, 0)
    meshgrid_x = tf.tile(meshgrid_x, [num_g, 1, 1])

    keyp_penalty_reduce = tf.exp(-((gbbox_y-meshgrid_y)**2 + (gbbox_x-meshgrid_x)**2)/(2*sigma**2))
    zero_like_keyp = tf.expand_dims(tf.zeros(pshape, dtype=tf.float32), axis=-1)
    reduction = []
    gt_keypoints = []
    for i in range(self.num_classes):
        exist_i = tf.equal(classid, i)
        reduce_i = tf.boolean_mask(keyp_penalty_reduce, exist_i, axis=0)
        reduce_i = tf.cond(
            tf.equal(tf.shape(reduce_i)[0], 0),
            lambda: zero_like_keyp,
            lambda: tf.expand_dims(tf.reduce_max(reduce_i, axis=0), axis=-1)
        )
        reduction.append(reduce_i)

        gbbox_yx_i = tf.boolean_mask(gbbox_yx, exist_i)
        gt_keypoints_i = tf.cond(
            tf.equal(tf.shape(gbbox_yx_i)[0], 0),
            lambda: zero_like_keyp,
            lambda: tf.expand_dims(tf.sparse.to_dense(tf.sparse.SparseTensor(gbbox_yx_i, tf.ones_like(gbbox_yx_i[..., 0], tf.float32), dense_shape=pshape), validate_indices=False),
                                    axis=-1)
        )
        gt_keypoints.append(gt_keypoints_i)
    reduction = tf.concat(reduction, axis=-1)
    gt_keypoints = tf.concat(gt_keypoints, axis=-1)
    keypoints_pos_loss = -tf.pow(1.-tf.sigmoid(keypoints), 2.) * tf.log_sigmoid(keypoints) * gt_keypoints
    keypoints_neg_loss = -tf.pow(1.-reduction, 4) * tf.pow(tf.sigmoid(keypoints), 2.) * (-keypoints+tf.log_sigmoid(keypoints)) * (1.-gt_keypoints)
    keypoints_loss = tf.reduce_sum(keypoints_pos_loss) / tf.cast(num_g, tf.float32) + tf.reduce_sum(keypoints_neg_loss) / tf.cast(num_g, tf.float32)
    return keypoints_loss

# from cornernet height width 是目标框的长宽
def _gaussian_radius(self, height, width, min_overlap=0.7):
    a1 = 1.
    b1 = (height + width)
    c1 = width * height * (1. - min_overlap) / (1. + min_overlap)
    sq1 = tf.sqrt(b1 ** 2. - 4. * a1 * c1)
    r1 = (b1 + sq1) / 2.

    a2 = 4.
    b2 = 2. * (height + width)
    c2 = (1. - min_overlap) * width * height
    sq2 = tf.sqrt(b2 ** 2. - 4. * a2 * c2)
    r2 = (b2 + sq2) / 2.

    a3 = 4. * min_overlap
    b3 = -2. * min_overlap * (height + width)
    c3 = (min_overlap - 1.) * width * height
    sq3 = tf.sqrt(b3 ** 2. - 4. * a3 * c3)
    r3 = (b3 + sq3) / 2.
    return tf.reduce_min([r1, r2, r3])




