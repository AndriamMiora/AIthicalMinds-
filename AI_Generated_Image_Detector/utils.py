import numpy as np
import cv2
import PIL.Image
from scipy.interpolate import griddata

def RGB2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def img_to_patches(img: PIL.Image.Image) -> tuple:
    patch_size = 16
    img = img.convert('RGB')

    grayscale_imgs = []
    imgs = []
    coordinates = []

    for i in range(0, img.height, patch_size):
        for j in range(0, img.width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            img_color = np.asarray(img.crop(box))
            grayscale_image = cv2.cvtColor(src=img_color, code=cv2.COLOR_RGB2GRAY)
            grayscale_imgs.append(grayscale_image.astype(dtype=np.int32))
            imgs.append(img_color)
            normalized_coord = (i + patch_size // 2, j + patch_size // 2)
            coordinates.append(normalized_coord)

    return grayscale_imgs, imgs, coordinates, (img.height, img.width)

def get_l1(v):
    return np.sum(np.abs(v[:, :-1] - v[:, 1:]))

def get_l2(v):
    return np.sum(np.abs(v[:-1, :] - v[1:, :]))

def get_l3l4(v):
    l3 = np.sum(np.abs(v[:-1, :-1] - v[1:, 1:]))
    l4 = np.sum(np.abs(v[1:, :-1] - v[:-1, 1:]))
    return l3 + l4

def get_pixel_var_degree_for_patch(patch: np.array) -> int:
    l1 = get_l1(patch)
    l2 = get_l2(patch)
    l3l4 = get_l3l4(patch)
    return l1 + l2 + l3l4

def get_rich_poor_patches(img: PIL.Image.Image, coloured=True):
    gray_scale_patches, color_patches, coordinates, img_size = img_to_patches(img)
    var_with_patch = []
    for i, patch in enumerate(gray_scale_patches):
        if coloured:
            var_with_patch.append((get_pixel_var_degree_for_patch(patch), color_patches[i], coordinates[i]))
        else:
            var_with_patch.append((get_pixel_var_degree_for_patch(patch), patch, coordinates[i]))

    var_with_patch.sort(reverse=True, key=lambda x: x[0])
    mid_point = len(var_with_patch) // 2
    r_patch = [(patch, coor) for var, patch, coor in var_with_patch[:mid_point]]
    p_patch = [(patch, coor) for var, patch, coor in var_with_patch[mid_point:]]
    p_patch.reverse()
    return r_patch, p_patch, img_size

def azimuthalAverage(image, center=None):
    y, x = np.indices(image.shape)
    if not center:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    r = np.hypot(x - center[0], y - center[1])
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]
    r_int = r_sorted.astype(int)
    deltar = r_int[1:] - r_int[:-1]
    rind = np.where(deltar)[0]
    nr = rind[1:] - rind[:-1]
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]
    radial_prof = tbin / nr
    return radial_prof

def azimuthal_integral(img, epsilon=1e-8, N=50):
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = RGB2gray(img)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift += epsilon
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    psd1D = azimuthalAverage(magnitude_spectrum)
    points = np.linspace(0, N, num=psd1D.size)
    xi = np.linspace(0, N, num=N)
    interpolated = griddata(points, psd1D, xi, method='cubic')
    interpolated = (interpolated - np.min(interpolated)) / (np.max(interpolated) - np.min(interpolated))
    return interpolated.astype(np.float32)

def positional_emb(coor, im_size, N):
    img_height, img_width = im_size
    center_y, center_x = coor
    normalized_y = center_y / img_height
    normalized_x = center_x / img_width
    pos_emb = np.zeros(N)
    indices = np.arange(N)
    div_term = 10000 ** (2 * (indices // 2) / N)
    pos_emb[0::2] = np.sin(normalized_y / div_term[0::2]) + np.sin(normalized_x / div_term[0::2])
    pos_emb[1::2] = np.cos(normalized_y / div_term[1::2]) + np.cos(normalized_x / div_term[1::2])
    return pos_emb

def azi_diff(img: PIL.Image.Image, patch_num, N):
    r, p, im_size = get_rich_poor_patches(img)
    r_len = len(r)
    p_len = len(p)
    patch_emb_r = np.zeros((patch_num, N))
    patch_emb_p = np.zeros((patch_num, N))
    positional_emb_r = np.zeros((patch_num, N))
    positional_emb_p = np.zeros((patch_num, N))
    coor_r = []
    coor_p = []
    if r_len != 0:
        for idx in range(patch_num):
            tmp_patch1 = r[idx % r_len][0]
            tmp_coor1 = r[idx % r_len][1]
            patch_emb_r[idx] = azimuthal_integral(tmp_patch1, N=N)
            positional_emb_r[idx] = positional_emb(tmp_coor1, im_size, N)
            coor_r.append(tmp_coor1)
    if p_len != 0:
        for idx in range(patch_num):
            tmp_patch2 = p[idx % p_len][0]
            tmp_coor2 = p[idx % p_len][1]
            patch_emb_p[idx] = azimuthal_integral(tmp_patch2, N=N)
            positional_emb_p[idx] = positional_emb(tmp_coor2, im_size, N)
            coor_p.append(tmp_coor2)
    output = {"total_emb": [patch_emb_r + positional_emb_r / 5, patch_emb_p + positional_emb_p / 5],
              "positional_emb": [positional_emb_r / 5, positional_emb_p / 5], "coor": [coor_r, coor_p],
              "image_size": im_size}
    return output