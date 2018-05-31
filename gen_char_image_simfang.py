# -*- coding: utf-8 -*-

import sys, os, shutil, cv2,gc,h5py
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import datetime
import numpy as np
from cProfile import Profile
from logic_tool import gen_dir, create_subdir, create_dir, get_main_sub_files


def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def rd(num2=10):
    return random.randint(0,num2)


def resize_img(img, size):
    resized_img=cv2.resize(img,size)
    resized_img=resized_img.astype(float)
    return resized_img


def getLTRB_c255b0(img):
    img2=np.array(img)
    top = 0
    bottom = 0

    left = 0
    right = 0

    horizontal_sum = np.sum(img2, axis=1)
    start=False
    for i,x in enumerate(horizontal_sum):
        if not start and x:
            top=i
            start=True
        if start and x:
            bottom=i-1

    vertical_sum = np.sum(img2, axis=0)
    start=False
    for i,x in enumerate(vertical_sum):
        if not start and x:
            left=i
            start=True
        if start and x:
            right=i-1

    return (left, top, right, bottom)


def img_color_reverse(img):
    w, h = img.size
    img_r = Image.new('L', (w, h), 255)
    for j in range(h):
        for i in range(w):
            v = img.getpixel((i, j))
            img_r.putpixel((i, j), 255 - v)
    return img_r


def rndColor(base, noise):
    left = base - noise
    right = base + noise

    if left < 0:
        left = 0

    if right > 255:
        right = 255

    return random.randint(left, right)


def gen_bkc_fgc_img(img_neg, bkc, fgc):
    w, h = img_neg.size
    img = Image.new('L', (w, h), bkc)
    for j in range(h):
        for i in range(w):
            v = img_neg.getpixel((i, j))
            if v:
                v = 255 - v

                vn = int(fgc + v / 255.0 * (bkc - fgc) + 0.5)

                img.putpixel((i, j), vn)

    return img


def my_gray_image_add_noise(im, noise):
    w, h = im.size
    img = Image.new('L', (w, h), 255)
    for x in range(w):
        for y in range(h):
            gray = im.getpixel((x, y))
            img.putpixel((x, y), rndColor(gray, noise))
    return img


def gen_my_key_value_map_file(input_root, output_root):
    png_list = get_main_sub_files(input_root, '.png')

    pf = open(output_root + '/' + 'char_0x_intv_eng_else.map', 'w')
    for png_name in png_list:
        array = png_name.split('_')
        arr = array[0].lower()

        int_v = int(arr, 16)  # int value
        u16_code = hex(int_v)  # 0x****
        unicode_code = str(int_v)  # han zi

        if 6 != len(arr):
            print(unicode_code, ' ', u16_code, ' ', int_v)

        pf.write('%s %s %d\n' % (unicode_code.encode('utf-8'), u16_code, int_v))
    pf.close()


def gen_train_data(map_file, output_root):
    #######
    # load dic
    l_char_unicode = []
    pf = open(map_file, 'r')
    line = True
    while line:
        line = pf.readline()
        if '' == line:
            break

        char_utf8 = line.split(' ')[0]
        ###print char_utf8
        l_char_unicode.append(str(char_utf8, 'utf-8'))
        font_char_date_root = create_subdir(output_root, str(char_utf8, 'utf-8'))

    n_labels = len(l_char_unicode)
    n_train_samples = 3000 * n_labels
    n_test_samples = 1000 * n_labels


    edge_noise = 7
    noise = 8
    score_pos=0
    size=(32,32)

    HDF5_FILE_PATH = '/media/xiaolong/f00e08ae-0ab6-4e2a-a98c-1fc206151090/samples_%d_%d_%d.h5' % (
    n_labels, n_train_samples, n_test_samples)
    # HDF5_FILE_PATH = '/media/xiaolong/f00e08ae-0ab6-4e2a-a98c-1fc206151090/samples75.h5'
    # HDF5_FILE_PATH = '/media/xiaolong/f00e08ae-0ab6-4e2a-a98c-1fc206151090/samples10.h5'

    f = h5py.File(HDF5_FILE_PATH, 'w')

    X_train = f.create_dataset('X_train', (n_train_samples, size[0], size[1]), dtype='f')
    X_test = f.create_dataset('X_test', (n_test_samples, size[0], size[1]), dtype='f')
    Y_train = f.create_dataset('Y_train', (n_train_samples, n_labels), dtype='f')
    Y_test = f.create_dataset('Y_test', (n_test_samples, n_labels), dtype='f')

    l_char_labels = np.zeros(((n_train_samples+n_test_samples)/(5*n_labels), n_labels), 'int')
    for u in range(l_char_labels.shape[0]):
        l_char_labels[u] = range(n_labels)
    l_char_labels = l_char_labels.flatten()
    np.random.shuffle(l_char_labels)

    l_char_times = np.zeros(n_labels, 'int')
    n_trains, n_tests = 0, 0
    for char_label in l_char_labels:
        char_unicode = l_char_unicode[char_label]
        l_char_times[char_label] += 5
        a_category = np.zeros(n_labels, dtype=float)
        a_category[char_label] = 1.0
        # date_root = create_subdir(output_root, date_str)
        data_char=[]

        bkc = random.randint(125, 255)
        fgc = random.randint(0, 125)
        while bkc - fgc < 120:
            bkc = random.randint(125, 255)
            fgc = random.randint(0, 125)

        font_name = random.choice(font_names)
        font_size = random.randint(36, 48)
        ImageSize = font_size * 4
        rotate = random.randint(0,1)
        perspective = random.randint(0,1)
        fftt = ImageFont.truetype(TTF_location + '/' + font_name, font_size)

        ### 黑底
        paper = Image.new('L', (ImageSize, ImageSize), 0)
        drawer = ImageDraw.Draw(paper)

        drawer.text((font_size, font_size), char_unicode, font=fftt, fill=255)

        if rotate:
            im_r = paper.rotate(random.randint(-10,10), Image.BICUBIC, True)
        else:
            im_r = paper.rotate(random.randint(-5,5), Image.BICUBIC, True)

        width, height = im_r.size
        if perspective:
            h = min(width, height) / 5
            coeffs = find_coeffs(
                [(0 + rd(h), 0 + rd(h)), (width - rd(h), 0 + rd(h)),
                 (width - rd(h), height - rd(h)), (0 + rd(h), height - rd(h))],
                [(0, 0), (width, -0), (width, height), (-0, height)])
            im_crop_transformed = im_r.transform((width, height), Image.PERSPECTIVE, coeffs,
                                                    Image.BICUBIC)
        else:
            im_crop_transformed = im_r

        width, height = im_crop_transformed.size

        im_name = output_root + '/' + char_unicode + '/' + char_unicode

        (left, top, right, bottom) = getLTRB_c255b0(im_crop_transformed)
        roi_w = right - left + 1
        roi_h = bottom - top + 1

        add_xL = edge_noise
        add_xR = add_xL
        add_yT = edge_noise
        add_yB = add_xR

        left -= add_xL
        right += add_xR
        top -= add_yT
        bottom += add_yB

        if left < 0 or top < 0 or right > width or bottom > height:
            print('%s r = %d too big !!!!!!' % (char_unicode.encode('utf-8'), rotate))

        im_crop = im_crop_transformed.crop((left, top, right + 1, bottom + 1))


        w_t, h_t = im_crop.size
        box_t = (random.randint(3,6), random.randint(3,6),
                 w_t - random.randint(3,6), h_t - random.randint(3,6))

        ### raw
        im_bkc_fgc = gen_bkc_fgc_img(im_crop, bkc, fgc)

        im_t = im_bkc_fgc.crop(box_t)
        # im_t.show()
        im_t.save('%s_%s_r%d_b%d_f%d_t%d.png' % (im_name, font_name.split('.')[0], rotate, bkc, fgc, perspective), 'png')
        resized_img = resize_img(np.array(im_t),size)
        data_char.append((resized_img, a_category))

        ### noise from raw
        im_noise = my_gray_image_add_noise(im_bkc_fgc, noise)

        im_t = im_noise.crop(box_t)
        im_t.save('%s_%s_r%d_b%d_f%d_t%d_n%d.png' % (im_name, font_name.split('.')[0], rotate, bkc, fgc, perspective, noise), 'png')
        resized_img = resize_img(np.array(im_t), size)
        data_char.append((resized_img, a_category))

        ### smooth from noise
        im_smooth = im_noise.filter(ImageFilter.SMOOTH)

        im_t = im_smooth.crop(box_t)
        im_t.save('%s_%s_r%d_b%d_f%d_t%d_n%d_smooth.png' % (im_name, font_name.split('.')[0], rotate, bkc, fgc, perspective, noise), 'png')
        resized_img = resize_img(np.array(im_t), size)
        data_char.append((resized_img, a_category))

        ### smooth more from noise
        im_smooth_more = im_noise.filter(ImageFilter.SMOOTH_MORE)

        im_t = im_smooth_more.crop(box_t)
        im_t.save('%s_%s_r%d_b%d_f%d_t%d_n%d_smoothmore.png' % (im_name, font_name.split('.')[0], rotate, bkc, fgc, perspective, noise), 'png')
        resized_img = resize_img(np.array(im_t), size)
        data_char.append((resized_img, a_category))

        ### gaussion r = 1 from noise
        r = 1
        im_gaussian1 = im_noise.filter(ImageFilter.GaussianBlur(radius=r))

        im_t = im_gaussian1.crop(box_t)
        im_t.save('%s_%s_r%d_b%d_f%d_t%d_n%d_gaussian%d.png' % (im_name, font_name.split('.')[0], rotate, bkc, fgc, perspective, noise, r), 'png')
        resized_img = resize_img(np.array(im_t), size)
        data_char.append((resized_img, a_category))

        ### gaussion r = 1 from noise
        # r = 2
        # im_gaussian2 = im_noise.filter(ImageFilter.GaussianBlur(radius=r))
        #
        # im_t = im_gaussian2.crop(box_t)
        # im_t.save('%s_%s_r%d_b%d_f%d_t%d_n%d_gaussian%d.png' % (im_name, font_name.split('.')[0], rotate, bkc, fgc, perspective, noise, r), 'png')

        # random.shuffle(data_char)
        for n in range(5):


            if l_char_times[char_label] <= n_train_samples/n_labels:
                X_train[n_trains] = data_char[n][0]
                Y_train[n_trains] = data_char[n][1]
                n_trains += 1
            else:
                X_test[n_tests] = data_char[n][0]
                Y_test[n_tests] = data_char[n][1]
                n_tests+=1

        score_pos+=5

    f.close()
    # print 'n_sample =',n, len(np_data[0]),len(np_data[1])
    # import tes2
    # tes2.run(HDF5_FILE_PATH,n_train_samples, n_test_samples, n_labels)


TTF_location = u'/home/xiaolong/下载'
font_names = ['simsun.ttc', 'STSONG.TTF', 'STZHONGS.TTF', 'STFANGSO.TTF', 'simfang.ttf']

if __name__ == '__main__':

    ### 时间标签
    date_str = '20171029'
    ### 主输出路径
    output_root = '/media/xiaolong/f00e08ae-0ab6-4e2a-a98c-1fc206151090/char_datasets'
    # output_root = '/media/xiaolong/f00e08ae-0ab6-4e2a-a98c-1fc206151090/char10'
    map_file = './char_set/char_0x_intv2.map'
    # map_file = './char_set/char_0x_intv_test.map'
    # map_file = './char_set/test.map'
    # dic_path = "dic_test.txt"
    gen_train_data(map_file, output_root)
    # prof = Profile()
    # prof.run("gen_train_data(map_file, font_size, output_root, date_str)")
    # prof.print_stats()
