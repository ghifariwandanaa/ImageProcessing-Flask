import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from collections import Counter
from pylab import savefig
import cv2
import matplotlib
from skimage.morphology import skeletonize
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
matplotlib.use('Agg') 


def grayscale():
    if not is_grey_scale("static/img/img_now.jpg"):
        img = Image.open("static/img/img_now.jpg")
        img_arr = np.asarray(img)
        r = img_arr[:, :, 0]
        g = img_arr[:, :, 1]
        b = img_arr[:, :, 2]
        new_arr = r.astype(int) + g.astype(int) + b.astype(int)
        new_arr = (new_arr/3).astype('uint8')
        new_img = Image.fromarray(new_arr)
        new_img.save("static/img/img_now.jpg")


def is_grey_scale(img_path):
    im = Image.open(img_path).convert('RGB')
    w, h = im.size
    for i in range(w):
        for j in range(h):
            r, g, b = im.getpixel((i, j))
            if r != g != b:
                return False
    return True

def zoomin():
    if is_grey_scale("static/img/img_now.jpg"):
        img = Image.open("static/img/img_now.jpg")
        img_arr = np.asarray(img)
        
        if len(img_arr.shape) == 3:
            # Convert grayscale to 2D array
            img_arr = img_arr[:, :, 0]

        new_size = (img_arr.shape[0] * 2, img_arr.shape[1] * 2)
        new_arr = np.full(new_size, 255, dtype=np.uint8)

        for i in range(len(img_arr)):
            for j in range(len(img_arr[i])):
                new_arr[2*i, 2*j] = img_arr[i, j]
                new_arr[2*i, 2*j+1] = img_arr[i, j]
                new_arr[2*i+1, 2*j] = img_arr[i, j]
                new_arr[2*i+1, 2*j+1] = img_arr[i, j]

        new_img = Image.fromarray(new_arr)
        new_img.save("static/img/img_now.jpg")
    else:
        img = Image.open("static/img/img_now.jpg")
        img = img.convert("RGB")
        img_arr = np.asarray(img)
        new_size = ((img_arr.shape[0] * 2),
                    (img_arr.shape[1] * 2), img_arr.shape[2])
        new_arr = np.full(new_size, 255)
        new_arr.setflags(write=1)

        r = img_arr[:, :, 0]
        g = img_arr[:, :, 1]
        b = img_arr[:, :, 2]

        new_r = []
        new_g = []
        new_b = []

        for row in range(len(r)):
            temp_r = []
            temp_g = []
            temp_b = []
            for i in r[row]:
                temp_r.extend([i, i])
            for j in g[row]:
                temp_g.extend([j, j])
            for k in b[row]:
                temp_b.extend([k, k])
            for _ in (0, 1):
                new_r.append(temp_r)
                new_g.append(temp_g)
                new_b.append(temp_b)

        for i in range(len(new_arr)):
            for j in range(len(new_arr[i])):
                new_arr[i, j, 0] = new_r[i][j]
                new_arr[i, j, 1] = new_g[i][j]
                new_arr[i, j, 2] = new_b[i][j]

        new_arr = new_arr.astype('uint8')
        new_img = Image.fromarray(new_arr)
        new_img.save("static/img/img_now.jpg")


def zoomout():
    if is_grey_scale("static/img/img_now.jpg"):
        img = Image.open("static/img/img_now.jpg")
        img_arr = np.asarray(img)
        
        if len(img_arr.shape) == 3:
            # Convert grayscale to 2D array
            img_arr = img_arr[:, :, 0]

        x, y = img_arr.shape
        new_arr = np.zeros((int(x / 2), int(y / 2)), dtype=np.uint8)

        for i in range(0, int(x/2)):
            for j in range(0, int(y/2)):
                new_arr[i, j] = np.mean(img_arr[2*i:2*i+2, 2*j:2*j+2])

        new_img = Image.fromarray(new_arr)
        new_img.save("static/img/img_now.jpg")
    else:
        img = Image.open("static/img/img_now.jpg")
        img = img.convert("RGB")
        x, y = img.size
        new_arr = Image.new("RGB", (int(x / 2), int(y / 2)))
        r = [0, 0, 0, 0]
        g = [0, 0, 0, 0]
        b = [0, 0, 0, 0]

        for i in range(0, int(x/2)):
            for j in range(0, int(y/2)):
                r[0], g[0], b[0] = img.getpixel((2 * i, 2 * j))
                r[1], g[1], b[1] = img.getpixel((2 * i + 1, 2 * j))
                r[2], g[2], b[2] = img.getpixel((2 * i, 2 * j + 1))
                r[3], g[3], b[3] = img.getpixel((2 * i + 1, 2 * j + 1))
                new_arr.putpixel((int(i), int(j)), (int((r[0] + r[1] + r[2] + r[3]) / 4), int(
                    (g[0] + g[1] + g[2] + g[3]) / 4), int((b[0] + b[1] + b[2] + b[3]) / 4)))
        new_arr = np.uint8(new_arr)
        new_img = Image.fromarray(new_arr)
        new_img.save("static/img/img_now.jpg")

def move_left():
    img_path = "static/img/img_now.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img)
    if is_grey_scale(img_path):
        g = img_arr[:, :]
        g = np.pad(g, ((0, 0), (0, 50)), 'constant')[:, 50:]
        new_arr = g
    else:
        r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
        r = np.pad(r, ((0, 0), (0, 50)), 'constant')[:, 50:]
        g = np.pad(g, ((0, 0), (0, 50)), 'constant')[:, 50:]
        b = np.pad(b, ((0, 0), (0, 50)), 'constant')[:, 50:]
        new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_right():
    img_path = "static/img/img_now.jpg"
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    if is_grey_scale(img_path):
        g = img_arr[:, :]
        g = np.pad(g, ((0, 0), (50, 0)), 'constant')[:, :-50]
        new_arr = g
    else:
        r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
        r = np.pad(r, ((0, 0), (50, 0)), 'constant')[:, :-50]
        g = np.pad(g, ((0, 0), (50, 0)), 'constant')[:, :-50]
        b = np.pad(b, ((0, 0), (50, 0)), 'constant')[:, :-50]
        new_arr = np.dstack((r, g, b))        
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_up():
    img_path = "static/img/img_now.jpg"
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    if is_grey_scale(img_path):
        r, g, b = img_arr[:, :], img_arr[:, :], img_arr[:, :]
    else:
        r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 50), (0, 0)), 'constant')[50:, :]
    g = np.pad(g, ((0, 50), (0, 0)), 'constant')[50:, :]
    b = np.pad(b, ((0, 50), (0, 0)), 'constant')[50:, :]
    if is_grey_scale(img_path):
        new_arr = r
    else:
        new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_down():
    img_path = "static/img/img_now.jpg"
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    if is_grey_scale(img_path):
        g = img_arr[:, :]
        g = np.pad(g, ((50, 0), (0, 0)), 'constant')[0:-50, :]
        new_arr = r
    else:
        r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
        r = np.pad(r, ((50, 0), (0, 0)), 'constant')[0:-50, :]
        g = np.pad(g, ((50, 0), (0, 0)), 'constant')[0:-50, :]
        b = np.pad(b, ((50, 0), (0, 0)), 'constant')[0:-50, :]
        new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_addition():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img).astype('uint16')
    img_arr = img_arr+100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_substraction():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img).astype('int16')
    img_arr = img_arr-100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_multiplication():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    img_arr = img_arr*1.25
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_division():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    img_arr = img_arr/1.25
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def convolution(img, kernel):
    if len(img.shape) == 3 and img.shape[2] == 3:  # Color image
        h_img, w_img, _ = img.shape
        out = np.zeros((h_img-2, w_img-2), dtype=np.float64)
        new_img = np.zeros((h_img-2, w_img-2, 3))
        if np.array_equal((img[:, :, 1], img[:, :, 0]), img[:, :, 2]) == True:
            array = img[:, :, 0]
            for h in range(h_img-2):
                for w in range(w_img-2):
                    S = np.multiply(array[h:h+3, w:w+3], kernel)
                    out[h, w] = np.sum(S)
            out_ = np.clip(out, 0, 255)
            for channel in range(3):
                new_img[:, :, channel] = out_
        else:
            for channel in range(3):
                array = img[:, :, channel]
                for h in range(h_img-2):
                    for w in range(w_img-2):
                        S = np.multiply(array[h:h+3, w:w+3], kernel)
                        out[h, w] = np.sum(S)
                out_ = np.clip(out, 0, 255)
                new_img[:, :, channel] = out_
        new_img = np.uint8(new_img)
        return new_img

    elif len(img.shape) == 2:  # Grayscale image
        h_img, w_img = img.shape
        out = np.zeros((h_img-2, w_img-2), dtype=np.float64)

        for h in range(h_img-2):
            for w in range(w_img-2):
                S = np.multiply(img[h:h+3, w:w+3], kernel)
                out[h, w] = np.sum(S)

        out_ = np.clip(out, 0, 255)
        new_img = np.uint8(out_)
        return new_img

    else:
        raise ValueError("Unsupported image format")

def edge_detection():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img, dtype=np.int64)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")

def blur():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img, dtype=np.int64)
    kernel = np.array(
        [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def sharpening():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img, dtype=np.int64)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def histogram_rgb():
    img_path = "static/img/img_now.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img)
    if is_grey_scale(img_path):
        g = img_arr[:, :].flatten()
        data_g = Counter(g)
        plt.bar(list(data_g.keys()), data_g.values(), color='black')
        plt.savefig(f'static/img/grey_histogram.jpg', dpi=300)
        plt.clf()
    else:
        r = img_arr[:, :, 0].flatten()
        g = img_arr[:, :, 1].flatten()
        b = img_arr[:, :, 2].flatten()
        data_r = Counter(r)
        data_g = Counter(g)
        data_b = Counter(b)
        data_rgb = [data_r, data_g, data_b]
        warna = ['red', 'green', 'blue']
        data_hist = list(zip(warna, data_rgb))
        for data in data_hist:
            plt.bar(list(data[1].keys()), data[1].values(), color=f'{data[0]}')
            plt.savefig(f'static/img/{data[0]}_histogram.jpg', dpi=300)
            plt.clf()


def df(img):  # to make a histogram (count distribution frequency)
    values = [0]*256
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            values[img[i, j]] += 1
    return values


def cdf(hist):  # cumulative distribution frequency
    cdf = [0] * len(hist)  # len(hist) is 256
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i-1]+hist[i]
    # Now we normalize the histogram
    # What your function h was doing before
    cdf = [ele*255/cdf[-1] for ele in cdf]
    return cdf


def histogram_equalizer():
    img = cv2.imread('static\img\img_now.jpg', 0)
    my_cdf = cdf(df(img))
    # use linear interpolation of cdf to find new pixel values. Scipy alternative exists
    image_equalized = np.interp(img, range(0, 256), my_cdf)
    cv2.imwrite('static/img/img_now.jpg', image_equalized)


def threshold(lower_thres, upper_thres):
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    condition = np.logical_and(np.greater_equal(img_arr, lower_thres),
                               np.less_equal(img_arr, upper_thres))
    print(lower_thres, upper_thres)
    img_arr = np.asarray(img).copy()
    img_arr[condition] = 255
    new_img = Image.fromarray(img_arr)
    new_img.save("static/img/img_now.jpg")

def dilasi():
    # Baca citra
    # Baca dalam skala BINER
    img = cv2.imread("static/img/img_now.jpg", cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
    kernel_size = 3

    # Definisikan kernel untuk erosi
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Lakukan erosi
    dilasi_img = cv2.dilate(binary_image, kernel, iterations=3)

    # Simpan citra yang telah dierosi
    cv2.imwrite("static/img/img_now.jpg", dilasi_img)


def erosi():
    # Baca citra
    # Baca dalam skala BINER
    img = cv2.imread("static/img/img_now.jpg", cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
    kernel_size = 3

    # Definisikan kernel untuk erosi
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Lakukan erosi
    erosi_img = cv2.erode(binary_image, kernel, iterations=3)

    # Simpan citra yang telah dierosi
    cv2.imwrite("static/img/img_now.jpg", erosi_img)


def Opening():
    # Baca citra
    # Baca dalam skala BINER
    img = cv2.imread("static/img/img_now.jpg", cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
    kernel_size = 3

    # Definisikan kernel untuk erosi
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Lakukan erosi
    erosi_img = cv2.erode(binary_image, kernel, iterations=3)

    opening_img = cv2.dilate(erosi_img, kernel, iterations=3)

    # Simpan citra yang telah dierosi
    cv2.imwrite("static/img/img_now.jpg", opening_img)


def Closing():
    # Baca citra
    # Baca dalam skala BINER
    img = cv2.imread("static/img/img_now.jpg", cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
    kernel_size = 3

    # Definisikan kernel untuk erosi
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    dilasi_img = cv2.dilate(binary_image, kernel, iterations=3)

    # Lakukan erosi
    closing_img = cv2.erode(dilasi_img, kernel, iterations=3)

    # Simpan citra yang telah dierosi
    cv2.imwrite("static/img/img_now.jpg", closing_img)


def count_white_objects():
    # Baca citra dalam skala keabuan
    img = cv2.imread("static/img/img_now.jpg", cv2.IMREAD_GRAYSCALE)

    _, img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Lakukan operasi morfologi untuk membersihkan gambar
    kernel = np.ones((5, 5), np.uint8)
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)

    # Temukan kontur dalam citra
    contours, _ = cv2.findContours(
        img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hitung jumlah objek putih
    num_white_objects = len(contours)
    print("Jumlah objek putih:", num_white_objects)
    return num_white_objects


# # Fungsi untuk pra-pemrosesan citra
def pra_pemrosesan_citra(lokasi_citra):
    # Membaca citra
    citra = cv2.imread(lokasi_citra, cv2.IMREAD_GRAYSCALE)
    # Mengubah citra menjadi citra biner menggunakan metode deteksi tepi
    _, citra_biner = cv2.threshold(citra, 128, 255, cv2.THRESH_BINARY_INV)
    # Mencari kontur citra
    kontur, _ = cv2.findContours(citra_biner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Memilih kontur terbesar
    kontur_terbesar = max(kontur, key=cv2.contourArea)
    # print(kontur_terbesar)
    return kontur_terbesar

def hitung_freeman_chain_code(kontur):
    kode_chain = []
    titik_valid = []
    arah = [0, 7, 6, 5, 4, 3, 2, 1]
    perubahan = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

    # Mendapatkan titik-titik kontur
    titik = [tuple(poin[0]) for poin in kontur]

    # Menemukan titik awal (titik paling kiri)
    indeks_awal = titik.index(min(titik, key=lambda x: x[0]))

    titik_sekarang = titik[indeks_awal]
    titik_awal = titik[indeks_awal]

    # Mengikuti kontur dan menghasilkan kode chain
    while True:
        found = False
        ra={0,1,2,3,4,5,6,7}
        for i in ra:
            next_point = (titik_sekarang[0] + perubahan[i][0], titik_sekarang[1] + perubahan[i][1])
            if next_point in titik:
                found = True
                if next_point in titik_valid:
                    continue
                else :
                    titik_valid.append(next_point)
                    break
        if not found:
            break
        kode_chain.append(arah[i])  
        titik_sekarang = next_point
        if titik_sekarang == titik_awal:
            break

    return kode_chain

# Fungsi untuk penipisan citra menggunakan thinning
def penipisan_citra(lokasi_citra):
    citra = cv2.imread(lokasi_citra, cv2.IMREAD_GRAYSCALE)
    _, citra_biner = cv2.threshold(citra, 128, 255, cv2.THRESH_BINARY_INV)
    citra_penipisan = skeletonize(citra_biner)
    # Konversi tipe data citra menjadi uint8
    citra_penipisan = citra_penipisan.astype(np.uint8)
    kontur, _ = cv2.findContours(citra_penipisan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    citra_penipisan = max(kontur, key=cv2.contourArea)
    return citra_penipisan

# Fungsi untuk pengenalan digit
def kenali_digit(kode_chain, basis_pengetahuan):
    jarak_minimum = float('inf')
    digit_terkenali = None
    for digit, referensi_kode_chain in basis_pengetahuan.items():
        jarak = sum(1 for a, b in zip(kode_chain, referensi_kode_chain) if a != b)
        if jarak < jarak_minimum:
            jarak_minimum = jarak
            digit_terkenali = digit
    return digit_terkenali

# Fungsi untuk melakukan pengujian pengenalan angka
def uji_pengenalan_angka(daftar_citra_uji, basis_pengetahuan):
    digit=[]
    for lokasi_citra_uji in daftar_citra_uji:
        kode_chain_uji = hitung_freeman_chain_code(pra_pemrosesan_citra(lokasi_citra_uji))
        digit_terkenali = kenali_digit(kode_chain_uji, basis_pengetahuan)
        digit.append(digit_terkenali)
        # print(f"Digit terkenali untuk {lokasi_citra_uji}: {digit_terkenali}")
    joined_digits = ''.join(map(str, digit))
    print("digit dikenali =", joined_digits)
    

    # Fungsi untuk menyimpan basis pengetahuan ke dalam file JSON
def     simpan_ke_json(basis_pengetahuan, nama_file):
    with open(nama_file, "w") as file:
        json.dump(basis_pengetahuan, file)

def segmentasi_digit(lokasi_citra):
    # Membaca citra
    citra = cv2.imread(lokasi_citra, cv2.IMREAD_GRAYSCALE)
    # Mengubah citra menjadi citra biner menggunakan metode deteksi tepi
    _, citra_biner = cv2.threshold(citra, 128, 255, cv2.THRESH_BINARY_INV)
    # Mencari kontur citra
    kontur, _ = cv2.findContours(citra_biner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Mengurutkan kontur berdasarkan koordinat x dari kiri ke kanan
    kontur_terurut = sorted(kontur, key=lambda c: cv2.boundingRect(c)[0])
    return kontur_terurut


def kenali_digit_segmentasi(daftar_kontur, basis_pengetahuan):
    digit = []
    for kontur in daftar_kontur:
        kode_chain = hitung_freeman_chain_code(kontur)
        digit_terkenali = kenali_digit(kode_chain, basis_pengetahuan)
        digit.append(digit_terkenali)
    joined_digits = ''.join(map(str, digit))
    return joined_digits

def uji_pengenalan_angka_segmentasi(daftar_citra_uji, basis_pengetahuan):
    for lokasi_citra_uji in daftar_citra_uji:
        daftar_kontur = segmentasi_digit(lokasi_citra_uji)
        digit_dikenali = kenali_digit_segmentasi(daftar_kontur, basis_pengetahuan)
        # print(f"Digit yang dikenali dari {lokasi_citra_uji}: {digit_dikenali}")
        return digit_dikenali

def training_data():
    kode_chain_freeman_digit = {}
    kode_chain_penipisan_digit = {}
    nama_citra = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    for digit, nama in enumerate(nama_citra):
        lokasi_citra = f"angkas/{nama}.png"
        kontur = pra_pemrosesan_citra(lokasi_citra)
        kode_chain = hitung_freeman_chain_code(kontur)
        kode_chain_freeman_digit[digit] = kode_chain
        
        citra_penipisan = penipisan_citra(lokasi_citra)
        kode_chain = hitung_freeman_chain_code(citra_penipisan)
        kode_chain_penipisan_digit[digit] = kode_chain

    simpan_ke_json(kode_chain_freeman_digit, "knowledge_base_metode1.json")
    simpan_ke_json(kode_chain_penipisan_digit, "knowledge_base_metode2.json")

def load_knowledge_base(filename):
    with open(filename, "r") as file:
        knowledge_base = json.load(file)
    return knowledge_base

def indent_citra():
    kode_chain_freeman_digit = load_knowledge_base("knowledge_base_metode1.json")
    daftar_citra_uji = ["static/img/img_now.jpg"]
    digit = uji_pengenalan_angka_segmentasi(daftar_citra_uji, kode_chain_freeman_digit)
    print("digit yang dikenali",digit)
    return digit

def load_data(dataset_path):
    images = []
    labels = []
    for image_name in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_name)
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)
            # Menggunakan nama file (tanpa ekstensi) sebagai label
            label = image_name.split('.')[0]  
            labels.append(label)
    return np.array(images), np.array(labels)


# Fungsi untuk pra-pemrosesan gambar
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    resized = cv2.resize(gray, (64, 64))  # Ensure consistent image size
    return resized.flatten() / 255.0

# Fungsi untuk melatih model SVM
def train_model(X_train, y_train):
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model

# Fungsi untuk mendeteksi emoji dari gambar baru
def detect_emoji(model, image):
    processed_image = preprocess_image(image)
    prediction = model.predict([processed_image])
    return prediction[0]

# Fungsi utama untuk menjalankan seluruh pipeline
def deteksi_emoji():
    dataset_path = 'emoji'  # Path to your emoji dataset folder
    images, labels = load_data(dataset_path)

    if len(images) == 0 or len(labels) == 0:
        print("Dataset is empty. Please check the dataset path and structure.")
        return
    
    print(f"Loaded {len(images)} images with {len(np.unique(labels))} unique labels.")

    images = np.array([preprocess_image(img) for img in images])
    
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    
    # Evaluasi model
    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    
    # Testing on a new image
    new_image_path = 'static/img/img_now.jpg' # Update this path
    new_image = cv2.imread(new_image_path)
    result = detect_emoji(model, new_image)
    print(f'Detected emoji: {result}')
    return result
