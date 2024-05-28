#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from cv2 import aruco
import cv2 as cv
from cv2 import aruco
import numpy as np
from tkinter import Tk, Label, Button, OptionMenu, StringVar, IntVar, Checkbutton
from PIL import Image as Img
from PIL import ImageTk

# Библиотеки, которые использовались для подключения и взаимодействия с роботом
# import rospy
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# from std_msgs.msg import String

import time


global_dict = {  # Словарь со всеми глобальными переменными
    "root": None,  # Переменная для окна интерфейса
    'width': 0,  # Ширина изображения с камеры
    'height': 0,  # Высота изображения с камеры
    "start_flag": False,  # Bool переменная для начала движения
    "clear_frame": None,  # "Чистое" изображение с камеры для поиска
    "final_frame": None,  # "Чистое" изображение с камеры для наглядного вывода после поиска
    "aruco_flag": False,  # Bool переменная: искать ли Aruco
    "vegie_var": 0,  # 0 - None, 1 - Tomato, 2 - Eggplant, 3 - Both
    "aruco_detector": None,  # Детектор Aruco, нужен для работы библиотеки
    'performance_flag': False,  # Bool переменная: используем ли режим "выступления"
    "performance_last_aruco": -1,  # Последняя найденная Aruco
    "can_see_aruco_rn": False,  # Bool переменная: проверяем, видим ли мы прямо сейчас Aruco (иначе Aruco принимается за баклажаны)
    "vegie_counter": 0,
    "current_aruco": 'none',
    "current_vegie": 'none',
    'vegie_label': None
}

# Класс, используемый для подключения к роботу
# class SighDetector:
#     def __init__(self) -> None:
#         self.image_sub = None
#         self.pub_detect = None
#         self.bridge = None
#         self.rate = None
#         self.image = None
#
#
#     def r_init_node(self):
#         rospy.init_node('sigh_detector')
#         self.rate = rospy.Rate(0.5)
#         self.bridge = CvBridge()
#         self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.callback)
#         self.pub_detect_plant = rospy.Publisher('/plant_result', String, queue_size=10)
#         self.pub_detect_aruco = rospy.Publisher('/last_aruco', String, queue_size=10)
#
#     def callback(self, data):
#         try:
#             cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
#         except CvBridgeError as e:
#             print(e)
#
#         cv_image = cv.rotate(cv_image, cv.ROTATE_90_CLOCKWISE)
#         self.image = cv_image
#
#         # cv.imshow("Image window", self.image)
#         # cv.waitKey(1)
#
#     def start(self):
#         while not rospy.is_shutdown():
#             self.rate.sleep()
#
#         cv.destroyAllWindows()


def drawing_vegies(boxes_lst, txt_var):  # Функция для рисования окантовки вокруг найденных овощей
    global global_dict
    if abs(global_dict["vegie_counter"]) == 20: # Проверка, что мы видим один и тот же овощ на протяжении 20 кадров подряд
        print(f"I'm certain that it's a {txt_var}")
        global_dict["vegie_counter"] = 0
        global_dict['current_vegie'] = txt_var
        global_dict['vegie_label'].config(text=str("Последний сканированный овощ: " + txt_var))
    boxes = boxes_lst
    if len(boxes) != 0:
        left, top = boxes[2], boxes[3]  # Находим координаты углов для окантовки
        right, bottom = boxes[0], boxes[1]  # Находим координаты углов для окантовки

        cv.rectangle(global_dict['final_frame'], (left, top), (right, bottom), (255, 0, 0), 2)  # Отрисовка окантовки
        cv.putText(global_dict['final_frame'], txt_var, (left, bottom), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))  # Подписывание овоща


def drawing_aruco(marker_corners, marker_IDs):  # Функция для рисования окантовки вокруг найденных Aruco
    global global_dict

    for ids, corners in zip(marker_IDs, marker_corners):  # Проходимся циклом по массиву всех найденных ARuco
        corners_check = corners.reshape(4, 2)
        corners_check = corners_check.astype(int)  # Находим координаты углов для окантовки
        top_right = corners_check[0].ravel()
        top_left = corners_check[1].ravel()
        bottom_right = corners_check[2].ravel()
        bottom_left = corners_check[3].ravel()

        global_dict["current_aruco"] = str(ids[0])  # Запоминаем Aruco, который только что увидели

        cv.polylines(global_dict['final_frame'], [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA)  # Рисуем окантовку
        if str(ids[0]) == '20':
            cv.putText(global_dict['final_frame'], 'start', (bottom_left[0], bottom_left[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))  # Подписываем Aruco
        elif str(ids[0]) == '10':
            cv.putText(global_dict['final_frame'], 'finish', (bottom_left[0], bottom_left[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))  # Подписываем Aruco
        else:
            cv.putText(global_dict['final_frame'], str(ids[0]), (bottom_left[0], bottom_left[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))  # Подписываем Aruco
        if global_dict['performance_flag']:
            global_dict['performance_last_aruco'] = ids[0]  # Если мы в режиме "выступления", запоминаем последнюю найденную Aruco


def image_analysis():
    global global_dict

    def img_blur(img):  # Блюр изображения
        blurred = cv.medianBlur(img, 15)
        blurred = cv.blur(blurred, (5, 5))
        return blurred

    def img_gray(img):  # Делаем изображение черно-белым
        grayed = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        return grayed

    def img_thresh(img):  # Threshholding. Если значеи серого пикселя ниже 1 указанного в скобках, оно ставится на 2 в скобках
        global global_dict
        ret, thresh = cv.threshold(img, 160, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY)
        return thresh

    def img_invert(img):  # Инвертирования черно-белого в бело-черное, использовалось раньше, сейчас не нужно
        inverted = cv.bitwise_not(img)
        return inverted

    def img_edges(img):  # Выделяем края цветов черно-белого изображения
        global global_dict
        edged = cv.Canny(img, 60, 125)
        return edged

    starting_img = global_dict["clear_frame"].copy()  # Создаем копию "чистого" изображения

    if global_dict['aruco_flag']:  # Если нам нужно искать Aruco
        img_grayed_no_blur = img_gray(starting_img)  # Делаем изображение серым
        marker_corners, marker_IDs, reject = global_dict["aruco_detector"].detectMarkers(
            img_grayed_no_blur)  # Находим Aruco
        if marker_corners:
            global_dict["can_see_aruco_rn"] = True
            drawing_aruco(marker_corners, marker_IDs)  # Если нашлись, вызываем функцию для их рисования
        else:
            global_dict["can_see_aruco_rn"] = False
            global_dict["current_aruco"] = 'none'

    if (global_dict['vegie_var'] != 0 and not global_dict['performance_flag']) or (
        global_dict['performance_flag'] and global_dict['performance_last_aruco'] == 2):  # Либо ищем нужные овощи, либо мы в режиме "выступления" и уже нашли Aruco с номером 1 (он стоит перед началом каждой грядки)
        img_blurred = img_blur(starting_img)
        img_grayed = img_gray(img_blurred)
        img2analyse = img_thresh(img_grayed)
        copy4mask = global_dict['clear_frame'].copy()
        contours, hierarchy = cv.findContours(img2analyse, mode=cv.RETR_TREE,
                                                method=cv.CHAIN_APPROX_NONE)  # Находим контуры на финальном изображении
        mask = np.zeros(img2analyse.shape, np.uint8)
        boxes = []
        mean_clr_lst = []
        for i in range(0, len(contours)):
            mask[...] = 0
            (x, y, w, h) = cv.boundingRect(contours[i])  # Границы контуров (для обводки)
            if global_dict["width"] * 0.2 <= (x+w)//2 <= global_dict['width']*0.8:
                if not(abs(w-h)<15) and w>100 and h>100 and not h > global_dict['height']*0.75:  # Если размеры найденного контура не больше 100 и контур не похож на квадрат (отсечение нераспознанных Aruco)
                    if boxes == [] or boxes[2]-boxes[0] < w:
                        cv.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
                        masked = cv.bitwise_and(copy4mask, copy4mask, mask=img2analyse)
                        boxes=[x, y, x + w, y + h]
                        mean_clr = cv.mean(masked)  # Считаем средний цвет внутри контуров с помощью накладывания маски на "чистое" изображение
                        mean_clr_lst = [round(mean_clr[0]), round(mean_clr[1]), round(mean_clr[2])] # Удобный для чтения лист RGB значений среднего цвета внутри контуров

        txt_var = 'none'
        if len(mean_clr_lst) != 0:  # Если вообще было найдено хоть что-то
            print(mean_clr_lst)
            if max(mean_clr_lst) == mean_clr_lst[0] and not (mean_clr_lst[1]/mean_clr_lst[0] >= 0.7 or mean_clr_lst[2]/mean_clr_lst[0] >= 0.7):  # Проверяем, что красный цвет - превалирующий в контуре, а количество синего и зеленого знАчимо меньше, чем красного
                if global_dict['vegie_var'] == 1 or global_dict['vegie_var'] == 3:  # Проверяем, что нам этот помидор вообще нужен
                    txt_var = "Tomato"
                    global_dict['vegie_counter'] += 1
                    drawing_vegies(boxes, txt_var)  # Функция отображения окантовки
            else:
                if global_dict['vegie_var'] == 2 or global_dict['vegie_var'] == 3:  # Проверяем, что нам этот баклажан вообще нужен
                    txt_var = "Eggplant"
                    global_dict['vegie_counter'] -= 1
                    drawing_vegies(boxes, txt_var)  # Функция отображения окантовки
        else:
            global_dict['current_vegie'] = 'none'


def create_GUI(cap_width, cap_height):  # Функция, создающая интерфейс
    global global_dict

    def aruco_checkbox_command():  # Функция для ARuco yes/no
        global global_dict
        if with_aruco.get() == 1:
            global_dict["aruco_flag"] = True
        else:
            global_dict["aruco_flag"] = False

    def performance_mode_checkbox():  # Функция для режима "выступления" yes/no
        global global_dict
        if performance_mode.get() == 1:
            global_dict["performance_flag"] = True
            global_dict["aruco_flag"] = True  # Автоматически включает Aruco
        else:
            global_dict["performance_flag"] = False

    def start_button_command():  # Функция для выбора желаемого овоща для поиска
        global global_dict
        v_flag = search_var.get()
        if v_flag == '':
            global_dict['vegie_var'] = 0
        elif v_flag == "Только Помидоры":
            global_dict['vegie_var'] = 1
        elif v_flag == "Только Баклажаны":
            global_dict['vegie_var'] = 2
        else:
            global_dict['vegie_var'] = 3
        global_dict["start_flag"] = True

    root = Tk()  # Создание сущности окна для интерфейса
    root.title('MISIS_Bananabot_GUI')
    root.geometry("800x480")

    search_options_label = Label(text="Выберите опцию для поиска:",  borderwidth=2, relief="ridge")  # Надпись над выбором овоща для поиска
    search_options_label.place(x=int(cap_width)+25, y=5)

    search_options = ["", "Только Помидоры", "Только Баклажаны", "Всё"]  # Сам выбор овоща для поиска
    search_var = StringVar(root)
    search_var.set("")
    search_options_menu = OptionMenu(root, search_var, *search_options)
    search_options_menu.place(x=int(cap_width)+25, y=30)

    start_button = Button(root, text="Старт", bd=5, command=start_button_command)  # Команда для старта поиска
    start_button.place(x=int(cap_width)+85, y=cap_height-35)

    with_aruco = IntVar()  # Aruco yes/no
    with_aruco_checkbox = Checkbutton(root, text="Искать Aruco?", variable=with_aruco, onvalue=1, offvalue=0, command=aruco_checkbox_command)
    with_aruco_checkbox.place(x=int(cap_width)+20, y=80)

    performance_mode = IntVar()  # Режим "выступления" yes/no
    performance_checkbox = Checkbutton(root, text="Режим для выступления", variable=performance_mode, onvalue=1, offvalue=0,
                                      command=performance_mode_checkbox)
    performance_checkbox.place(x=int(cap_width) + 20, y=120)

    last_vegie_label = Label(text="Whatever", borderwidth=2, relief="ridge")  # Надпись над выбором овоща для поиска
    last_vegie_label.place(x=int(cap_width) + 25, y=160)
    global_dict['vegie_label'] = last_vegie_label

    root.geometry(str(cap_width + 300) + 'x' + str(cap_height+5))  # Изменяем размер окна на немного увеличенное, в сравнении с получаемым с камеры изображением

    global_dict["root"] = root  # Сохраняем ссылку на окно в глобальный словарь


def create_aruco_detector():  # Создаем и сохраняем в глобальный словарь детектор для Aruco
    global global_dict
    
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_250)
    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(dictionary, parameters)
    global_dict["aruco_detector"] = detector
    

def start():  # Функция старта всех необходимых прочих функций перед началом основного цикла

    create_aruco_detector()  # Создаем детектор для Aruco
    cap = cv.VideoCapture('C:/Users/79875/Desktop/course_work/video.mp4')  # Открываем видео по ссылке
    main(cap)  # Запускаем основной цикл


def main(cap):  # Основной цикл

    global global_dict

    def image_callback(img):
        global global_dict

        clear_frame = img[int(global_dict["height"]*0.05) : int(global_dict["height"]*0.9), 0:global_dict["width"] ]  # Считываем видео, обрезаем вверх и низ
        clear_frame = cv.cvtColor(cv.convertScaleAbs(clear_frame, alpha=1.5, beta=1), cv.COLOR_BGR2RGB)  # Перевод из BRG в RGB

        global_dict["clear_frame"] = clear_frame  # Сохраняем "чистое" видео в глобальный словарь

        global_dict["final_frame"] = clear_frame.copy()
        image_analysis()  # Анализируем видео

        # Форматирование финального изображения для вывода, чтобы оно могло быть помещено в интерфейс
        img = Img.fromarray(global_dict["final_frame"])
        imgtk = ImageTk.PhotoImage(image=img)

        # Вставляем видео
        video.place(x=0, y=0)
        video.config(image=imgtk)
        global_dict["root"].update()

        cv.destroyAllWindows()

    ret, clear_frame = cap.read()
    global_dict['height'], global_dict["width"] = clear_frame.shape[:2]  # Считываем и запоминаем размерность видео

    create_GUI(global_dict["width"], global_dict['height'])  # Создаем интерфейс
    video = Label(global_dict["root"])  # Обозначаем место в интерфейсе, в которое поместим финальное видео

    while True:
        ret, clear_frame = cap.read()  # Считываем полученное видео в формат, подходящий для обработки
        image_callback(clear_frame)


start()
