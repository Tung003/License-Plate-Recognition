import tkinter as tk
import cv2
import torch
import os
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
from ultralytics import YOLO
from detect_license_plate import cls_char,detect_char
from model import CNN   #model CNN
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
model_detect_path='/src/model/best_detect.pt'
model_OCR_path='/src/model/best_OCR.pt'
model_classification_path='/src/model/best_cls.pth'
# Load model 1 lần duy nhất ở đây
model_detect = YOLO(model_detect_path).to(device)
model_OCR = YOLO(model_OCR_path).to(device)
model_classification=CNN().to(device)
model_classification.load_state_dict(torch.load(model_classification_path,map_location=device))
#chuyển mode eval
model_detect.eval()
model_OCR.eval()
model_classification.eval()
#model dự đoán cho ảnh
def run_model(img_path):
    # đường dẫn của ảnh
    img = cv2.imread(img_path)
    # kết quả từ model
    result_text=cls_char(model_detect,model_OCR,model_classification,img,device)      
    return img,result_text
#hiển thị kết quả cho ảnh chọn
def open_image():
    canvas.delete("all")
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    img, result_text = run_model(file_path)
    #ảnh crop biển số từ ảnh gốc
    # cropped=detect_license_plate(model_detect,img)
    _,result_ocr=detect_char(model_detect,model_OCR,img)  
    result_ocr=cv2.resize(result_ocr,(600,400))
    # h,w,c=img.shape
    img = cv2.resize(img, (1180,720))
    # cropped=cv2.resize(cropped,(600,400))
    # result_ocr=cv2.re

    #ảnh gốc
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = ImageOps.expand(img_pil, border=5, fill='green') 
    img_tk = ImageTk.PhotoImage(img_pil)

    #ảnh OCR
    OCR_rgb = cv2.cvtColor(result_ocr, cv2.COLOR_BGR2RGB)
    OCR_pil = Image.fromarray(OCR_rgb)
    OCR_pil = ImageOps.expand(OCR_pil, border=5, fill='green') 
    OCR_tk = ImageTk.PhotoImage(OCR_pil)
    #khung ảnh gốc ở vị trí 00 neo ở góc trên trái
    canvas.create_image(0, 0, anchor="nw", image=img_tk)
    canvas.image_main = img_tk

    #khung ảnh OCR ở vị trí 1580 500
    canvas.create_image( 1810,0, anchor="ne", image=OCR_tk)
    canvas.image_OCR = OCR_tk
    result_text = ''.join(result_text)
    plate_main = result_text[0:2] + '-' + result_text[2:4]

    if len(result_text) == 11:
        # print("độ dài: ",len(result_text),'giá trị',result_text)
        plate_tail = result_text[6:]
    elif len(result_text) == 10:
        # print("độ dài: ",len(result_text),'giá trị',result_text)
        plate_tail = result_text[5:]
    else:
        plate_tail = result_text[4:]
        # print("độ dài: ",len(result_text),'giá trị',result_text)

    result_text = plate_main + '\n' + plate_tail

    canvas.create_text(1580, 650, text=result_text, anchor="center", font=("Arial", 60), fill="green")
#chức năng chưa define
def accept_open_barie():
    pass

#hàm chuyển sang ảnh khác trong thư mục để dự đoán và hiển thị kết quả
def next_image():
    global current_index

    if current_index >= len(image_paths):
        label_result.config(text='This is the last photo!',)
        return
    label_result.config(text="")  # <--- xoá thông báo cũ nếu có
    canvas.delete("all")

    file_path = image_paths[current_index]
    current_index += 1  # chuẩn bị cho lần nhấn tiếp theo

    
    img, result_text = run_model(file_path)
    # cropped=detect_license_plate(model_detect,img)
    _,result_ocr=detect_char(model_detect,model_OCR,img)
    result_ocr=cv2.resize(result_ocr,(600,400))  
    # h,w,c=img.shape
    img = cv2.resize(img, (1180,720))
    # cropped=cv2.resize(cropped,(600,400))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = ImageOps.expand(img_pil, border=5, fill='green') 
    img_tk = ImageTk.PhotoImage(img_pil)

    #ảnh OCR
    OCR_rgb = cv2.cvtColor(result_ocr, cv2.COLOR_BGR2RGB)
    OCR_pil = Image.fromarray(OCR_rgb)
    OCR_pil = ImageOps.expand(OCR_pil, border=5, fill='green') 
    OCR_tk = ImageTk.PhotoImage(OCR_pil)

    canvas.create_image(0, 0, anchor="nw", image=img_tk)
    canvas.image_main = img_tk

    #khung ảnh OCR ở vị trí 1580 500
    canvas.create_image( 1810,0, anchor="ne", image=OCR_tk)
    canvas.image_OCR = OCR_tk
    #kết quả
    result_text = ''.join(result_text)
    plate_main = result_text[0:2] + '-' + result_text[2:4]

    if len(result_text) == 11:
        # print("độ dài: ",len(result_text),'giá trị',result_text)
        plate_tail = result_text[6:]
    elif len(result_text) == 10:
        # print("độ dài: ",len(result_text),'giá trị',result_text)
        plate_tail = result_text[5:]
    else:
        plate_tail = result_text[4:]
        # print("độ dài: ",len(result_text),'giá trị',result_text)

    result_text = plate_main + '\n' + plate_tail

    canvas.create_text(1580, 650, text=result_text, anchor="center", font=("Arial", 60), fill="green")

def prev_image():
    global current_index

    if current_index <= 1:
        # print("This is the first photo!")
        label_result.config(text='This is the first photo!')
        return

    current_index -= 2  # vì `next_image()` sẽ +1 ngay sau

    next_image()        # dùng lại logic hiển thị của next_image()
if __name__=='__main__':

    #folder chọn để test 
    folder_path='/src/data/test'
    current_index=0
    image_paths = [os.path.join(folder_path, f)     #add các đường dẫn của ảnh vào mảng
                for f in sorted(os.listdir(folder_path))
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
    # Giao diện
    root = tk.Tk()
    root.geometry("1810x900") 
    root.title("License Plate")

    button_frame = tk.Frame(root)
    button_frame.pack(side="top", pady=10) # button phía trên frame
    #nút nhấn chọn ảnh
    btn_chose = tk.Button(button_frame, command=open_image,
                    text="Chose img",
                    font=("Arial", 18),    # tăng cỡ chữ
                    width=15,              # chiều ngang nút (số ký tự)
                    height=2,              # chiều cao nút (số dòng)
                    bg="lightblue",        # màu nền
                    fg="black" )
    btn_chose.pack(side="left", padx=10)
    #nút nhấn chấp nhận
    btn_accept = tk.Button(button_frame, command=open_image,
                    text="Accept",
                    font=("Arial", 18),    # tăng cỡ chữ
                    width=15,              # chiều ngang nút (số ký tự)
                    height=2,              # chiều cao nút (số dòng)
                    bg="lightblue",        # màu nền
                    fg="black" )
    btn_accept.pack(side="left", padx=10)
    #nut nhấn ảnh tiếp theo
    btn_next_img = tk.Button(button_frame, command=next_image,
                    text="Next",
                    font=("Arial", 18),    # tăng cỡ chữ
                    width=15,              # chiều ngang nút (số ký tự)
                    height=2,              # chiều cao nút (số dòng)
                    bg="lightblue",        # màu nền
                    fg="black" )
    btn_next_img.pack(side="left", padx=10)
    #nut nhấn ảnh trước đó
    btn_prev = tk.Button(button_frame,
                    text="Prev",
                    font=("Arial", 18),
                    width=15,
                    height=2,
                    bg="lightblue",
                    fg="black",
                    command=prev_image)
    btn_prev.pack(side="left", padx=10)

    canvas = tk.Canvas(root, width=1810, height=900)
    canvas.pack()

    label_result = tk.Label(root, text="", font=("Arial", 24),fg='red')
    label_result.place(x=905, y=880)

    root.mainloop()
