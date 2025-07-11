import cv2
from torchvision import transforms
import torch

def detect_license_plate(model_detect,img):
    """Dùng model đã train với bộ dữ liệu biển số xe để cắt được biển số xe và send đến model OCR phân tích"""
    result = model_detect.predict(img, device='cpu',verbose=False)
    boxes=result[0].boxes.xyxy# return duy nhất 1 box
    x1,y1,x2,y2=boxes[0]
    x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
    cropped=img[y1:y2,x1:x2]
    # print(cropped.shape)
    return cropped

def detect_char(model_detect,model_OCR,img):
    """Dùng model OCR đã train với bộ dữ liệu biển số được cắt ra từ model detect """
    cropped=detect_license_plate(model_detect,img)
    cropped=cv2.resize(cropped,(640,540))
    result=model_OCR.predict(cropped,device='cpu',verbose=False)
    boxes=result[0].boxes.xyxy
    # result_img = result[0].plot() 
    result_img = cropped.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # màu xanh lá, độ dày 2

    # result[0].show()

    #mảng chứa box và điểm trung tâm box
    boxes_with_center = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)  # box từ YOLO chuyển về int để có thể cắt được
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        boxes_with_center.append([[x1,y1,x2,y2],xc,yc]) #add vào mảng phẩn tử trung tâm box

    #mảng tính toán chia dòng
    yc_values = [b[2] for b in boxes_with_center]  #mảng ys chứa các phần tử yc từ mảng boxes_with_center (chỉ số b[2])
    threshold_y = (max(yc_values) + min(yc_values)) / 2   #khởi tạo ngưỡng chia dòng

    #chia dòng trên và dưới
    line1 = [b for b in boxes_with_center if b[2] < threshold_y]  # dòng trên (nếu yc nhỏ hơn ngưỡng thì ở trên)
    line2 = [b for b in boxes_with_center if b[2] >= threshold_y] # dòng dưới (nếu yc lớn hơn ngưỡng thì ở dưới)

    #sắp xếp tăng dần theo x_center từ trái qua phải
    line1_sorted = sorted(line1, key=lambda b: b[1])
    line2_sorted = sorted(line2, key=lambda b: b[1])
    # out=[[x1,y1,x2,y2],xc,yc]

    # dòng trên
    box_line1=[box[0] for box in line1_sorted]
    #dòng dưới
    box_line2=[box[0] for box in line2_sorted]

    #hàm xử lý box resize và convert channels
    def crop_and_preprocess_box(boxes):
        return [cv2.cvtColor
                    (cv2.resize(cropped[y1:y2,x1:x2],(35,150)),cv2.COLOR_BGR2RGB)
                for x1,y1,x2,y2 in boxes
                ]
    
    # trả về các box được resize đúng kích thước và chuyển về đúng kênh màu khi train (trong custom_dataset)
    char_line1=crop_and_preprocess_box(box_line1)
    char_line2=crop_and_preprocess_box(box_line2)
    #unpack values
    return char_line1 +char_line2,result_img

def cls_char(model_detect,model_OCR,model_classification,img,device):
    dict_label={   0:  '0', 1:  '1', 2:  '2', 3:  '3', 4:  '4', 5:  '5',
                6:  '6', 7:  '7', 8:  '8', 9:  '9', 10: 'A', 11: 'B',
                12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G',
                17: 'H', 18: 'K', 19: 'L', 20: 'M', 21: 'N',
                22: 'P', 23: 'R', 24: 'S', 25: 'T', 26: 'U',
                27: 'V', 28: 'X', 29: 'Y', 30: 'Z'}
    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    chars,_=detect_char(model_detect,model_OCR,img) 
    inp=[transform(char).unsqueeze(0).to(device) for char in chars]

    #chuyển mode test
    model_classification.eval()

    # print("bắt đầu TEST")

    with torch.no_grad():
        """result = tensor([[ -8.9597,  -7.4596,  -1.4715,  -1.0938,  -4.3297,
            -5.1017,  -9.8162,  -1.3908,  -8.5262,  -9.8866, -12.7221,
            -17.4604,  -9.7798, -17.4732, -20.3030, -12.2342, -10.4305,
            -18.7969, -16.7061, -20.4210, -16.3622, -21.1948,  -6.6170,
            -21.9711, -13.1088, -14.3279, -20.4130, -17.9417, -15.9735,
            -15.4930, -10.2099]],
        tuy nhiên có vị trị biển số chắc chắn và chữ cái là vị trí thứ 3 dòng trên
        nên chỉ lấy giá trị dự đoán cao nhất từ vị trí chữ trở đi
        """
        # preds=[model_classification(inp[0])]
        def get_pred():
            results=[]
            for inpu in inp:
                results.append(model_classification(inpu))
            return results  #return mảng phần tử đã được model tính toán cho từng label
        # ký tự thứ 2 3 trong biển số xe ở hàng trên có ký tự đầu tiên chắc chắn là chữ
        predict=get_pred()

        pred23=[torch.argmax(predict[2][0 , 10:])+10,
                torch.argmax(predict[3][0 , :])]
        # các ký tự còn lại đều là số lên tìm các số có xác xuất cao nhất
        preds=[torch.argmax(result[0,0:10])  for result in predict[0:2]]+pred23+[torch.argmax(result[0,0:10]) for result in predict[4:]]
        preds=[dict_label[pred.item()] for pred in preds]
        
    return preds
    #/home/chu-tung/Desktop/Deep_learning/LICENSE_PLATE_PIPELINE/PIPELINE/data/raw/0509_04094_b.jpg
# while True:
#     count=0
#     img_file=input('nhập tên file ảnh: ')
#     img_path='/home/chu-tung/Desktop/Deep_learning/LICENSE_PLATE_PIPELINE/PIPELINE/data/raw/'
#     img=cv2.imread(os.path.join(img_path,img_file))
#     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#     model_detect_path='./LICENSE_PLATE_PIPELINE/PIPELINE/model/best_detect.pt'
#     model_OCR_path='./LICENSE_PLATE_PIPELINE/PIPELINE/model/best_OCR.pt'
#     model_classification_path='./LICENSE_PLATE_PIPELINE/PIPELINE/model/best_cls.pth'

#     model_detect = YOLO(model_detect_path).to('cuda')
#     model_OCR = YOLO(model_OCR_path).to('cuda')
#     model_classification=CNN().to(device)
#     model_classification.load_state_dict(torch.load("./LICENSE_PLATE_PIPELINE/Predict_Char/checkpoint/best.pth",map_location=device))
#     start_time=time.time()
#     cls_char(model_detect,model_OCR,model_classification,img,device)       
#     end_time=time.time()
#     print("tổng thời gian: ",end_time-start_time)
#     count+=1
#     if count>6:
#         print("thoát vòng lặp.")
#         break
