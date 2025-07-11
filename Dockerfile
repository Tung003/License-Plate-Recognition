#dùng image ubuntu cơ bản 
FROM ubuntu
#cập nhật và cài python+pip
RUN apt-get update 
RUN apt-get -y install python3 python3-pip python3-tk libgl1 libglib2.0-0
RUN apt-get -y install git 
RUN apt-get clean

#chỉ thư mục làm việc
WORKDIR /src
#cập nhật để cài venv
RUN apt update && apt install -y python3.12-venv
#copy file thư viện yêu cầu
COPY requirements.txt .
#chạy lệnh cài đặt các thư viện cần có; 
RUN python3 -m venv venv && \
    venv/bin/pip install -r requirements.txt

COPY . .

CMD ["venv/bin/python3","src/GUI.py"]