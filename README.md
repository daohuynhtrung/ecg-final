# Sử dụng điện tim để phát hiện bất thường ở người sử dụng học máy
## Giới thiệu
Project dùng để phát hiện bất thường ở người có bệnh tim thông qua việc phân loại điện tâm đồ.
## Nguồn dữ liệu sử dụng
[Dữ liệu MIT-BIH database](https://physionet.org/physiobank/database/html/mitdbdir/mitdbdir.htm)

[Dữ liệu đang sử dụng](https://drive.google.com/drive/folders/1q4b4U13_XDivYBoy2-HzXv1fLoPz8KCS?usp=sharing)
## Cấu trúc project
### Configure
Bao gồm những file json có tác dụng để điều chỉnh thông số của các hàm đang hiện thực. Bao gồm 2 file:
* **colab_classification**: cấu hình project chạy trên google colab 
* **local_classification**: cấu hình project chạy trên máy local

Các tham số trong file configure là:
* data_path: vị trí lưu dataset
* epoch: số lần tính toán
* batchsize
* timestep
* checkpoint: vị trí lưu file checkpoint của model khi training (hdf5), các file plot accuracy-loss, file configure,...
* model: Model được sử dụng để training
* data_processing: hàm xử lý data
* data_prepare_function: hàm xử lý data
* rr_processin_function: hàm xử lý khoảng R-R
### Source
Bao gồm 3 file:
* **classification.py**: phân loại tín hiệu điện tâm đồ
* **load_model.py** : load model đã lưu và kiểm thử
* **utils.py**: xử lý tác vụ liên quan
Và 2 folder:
#### Data
Bao gồm 2 file xử lý data:
* **data_stuff.py**: xử lý data chuẩn bị cho mô hình phân loại
* **plot_data.py**: xử lý data từ matlab chuyển về csv, đánh dấu đỉnh, ghi chú 
trong data góc,...
* **pca.py**: xử lý nhiễu, thu giảm chiều dữ liệu,...
#### Model
Chứa model cho mô hình phân loại.
## Run project
* cd ecg-final
* python source/classification.py
