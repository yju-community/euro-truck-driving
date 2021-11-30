# Setting

1. git clone https://github.com/yju-community/euro-truck-driving
2. python -m venv euro-truck-driving
3. cd euro-truck-driving
4. Scripts\activate.bat
5. python -m pip install --upgrade pip
6. pip install opencv-python cvzone numpy pandas tflearn mediapipe pywin32
7. pip install --upgrade tensorflow

### pywin32 error

- ImportError: DLL load failed while importing win32api: 지정된 모듈을 찾을 수 없습니다.
- 위 에러가 뜰 시 Scripts 폴더에 들어가서 python pywin32_postinstall.py -install 명령어 사용

### self-driving

- 1 ~ 4 번까지의 VERSION 변수를 동일하게 맞춰야 함
- 경로를 찾을 수 없을 경우, Models와 Dataset폴더 경로 앞에 ../ 를 붙여야 함

# Library

- opencv-python 4.5.4.58
- cvzone 1.5.3
- numpy 1.21.4
- pandas 1.3.4
- tflearn 0.5.0
- mediapipe 0.8.9
- tensorflow 2.7.0
- pywin32 302
