# Setting

1. python -m venv project name
2. scripts\python.exe -m pip install --upgrade pip 2.

# Library

- pip install opencv-python
- pip install cvzone
- pip install numpy
- pip install pandas
- pip install tflearn
- pip install mediapipe
- pip install --upgrade tensorflow
- pip install pywin32

pywin32 error

- ImportError: DLL load failed while importing win32api: 지정된 모듈을 찾을 수 없습니다.
- 위 에러가 뜰 시 Scripts 폴더에 들어가서 python pywin32_postinstall.py -install 명령어 사용

# self-driving

root폴더에서 mkdir Dataset을 해주어야지 createData를 할 때 저장할 디렉토리가 생김
createData를 할때마다 VERSION을 1씩 올려주어야됨
