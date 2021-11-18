import cv2
import numpy as np
import win32gui
import win32ui
import win32con
import win32api


def grab_screen(region=None):

    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)


# Region of Interest
def roi(img, vertices):
    # img 크기만큼의 영행렬을 mask 변수에 저장
    mask = np.zeros_like(img)

    # vertices 영역만큼의 Polygon 형상에만 255
    cv2.fillPoly(mask, vertices, 255)

    # img와 mask 변수를 and (비트연산) 해서 나온 값들을 masked에 넣고 반환
    masked = cv2.bitwise_and(img, mask)

    return masked


def set_screen():
    # 1920x1080 full screen mode
    screen = grab_screen(region=(0, 0, 1920, 1080))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

    # select area
    vertices = np.array([[0, 1080], [0, 640], [960-240, 540], [
                        960+240, 540], [1920, 640], [1920, 1080]])

    # masking
    screen = roi(screen, [vertices])

    # resize to something a bit more acceptable for a CNN
    screen = cv2.resize(screen, (256, 160))

    return screen
