import numpy as np
import math
import cv2

# 보간 마스크
def itp_kernel(s,a):
    if (abs(s) >=0) & (abs(s) <=1): # s의 절댓값이 0이상 1이하
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2): # s의 절댓값이 1초과 2이하
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0

# 영상 스케일링 함수 ( 영상을 512*512에서 1024*1024로 확장 )
def scaling_img(img,H,W):
    res = np.zeros((H+4,W+4)) # zeros로 0으로 초기화된 2차원 배열 생성
    res[2:H+2,2:W+2] = img    # 초기화된 배열에 Lenna 영상의 화소값들로 초기화

    # 4방향으로 연산
    res[2:H+2,0:2]=img[:,0:1]  
    res[H+2:H+4,2:W+2]=img[H-1:H,:]
    res[2:H+2,W+2:W+4]=img[:,W-1:W]
    res[0:2,2:W+2]=img[0:1]
    
    # 4방향으로 연산
    res[0:2,0:2]=img[0,0]
    res[H+2:H+4,0:2]=img[H-1,0]
    res[H+2:H+4,W+2:W+4]=img[H-1,W-1]
    res[0:2,W+2:W+4]=img[0,W-1]
    
    # 스케일링된 영상 반환
    return res

# 바이큐빅 연산 함수
def bicubic_interpolation(img, ratio, a):
    H,W = img.shape

    img = scaling_img(img,H,W) # 영상 스케일링

    # math를 사용하여 floor를 통해 연산된 화소값을 반올림하여 소수를 제거
    dH = math.floor(H*ratio)
    dW = math.floor(W*ratio)
    dst = np.zeros((dH, dW)) # destination에 0으로 초기화된 (dH * dW) 2차원 배열 삽입 

    h = 1/ratio  # ratio는 배율 ( 2배로 크게 )

    print('Plz Wait, Bicubic Interpolation takes a lot of time to apply...')
    count = 0
    for j in range(dH):
        for i in range(dW):
            x, y = i * h + 2 , j * h + 2
                
            x1 = 1 + x - math.floor(x)
            x2 = x - math.floor(x)
            x3 = math.floor(x) + 1 - x
            x4 = math.floor(x) + 2 - x

            y1 = 1 + y - math.floor(y)
            y2 = y - math.floor(y)
            y3 = math.floor(y) + 1 - y
            y4 = math.floor(y) + 2 - y
            
            # 바이큐빅 보간법 적용 (numpy의 matrix를 사용하여 행렬 생성)
            mat_l = np.matrix([[itp_kernel(x1,a),itp_kernel(x2,a),itp_kernel(x3,a),itp_kernel(x4,a)]])
            
            mat_m = np.matrix([[img[int(y-y1),int(x-x1)],img[int(y-y2),int(x-x1)],img[int(y+y3),int(x-x1)],img[int(y+y4),int(x-x1)]],
                                [img[int(y-y1),int(x-x2)],img[int(y-y2),int(x-x2)],img[int(y+y3),int(x-x2)],img[int(y+y4),int(x-x2)]],
                                [img[int(y-y1),int(x+x3)],img[int(y-y2),int(x+x3)],img[int(y+y3),int(x+x3)],img[int(y+y4),int(x+x3)]],
                                [img[int(y-y1),int(x+x4)],img[int(y-y2),int(x+x4)],img[int(y+y3),int(x+x4)],img[int(y+y4),int(x+x4)]]])
            
            mat_r = np.matrix([[itp_kernel(y1,a)],[itp_kernel(y2,a)],[itp_kernel(y3,a)],[itp_kernel(y4,a)]])
            
            dst[j, i] = np.dot(np.dot(mat_l, mat_m),mat_r) # mat_l과 mat_m의 내적 곱 결과를 다시 mat_r의 내적곱 결과를 저장
            count+=1;
            
    return dst.astype(np.uint8) # 형 변환

# 이미지 그레이스케일로 읽기 ( Lenna512.png 파일을 반드시 소스코드와 같은 폴더 안에 두고 실행 )
img = cv2.imread("Lenna512.png", cv2.IMREAD_GRAYSCALE)

# 보간법 적용
dst = bicubic_interpolation(img, 2, -1/2) # 2배 확대 ( 512* 512 -> 1024*1024 , ratio 값을 2로 준다 )
print("                    < 16 By 16 Pixels >")

# (256, 256) 관심영역에 영상의 화소값 16 by 16 출력
for i in range(dst.shape[0]//4, dst.shape[0]//4+15): #(1024/4, 1024/4+15)
    for j in range(dst.shape[1]//4, dst.shape[1]//4+15): #(1024/4, 1024/4+15)
        print("%3d"%dst[i,j] , end=' ')
    print()

cv2.imshow("Scaling-Bicubic", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()