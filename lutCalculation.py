import numpy as np
from math import pi
import matplotlib.colors as mcolors
import pandas as pd

class ChangeVal:
    rgbColor : str
    index : int
    value : int
    init : bool

LUTSIZE = 17

#Color Contorl 관련 부분
def saveLUTfile(array : np.array):
        with open("./makeLUT.CUBE",'w') as f:
            f.write('#LUT size\n')
            f.write('LUT_3D_SIZE 17\n\n')
            """
            f.write('#Created by: python\n')
            f.write('TITLE \"generateLUT\"\n')
            f.write('\n#data domain\n')
            f.write('DOMAIN_MIN 0.0 0.0 0.0\n')
            f.write('DOMAIN_MAX 1.0 1.0 1.0\n\n')
            """
            np.savetxt(f,array,fmt='%f',delimiter=' \t',newline=' \n')
        f.close()

def saveLUTfileBin(array : np.array):
    with open("./lutArrayBin",'wb') as f:
        array.tofile(f)
    f.close()

def getRed(idx:int):
    return int(idx%17)
def getGreen(idx:int):
    return int(idx/17)%17
def getBlue(idx:int):
    return int(idx/int(17*17))%17

def getLinearArrayIndex(r:int,g:int,b:int,size:int):
    return (b * size * size) + (g * size) + r

def colorControl(color : str, idx : int, gain : float, array : np.array):
    if color == "r":
        for i in range(17*17*17):
            if getRed(i) == idx:
                array[i][0] = array[i][0] * gain
    elif color == "g":
        for i in range(17*17*17):
            if getGreen(i) == idx:
                array[i][1] = array[i][1] * gain
    else:
        for i in range(17*17*17):
            if getBlue(i) == idx:
                array[i][2] = array[i][2] * gain

def makeLUT(values : ChangeVal):
    gainVal = float(values.value) / 10
    colorControl(values.rgbColor,values.index,gainVal)
    saveLUTfile()

def create_hsv_background(xlim, ylim):
    x = np.linspace(xlim[0], xlim[1], 500)
    y = np.linspace(ylim[0], ylim[1], 500)
    xv, yv = np.meshgrid(x, y)
    r = np.sqrt(xv**2 + yv**2)
    theta = np.arctan2(yv, xv)
    h = (theta + np.pi) / (2 * np.pi)  # 기본적으로 빨간색이 오른쪽에 위치
    h = (h + 0.5) % 1  # 빨간색이 왼쪽에 오도록 조정
    s = np.clip(r / np.max(r), 0, 1)  # 중심에서 멀어질수록 채도가 증가
    v = np.ones_like(r)  # 밝기를 일정하게 유지
    hsv = np.stack((h, s, v), axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)

"""
					R		G		B
0	0		R		1		0		0
1	30		RRG		0.75	0.25	0
2	60		RG		0.5		0.5		0
3	90		RGG		0.25	0.75	0
4	120		G		0		1		0
5	150		GGB		0		0.75	0.25
6	180		GB		0		0.5		0.5
7	210		GBB		0		0.25	0.75
8	240		B		0		0		1
9	270		BBR		0.25	0		0.75
10	300		BR		0.5		0		0.5
11	330		BRR		0.75	0		0.25
"""
# 단순히 치환으로 하면 될듯? 변경 좌표의 Color로 표시되게 하면 됨.

index_data = np.array([
    # 0
    [
        [16,	14,	14],
        [15,	14,	14],
        [15,	15,	14],
        [14,	15,	14],
        [14,	16,	14],
        [14,	15,	14],
        [14,	15,	15],
        [14,	14,	15],
        [14,	14,	16],
        [14,	14,	15],
        [15,	14,	15],
        [15,	14,	14]
    ],
    # 1
    [
        [16,	12,	12],
        [15,	13,	12],
        [14,	14,	12],
        [13,	15,	12],
        [12,	16,	12],
        [12,	15,	13],
        [12,	14,	14],
        [12,	13,	15],
        [12,	12,	16],
        [13,	12,	15],
        [14,	12,	14],
        [15,	12,	13]
    ],
    # 2
    [
        [16,	10,	10],
        [15,	12,	10],
        [13,	13,	10],
        [12,	15,	10],
        [10,	16,	10],
        [10,	15,	12],
        [10,	13,	13],
        [10,	12,	15],
        [10,	10,	16],
        [12,	10,	15],
        [13,	10,	13],
        [15,	10,	12]
    ],
    # 3
    [
        [16,	8,	8],
        [14,	10,	8],
        [12,	12,	8],
        [10,	14,	8],
        [8,	16,	8],
        [8,	14,	10],
        [8,	12,	12],
        [8,	10,	14],
        [8,	8,	16],
        [10,	8,	14],
        [12,	8,	12],
        [14,	8,	10]
    ],
    # 4
    [
        [16,	6,	6],
        [14,	9,	6],
        [11,	11,	6],
        [9,	14,	6],
        [6,	16,	6],
        [6,	14,	9],
        [6,	11,	11],
        [6,	9,	14],
        [6,	6,	16],
        [9,	6,	14],
        [11,	6,	11],
        [14,	6,	9]
    ],
    # 5
    [
        [16,	4,	4],
        [13,	7,	4],
        [10,	10,	4],
        [7,	13,	4],
        [4,	16,	4],
        [4,	13,	7],
        [4,	10,	10],
        [4,	7,	13],
        [4,	4,	16],
        [7,	4,	13],
        [10,	4,	10],
        [13,	4,	7]
    ],
    # 6
    [
        [16,	2,	2],
        [13,	6,	2],
        [9,	9,	2],
        [6,	13,	2],
        [2,	16,	2],
        [2,	13,	6],
        [2,	9,	9],
        [2,	6,	13],
        [2,	2,	16],
        [6,	2,	13],
        [9,	2,	9],
        [13,	2,	6]
    ],
    # 7
    [
        [16,	0,	0],
        [12,	4,	0],
        [8,	8,	0],
        [4,	12,	0],
        [0,	16,	0],
        [0,	12,	4],
        [0,	8,	8],
        [0,	4,	12],
        [0,	0,	16],
        [4,	0,	12],
        [8,	0,	8],
        [12,	0,	4]
    ]
])
lutData = pd.DataFrame({'rI':[],'gI':[],'bI':[],'R':[],'G':[],'B':[],'theta':[],'r':[],'v':[]})

def find_close_lut(theta,r,v,radius):
    
    if r > radius:
        r = radius

    h = (theta + np.pi) / (2 * np.pi)  # 기본적으로 빨간색이 오른쪽에 위치
    h = (h + 0.5) % 1  # 빨간색이 왼쪽에 오도록 조정

    hsv = np.stack((h, r/radius, v), axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)
    #print(f"input = {theta:1.4f}, {r}, {v}, {radius}, hsv = {hsv} => rgb = {rgb}")
    rgbIndex = []
    for i in range(3):
        base = (int)(rgb[i] / (1/(LUTSIZE-1)))
        if ((rgb[i] % (1/(LUTSIZE-1)))/(1/(LUTSIZE-1))) >= 0.5:
            base +=1
        rgbIndex.append(base)
    #print(f"rgbIndex = {rgbIndex}")
    return rgbIndex

def find_lut_hsv(lutVal):

    rgb = []
    for i in range(3):
        rgb.append((float)(lutVal[i]/(LUTSIZE-1)))

    hsv = mcolors.rgb_to_hsv(rgb)
    return hsv

def getLutIndexValue(polygonIndex : int, vertex : int):
    return index_data[polygonIndex][vertex]

def getTargetLutIndexValue(r,theta):
    degree_index = 0
    is_upper = 0
    dpi = 2*pi
    if theta < 0:
        theta += 2*pi

    if theta == 0 or theta == (2*pi):
        degree_index = 0
    else:
        if(theta % (6/pi)) >= 12/pi:
            is_upper = 1
        
        theta360 = (theta/dpi) * 360

        degree_index = (int)(theta360/30)
        if is_upper == 1:
            if degree_index == 11:
                degree_index = 0
            else:
                degree_index += 1
    
    target_radius = (int)(r/2)
    if target_radius < 1:
        target_radius = 1
    if target_radius > 8:
        target_radius = 8

    target_radius -= 1

    print(f"target_index = {degree_index}, radius = {target_radius}")

    return [degree_index,target_radius]