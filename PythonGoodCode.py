import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.widgets import CheckButtons, Slider
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image, ImageEnhance
from pillow_lut import load_cube_file, identity_table, load_hald_image, rgb_color_enhance
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import lutCalculation
from math import pi
from matplotlib.widgets import Button
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom
from scipy.interpolate import griddata

# debug
debug_print = False

# Size Parameter 설정
polygons = []
selected_polygon_info = None
vertex_select_threshold = 0.5  # 꼭짓점 선택 임계값
move_all_vertices = False  # 다른 다각형의 꼭짓점도 같이 움직일지 여부
weight = 1.0  # 가중치
plt_size = 17
background_size = 400
num_polygons = 8
num_vertices = 12
volume = 0.5 # 밝기
polygon_control = False
# Polygon 좌표 저장
prev_polgon_coordinate = np.empty((num_polygons, num_vertices,2))
original_polgon_coordinate = np.empty((num_polygons, num_vertices,2))
#LUT Data 설정
values=lutCalculation.ChangeVal()
lutdata = pd.read_csv("./bypasslut.txt",sep="\t")
#print(lutdata)
lutArray = lutdata.to_numpy()
originalLut = lutArray.copy()
lutCalculation.saveLUTfile(lutArray)
# image
image_path = 'colorPatch.png'
image_path2 = '_red.png'
#image_path = 'newjeans.png'

img1 = Image.open(image_path)

# 다각형 정보와 선택된 꼭짓점 정보를 저장할 클래스
class PolygonInfo:
    def __init__(self, index, vertices):
        self.index = index
        self.vertices = vertices
        self.selected_vertex_index = None

    def update_vertex(self, vertex_index, new_position):
        self.vertices[vertex_index] = new_position

    def select_vertex(self, vertex_index):
        self.selected_vertex_index = vertex_index

    def deselect_vertex(self):
        self.selected_vertex_index = None

def to_cartesian(r, theta):
    return r * np.cos(theta), r * np.sin(theta)

def to_polar(x, y):
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return r, theta

"""
def create_hsv_background(xlim, ylim):
    x = np.linspace(xlim[0], xlim[1], background_size)
    y = np.linspace(ylim[0], ylim[1], background_size)
    xv, yv = np.meshgrid(x, y)
    r = np.sqrt(xv**2 + yv**2)
    theta = np.arctan2(yv, xv)
    h = (theta + np.pi) / (2 * np.pi)  # 기본적으로 빨간색이 오른쪽에 위치
    h = (h + 0.5) % 1  # 빨간색이 왼쪽에 오도록 조정
    s = np.clip(r / np.max(r), 0, 1)  # 중심에서 멀어질수록 채도가 증가
    v = np.ones_like(r)  # 밝기를 일정하게 유지
    hsv = np.stack((h, s, v), axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)
    return rgb
"""

def create_hsv_background(xlim, ylim, volume):
    x = np.linspace(xlim[0], xlim[1], background_size)
    y = np.linspace(ylim[0], ylim[1], background_size)
    xv, yv = np.meshgrid(x, y)
    r = np.sqrt(xv**2 + yv**2)
    theta = np.arctan2(yv, xv)
    h = (theta + np.pi) / (2 * np.pi)  # 기본적으로 빨간색이 오른쪽에 위치
    h = (h + 0.5) % 1  # 빨간색이 왼쪽에 오도록 조정
    s = np.clip(r / np.max(r), 0, 1)  # 중심에서 멀어질수록 채도가 증가
    v = np.clip(volume + (1 - volume) * (1 - r / np.max(r)), 0, 1)  # 중심에서 멀어질수록 밝기가 감소, 기본 밝기 추가
    hsv = np.stack((h, s, v), axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)
    return rgb
    
def generate_polygons():
    global polygons
    radii = np.arange(2, 2 * (num_polygons + 1), 2)  # 다각형의 반지름을 2씩 증가시켜 설정
    for i, radius in enumerate(radii):
        angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
        vertices = [(radius, angle) for angle in angles]
        polygons.append(PolygonInfo(i, vertices))

# 클릭 이벤트 핸들러 함수
def onpress(event):
    global selected_polygon_info
    global polygon_control
    if event.button == 1:  # 왼쪽 클릭
        # 클릭한 좌표와 가까운 꼭짓점을 찾습니다.
        min_dist = float('inf')
        for polygon_info in polygons:
            for j, vertex in enumerate(polygon_info.vertices):
                vx, vy = to_cartesian(*vertex)
                dist = np.hypot(vx - event.xdata, vy - event.ydata)
                if dist < min_dist and dist < vertex_select_threshold:
                    min_dist = dist
                    selected_polygon_info = polygon_info
                    selected_polygon_info.select_vertex(j)
                    polygon_control = True

def onrelease(event):
    global polygon_control

    if polygon_control != True:
        return

    global selected_polygon_info
    update_coordinate(selected_polygon_info.selected_vertex_index)
    if selected_polygon_info is not None:
        selected_polygon_info.deselect_vertex()
        polygon_control = False
    selected_polygon_info = None
    
    update_image()

    update_coord_text('')  # 선택된 꼭짓점이 없을 때 빈 문자열 표시
    
    plt.draw()

def onmove(event):
    global selected_polygon_info, hsv_background
    if selected_polygon_info is not None and selected_polygon_info.selected_vertex_index is not None:
        # 선택된 꼭짓점을 이동시킵니다.
        selected_vertex_index = selected_polygon_info.selected_vertex_index
        old_position = selected_polygon_info.vertices[selected_vertex_index]
        new_position = to_polar(event.xdata, event.ydata)
        selected_polygon_info.update_vertex(selected_vertex_index, new_position)

        # 이동 거리를 계산합니다.
        old_cartesian = np.array(to_cartesian(*old_position))
        new_cartesian = np.array(to_cartesian(*new_position))
        move_vector = new_cartesian - old_cartesian

        if move_all_vertices:
            # 다른 다각형의 같은 방향의 꼭짓점을 이동시킵니다.
            for polygon_info in polygons:
                if polygon_info != selected_polygon_info:
                    old_vertex_position = polygon_info.vertices[selected_vertex_index]
                    old_vertex_cartesian = np.array(to_cartesian(*old_vertex_position))
                    new_vertex_cartesian = old_vertex_cartesian + move_vector * (polygon_info.index + 1) / (selected_polygon_info.index + 1) * weight
                    new_vertex_position = to_polar(*new_vertex_cartesian)
                    polygon_info.update_vertex(selected_vertex_index, new_vertex_position)

        # 현재 플롯을 지우고
        ax.clear()
        # HSV 배경을 그립니다.
        ax.imshow(hsv_background, extent=(-plt_size, plt_size, -plt_size, plt_size), origin='lower')
        # 모든 다각형을 다시 그립니다.
        for polygon_info in polygons:
            cartesian_polygon = [to_cartesian(*vertex) for vertex in polygon_info.vertices]
            facecolor = (volume, volume, volume, 0.1)  # volume 값에 따라 밝기 조절
            poly_patch = Polygon(cartesian_polygon, closed=True, edgecolor='r', facecolor=facecolor, alpha=0.1)
            ax.add_patch(poly_patch)
            for vertex in cartesian_polygon:
                ax.plot(vertex[0], vertex[1], 'bo')  # 파란색 점으로 표시
        # 선택된 꼭짓점을 빨간색으로 강조합니다.
        if selected_polygon_info is not None and selected_polygon_info.selected_vertex_index is not None:
            selected_point = selected_polygon_info.vertices[selected_polygon_info.selected_vertex_index]
            index_info = f'Polygon {selected_polygon_info.index}, Vertex {selected_polygon_info.selected_vertex_index}'
            sx, sy = to_cartesian(*selected_point)
            ax.plot(sx, sy, 'ro')  # 빨간색 점으로 표시
            # 선택된 꼭짓점의 극좌표를 화면 왼쪽 위에 표시합니다.
            r, theta = selected_point
            lutPoint = lutCalculation.find_close_lut(theta,r,volume,num_polygons*2)
            update_coord_text(f'{index_info}\nr: {r:.2f}, θ: {theta:.2f} rad\n LUT : [{lutPoint}], R : {(lutPoint[0]*16)-1}, G : {(lutPoint[1]*16)-1}, R : {(lutPoint[2]*16)-1}')

        # 같은 방향의 꼭짓점들을 점선으로 연결합니다.
        for vertex_index in range(len(polygons[0].vertices)):
            x_coords = []
            y_coords = []
            for polygon_info in polygons:
                vx, vy = to_cartesian(*polygon_info.vertices[vertex_index])
                x_coords.append(vx)
                y_coords.append(vy)
            ax.plot(x_coords, y_coords, 'k--', alpha=0.5)  # 점선으로 연결

        # 마우스 위치의 색상을 추출하여 표시합니다.
        if event.inaxes is not None:
            ix, iy = int(event.xdata * background_size / (plt_size*2) + (background_size/2)), int(event.ydata * background_size / (plt_size*2) + (background_size/2))
            if 0 <= ix < background_size and 0 <= iy < background_size:
                hovered_color = hsv_background[iy, ix]
                update_color_patch(hovered_color)
        ax.add_patch(color_patch)

        # 축의 범위를 설정합니다.
        ax.set_xlim(-plt_size, plt_size)
        ax.set_ylim(-plt_size, plt_size)
        ax.set_title('Color Grading Graph',pad=6,size=20)

        # 그래프를 업데이트합니다.
        plt.draw()

# 클릭 이벤트 핸들러 함수
def onclick(event):
    global selected_polygon_info
    if event.button == 1:  # 왼쪽 클릭
        # 클릭한 좌표와 가까운 꼭짓점을 찾습니다.
        min_dist = float('inf')
        for polygon_info in polygons:
            for j, vertex in enumerate(polygon_info.vertices):
                vx, vy = to_cartesian(*vertex)
                dist = np.hypot(vx - event.xdata, vy - event.ydata)
                if dist < min_dist and dist < vertex_select_threshold:
                    min_dist = dist
                    selected_polygon_info = polygon_info
                    selected_polygon_info.select_vertex(j)

        # 선택된 꼭짓점이 없으면 초기화합니다.
        if selected_polygon_info is None or selected_polygon_info.selected_vertex_index is None:
            selected_polygon_info = None

        # 현재 플롯을 지우고
        ax.clear()

        # HSV 배경을 그립니다.
        ax.imshow(hsv_background, extent=(-plt_size, plt_size, -plt_size, plt_size), origin='lower')

        # 모든 다각형을 다시 그립니다.
        for polygon_info in polygons:
            cartesian_polygon = [to_cartesian(*vertex) for vertex in polygon_info.vertices]
            facecolor = (volume, volume, volume, 0.1)  # volume 값에 따라 밝기 조절
            poly_patch = Polygon(cartesian_polygon, closed=True, edgecolor='r', facecolor=facecolor, alpha=0.1)
            ax.add_patch(poly_patch)
            for vertex in cartesian_polygon:
                ax.plot(vertex[0], vertex[1], 'bo')  # 파란색 점으로 표시

        # 선택된 꼭짓점을 빨간색으로 강조합니다.
        if selected_polygon_info is not None and selected_polygon_info.selected_vertex_index is not None:
            selected_point = selected_polygon_info.vertices[selected_polygon_info.selected_vertex_index]
            index_info = f'Polygon {selected_polygon_info.index}, Vertex {selected_polygon_info.selected_vertex_index}'
            sx, sy = to_cartesian(*selected_point)
            ax.plot(sx, sy, 'ro')  # 빨간색 점으로 표시

            # 선택된 꼭짓점의 극좌표를 화면 왼쪽 위에 표시합니다.
            r, theta = selected_point
            lutPoint = lutCalculation.find_close_lut(theta,r,volume,num_polygons*2)
            update_coord_text(f'{index_info}\nr: {r:.2f}, θ: {theta:.2f} rad\n LUT : [{lutPoint}], R : {(lutPoint[0]*16)-1}, G : {(lutPoint[1]*16)-1}, R : {(lutPoint[2]*16)-1}')
        else:
            update_coord_text('')  # 선택된 꼭짓점이 없을 때 빈 문자열 표시

        # 같은 방향의 꼭짓점들을 점선으로 연결합니다.
        for vertex_index in range(len(polygons[0].vertices)):
            x_coords = []
            y_coords = []
            for polygon_info in polygons:
                vx, vy = to_cartesian(*polygon_info.vertices[vertex_index])
                x_coords.append(vx)
                y_coords.append(vy)
            ax.plot(x_coords, y_coords, 'k--', alpha=0.2)  # 점선으로 연결

        # 마우스 위치의 색상을 추출하여 표시합니다.
        if event.inaxes is not None:
            ix, iy = int(event.xdata * background_size / (plt_size*2) + (background_size/2)), int(event.ydata * background_size / (plt_size*2) + (background_size/2))
            if 0 <= ix < background_size and 0 <= iy < background_size:
                hovered_color = hsv_background[iy, ix]
                update_color_patch(hovered_color)
        ax.add_patch(color_patch)

        # 축의 범위를 설정합니다.
        ax.set_xlim(-plt_size, plt_size)
        ax.set_ylim(-plt_size, plt_size)

        # 그래프를 업데이트합니다.
        plt.draw()

def onclick2(event):
    global img3, text3
    tImg = img3.convert('RGB')
    # 클릭한 위치의 좌표
    x, y = int(event.xdata), int(event.ydata)
    
    # 이미지의 R, G, B 값 가져오기
    if x >= 0 and y >= 0 and x < tImg.width and y < tImg.height:
        r, g, b = tImg.getpixel((x, y))  # RGB 값 가져오기
        text3.set_text(f'R: {r}, G: {g}, B: {b}')  # 텍스트 업데이트
        color_box.set_color((r/255, g/255, b/255))  # 색상 박스 업데이트
        if debug_print :
            print(f'Clicked at ({x}, {y}) - R: {r}, G: {g}, B: {b}')
    else:
        text3.set_text('')  # 클릭이 이미지 범위를 벗어난 경우 텍스트 초기화
        color_box.set_color('white')  # 색상 박스 초기화
        if debug_print :
            print('Click is out of image bounds.')

    # 텍스트 위치 업데이트
    plt.draw()  # 그래프 업데이트

# Reset 버튼 클릭 시 호출될 함수
def reset(event):
    global lutArray, originalLut, image_path, ax, fig, hsv_background
    
    if debug_print :
        print("Reset")
    for vertex_index in range(len(polygons[0].vertices)):
        for polygon_info in polygons:
            vertex = original_polgon_coordinate[polygon_info.index][vertex_index]
            polygon_info.vertices[vertex_index] = [vertex[0], vertex[1]]

    # 다각형을 다시 그리기 위해 축을 초기화하고 다각형을 다시 추가합니다.
    ax.clear()
    ax.set_xlim(-plt_size, plt_size)
    ax.set_ylim(-plt_size, plt_size)
    ax.set_title('Color Grading Graph', pad=3, size=20)
    ax.set_aspect('equal')
    ax.imshow(hsv_background, extent=(-plt_size, plt_size, -plt_size, plt_size), origin='lower')
    
    for polygon_info in polygons:
        cartesian_polygon = [to_cartesian(*vertex) for vertex in polygon_info.vertices]
        facecolor = (volume, volume, volume, 0.1)  # volume 값에 따라 밝기 조절
        poly_patch = Polygon(cartesian_polygon, closed=True, edgecolor='r', facecolor=facecolor, alpha=0.1)
        ax.add_patch(poly_patch)
        for vertex in cartesian_polygon:
            ax.plot(vertex[0], vertex[1], 'bo')
    
    # 같은 방향의 꼭짓점들을 점선으로 연결합니다.
    for vertex_index in range(len(polygons[0].vertices)):
        x_coords = []
        y_coords = []
        for polygon_info in polygons:
            vx, vy = to_cartesian(*polygon_info.vertices[vertex_index])
            x_coords.append(vx)
            y_coords.append(vy)
        ax.plot(x_coords, y_coords, 'k--', alpha=0.5)

    for vertex_index in range(len(polygons[0].vertices)):
        for polygon_info in polygons:
            vertex = polygon_info.vertices[vertex_index]
            prev_polgon_coordinate[polygon_info.index][vertex_index] =  original_polgon_coordinate[polygon_info.index][vertex_index]
    
    
    # LUT 초기화
    lutArray = pd.read_csv("./bypasslut.txt",sep="\t").to_numpy()
    lutCalculation.saveLUTfile(lutArray)
    update_image()

    update_coord_text('')  # 선택된 꼭짓점이 없을 때 빈 문자열 표시

    plt.draw()

def update_coord_text(text):
    global coord_text
    # 기존 텍스트 객체를 제거하고 새로 생성
    for txt in ax.texts:
        txt.set_visible(False)
    coord_text = ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12, verticalalignment='top')

# 마우스 위치의 색상을 화면에 표시하는 함수
def update_color_patch(color):  # 새로 추가된 부분
    color_patch.set_facecolor(color * np.clip((volume + 0.3), 0.3, 1))  # 새로 추가된 부분

def toggle_move_all_vertices(label):
    global move_all_vertices
    move_all_vertices = not move_all_vertices

def update_weight(val):
    global weight
    weight = val

def update_volume(val):
    global volume, hsv_background, ax
    # 다각형을 다시 그리기 위해 축을 초기화하고 다각형을 다시 추가합니다.
    ax.clear()
    ax.set_xlim(-plt_size, plt_size)
    ax.set_ylim(-plt_size, plt_size)
    ax.set_title('Color Grading Graph', pad=3, size=20)
    ax.set_aspect('equal')
    ax.imshow(hsv_background, extent=(-plt_size, plt_size, -plt_size, plt_size), origin='lower')
    
    for polygon_info in polygons:
        cartesian_polygon = [to_cartesian(*vertex) for vertex in polygon_info.vertices]
        # facecolor를 volume 값에 따라 설정
        facecolor = (volume, volume, volume, 1.0)  # volume 값에 따라 밝기 조절, alpha 값을 1.0으로 설정하여 불투명하게 만듦
        poly_patch = Polygon(cartesian_polygon, closed=True, edgecolor='r', facecolor=facecolor)
        ax.add_patch(poly_patch)
        for vertex in cartesian_polygon:
            ax.plot(vertex[0], vertex[1], 'bo')
    
    # 같은 방향의 꼭짓점들을 점선으로 연결합니다.
    for vertex_index in range(len(polygons[0].vertices)):
        x_coords = []
        y_coords = []
        for polygon_info in polygons:
            vx, vy = to_cartesian(*polygon_info.vertices[vertex_index])
            x_coords.append(vx)
            y_coords.append(vy)
        ax.plot(x_coords, y_coords, 'k--', alpha=0.5)
        
    volume = val
    hsv_background = create_hsv_background((-plt_size, plt_size), (-plt_size, plt_size), volume)
    ax.imshow(hsv_background, extent=(-plt_size, plt_size, -plt_size, plt_size), origin='lower')
    fig.canvas.draw_idle()
    # 여기서 volume 값을 사용하여 필요한 작업을 수행합니다.
    if debug_print :
        print(f"Volume updated to: {volume}")

def update_image():
    global lut,fig2,fig4,img1,img3,image_display, image_display2
    lut = load_cube_file("./makeLUT.CUBE")
    img2 = img1.filter(lut)
    image_display.set_data(img2)
    fig2.canvas.draw()
    img4 = img3.filter(lut)
    image_display2.set_data(img4)
    fig4.canvas.draw()

def domain_change(near, ori, target):
    global lutArray
    global originalLut

    if debug_print :
        print(f"[ori] near : {near}, ori : {ori}, target : {target}")

    # Ratio based method
    oDiff = []
    tDiff = []
    for i in range(3):
        oDiff.append(near[i] - ori[i])
        tDiff.append(near[i] - target[i])

    oTmpList = []
    for i in range(3):
        if oDiff[i] == 0:
            oTmpList.append([near[i]])
        elif oDiff[i] < 0:
            tmp = []
            for j in range(0,-(oDiff[i]-1)):
                tmp.append(near[i] + j)
            oTmpList.append(tmp)
        else:
            tmp = []
            for j in range(0,oDiff[i]+1):
                tmp.append(near[i] - j)
            oTmpList.append(tmp)

    if debug_print :
        print(f"oTmpList : r:{oTmpList[0]}, g:{oTmpList[1]}, b:{oTmpList[2]}")

    oLutList = []
    for i in oTmpList[0]:
        for j in oTmpList[1]:
            for k in oTmpList[2]:
                # need to check v Value.
                oLutList.append([i,j,k])

    # should remove first, last. the original, target
    if len(oLutList) >= 3:
        del oLutList[len(oLutList)-1]
        del oLutList[0]

    oLutRatio = []
    for idx, oLut in enumerate(oLutList):
        tmpRatio = []
        for i in range(3):
            if oDiff[i] == 0:
                # Same
                tmpRatio.append(1.0)
            else:
                tmpRatio.append((float)(abs(near[i] - oLut[i]) / abs(oDiff[i])))
        if debug_print :
            print(f"{oLutList[idx]} => {tmpRatio}")
        oLutRatio.append(tmpRatio)

    if debug_print :
        print(f"[target] near : {near}, ori : {ori}, target : {target}")

    tTmpList = []
    for i in range(3):
        if tDiff[i] == 0:
            tTmpList.append([near[i]])
        elif tDiff[i] < 0:
            tmp = []
            for j in range(0,-(tDiff[i]-1)):
                tmp.append(near[i] + j)
            tTmpList.append(tmp)
        else:
            tmp = []
            for j in range(0,tDiff[i]+1):
                tmp.append(near[i] - j)
            tTmpList.append(tmp)

    if debug_print :
        print(f"tTmpList : r:{tTmpList[0]}, g:{tTmpList[1]}, b:{tTmpList[2]}")

    tLutList = []
    for i in tTmpList[0]:
        for j in tTmpList[1]:
            for k in tTmpList[2]:
                tLutList.append([i,j,k])
    #print(lutList)
    # need to match new target
    # diff = | ori - near | target - near |
    # r ratio
    tRatio = []
    for i in range(3):
        if len(tTmpList[i]) != 0:
            if tDiff[i] != 0:
                tRatio.append((1 / abs(tDiff[i])))
            else:
                tRatio.append(0.0)
        else:
            tRatio.append(0.0)
    if debug_print :
        print(f"tRatio = {tRatio}")

    # set LUT
    for idx, lut in enumerate(oLutList):
        # get target LUT
        tLut = []
        for rgb in range(3):
            # ratio 0 case -> same coordinate
            if abs(tRatio[rgb]) <= 1e-5:
                tLut.append(near[rgb])
            elif abs(tRatio[rgb] - 1.0) <= 1e-5:
                if oLutRatio[idx][rgb] >= 0.5:
                    tLut.append(target[rgb])
                else:
                    tLut.append(near[rgb])
            else:
                # 1.0 case -> match to the target LUT
                if abs(oLutRatio[idx][rgb] - 1.0) <= 1e-5:
                    tLut.append(target[rgb])
                # 0 case
                elif abs(oLutRatio[idx][rgb]) <= 1e-5:
                    tLut.append(near[rgb])
                else:
                    resultVal = 0.0
                    # near bigger case
                    if near[rgb] - target[rgb] > 0:
                        resultVal = round(near[rgb] - (oLutRatio[idx][rgb] * abs(tDiff[rgb])))
                    # near smaller case
                    else:
                        resultVal = round(near[rgb] + (oLutRatio[idx][rgb] * abs(tDiff[rgb])))
                    #print(f"{lut} : {lut[rgb]:2d} -> {oLutRatio[idx][rgb]:1.4f} * {abs(tDiff[rgb]):1.4f} = {oLutRatio[idx][rgb] * abs(tDiff[rgb]):1.4f} near = {near[rgb]:2d}, resultVal = {resultVal:2d}")
                    tLut.append(resultVal)

        arrayTarget = lutCalculation.getLinearArrayIndex(tLut[0],tLut[1],tLut[2],17)
        originalIdx = lutCalculation.getLinearArrayIndex(lut[0],lut[1],lut[2],17)
        oriHsv = lutCalculation.find_lut_hsv(lut)
        tarHsv = lutCalculation.find_lut_hsv(tLut)
        if debug_print :
            print(f"original : {lut} = target {tLut}. original coordinate = [theta = {oriHsv[0]*2*pi}, r = {oriHsv[1]*2*num_polygons}, v = {oriHsv[2]}, idx = {arrayTarget}, LUT value = {originalLut[arrayTarget]}")
        lutArray[originalIdx] = originalLut[arrayTarget]

def update_lut_gain(polygon_index,vertex_index,target_r,target_theta):
    global lutArray
    global originalLut

    if debug_print :
        print(f"selected Point = p,v -> [{polygon_index},{vertex_index}]")
    #1. Get Original LUT
    ori_r = original_polgon_coordinate[polygon_index][vertex_index][0]
    ori_theta = original_polgon_coordinate[polygon_index][vertex_index][1]
    #print(f"p_index = {polygon_index}, v_index = {vertex_index}, ori_r = {ori_r}, ori_theta = {ori_theta}, r = {target_r}, theta = {target_theta}")
    originalLutIndex = lutCalculation.find_close_lut(ori_theta,ori_r,volume,num_polygons*2)

    #2. Get Target LUT
    targetlLutIndex = lutCalculation.find_close_lut(target_theta,target_r,volume,num_polygons*2)
    #print(f"[{vertex_index}-{polygon_index}], original LUT = [{originalLutIndex[0]}, {originalLutIndex[1]}, {originalLutIndex[2]}], target LUT = [{targetlLutIndex[0]}, {targetlLutIndex[1]}, {targetlLutIndex[2]}]")
    
    #3. Calculate nearby LUT
    # polygon -1 0 1
    # vertial -1 0 1

    originalLUT = []
    targetLUT = []
    for i in range(3):
        pIdx = i - 1 + polygon_index
        if pIdx < 0:
            pIdx = 0
        elif pIdx >= num_polygons:
            pIdx = num_polygons - 1
        tmpSlot1 = []
        tmpSlot2 = []
        for j in range(3):
            vIdx = ((j - 1 + vertex_index) + num_vertices) % num_vertices
            if debug_print :
                print(f"[{pIdx},{vIdx}] = [{polygons[pIdx].vertices[vIdx][1]},{polygons[pIdx].vertices[vIdx][0]}]")
            originalIndx = lutCalculation.find_close_lut(original_polgon_coordinate[pIdx][vIdx][1],original_polgon_coordinate[pIdx][vIdx][0],volume,num_polygons*2)
            #need to check it's original or moved point
            targetIdx = lutCalculation.find_close_lut(polygons[pIdx].vertices[vIdx][1],polygons[pIdx].vertices[vIdx][0],volume,num_polygons*2)
            tmpSlot1.append(targetIdx)
            tmpSlot2.append(originalIndx)
        targetLUT.append(tmpSlot1)
        originalLUT.append(tmpSlot2)
        
#LUT Base calculate
    if debug_print :
        print(f"{originalLUT[0]} : {targetLUT[0]}\n{originalLUT[1]} : {targetLUT[1]}\n{originalLUT[2]} : {targetLUT[2]}\n")
    
    # should calculate 
    # array compare original
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue
            # need to fix
            domain_change(originalLUT[i][j],originalLUT[1][1],targetLUT[1][1])
    #domain_change(originalLUT[1][0],originalLUT[1][1],targetLUT[1][1])
    lutArray[lutCalculation.getLinearArrayIndex(originalLUT[0],originalLUT[1],originalLUT[2],17)] = originalLut[lutCalculation.getLinearArrayIndex(targetLUT[0],targetLUT[1],targetLUT[2],17)]

    lutCalculation.saveLUTfile(lutArray)

def update_coordinate(selected_vertice):
    for polygon_info in polygons:
        vertex = polygon_info.vertices[selected_vertice]
        prev_coordinate = [prev_polgon_coordinate[polygon_info.index][selected_vertice][0],prev_polgon_coordinate[polygon_info.index][selected_vertice][1]]
        if not((abs(prev_coordinate[0] - vertex[0]) <= 1e-5) & (abs(prev_coordinate[1] - vertex[1])  <= 1e-5)):
            update_lut_gain(polygon_info.index,selected_vertice,vertex[0],vertex[1])
            prev_polgon_coordinate[polygon_info.index][selected_vertice][0] = vertex[0]
            prev_polgon_coordinate[polygon_info.index][selected_vertice][1] = vertex[1]

def lut_binary_save(event):
    if debug_print :
        print("Original shape:", lutArray.shape)
        print(lutArray)

    # 원본 격자 생성
    x = np.linspace(0, 1, 17)
    y = np.linspace(0, 1, 17)
    z = np.linspace(0, 1, 17)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z)

    # 새로운 격자 생성 (33x33x33)
    new_x = np.linspace(0, 1, 33)
    new_y = np.linspace(0, 1, 33)
    new_z = np.linspace(0, 1, 33)
    new_grid_x, new_grid_y, new_grid_z = np.meshgrid(new_x, new_y, new_z)

    # 원본 데이터의 RGB 값을 3D 격자에 맞게 보간
    # 원본 데이터의 RGB 값
    values = lutArray.reshape((17, 17, 17, 3))

    # 보간 수행
    new_values = griddata((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), 
                        values.reshape(-1, 3), 
                        (new_grid_x.flatten(), new_grid_y.flatten(), new_grid_z.flatten()), 
                        method='linear')

    # 결과를 33x33x33 형태로 재구성
    reshaped_new_data = new_values.reshape((33, 33, 33, 3))

    # (35937, 3) 형태로 변환
    final_data = reshaped_new_data.reshape(-1, 3)

    # 결과 확인
    if debug_print :
        print("Final data shape:", final_data.shape)
        print(final_data)
    binArray = np.empty(33 * 33 * 33 *3, dtype=np.uint16)

    for i in range(33 * 33 * 33):
        binArray[3*i] = final_data[i][0] * 4095
        binArray[(3*i)+1] = final_data[i][1] * 4095
        binArray[(3*i)+2] = final_data[i][2] * 4095

    if debug_print :
        print("binArray data shape : ",binArray.shape)
        print(binArray[:50])
        print(binArray[(33 * 33 * 33 * 3)-50:])
    lutCalculation.saveLUTfileBin(binArray)
    
    print("Save Complete")

# Matplotlib 설정
fig, ax = plt.subplots()
#fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.25)  # 그래프와 다른 요소들 간의 간격 조정

ax.set_xlim(-plt_size, plt_size)
ax.set_ylim(-plt_size, plt_size)
ax.set_title('Color Grading Graph',pad=3,size=20)
ax.set_aspect('equal')

# HSV 배경 생성
hsv_background = create_hsv_background((-plt_size, plt_size), (-plt_size, plt_size),volume)
ax.imshow(hsv_background, extent=(-plt_size, plt_size, -plt_size, plt_size), origin='lower')

# 다각형 생성
generate_polygons()

# 다각형 그리기
for polygon_info in polygons:
    cartesian_polygon = [to_cartesian(*vertex) for vertex in polygon_info.vertices]
    # facecolor를 volume 값에 따라 설정
    facecolor = (volume, volume, volume, 0.1)  # volume 값에 따라 밝기 조절
    poly_patch = Polygon(cartesian_polygon, closed=True, edgecolor='r', facecolor=facecolor, alpha=0.1)
    ax.add_patch(poly_patch)
    for vertex in cartesian_polygon:
        ax.plot(vertex[0], vertex[1], 'bo')  # 파란색 점으로 표시

# 좌표 텍스트 표시
coord_text = ax.text(-plt_size, plt_size, '', fontsize=9, verticalalignment='top')

# 마우스 위치의 색상을 표시할 사각형 패치 객체
color_patch = Rectangle((-plt_size, plt_size-10), 2, 2, linewidth=1, edgecolor='black', facecolor='none') 
ax.add_patch(color_patch)

# 체크박스 추가
rax = plt.axes([0.1, 0.05, 0.1, 0.05], facecolor='lightgoldenrodyellow')
check = CheckButtons(rax, ['Move All'], [move_all_vertices])
check.on_clicked(toggle_move_all_vertices)

# save 버튼 추가
save_ax = plt.axes([0.1, 0.1, 0.1, 0.05])  # Adjust the position as needed
save_button = Button(save_ax, 'Save')
save_button.on_clicked(lut_binary_save)

# 슬라이더 추가
rax_slider = plt.axes([0.3, 0.05, 0.4, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(rax_slider, 'Weight', 0.1, 2.0, valinit=weight)
slider.on_changed(update_weight)

# Volume 슬라이더 추가 (오른쪽에 위치)
rax_volume_slider = plt.axes([0.95, 0.25, 0.03, 0.6], facecolor='lightgoldenrodyellow')
volume_slider = Slider(rax_volume_slider, 'Volume', 0.0, 1.0, valinit=volume, orientation='vertical')
volume_slider.on_changed(update_volume)

# Reset 버튼 추가
reset_ax = plt.axes([0.8, 0.05, 0.1, 0.05])
reset_button = Button(reset_ax, 'Reset')
reset_button.on_clicked(reset)

# 같은 방향의 꼭짓점들을 점선으로 연결
for vertex_index in range(len(polygons[0].vertices)):
    x_coords = []
    y_coords = []
    for polygon_info in polygons:
        vx, vy = to_cartesian(*polygon_info.vertices[vertex_index])
        x_coords.append(vx)
        y_coords.append(vy)
    ax.plot(x_coords, y_coords, 'k--', alpha=0.5)  # 점선으로 연결

# 첫 번째 이미지
#ax_inset1 = inset_axes(ax2, width="100%", height="100%", loc='upper left', bbox_to_anchor=(0.1, 0.5, 1, 0.5), bbox_transform=ax2.transAxes)
fig1, ax1 = plt.subplots()
img1 = Image.open(image_path)
ax1.imshow(img1)
ax1.axis('off')
#ax2.text(0.05, 0.95, "Original", transform=ax_inset1.transAxes, fontsize=12, ha='left', va='top', color='white', bbox=dict(facecolor='black', alpha=0.5))
#ax1.text(0.05, 0.95, "Original", fontsize=12, ha='left', va='top', color='white', bbox=dict(facecolor='black', alpha=0.5))
ax1.set_title("Original")
plt.tight_layout()
plt.show(block=False)

# 두 번째 이미지
#ax_inset2 = inset_axes(ax2, width="100%", height="100%", loc='lower left', bbox_to_anchor=(0.1, 0.0, 1, 0.5), bbox_transform=ax2.transAxes)
lut = load_cube_file("./makeLUT.CUBE")
img2 = img1.filter(lut)
fig2, ax2 = plt.subplots()
img2 = Image.open(image_path)
image_display = ax2.imshow(img2)
ax2.axis('off')
#ax_inset2.text(0.05, 0.95, "Result", transform=ax_inset2.transAxes, fontsize=12, ha='left', va='top', color='white', bbox=dict(facecolor='black', alpha=0.5))
#ax2.text(0.05, 0.95, "Result", fontsize=12, ha='left', va='top', color='white', bbox=dict(facecolor='black', alpha=0.5))
ax2.set_title("Result")
plt.tight_layout()
plt.show(block=False)

# Real Image
fig3, ax3 = plt.subplots()
img3 = Image.open(image_path2)
ax3.imshow(img3)
ax3.axis('off')
ax3.set_title("Original")
# 텍스트 객체 생성 (초기값은 빈 문자열)
text3 = ax3.text(0.70, 0.95, '', fontsize=10, verticalalignment='top', transform=ax3.transAxes, bbox=dict(facecolor='white', alpha=0.5))
color_box = plt.Rectangle((0.9, 0.8), 0.1, 0.1, color='black', transform=ax3.transAxes)
ax3.add_patch(color_box)
plt.show(block=False)

img4 = img3.filter(lut)
fig4, ax4 = plt.subplots()
img4 = Image.open(image_path2)
image_display2 = ax4.imshow(img4)
ax4.axis('off')
ax4.set_title("Result")
plt.tight_layout()
plt.show(block=False)

# 이벤트 연결
fig.canvas.mpl_connect('button_press_event', onpress)
fig.canvas.mpl_connect('button_release_event', onrelease)
fig.canvas.mpl_connect('motion_notify_event', onmove)
fig.canvas.mpl_connect('button_press_event', onclick)
fig3.canvas.mpl_connect('button_press_event', onclick2)

for vertex_index in range(len(polygons[0].vertices)):
    for polygon_info in polygons:
        vertex = polygon_info.vertices[vertex_index]
        prev_polgon_coordinate[polygon_info.index][vertex_index] =  [vertex[0],vertex[1]]
        original_polgon_coordinate[polygon_info.index][vertex_index] =  [vertex[0],vertex[1]]
        originalLutIndex = lutCalculation.find_close_lut(vertex[1],vertex[0],volume,num_polygons*2)
        if debug_print :
            print(f"[{vertex_index}-{polygon_info.index}] r = {vertex[0]}, theta = {vertex[1]}, LUT : [{originalLutIndex[0]},{originalLutIndex[1]},{originalLutIndex[2]}]")

plt.tight_layout()
plt.show()