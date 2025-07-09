import cv2
import numpy as np
import pygame
from pygame.locals import *
import pygame.font
from OpenGL.GL import *
from OpenGL.GLU import *
from radar_module import HLKLD2451ShortRange
from ultralytics import YOLO
import os
import ctypes

# 全局存储最近目标信息
latest_distance = None
latest_angle = None
radar_running = False  # 控制雷达状态

# 全局变量保存雷达设置
radar_setting = {
    'enabled': True,  # 是否启用雷达
    'mode': None     # 当前选择的模式（1-4），None为未设置
}

# 回调函数，更新全局变量
def radar_callback(targets):
    global latest_distance, latest_angle
    if targets:
        closest = min(targets, key=lambda t: t['distance'])
        latest_distance = closest['distance']
        latest_angle = closest['angle']
    else:
        latest_distance = None
        latest_angle = None

# 绘制雷达信息覆盖
def draw_radar_overlay_pygame():
    from OpenGL.GL import glDrawPixels, GL_RGBA, GL_UNSIGNED_BYTE
    info = "未检测到目标" if latest_distance is None else f"车辆靠近 | 距离: {latest_distance:.1f}m | 角度: {latest_angle:.1f}°"
    font = pygame.font.Font("NotoSansSC-Black.ttf", 18)
    text_surface = font.render(info, True, (255, 255, 255))
    text_data = pygame.image.tostring(text_surface, "RGBA", True)
    w, h = text_surface.get_size()
    
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, 800, 0, 600, -1, 1)  # 与屏幕坐标匹配
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glRasterPos2f(10, 580)
    glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

# 加载 YOLO 模型
model = YOLO("./yolo11n_rknn_model")

# 车辆模型的顶点及面定义
car_vertices = [
    # 车身顶面
    [ 1.0, 0.5, -2.0],
    [ 1.0, 0.5,  2.0],
    [-1.0, 0.5,  2.0],
    [-1.0, 0.5, -2.0],
    # 车身底面
    [ 1.0, -0.5, -2.0],
    [ 1.0, -0.5,  2.0],
    [-1.0, -0.5,  2.0],
    [-1.0, -0.5, -2.0],
    # 车顶顶面
    [ 0.8, 1.0, -0.5],
    [ 0.8, 1.0,  0.5],
    [-0.8, 1.0,  0.5],
    [-0.8, 1.0, -0.5],
]

# 车辆模型面的索引
car_surfaces = [
    (4, 5, 6, 7),
    (0, 1, 2, 3),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
    (8, 9, 10, 11),
    (0, 3, 11, 8),
    (1, 0, 8, 9),
    (2, 1, 9, 10),
    (3, 2, 10, 11),
]

# 车身配色
car_colors = [
    (0.8, 0.0, 0.0),  # 深红
    (0.8, 0.0, 0.0),
    (0.8, 0.0, 0.0),
    (0.8, 0.0, 0.0),
    (0.8, 0.0, 0.0),
    (0.8, 0.0, 0.0),
    (0.5, 0.5, 0.5),  # 车顶灰
    (0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5),
]

def draw_car(position, rotation=0, scale=1):
    """绘制简易轿车模型"""
    glPushMatrix()
    glTranslatef(position[0], position[1], position[2])
    glScalef(scale, scale, scale)
    glRotatef(rotation, 0, 1, 0)

    glBegin(GL_QUADS)
    for i, surface in enumerate(car_surfaces):
        color = car_colors[i % len(car_colors)]
        glColor3fv(color)
        for idx in surface:
            glVertex3fv(car_vertices[idx])
    glEnd()

    glColor3f(0, 0, 0)
    glBegin(GL_LINES)
    for surface in car_surfaces:
        for j in range(4):
            v0 = car_vertices[surface[j]]
            v1 = car_vertices[surface[(j + 1) % 4]]
            glVertex3fv(v0)
            glVertex3fv(v1)
    glEnd()

    glPopMatrix()

# 行人模型数据
person_color = (0.0, 0.0, 1.0)  # 蓝色
person_vertices = [
    [0.2, 0.0, 0.1], [0.2, 0.0, -0.1], [-0.2, 0.0, -0.1], [-0.2, 0.0, 0.1],
    [0.2, 1.0, 0.1], [0.2, 1.0, -0.1], [-0.2, 1.0, -0.1], [-0.2, 1.0, 0.1],
    [0.15, 1.0, 0.15], [0.15, 1.0, -0.15], [-0.15, 1.0, -0.15], [-0.15, 1.0, 0.15],
    [0.15, 1.3, 0.15], [0.15, 1.3, -0.15], [-0.15, 1.3, -0.15], [-0.15, 1.3, 0.15],
    [-0.25, 0.8, 0.05], [-0.25, 0.8, -0.05], [-0.25, 0.2, -0.05], [-0.25, 0.2, 0.05],
    [-0.15, 0.8, 0.05], [-0.15, 0.8, -0.05], [-0.15, 0.2, -0.05], [-0.15, 0.2, 0.05],
    [0.15, 0.8, 0.05], [0.15, 0.8, -0.05], [0.15, 0.2, -0.05], [0.15, 0.2, 0.05],
    [0.25, 0.8, 0.05], [0.25, 0.8, -0.05], [0.25, 0.2, -0.05], [0.25, 0.2, 0.05],
    [-0.15, 0.0, 0.05], [-0.15, 0.0, -0.05], [-0.15, -0.6, -0.05], [-0.15, -0.6, 0.05],
    [-0.05, 0.0, 0.05], [-0.05, 0.0, -0.05], [-0.05, -0.6, -0.05], [-0.05, -0.6, 0.05],
    [0.05, 0.0, 0.05], [0.05, 0.0, -0.05], [0.05, -0.6, -0.05], [0.05, -0.6, 0.05],
    [0.15, 0.0, 0.05], [0.15, 0.0, -0.05], [0.15, -0.6, -0.05], [0.15, -0.6, 0.05],
]
person_model_parts = [
    (0, 8),    # 身体
    (8, 8),    # 头部
    (16, 8),   # 左臂
    (24, 8),   # 右臂
    (32, 8),   # 左腿
    (40, 8),   # 右腿
]
person_surfaces = [
    (0, 1, 2, 3), (4, 5, 6, 7),
    (0, 1, 5, 4), (1, 2, 6, 5),
    (2, 3, 7, 6), (3, 0, 4, 7)
]
def draw_person(position, scale=1):
    glPushMatrix()
    glTranslatef(*position)
    glScalef(scale, scale, scale)
    glColor3fv(person_color)
    for base_idx, _ in person_model_parts:
        glBegin(GL_QUADS)
        for face in person_surfaces:
            for idx in face:
                glVertex3fv(person_vertices[base_idx + idx])
        glEnd()
    glColor3f(0, 0, 0)
    for base_idx, _ in person_model_parts:
        glBegin(GL_LINES)
        for face in person_surfaces:
            for i in range(4):
                v0 = person_vertices[base_idx + face[i]]
                v1 = person_vertices[base_idx + face[(i + 1) % 4]]
                glVertex3fv(v0)
                glVertex3fv(v1)
        glEnd()
    glPopMatrix()

def detect_vehicles(frame):
    results = model(frame)
    return results

def draw_results(frame, results, scale=0.5):
    vehicle_positions = []
    pedestrian_positions = []
    height, width = frame.shape[:2]
    cx, cy = width // 2, height // 2
    real_width = 1.75
    focal_length = 800
    camera_height = 1.1
    min_distance = 1.0
    for result in results:
        boxes = result.boxes.xyxy
        confidences = result.boxes.conf
        class_ids = result.boxes.cls
        # 获取类别id
        name2id = {name: idx for idx, name in model.names.items()}
        car_id = name2id.get('car', 2)
        person_id = name2id.get('person', 0)
        for i in range(len(boxes)):
            if confidences[i] > 0.5:
                x1, y1, x2, y2 = map(int, boxes[i])
                x_bottom = (x1 + x2) // 2
                y_bottom = y2
                dy = y_bottom - cy
                if dy > 0:
                    distance = (camera_height * focal_length) / dy
                    dx = x_bottom - cx
                    lateral_offset = (dx * distance) / focal_length
                    position = [
                        lateral_offset / 5.0,
                        0,
                        (-distance / 5.0)
                    ]
                    if int(class_ids[i]) == car_id:
                        for pos in vehicle_positions:
                            dist_between = np.linalg.norm(np.array(position[:2]) - np.array(pos[:2]))
                            if dist_between < min_distance:
                                adjustment = min_distance - dist_between
                                position[2] -= adjustment / 10.0
                        vehicle_positions.append(position)
                    elif int(class_ids[i]) == person_id:
                        for pos in pedestrian_positions:
                            dist_between = np.linalg.norm(np.array(position[:2]) - np.array(pos[:2]))
                            if dist_between < min_distance:
                                adjustment = min_distance - dist_between
                                position[2] -= adjustment / 10.0
                        pedestrian_positions.append(position)
    return vehicle_positions, pedestrian_positions

def draw_button(text, x, y, width, height, color):
    """绘制按键"""
    pygame.draw.rect(screen, color, (x, y, width, height))
    font = pygame.font.Font("NotoSansSC-Black.ttf", 24)  # 使用带有中文的字体
    text_surface = font.render(text, True, (255, 255, 255))
    screen.blit(text_surface, (x + 10, y + 10))

def home_screen():
    """车载终端首页UI"""
    global screen, radar_setting
    pygame.init()
    info = pygame.display.Info()
    width, height = info.current_w, info.current_h
    if width / height > 16/9:
        height = int(width * 9 / 16)
    else:
        width = int(height * 16 / 9)
    screen = pygame.display.set_mode((width, height), pygame.FULLSCREEN)
    pygame.display.set_caption("智能车载终端")
    clock = pygame.time.Clock()
    bg_path = os.path.join(os.getcwd(), "背景.jpg")
    bg_img = pygame.image.load(bg_path)
    bg_img = pygame.transform.scale(bg_img, (width, height))
    btn_w, btn_h = 260, 80
    btn_x = 80
    btn1_y = int(height * 0.35)
    btn2_y = btn1_y + btn_h + 40
    btn_radius = 40
    btn_color = (0, 102, 204)
    font = pygame.font.Font("NotoSansSC-Black.ttf", 36)
    # 雷达设置弹出菜单参数
    menu_open = False
    menu_btn_w, menu_btn_h = 260, 60
    menu_btn_gap = 20
    menu_x = btn_x + btn_w + 40
    menu_items = [
        ("普通道路预警模式", 1),
        ("高速道路预警模式", 2),
        ("倒车预警模式", 3),
        ("默认短距离模式", 4),
        ("雷达开关", 'switch')
    ]
    # 计算菜单整体高度并居中向上偏移
    menu_total_height = len(menu_items) * menu_btn_h + (len(menu_items)-1) * menu_btn_gap
    menu_y = btn2_y - menu_total_height//2 + btn_h//2
    menu_radius = 30
    menu_color = (0, 102, 204)
    menu_font = pygame.font.Font("NotoSansSC-Black.ttf", 28)
    setting_msg = ""
    msg_timer = 0
    running = True
    while running:
        screen.blit(bg_img, (0, 0))
        # 道路智能识别按钮
        pygame.draw.rect(screen, btn_color, (btn_x, btn1_y, btn_w, btn_h), border_radius=btn_radius)
        text1 = font.render("道路智能识别", True, (255, 255, 255))
        screen.blit(text1, (btn_x + (btn_w-text1.get_width())//2, btn1_y + (btn_h-text1.get_height())//2))
        # 雷达设置按钮
        pygame.draw.rect(screen, btn_color, (btn_x, btn2_y, btn_w, btn_h), border_radius=btn_radius)
        text2 = font.render("雷达设置", True, (255, 255, 255))
        screen.blit(text2, (btn_x + (btn_w-text2.get_width())//2, btn2_y + (btn_h-text2.get_height())//2))
        # 展开菜单
        if menu_open:
            for i, (label, val) in enumerate(menu_items):
                my = menu_y + i * (menu_btn_h + menu_btn_gap)
                pygame.draw.rect(screen, menu_color, (menu_x, my, menu_btn_w, menu_btn_h), border_radius=menu_radius)
                # 高亮当前选择
                if val == radar_setting['mode'] or (val == 'switch' and not radar_setting['enabled']):
                    pygame.draw.rect(screen, (0,180,255), (menu_x, my, menu_btn_w, menu_btn_h), 4, border_radius=menu_radius)
                label_show = label
                if val == 'switch':
                    label_show += "(开)" if radar_setting['enabled'] else "(关)"
                t = menu_font.render(label_show, True, (255,255,255))
                screen.blit(t, (menu_x + 20, my + (menu_btn_h-t.get_height())//2))
        # 设置成功提示
        if setting_msg:
            msg_font = pygame.font.Font("NotoSansSC-Black.ttf", 32)
            msg = msg_font.render(setting_msg, True, (0,255,0))
            screen.blit(msg, (menu_x, menu_y-60))
            if pygame.time.get_ticks() - msg_timer > 1200:
                setting_msg = ""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                # 道路智能识别按钮
                if btn_x <= mx <= btn_x+btn_w and btn1_y <= my <= btn1_y+btn_h:
                    # 读取雷达设置，传递给检测视图
                    radar_enable = radar_setting['enabled']
                    radar_mode = radar_setting['mode']
                    detect3d_screen(radar_enable, radar_mode)
                # 雷达设置按钮
                if btn_x <= mx <= btn_x+btn_w and btn2_y <= my <= btn2_y+btn_h:
                    menu_open = not menu_open
                # 菜单项
                if menu_open:
                    for i, (label, val) in enumerate(menu_items):
                        my_btn = menu_y + i * (menu_btn_h + menu_btn_gap)
                        if menu_x <= mx <= menu_x+menu_btn_w and my_btn <= my <= my_btn+menu_btn_h:
                            if val == 'switch':
                                radar_setting['enabled'] = not radar_setting['enabled']
                                setting_msg = f"雷达{'已开启' if radar_setting['enabled'] else '已关闭'}"
                            else:
                                radar_setting['mode'] = val
                                setting_msg = f"{label}设置成功"
                            msg_timer = pygame.time.get_ticks()
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

def draw_text_opengl(x, y, text, font, color, win_w, win_h):
    # 用pygame生成文字表面，然后贴到OpenGL
    text_surface = font.render(text, True, color)
    text_data = pygame.image.tostring(text_surface, "RGBA", True)
    w, h = text_surface.get_size()
    # OpenGL正交投影绘制
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, win_w, win_h, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glRasterPos2f(x, y + h)
    glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
    glDisable(GL_BLEND)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def draw_rounded_rect_opengl(x, y, w, h, radius, color, win_w, win_h):
    # 只画矩形（无圆角），用OpenGL正交投影
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, win_w, win_h, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glColor4f(color[0]/255, color[1]/255, color[2]/255, 1)
    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x+w, y)
    glVertex2f(x+w, y+h)
    glVertex2f(x, y+h)
    glEnd()
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def detect3d_screen(radar_enable, radar_mode):
    global radar_running, radar
    pygame.display.set_caption("智能检测3D视图")
    info = pygame.display.Info()
    width, height = info.current_w, info.current_h
    if width / height > 16/9:
        height = int(width * 9 / 16)
    else:
        width = int(height * 16 / 9)
    screen = pygame.display.set_mode((width, height), pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.OPENGL)
    cap = cv2.VideoCapture(11)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    clock = pygame.time.Clock()
    running = True
    if radar_enable:
        radar = HLKLD2451ShortRange('/dev/ttyS9', 115200)
        if radar.connect():
            if radar_mode == 1:
                radar.configure_normal_road_mode()
            elif radar_mode == 2:
                radar.configure_highway_mode()
            elif radar_mode == 3:
                radar.configure_reverse_mode()
            elif radar_mode == 4 or radar_mode is None:
                radar.configure_short_range()
            radar.start_data_reception(radar_callback)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, width / height, 0.1, 100.0)
    # UI参数
    btn_w = width // 5
    btn_h = 60
    btn_x = width - btn_w - 40
    btn_y = 40
    btn_radius = 30
    btn_font = pygame.font.Font("NotoSansSC-Black.ttf", 24)
    # 状态条参数
    bar_w = width // 2
    bar_h = 60
    bar_x = (width - bar_w) // 2
    bar_y = 40
    bar_radius = 30
    bar_font = pygame.font.Font("NotoSansSC-Black.ttf", 28)
    # 行人提示框参数
    person_w = width // 5
    person_h = 60
    person_x = 40
    person_y = 40
    person_radius = 30
    person_color = (255, 165, 0)  # 橙色
    person_font = pygame.font.Font("NotoSansSC-Black.ttf", 24)
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False
            if e.type == pygame.MOUSEBUTTONDOWN:
                mx, my = e.pos
                if btn_x <= mx <= btn_x+btn_w and btn_y <= my <= btn_y+btn_h:
                    running = False
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, -1.4, -2.0)
        glScalef(3.0, 3.0, 3.0)
        glColor3f(0.3, 0.3, 0.3)
        glBegin(GL_QUADS)
        glVertex3f(-10, 0, -10)
        glVertex3f( 10, 0, -10)
        glVertex3f( 10, 0, 10)
        glVertex3f(-10, 0, 10)
        glEnd()
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detect_vehicles(frame)
        vehicle_positions, pedestrian_positions = draw_results(frame, results)
        sorted_positions = sorted(vehicle_positions, key=lambda pos: pos[2])
        for pos in sorted_positions:
            draw_car(pos, scale=0.15)
        for pos in pedestrian_positions:
            draw_person(pos, scale=0.18)
        draw_radar_overlay_pygame()
        # --- UI绘制（OpenGL） ---
        # 关闭按钮
        draw_rounded_rect_opengl(btn_x, btn_y, btn_w, btn_h, btn_radius, (220,60,60), width, height)
        draw_text_opengl(btn_x + (btn_w-btn_font.size("关闭并返回主菜单")[0])//2, btn_y + (btn_h-btn_font.get_height())//2, "关闭并返回主菜单", btn_font, (255,255,255), width, height)
        # 顶部状态条
        draw_rounded_rect_opengl(bar_x, bar_y, bar_w, bar_h, bar_radius, (0,102,204), width, height)
        if radar_enable and latest_distance is not None:
            bar_msg = f"请注意车尾距离!速度:-- 距离:{latest_distance:.1f}m"
        elif radar_enable:
            bar_msg = "后方空旷"
        else:
            bar_msg = "雷达未启用"
        draw_text_opengl(bar_x + (bar_w-bar_font.size(bar_msg)[0])//2, bar_y + (bar_h-bar_font.get_height())//2, bar_msg, bar_font, (255,255,255), width, height)
        # 行人提示框（左侧）
        if pedestrian_positions:  # 有行人时显示
            draw_rounded_rect_opengl(person_x, person_y, person_w, person_h, person_radius, person_color, width, height)
            draw_text_opengl(person_x + (person_w-person_font.size("小心前方行人")[0])//2, person_y + (person_h-person_font.get_height())//2, "小心前方行人", person_font, (255,255,255), width, height)
        pygame.display.flip()
        clock.tick(60)
        if latest_distance is not None:
            caption = f"距离: {latest_distance:.1f}m, 角度: {latest_angle:.1f}°"
        else:
            caption = "无检测目标"
        pygame.display.set_caption(caption)
    if radar_enable and radar:
        radar.stop_data_reception()
        radar.disconnect()
    cap.release()
    # 返回主菜单并保持全屏
    home_screen()

def radar_loop():
    """雷达循环"""
    global radar_running, radar
    
    # 选择雷达模式
    mode_choice = select_radar_mode()
    
    radar = HLKLD2451ShortRange('/dev/ttyS9', 115200)
    if not radar.connect():
        return
    
    # 根据选择配置雷达模式
    if mode_choice == 1:
        print("正在配置普通道路预警模式...")
        if not radar.configure_normal_road_mode():
            print("配置失败，使用默认模式")
            radar.configure_short_range()
    elif mode_choice == 2:
        print("正在配置高速道路预警模式...")
        if not radar.configure_highway_mode():
            print("配置失败，使用默认模式")
            radar.configure_short_range()
    elif mode_choice == 3:
        print("正在配置倒车预警模式...")
        if not radar.configure_reverse_mode():
            print("配置失败，使用默认模式")
            radar.configure_short_range()
    else:
        print("使用默认短距离模式...")
        radar.configure_short_range()
    
    radar.start_data_reception(radar_callback)
    
    pygame.init()
    pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("雷达数据接收中...")
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 800 / 600, 0.1, 100.0)
    cap = cv2.VideoCapture("./car2.mp4")
    clock = pygame.time.Clock()
    running = True

    while running:
        for e in pygame.event.get():
            if e.type == QUIT:
                running = False
            if e.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = e.pos
                if 300 <= mouse_x <= 500 and 200 <= mouse_y <= 250:
                    radar_running = not radar_running
                    if radar_running:
                        radar.start_data_reception(radar_callback)
                        print("雷达开启")
                    else:
                        radar.stop_data_reception()
                        print("雷达关闭")
                if 300 <= mouse_x <= 500 and 300 <= mouse_y <= 350:
                    radar.stop_data_reception()
                    radar.disconnect()
                    cap.release()
                    home_screen()

        # OpenGL 渲染部分
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, -1.4, -2.0)
        glScalef(3.0, 3.0, 3.0)

        # 绘制地面
        glColor3f(0.3, 0.3, 0.3)
        glBegin(GL_QUADS)
        glVertex3f(-10, 0, -10)
        glVertex3f( 10, 0, -10)
        glVertex3f( 10, 0, 10)
        glVertex3f(-10, 0, 10)
        glEnd()

        # 绘制车辆
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detect_vehicles(frame)
        vehicle_positions, pedestrian_positions = draw_results(frame, results)
        
        sorted_positions = sorted(vehicle_positions, key=lambda pos: pos[2])
        for pos in sorted_positions:
            draw_car(pos, scale=0.15)

        draw_radar_overlay_pygame()  # 显示雷达信息

        # Pygame 绘制按钮
        draw_button("切换雷达", 300, 200, 200, 50, (0, 128, 0) if radar_running else (128, 0, 0))
        draw_button("返回主菜单", 300, 300, 200, 50, (128, 0, 0))

        pygame.display.flip()  # 更新整个显示
        clock.tick(60)

        # 在窗口标题上显示最近检测到的距离和角度
        if latest_distance is not None:
            caption = f"距离: {latest_distance:.1f}m, 角度: {latest_angle:.1f}°"
        else:
            caption = "无检测目标"
        pygame.display.set_caption(caption)

    radar.stop_data_reception()
    radar.disconnect()
    cap.release()
    pygame.quit()

# 雷达模式选择
def select_radar_mode():
    """通过终端选择雷达预警模式"""
    print("\n" + "="*50)
    print("请选择雷达预警模式:")
    print("1. 普通道路预警模式 (0-20m)")
    print("2. 高速道路预警模式 (0-50m)") 
    print("3. 倒车预警模式 (0cm-10cm)")
    print("4. 默认短距离模式 (0.1m-10m)")
    print("="*50)
    
    while True:
        try:
            choice = input("请输入选择 (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return int(choice)
            else:
                print("无效选择，请输入 1-4 之间的数字")
        except KeyboardInterrupt:
            print("\n程序被用户中断")
            return 4
        except:
            print("输入错误，请重新输入")

if __name__ == "__main__":
    home_screen()
#pip3 install rknn-toolkit-lite2