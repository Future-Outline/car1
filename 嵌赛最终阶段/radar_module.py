# radar_module.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HLK-LD2451雷达模块 - 集成短距离检测与回调功能
"""
import serial
import struct
import time
import threading
from typing import Optional, List, Dict

class HLKLD2451ShortRange:
    """HLK-LD2451短距离检测雷达类"""
    def __init__(self, port='/dev/ttyS9', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.is_connected = False
        self.data_callback = None
        self.running = False
        self.data_thread = None

    def connect(self) -> bool:
        """连接雷达"""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            )
            self.is_connected = True
            print(f"✓ 雷达连接成功: {self.port}")
            return True
        except Exception as e:
            print(f"✗ 雷达连接失败: {e}")
            return False

    def disconnect(self):
        """断开连接并停止接收"""
        self.stop_data_reception()
        if self.serial and self.serial.is_open:
            self.serial.close()
        self.is_connected = False
        print("雷达连接已断开")

    def configure_short_range(self) -> bool:
        """配置短距离检测参数"""
        try:
            # 1. 使能配置模式
            self.serial.write(b'\xFD\xFC\xFB\xFA\x04\x00\xFF\x00\x01\x00\x04\x03\x02\x01')
            time.sleep(0.2)
            # 2. 设置检测参数 - 10米最大距离
            self.serial.write(b'\xFD\xFC\xFB\xFA\x06\x00\x02\x00\x0A\x02\x01\x01\x04\x03\x02\x01')
            time.sleep(0.2)
            # 3. 设置高灵敏度
            self.serial.write(b'\xFD\xFC\xFB\xFA\x06\x00\x03\x00\x01\x03\x00\x00\x04\x03\x02\x01')
            time.sleep(0.2)
            # 4. 结束配置
            self.serial.write(b'\xFD\xFC\xFB\xFA\x02\x00\xFE\x00\x04\x03\x02\x01')
            time.sleep(0.2)
            print("✓ 短距离检测配置完成: 0.1m-10m, 高灵敏度")
            return True
        except Exception as e:
            print(f"✗ 配置失败: {e}")
            return False

    def configure_normal_road_mode(self) -> bool:
        """配置普通道路预警模式 (0-20m)"""
        try:
            # 1. 使能配置模式
            self.serial.write(b'\xFD\xFC\xFB\xFA\x04\x00\xFF\x00\x01\x00\x04\x03\x02\x01')
            time.sleep(0.2)
            # 2. 设置检测参数 - 20米最大距离，中等灵敏度
            self.serial.write(b'\xFD\xFC\xFB\xFA\x06\x00\x02\x00\x14\x02\x01\x01\x04\x03\x02\x01')
            time.sleep(0.2)
            # 3. 设置中等灵敏度
            self.serial.write(b'\xFD\xFC\xFB\xFA\x06\x00\x03\x00\x01\x03\x00\x00\x04\x03\x02\x01')
            time.sleep(0.2)
            # 4. 结束配置
            self.serial.write(b'\xFD\xFC\xFB\xFA\x02\x00\xFE\x00\x04\x03\x02\x01')
            time.sleep(0.2)
            print("✓ 普通道路预警模式配置完成: 0.1m-20m, 中等灵敏度")
            return True
        except Exception as e:
            print(f"✗ 普通道路模式配置失败: {e}")
            return False

    def configure_highway_mode(self) -> bool:
        """配置高速道路预警模式 (0-50m)"""
        try:
            # 1. 使能配置模式
            self.serial.write(b'\xFD\xFC\xFB\xFA\x04\x00\xFF\x00\x01\x00\x04\x03\x02\x01')
            time.sleep(0.2)
            # 2. 设置检测参数 - 50米最大距离，低灵敏度
            self.serial.write(b'\xFD\xFC\xFB\xFA\x06\x00\x02\x00\x32\x02\x01\x01\x04\x03\x02\x01')
            time.sleep(0.2)
            # 3. 设置低灵敏度
            self.serial.write(b'\xFD\xFC\xFB\xFA\x06\x00\x03\x00\x01\x04\x00\x00\x04\x03\x02\x01')
            time.sleep(0.2)
            # 4. 结束配置
            self.serial.write(b'\xFD\xFC\xFB\xFA\x02\x00\xFE\x00\x04\x03\x02\x01')
            time.sleep(0.2)
            print("✓ 高速道路预警模式配置完成: 0.1m-50m, 低灵敏度")
            return True
        except Exception as e:
            print(f"✗ 高速道路模式配置失败: {e}")
            return False

    def configure_reverse_mode(self) -> bool:
        """配置倒车预警模式 (0cm-10cm)"""
        try:
            # 1. 使能配置模式
            self.serial.write(b'\xFD\xFC\xFB\xFA\x04\x00\xFF\x00\x01\x00\x04\x03\x02\x01')
            time.sleep(0.2)
            # 2. 设置检测参数 - 0.1米最大距离，超高灵敏度
            self.serial.write(b'\xFD\xFC\xFB\xFA\x06\x00\x02\x00\x01\x02\x01\x01\x04\x03\x02\x01')
            time.sleep(0.2)
            # 3. 设置超高灵敏度
            self.serial.write(b'\xFD\xFC\xFB\xFA\x06\x00\x03\x00\x01\x01\x00\x00\x04\x03\x02\x01')
            time.sleep(0.2)
            # 4. 结束配置
            self.serial.write(b'\xFD\xFC\xFB\xFA\x02\x00\xFE\x00\x04\x03\x02\x01')
            time.sleep(0.2)
            print("✓ 倒车预警模式配置完成: 0cm-10cm, 超高灵敏度")
            return True
        except Exception as e:
            print(f"✗ 倒车模式配置失败: {e}")
            return False

    def _parse_target_data(self, data: bytes) -> List[Dict]:
        """解析目标数据，返回列表"""
        targets = []
        if len(data) < 2:
            return targets
        count = data[0]
        offset = 2
        for i in range(count):
            if offset + 5 <= len(data):
                angle_raw, dist, _, _, _ = struct.unpack('<BBBBB', data[offset:offset+5])
                angle = angle_raw - 0x80
                # 距离单位: 米
                targets.append({'angle': angle, 'distance': dist})
                offset += 5
        return targets

    def _reception_loop(self):
        while self.running:
            try:
                if not self.serial.in_waiting:
                    time.sleep(0.01)
                    continue
                header = self.serial.read(4)
                if header != b'\xF4\xF3\xF2\xF1':
                    continue
                length = struct.unpack('<H', self.serial.read(2))[0]
                data = self.serial.read(length)
                self.serial.read(4)  # tail
                targets = self._parse_target_data(data)
                if self.data_callback:
                    self.data_callback(targets)
            except Exception:
                time.sleep(0.01)

    def start_data_reception(self, callback):
        """开始数据接收，回调targets"""
        if not self.is_connected:
            raise RuntimeError("雷达未连接")
        self.data_callback = callback
        self.running = True
        self.data_thread = threading.Thread(target=self._reception_loop, daemon=True)
        self.data_thread.start()
        print("✓ 已开始雷达数据接收")

    def stop_data_reception(self):
        self.running = False
        if self.data_thread:
            self.data_thread.join(timeout=1)
        print("✓ 停止雷达数据接收")