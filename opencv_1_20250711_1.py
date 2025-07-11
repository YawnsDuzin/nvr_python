#!/usr/bin/env python3
"""
Raspberry Pi NVR (Network Video Recorder) System
개발언어: Python
대상 HW: Raspberry Pi
라이브러리: PyQt5, OpenCV
"""

# ===========================================
# HISTORY
# - 2025.06.24 Err_FrameDrop
#   ㄴ 시간지나면 RTSP Streaming Frame Drop Error
# - 2025.06.24 Err_close_threading
#   ㄴ 종료 시, threading 오류 표시되는 부분

# - opencv_1_20250626_1.py
#   ㄴ "min_free_space_gb : 최소 여유 공간 (GB), min_free_space_percent : 최소 여유 공간 (%), auto_delete : 자동 삭제 활성화, delete_batch_size : 한 번에 삭제할 파일 수", 

# - opencv_1_20250626_3.py
#   ㄴ 

# - opencv_1_20250627_1.py
#   ㄴ  CCTV 화면에 글씨 표시 테스트
#     - VideoWidget() class에 추가

# - opencv_1_20250627_2.py
#   ㄴ ptz제어 관련 - 화면 오버레이

# - opencv_1_20250630_1.py
#   ㄴ usb백업 다이얼로그 추가작업

# - opencv_1_20250630_1.py
#   ㄴ usb백업 다이얼로그 추가작업
#   ㄴ NVRLogger 클래스 추가

# - opencv_1_20250701_1.py
#   ㄴ 백업기능 파일 선택 및 백업용량 확인가능하도록

# - opencv_1_20250702_1.py
#   ㄴ 

# - opencv_1_20250702_2.py
#   ㄴ 백업 - CCTV데이터백업 : 화면 사이즈 조절관련

# - opencv_1_20250703_1.py
#   ㄴ 백업 - CCTV데이터백업 : 화면 사이즈 조절관련
# ===========================================


import sys                                      # 시스템 관련 기능 (프로그램 종료, 명령줄 인자 등)
import os                                       # 운영체제 관련 기능 (파일/디렉토리 조작, 환경변수 등)
import json                                     # JSON 파일 읽기/쓰기 (설정 파일 관리용)

import time                                     # 시간 관련 기능 (타임스탬프, 지연 등)
import threading                                # 멀티스레딩 지원 (동시 작업 처리)
import numpy as np
import psutil                                   # 시스템 모니터링 (CPU, 메모리, 네트워크 사용량)
import subprocess                               # 외부 프로그램 실행 (라즈베리파이 온도 측정 명령)
from datetime import datetime                   # 날짜/시간 처리 (파일명 생성 등)
from pathlib import Path                        # 파일 경로 처리 (디렉토리 생성 등)
from queue import Queue                         # 스레드 간 데이터 전달용 큐 (비동기 작업 처리)

import cv2                                      # OpenCV - 비디오 처리 (RTSP 스트림 캡처, 녹화)

from PyQt5.QtWidgets import *                   # GUI 위젯들 (버튼, 레이블, 레이아웃 등)
from PyQt5.QtCore import *                      # Qt 코어 기능 (시그널/슬롯, 스레드 등)
from PyQt5.QtGui import *                       # GUI 관련 기능 (이미지, 색상, 그리기 등)


# 2025.06.23 duzn
import shutil  # 파일 및 디렉토리 작업을 위한 고수준 파일 연산 (파일 복사, 이동, 삭제 등)을 위해 사용됩니다.
# - 파일 이동: shutil.move() - 녹화 완료된 파일을 임시 디렉토리에서 최종 저장 디렉토리로 이동할 때 사용
# - 파일 복사: shutil.copy2() - 파일 메타데이터(생성일, 수정일 등)를 보존하면서 파일을 복사할 때 사용
# - 디렉토리 작업: 디렉토리 생성, 삭제, 복사 등에 사용

# 2025.06.24 Err_close_threading
import atexit  # 프로그램 종료 시 정리 작업용

# 2025.06.25_2
from typing import Dict, List, Tuple, Optional
import logging
import logging.handlers

# opencv_1_20250703_1.py
import queue
from threading import Thread

# opencv_1_20250702_2.py
# from window_utils import show_window_size, get_window_info, log_window_size

import platform                                 # OS 종류 감지용
IS_WINDOWS = platform.system() == 'Windows'     # Windows OS 여부
IS_LINUX = platform.system() == 'Linux'         # Linux OS 여부

# Windows에서 PyQt5 플러그인 경로 수동 설정 (한글 경로 포함)
# - 경로에 사용자가 PC마다 경로가 달라질 것으로 보임
if IS_WINDOWS:
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'C:/Users/박두진/AppData/Local/Programs/Python/Python313/Lib/site-packages/PyQt5/Qt5/plugins'

# 설정 파일 경로
CONFIG_FILE = "nvr_config.json"                 # 설정 파일 이름

# 기본 설정값
DEFAULT_CONFIG = {
    "camera": {
        "rtsp_url": "",                         # RTSP 스트림 주소 (예: rtsp://192.168.1.100:554/stream)
        "resolution": "1920x1080",              # 녹화 해상도
        "fps": 30,                              # FPS (초당 프레임 수)
        "save_path": r"./recordings",             # 녹화 파일 저장 경로
        "autostart_streaming": True,
        "autostart_recording": False,
        "message1": "테스트 중입니다. 끄지 마세요!!"
    },
    "storage": {
        "min_free_space_gb": 10,
        "min_free_space_percent": 10,
        "auto_delete": True,
        "delete_batch_size": 10
        },
    "keys": {},                                 # 단축키 설정 (미구현)
    "ptz": {},                                  # Pan-Tilt-Zoom 카메라 제어 설정 (미구현)
    "backup": {
            "src_path": r"/media/pi/NVR_MAIN/camera_1/completed",
            "dest_path": r"/media/pi/NVR_BACKUP/camera_1/completed",
            "ext" : "mp4",
            "delete_after_backup" : True,
            "verification": True,
            "sync_interval" : 5
        }
}



# # 2025.06.25_2
# # 로깅 설정
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(r'nvr_log.log', encoding='utf-8'),  # UTF-8 인코딩 명시
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)


class NVRLogger:
    """
    NVR(Network Video Recorder) 시스템을 위한 로깅 클래스입니다.
    로그는 파일과 콘솔에 동시에 출력되며, 로그 파일은 일별로 자동 생성됩니다.
    """
    def __init__(self, log_dir='./logs'):
        """
        NVRLogger 인스턴스를 초기화합니다.
        
        Args:
            log_dir (str): 로그 파일이 저장될 디렉토리 경로 (기본값: './logs')
        """
        self.log_dir = log_dir
        # 로그 디렉토리가 없는 경우 생성
        os.makedirs(log_dir, exist_ok=True)
        self.setup_logger()
    
    def setup_logger(self):
        """로거 설정을 초기화합니다. 파일 핸들러와 콘솔 핸들러를 설정합니다."""
        # 기본 로그 파일명 설정
        log_filename = "nvr.log"
        log_path = os.path.join(self.log_dir, log_filename)
        
        # 일별 로테이팅 파일 핸들러 설정
        # - when='midnight': 매일 자정에 새 파일 생성
        # - interval=1: 1일 간격
        # - backupCount=30: 최대 30개까지 백업 파일 유지 (30일치)
        # - encoding: UTF-8 인코딩 사용
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_path,
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )

        # 파일이 10MB를 초과하면 중간에도 새 파일 생성
        file_handler.maxBytes = 10*1024*1024  # 10MB
        
        # 로그 파일명에 날짜 형식 설정 (예: nvr.log.2025-07-11)
        file_handler.suffix = '%Y-%m-%d'
        
        # 콘솔 출력을 위한 핸들러
        console_handler = logging.StreamHandler()
        
        # 로그 포맷 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 핸들러에 포매터 설정
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 루트 로거 설정
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # 기존 핸들러 제거 (중복 로깅 방지)
        root_logger.handlers = []
        
        # 핸들러 추가
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    def get_logger(self, name):
        """
        지정된 이름의 로거 인스턴스를 반환합니다.
        
        Args:
            name (str): 로거 이름 (일반적으로 __name__ 사용)
            
        Returns:
            logging.Logger: 설정된 로거 인스턴스
        """
        return logging.getLogger(name)


# 전역 로거 인스턴스 생성
nvr_logger = NVRLogger()
# 'NVR_MAIN' 이름의 로거 인스턴스 생성
logger = nvr_logger.get_logger('NVR_MAIN')


class VideoThread(QThread):
    """비디오 스트리밍을 위한 쓰레드
    
    QThread를 상속받아 메인 GUI와 별도로 비디오 처리를 수행합니다.
    이렇게 하면 비디오 처리가 GUI를 멈추지 않습니다.
    """

    # PyQt시그널 정의 - 스레드에서 메인 윈도우로 데이터 전달용
    frame_ready = pyqtSignal(object)            # 새 프레임이 준비되면 발생 (GUI에 CCTV 화면 표시)
    fps_update = pyqtSignal(float)              # FPS가 변경되면 발생 (GUI에 FPS 표시)
    status_update = pyqtSignal(str)             # 상태 변경 시 발생 (GUI에 비디오 상태 표시)
    error_occurred = pyqtSignal(int, str)       # 오류 발생 시 발생
    
    storage_warning = pyqtSignal(str)           # 저장소 경고 시그널 추가

    def __init__(self):
        super().__init__()                      # QThread 초기화
        
        self.reset_thread_state()

        # 2025.06.24 Err_close_threading
        # 종료 시 자동 정리를 위한 등록
        atexit.register(self.cleanup_resources)

    def reset_thread_state(self):
        """스레드 상태 초기화"""

        print("VideoThread 상태 초기화")
        
        self.camera_id = 1
        self.rtsp_url = ""
        
        # 상태 플래그 초기화
        self.is_running = False
        self.is_recording = False
        self.force_stop = False         # 중요: 반드시 False로 초기화
        
        # 성능 측정 변수 초기화
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # 비디오 설정 초기화
        self.resolution = (1920, 1080)
        self.target_fps = 30
        self.save_path = "./recordings"
        
        # OpenCV 객체 초기화
        self.cap = None
        self.out = None
        
        # 연결 관리 변수 초기화
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 60 * 2                     # 재연결 횟수 (1회 30초) = 1시간 = (60 * 2)
        self.last_frame_time = time.time()
        self.frame_timeout = 10.0
        self.connection_stable_time = 0
        
        # 녹화 관련 초기화
        self.record_start_time = 0              # 녹화 시작 시간
        self.record_term = 120                  # 녹화파일 저장 단위 (300(초) = 5분)
        self.temp_filepath = None
        self.final_filepath = None
        self.usb_ismount = True
        self.usb_status = True
        
        # 저장소 관련 초기화
        self.storage_manager = None
        self.storage_check_interval = 30
        self.last_storage_check = 0
        
        # 스레드 동기화용 락
        self.recording_lock = threading.Lock()

    # 2025.06.24 Err_close_threading
    def cleanup_resources(self):
        """리소스 정리 - 안전한 종료를 위한 메서드"""
        try:
            if hasattr(self, 'is_running'):
                self.is_running = False
            if hasattr(self, 'force_stop'):
                self.force_stop = True
            
            # OpenCV 객체들 정리 및 파일 이동 처리
            if hasattr(self, 'out') and self.out is not None:
                try:
                    # 1. VideoWriter 해제
                    self.out.release()
                    self.out = None

                    # 2. 파일 이동 처리 (finalize_video_file과 유사한 로직)
                    if hasattr(self, 'temp_filepath') and hasattr(self, 'final_filepath'):
                        if os.path.exists(self.temp_filepath):
                            try:
                                # 대상 디렉토리 생성
                                os.makedirs(os.path.dirname(self.final_filepath), exist_ok=True)
                                # 파일 이동
                                shutil.move(self.temp_filepath, self.final_filepath)
                                print(f"강제 종료 시 파일 이동 완료: {self.final_filepath}")
                            except Exception as e:
                                print(f"강제 종료 시 파일 이동 실패: {e}")
                except:
                    pass
                    
            # VideoCapture 정리
            if hasattr(self, 'cap') and self.cap is not None:
                try:
                    self.cap.release()
                    self.cap = None
                except:
                    pass
        except:
            pass


    def set_config(self, config):
        """설정 적용
    
        JSON 설정 파일에서 읽은 설정을 비디오 스레드에 적용합니다.
        """

        self.rtsp_url = config["camera"]["rtsp_url"]

        # "1920x1080" 형식의 문자열을 튜플로 변환
        res = config["camera"]["resolution"].split('x')
        self.resolution = (int(res[0]), int(res[1]))

        self.target_fps = config["camera"]["fps"]
        self.save_path = config["camera"]["save_path"]

        # StorageManager 초기화 추가
        storage_config = config.get("storage", DEFAULT_CONFIG["storage"])
        self.storage_manager = StorageManager(self.save_path, storage_config)
        
    # 2025.06.24 Err_FrameDrop
    def create_video_capture(self):
        """개선된 VideoCapture 생성"""

        # 2025.06.24 Err_close_threading
        if self.force_stop:  # 강제 종료 상태면 생성하지 않음
            return None

        print(f"VideoCapture 생성 시도: {self.rtsp_url}")

        # 이전 연결 완전 정리
        if hasattr(self, 'cap') and self.cap:
            try:
                self.cap.release()
                time.sleep(0.2)
            except:
                pass

        cap = cv2.VideoCapture(self.rtsp_url)
        
        if cap is None or not cap.isOpened():
            print("VideoCapture 생성 실패")
            return None

        # 설정 적용
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # 연결 테스트
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print("VideoCapture 생성 성공")
                return cap
            else:
                print("VideoCapture 테스트 실패")
                cap.release()
                return None
                
        except Exception as e:
            print(f"VideoCapture 설정 실패: {e}")
            if cap:
                cap.release()
            return None
            
        
    # 2025.06.24 Err_FrameDrop
    def reconnect_camera(self):
        """카메라 재연결"""

        # 2025.06.24 Err_close_threading
        if self.force_stop:
            return False

        print(f"reconnect_camera ({self.reconnect_attempts + 1}/{self.max_reconnect_attempts})")
        # opencv_1_20250703_1.py
        # 현재 녹화 중인 파일 저장
        with self.recording_lock:
            if self.is_recording and self.out:
                self.usb_ismount = False
                self.usb_status = False
                self.usb_ismount = self.storage_manager.is_mount(self.save_path)
                if self.usb_ismount:
                    self.usb_status = self.check_storage_space()

                # usb가 정상 연결되어있고, 용량 및 상태가 정상이면..
                if self.usb_ismount and self.usb_status:
                        print(f"[REC] reconnect_camera ({self.reconnect_attempts + 1}/{self.max_reconnect_attempts})")
                        self.finalize_video_file()


        self.status_update.emit(f"재연결 시도 중... ({self.reconnect_attempts + 1}/{self.max_reconnect_attempts})")
        
        # 기존 연결 해제
        if self.cap:
            self.cap.release()
            self.cap = None
            
        # 잠시 대기 (점진적 백오프)
        wait_time = min(5 * (self.reconnect_attempts + 1), 30)  # 최대 30초
        
        # 2025.06.24 Err_close_threading
        # 강제 종료 확인하면서 대기
        for _ in range(int(wait_time * 10)):  # 0.1초씩 나누어 대기
            if self.force_stop:
                return False
            time.sleep(0.1)
        
        # 새 연결 시도
        self.cap = self.create_video_capture()
        
        if self.cap and self.cap.isOpened() and not self.force_stop:
            # 연결 테스트
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.reconnect_attempts = 0
                self.last_frame_time = time.time()
                self.connection_stable_time = time.time()
                self.status_update.emit("재연결 성공")
                return True
            else:
                if self.cap:
                    self.cap.release()
                    self.cap = None        

        # opencv_1_20250703_1.py
        if self.reconnect_attempts == 2:       # 1분동안 연결 안되는 경우, 일단 현재 파일 완료 처리
            with self.recording_lock:
                if self.is_recording and not self.force_stop and self.usb_ismount and self.usb_status:
                    if self.out:
                        self.finalize_video_file()


        self.reconnect_attempts += 1
        return False
        
    # 2025.06.24 Err_FrameDrop
    def is_connection_stable(self):
        """연결 안정성 확인"""

        # 2025.06.24 Err_close_threading
        if self.force_stop:
            return False

        current_time = time.time()
        
        # 프레임 타임아웃 확인
        # - 마지막으로 프레임을 수신한 시간(self.last_frame_time)과 현재 시간의 차이를 계산합니다.
        # - 이 차이가 frame_timeout(초)보다 크면, 프레임 수신이 지연되고 있다고 판단하여 False를 반환합니다.
        # - 즉, 일정 시간 동안 새 프레임이 도착하지 않으면 연결이 끊어진 것으로 간주합니다.
        if current_time - self.last_frame_time > self.frame_timeout:
            return False
            
        return True


    def run(self):
        """비디오 캡처 실행
    
        QThread의 run() 메서드를 오버라이드합니다.
        start() 호출 시 별도 스레드에서 이 메서드가 실행됩니다.
        """
        
        try:
            print(f"VideoThread 시작 - RTSP URL: {self.rtsp_url}")

            # 상태 초기화 확인
            if self.force_stop:
                print("스레드 시작 시 force_stop이 True입니다. 초기화합니다.")
                self.force_stop = False

            # RTSP URL 확인
            if not self.rtsp_url or self.force_stop:
                self.status_update.emit("RTSP URL이 설정되지 않음")
                return
                
            # 기존 연결이 있다면 정리
            self.cleanup_opencv_resources()

            # 초기 연결
            self.cap = self.create_video_capture()
            
            # RTSP 연결 확인
            if not self.cap or not self.cap.isOpened():
                self.status_update.emit("카메라 연결 실패")
                return
            
            # 초기 연결 테스트
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                self.status_update.emit("카메라에서 프레임을 읽을 수 없음")
                self.cleanup_opencv_resources()
                return
                
            # 스트리밍 시작
            self.is_running = True
            self.status_update.emit("스트리밍 중")
            self.start_time = time.time()
            self.last_frame_time = time.time()
            self.connection_stable_time = time.time()
            self.frame_count = 0

            print("비디오 스트리밍 루프 시작")

            # 실제 카메라 속성 가져오기 (요청한 값과 다를 수 있음)
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            if actual_fps == 0:
                actual_fps = self.fps

            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # FPS 계산용 변수
            fps_counter = 0
            last_fps_time = time.time()

            usb_chk_before = True            # 이전 usb 상태 저장 플래그

            # 메인 루프 - 계속해서 프레임을 읽고 처리
            while self.is_running and not self.force_stop:
                current_time = time.time()

                # 연결 안정성 확인
                if not self.is_connection_stable():
                    self.status_update.emit("연결 불안정 감지")
                    
                    if self.reconnect_attempts < self.max_reconnect_attempts and not self.force_stop:
                        if not self.reconnect_camera():
                            print(f"1 - reconnect_camera ({self.reconnect_attempts + 1}/{self.max_reconnect_attempts})")
                            continue
                    else:
                        self.status_update.emit("최대 재연결 시도 횟수 초과")
                        break

                try:
                    # 2025.06.24 Err_close_threading
                    if self.force_stop:  # 각 단계마다 강제 종료 확인
                        break

                    ret, frame = self.cap.read()        # 프레임 읽기

                    if ret and frame is not None and not self.force_stop:     # 프레임을 성공적으로 읽었으면
                        self.last_frame_time = current_time

                        self.frame_ready.emit(frame)    # GUI로 프레임 전송
                        
                        # FPS 계산 (1초마다)
                        fps_counter += 1
                        current_time = time.time()
                        if current_time - last_fps_time > 1.0:
                            actual_fps = fps_counter / (current_time - last_fps_time)
                            self.fps_update.emit(actual_fps)
                            fps_counter = 0

                            last_fps_time = current_time

                        # 녹화 처리 (5분마다 새 파일 생성)
                        # ======================================================================
                        # 스레드 안전성을 위한 락
                        # - lock 획득 (이 안의 코드는 한 번에 하나의 스레드만 실행 가능)
                        # - with 블록을 벗어나면 자동으로 lock 해제
                        with self.recording_lock:
                            # if self.is_recording and not self.force_stop:
                            if self.is_recording and not self.force_stop:
                                # 스트리밍이나 녹화에 영향이 없을까?
                                # 주기적으로 저장 공간 체크 (녹화 중일 때만)
                                self.usb_ismount = False
                                self.usb_status = False
                                
                                # opencv_1_20250702_1.py
                                # 저장 경로USB 연결여부 확인
                                self.usb_ismount = self.storage_manager.is_mount(self.save_path)
                                if (not self.usb_ismount):
                                    self.storage_warning.emit("⚠️ 저장 USB가 연결 되어있지 않습니다!")
                                else:
                                    self.usb_status = self.check_storage_space()

                                # usb가 정상 연결되어있고, 용량 및 상태가 정상이면..
                                if self.usb_ismount and self.usb_status:

                                    # USB 상태 : False => True 로 오면 현재 녹화중인 파일 emegency 폴더로 이동
                                    # 깨진 파일 임!!
                                    if self.out and usb_chk_before == False:
                                        self.finalize_video_file_emergency()

                                    usb_chk_before = True       # usb 상태플래그 = True

                                    # 5분(300초)마다 새 파일 생성
                                    # opencv_1_20250703_1.py
                                    if self.out is None or (current_time - self.record_start_time) > self.record_term:
                                        if self.out:
                                            self.finalize_video_file()

                                        # 새 파일 생성
                                        self.out = self.create_video_writer(frame_width, frame_height, actual_fps)
                                        self.record_start_time = current_time
                                        
                                    # if self.out and not self.force_stop:
                                    if self.out and not self.force_stop and self.usb_ismount and self.usb_status:
                                        try:
                                            self.out.write(frame)       # 프레임 저장
                                        except Exception as e:
                                            print(f"프레임 처리 오류: {e}")
                                else:
                                    usb_chk_before = False      # usb 상태플래그 = False

                                    self.storage_warning.emit("⚠️ 저장 USB가 잘못되었습니다!")
                                    
                            # ======================================================================
                    
                    else:
                        # 2025.06.24 Err_close_threading
                        if self.force_stop:
                            break

                        # 프레임 읽기 실패
                        self.status_update.emit("프레임 읽기 실패")
                        
                        # 연속 실패 시 재연결 시도
                        if (self.reconnect_attempts < self.max_reconnect_attempts):
                            print(f"2 - reconnect_camera ({self.reconnect_attempts + 1}/{self.max_reconnect_attempts})")
                            if not self.reconnect_camera():
                                continue
                        else:
                            break
                            
                except Exception as e:
                    
                    if self.force_stop:
                        break

                    print(f"프레임 처리 오류: {e}")
                    # OpenCV 오류 시에도 재연결 시도
                    if (self.reconnect_attempts < self.max_reconnect_attempts):
                        print(f"3 - reconnect_camera ({self.reconnect_attempts + 1}/{self.max_reconnect_attempts})")
                        if not self.reconnect_camera():
                            continue
                    else:
                        break
                        
                if not self.force_stop:
                    time.sleep(0.001)
                
        except Exception as e:
            if not self.force_stop:  # 강제 종료가 아닌 경우만 로그 출력
                print(f"비디오 스레드 오류: {e}")
                self.status_update.emit(f"스트리밍 오류: {e}")
        finally:
            # 안전한 리소스 정리
            self.cleanup_opencv_resources()
            if not self.force_stop:
                self.status_update.emit("스트리밍 종료")
            print("VideoThread 종료 완료")
        

    def cleanup_opencv_resources(self):
        """OpenCV 리소스 완전 정리"""
        try:
            print("OpenCV 리소스 정리 시작")

            # VideoWriter 정리
            if self.out is not None:
                try:
                    self.out.release()
                    time.sleep(0.1)  # 리소스 해제 대기
                except:
                    pass
                finally:
                    self.out = None
                    
            # VideoCapture 정리  
            if self.cap is not None:
                try:
                    self.cap.release()
                    time.sleep(0.1)  # 리소스 해제 대기
                except:
                    pass
                finally:
                    self.cap = None
                    
            print("OpenCV 리소스 정리 완료")
            
        except Exception as e:
            print(f"OpenCV 리소스 정리 오류: {e}")


    def stop(self):
        """스트리밍 종료"""
        print("VideoThread 종료 요청")

        self.force_stop = True  # 강제 종료 플래그 설정
        self.is_running = False

        # 녹화 중이면 파일 마무리
        if self.is_recording:
            self.stop_recording()
            
        # 스레드 종료 대기
        if self.isRunning():
            print("스레드 종료 대기 중...")
            self.wait(3000)  # 최대 5초 대기
            if self.isRunning():
                print("강제 종료")
                self.terminate()
                self.wait(1000)

        # 추가 정리 작업
        try:
            self.cleanup_resources()
        except:
            pass  # 종료 시점의 예외는 무시


    def start_recording(self):
        """녹화 시작
    
        Returns:
            bool: 녹화 시작 성공 여부
        """
        print(f"[DEBUG] is_running: {self.is_running}, force_stop: {self.force_stop}")

        if not self.is_running or self.force_stop:
            return False
            
        # 저장 디렉토리가 없으면 생성
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        
        if not self.save_path:
            # error_occurred 시그널이 정의되어 있지 않아 오류 발생 가능
            self.error_occurred.emit(self.camera_id, "저장 경로가 설정되지 않았습니다.")
            return False

        # 스레드 안전하게 녹화 시작
        with self.recording_lock:  # 스레드 안전을 위한 락 획득
            self.is_recording = True  # 녹화 상태를 True로 설정하여 녹화 시작
            self.record_start_time = time.time()

        return True
        
    def stop_recording(self):
        """녹화 종료"""

        with self.recording_lock:       # 스레드 안전을 위한 락 획득
            self.is_recording = False

            if self.out:
                self.finalize_video_file()
    

    def create_video_writer(self, frame_width, frame_height, fps):
        """비디오 작성기 생성
    
        Args:
            frame_width: 프레임 너비
            frame_height: 프레임 높이
            fps: 초당 프레임 수
            
        Returns:
            cv2.VideoWriter: 비디오 작성기 객체
        """
        
        # 2025.06.24 Err_close_threading
        if self.force_stop:
            return None


        # 체크 하는 시간 동안 녹화가 안되지 않을까?? 별로 영향 없을까??
        # 저장 공간 확인
        if not self.check_storage_space():
            self.error_occurred.emit(self.camera_id, "저장 공간 부족으로 녹화할 수 없습니다.")
            return None


        # 타임스탬프로 파일명 생성 (예: cam1_recording_20240115_143052.mp4)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 카메라별 디렉토리 생성 
        #  ./recordings/camera_1/recording/
        #  ./recordings/camera_1/completed/
        camera_path = os.path.join(self.save_path, f"camera_{self.camera_id}")
        Path(camera_path).mkdir(parents=True, exist_ok=True)
        
        # 임시 파일명과 최종 파일명
        temp_path = os.path.join(self.save_path, f"camera_{self.camera_id}", "recording")
        emergency_path = os.path.join(self.save_path, f"camera_{self.camera_id}", "emergency")
        final_path = os.path.join(self.save_path, f"camera_{self.camera_id}", "completed")
        Path(temp_path).mkdir(parents=True, exist_ok=True)
        Path(final_path).mkdir(parents=True, exist_ok=True)

        filename = f"cam{self.camera_id}_recording_{timestamp}.mp4"
        self.temp_filepath = os.path.join(temp_path, filename)
        self.emergency_filepath = os.path.join(emergency_path, filename)
        self.final_filepath = os.path.join(final_path, filename)
        
        # 비디오 코덱 설정 (mp4v는 범용적이지만 H.264가 더 효율적)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fourcc = cv2.VideoWriter_fourcc(*'H264')      # 오류남

        if fps > 65535:
            print(f"[create_video_writer] fps가 너무 높습니다: {fps}. 65535로 제한합니다.")
            fps = 65535
            
        # VideoWriter 객체 생성
        return cv2.VideoWriter(self.temp_filepath, fourcc, fps, (frame_width, frame_height))

    # opencv_1_20250703_1.py
    def finalize_video_file_emergency(self):
        """녹화 중, usb 이상 발견 시, recording 완료 처리 및 디렉토리에서 파일이름 변경"""

        if self.out:
            # 1. VideoWriter 객체 해제 (파일 닫기)
            self.out.release()
            self.out = None
            
            # 2. 파일 시스템이 완전히 해제할 시간을 줌
            time.sleep(0.5)  # 500ms 대기

            # 3. 파일 존재 확인
            if not os.path.exists(self.temp_filepath):
                print(f"임시 파일이 없습니다: {self.temp_filepath}")
                return

            # 4. 파일 크기 확인
            try:
                file_size = os.path.getsize(self.temp_filepath)
                if file_size == 0:
                    os.remove(self.temp_filepath)
                    return
            except Exception as e:
                print(f"파일 크기 확인 실패: {e}")
                return

            # 5. 대상 디렉토리 확인 및 생성
            final_dir = os.path.dirname(self.emergency_filepath)
            if not os.path.exists(final_dir):
                Path(final_dir).mkdir(parents=True, exist_ok=True)

            # 6. 재시도 로직으로 파일 이동
            max_retries = 5
            retry_delay = 1  # 초
            
            for attempt in range(max_retries):

                try:
                    # Windows에서 shutil.move가 실패하면 복사 후 삭제 시도
                    if os.name == 'nt':  # Windows
                        # 먼저 복사
                        shutil.copy2(self.temp_filepath, self.emergency_filepath)
                        
                        # 복사 성공 확인
                        if os.path.exists(self.emergency_filepath):
                            # 원본 삭제 시도
                            try:
                                os.remove(self.temp_filepath)
                            except PermissionError:
                                # 삭제 실패 시 나중에 정리
                                print(f"원본 파일 삭제 실패 (사용 중): {self.temp_filepath}")
                                self.mark_for_deletion(self.temp_filepath)
                    else:  # Linux/Mac
                        shutil.move(self.temp_filepath, self.emergency_filepath)
                    
                    status_msg = ("녹화 완료" if not self.force_stop 
                                else "강제 종료로 녹화 중단 (파일 정리 완료)")
                    print(f"{status_msg}: {self.emergency_filepath}")
                    self.status_update.emit(f"{status_msg}: {os.path.basename(self.emergency_filepath)}")

                    break  # 성공
                    
                except (PermissionError, OSError) as e:
                    if attempt < max_retries - 1:
                        print(f"파일 이동 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                        time.sleep(retry_delay)
                    else:
                        print(f"파일 이동 최종 실패: {e}")
                        self.status_update.emit("녹화 파일 이동 실패")



    def finalize_video_file(self):
        """녹화 완료 시 recording 디렉토리에서 completed 디렉토리로 파일 이동"""

        if self.out:
            # 1. VideoWriter 객체 해제 (파일 닫기)
            self.out.release()
            self.out = None
            
            # 2. 파일 시스템이 완전히 해제할 시간을 줌
            time.sleep(0.5)  # 500ms 대기

            # 3. 파일 존재 확인
            if not os.path.exists(self.temp_filepath):
                print(f"임시 파일이 없습니다: {self.temp_filepath}")
                return

            # 4. 파일 크기 확인
            try:
                file_size = os.path.getsize(self.temp_filepath)
                if file_size == 0:
                    os.remove(self.temp_filepath)
                    return
            except Exception as e:
                print(f"파일 크기 확인 실패: {e}")
                return

            # 5. 대상 디렉토리 확인 및 생성
            final_dir = os.path.dirname(self.final_filepath)
            if not os.path.exists(final_dir):
                Path(final_dir).mkdir(parents=True, exist_ok=True)

            # 6. 재시도 로직으로 파일 이동
            max_retries = 5
            retry_delay = 1  # 초
            
            for attempt in range(max_retries):

                try:
                    # Windows에서 shutil.move가 실패하면 복사 후 삭제 시도
                    if os.name == 'nt':  # Windows
                        # 먼저 복사
                        shutil.copy2(self.temp_filepath, self.final_filepath)
                        
                        # 복사 성공 확인
                        if os.path.exists(self.final_filepath):
                            # 원본 삭제 시도
                            try:
                                os.remove(self.temp_filepath)
                            except PermissionError:
                                # 삭제 실패 시 나중에 정리
                                print(f"원본 파일 삭제 실패 (사용 중): {self.temp_filepath}")
                                self.mark_for_deletion(self.temp_filepath)
                    else:  # Linux/Mac
                        shutil.move(self.temp_filepath, self.final_filepath)
                    
                    status_msg = ("녹화 완료" if not self.force_stop 
                                else "강제 종료로 녹화 중단 (파일 정리 완료)")
                    print(f"{status_msg}: {self.final_filepath}")
                    self.status_update.emit(f"{status_msg}: {os.path.basename(self.final_filepath)}")

                    break  # 성공
                    
                except (PermissionError, OSError) as e:
                    if attempt < max_retries - 1:
                        print(f"파일 이동 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                        time.sleep(retry_delay)
                    else:
                        print(f"파일 이동 최종 실패: {e}")
                        self.status_update.emit("녹화 파일 이동 실패")

    def check_storage_space(self):
        """저장 공간 확인 및 관리"""
        if not self.storage_manager:
            return True
            
        current_time = time.time()

        # 주기적 체크
        if current_time - self.last_storage_check < self.storage_check_interval:
            return True
            
        self.last_storage_check = current_time
        
        try:
            # 저장 공간 상태 확인
            total_gb, used_gb, free_gb = self.storage_manager.get_disk_usage()
            free_percent = self.storage_manager.get_free_space_percent()
            
            # 상태 정보 전송
            storage_info = f"저장소: {free_gb:.1f}GB 여유 ({free_percent:.1f}%)"
            self.storage_warning.emit(storage_info)
            
            # 공간 부족 시 처리
            if self.storage_manager.is_storage_low():
                self.storage_warning.emit("⚠️ 저장 공간 부족 - 오래된 파일 삭제 중...")
                
                # 5분 분량의 녹화 공간 확보 (대략적인 계산)
                # 1920x1080, 30fps, H264 기준 약 50MB/분
                # required_mb = 50 * 5  # 250MB
                required_mb = 10 * 5  # 50MB
                
                if self.storage_manager.ensure_free_space(required_mb):
                    self.storage_warning.emit("✓ 저장 공간 확보 완료")
                    return True
                else:
                    self.storage_warning.emit("❌ 저장 공간 확보 실패")
                    return False
                    
            return True
            
        except Exception as e:
            print(f"저장소 체크 오류: {e}")
            return True

    

class SystemMonitorThread(QThread):
    """시스템 모니터링 쓰레드
    
    CPU, 메모리, 네트워크, 디스크 사용량을 모니터링합니다.
    """
    
    update_signal = pyqtSignal(dict)            # 시스템 정보 업데이트 시그널
    
    def __init__(self):
        super().__init__()
        self.is_running = True
        self.force_stop = False  # 강제 종료 플래그 추가

        # 종료 시 자동 정리를 위한 등록
        atexit.register(self.cleanup_resources)
        
    def cleanup_resources(self):
        """리소스 정리"""
        try:
            if hasattr(self, 'is_running'):
                self.is_running = False
            if hasattr(self, 'force_stop'):
                self.force_stop = True
        except:
            pass

    def get_cpu_temperature(self):
        """CPU 온도 가져오기 (라즈베리파이)
    
        vcgencmd는 라즈베리파이 전용 명령어입니다.
        
        Returns:
            float: 섭씨 온도
        """
        
        if self.force_stop:
            return 0.0

        try:
            # 라즈베리파이 전용 명령 실행
            result = subprocess.run(
                ['vcgencmd', 'measure_temp'],
                capture_output=True,                # 출력 캡처
                text=True                           # 텍스트 모드
            )
            
            # 출력 형식: "temp=42.8'C"
            temp_str = result.stdout.strip()
            temp = float(temp_str.split('=')[1].split("'")[0])

            return temp
        except:
            return 0.0          # 오류 시 0 반환
            
    def run(self):
        """시스템 정보 수집
    
        1초마다 시스템 정보를 수집하여 메인 윈도우로 전송합니다.
        """

        while self.is_running and not self.force_stop:
            try:

                if self.force_stop:
                    break

                # psutil을 사용한 시스템 정보 수집

                # CPU 사용률 (%)
                cpu_percent = psutil.cpu_percent(interval=0.1)
                
                # 메모리 사용률 (%)
                mem = psutil.virtual_memory()
                mem_percent = mem.percent
                
                # CPU 온도 (라즈베리파이)
                cpu_temp = self.get_cpu_temperature()
                
                if self.force_stop:
                    break

                # 디스크 I/O
                disk_io = psutil.disk_io_counters()
                disk_read = disk_io.read_bytes          # 읽은 바이트
                disk_write = disk_io.write_bytes        # 쓴 바이트
                
                # 네트워크 I/O
                net_io = psutil.net_io_counters()
                net_sent = net_io.bytes_sent            # 전송된 바이트
                net_recv = net_io.bytes_recv            # 수신된 바이트
                
                if self.force_stop:
                    break

                # 딕셔너리로 정보 묶기
                info = {
                    'cpu': cpu_percent,                   # CPU 사용률 (%)
                    'memory': mem_percent,                  # 메모리 사용률 (%)
                    'temperature': cpu_temp,                # CPU 온도 (라즈베리파이)
                    'disk_read': disk_read,                 # 디스크 읽기 I/O
                    'disk_write': disk_write,               # 디스크 쓰기 I/O
                    'net_sent': net_sent,                   # 네트워크 전송 I/O
                    'net_recv': net_recv
                }
                
                # 메인 윈도우로 전송
                if not self.force_stop:
                    self.update_signal.emit(info)
                
            except Exception as e:
                if not self.force_stop:
                    print(f"System monitor error: {e}")
                
            # 강제 종료 확인하면서 대기
            for _ in range(10):  # 1초를 0.1초씩 나누어 대기
                if self.force_stop:
                    break
                time.sleep(0.1)
            
    def stop(self):
        """모니터링 종료"""
        
        self.force_stop = True
        self.is_running = False
        
        if self.isRunning():
            self.wait(2000)  # 최대 2초 대기
            if self.isRunning():
                print("Warning: SystemMonitorThread did not stop gracefully")


class VideoWidget(QWidget):
    """비디오 표시 위젯
    
    OpenCV 프레임을 Qt 위젯에 표시합니다.
    """

    # opencv_1_20250627_1.py
    # 텍스트 위치 상수 정의
    TEXT_TOP_LEFT = 0
    TEXT_TOP_RIGHT = 1
    TEXT_BOTTOM_LEFT = 2
    TEXT_BOTTOM_RIGHT = 3
    TEXT_CENTER = 4

    # opencv_1_20250627_2.py
    # PTZ 제어 신호 정의
    ptz_up_clicked = pyqtSignal()
    ptz_down_clicked = pyqtSignal()
    ptz_left_clicked = pyqtSignal()
    ptz_right_clicked = pyqtSignal()
    ptz_up_left_clicked = pyqtSignal()
    ptz_up_right_clicked = pyqtSignal()
    ptz_down_left_clicked = pyqtSignal()
    ptz_down_right_clicked = pyqtSignal()
    ptz_center_clicked = pyqtSignal()
    ptz_zoom_in_clicked = pyqtSignal()
    ptz_zoom_out_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.image = None               # 표시할 QImage
        
        # opencv_1_20250627_1.py - 다중 텍스트 지원
        # 텍스트들을 딕셔너리로 관리 (키: 텍스트 ID, 값: 텍스트 정보)
        self.texts = {}
        
        # 기본 텍스트 설정
        self.default_color = QColor(255, 255, 255)  # 흰색
        self.default_font = QFont("Arial", 12)      # 폰트 설정
        self.default_margin = 10                    # 텍스트 여백

        # opencv_1_20250627_2.py
        # PTZ 제어 관련 설정
        self.ptz_enabled = False                    # PTZ 제어 활성화 여부
        self.overlay_enabled = False                # PTZ 오버레이 표시 여부
        self.ptz_grid_size = 3                      # 3x3 그리드
        self.ptz_areas = {}                         # PTZ 영역들

        # opencv_1_20250627_2.py
        # PTZ 클릭 효과 관련
        self.clicked_area = None                    # 클릭된 영역
        self.click_effect_timer = QTimer()          # 클릭 효과 타이머
        self.click_effect_timer.timeout.connect(self._clear_click_effect)
        self.click_effect_duration = 200           # 클릭 효과 지속 시간 (ms)

        self.click_animation_duration = 20          # 애니메이션 속도 조절 (ms)
        self.click_animation_step = 0               # 애니메이션 단계
        self.click_animation_timer = QTimer()       # 애니메이션 타이머
        self.click_animation_timer.timeout.connect(self._animate_click_effect)
        
        # opencv_1_20250627_2.py
        # 호버 효과 관련
        self.hovered_area = None                    # 호버된 영역
        self.mouse_pos = None                       # 마우스 위치

        # 아이콘 관련 설정
        self.icon_size = 32  # 아이콘 크기
        self.use_builtin_icons = True  # 내장 아이콘 사용 여부
        self.icon_cache = {}  # 아이콘 캐시
        # 아이콘 초기화
        self._init_icons()

        # opencv_1_20250627_2.py
        # 마우스 이벤트 활성화
        self.setMouseTracking(True)

        # 2025.06.25_2
        # self.setMinimumSize(640, 480)   # 최소 크기 설정
        # self.setMinimumSize(480, 360)   # 최소 크기 설정
        self.setMinimumSize(360, 240)   # 최소 크기 설정

        # 기본 배경 설정
        self.setAutoFillBackground(True)
        self.setStyleSheet("background-color: black;")

    # opencv_1_20250627_2.py
    def _init_icons(self):
        """아이콘 초기화"""
        if self.use_builtin_icons:
            self._create_builtin_icons()
        else:
            self._load_external_icons()

    # opencv_1_20250627_2.py
    def _create_builtin_icons(self):
        """내장 그래픽으로 아이콘 생성"""
        size = self.icon_size
        
        # 각 방향별 아이콘 생성
        directions = {
            'up': self._create_arrow_icon(0, size),           # 위쪽 화살표
            'down': self._create_arrow_icon(180, size),       # 아래쪽 화살표
            'left': self._create_arrow_icon(270, size),       # 왼쪽 화살표
            'right': self._create_arrow_icon(90, size),       # 오른쪽 화살표
            'up_left': self._create_arrow_icon(315, size),    # 왼쪽 위 화살표
            'up_right': self._create_arrow_icon(45, size),    # 오른쪽 위 화살표
            'down_left': self._create_arrow_icon(225, size),  # 왼쪽 아래 화살표
            'down_right': self._create_arrow_icon(135, size), # 오른쪽 아래 화살표
            'center': self._create_home_icon(size),           # 홈 아이콘
            'zoom_in': self._create_zoom_icon(size, True),    # 줌 인 아이콘
            'zoom_out': self._create_zoom_icon(size, False)   # 줌 아웃 아이콘
        }
        
        self.icon_cache = directions
    
    def _create_arrow_icon(self, angle, size):
        """화살표 아이콘 생성"""
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 화살표 색상
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        
        # 중심점 (정수로 변환)
        center = QPoint(int(size // 2), int(size // 2))
        
        # 화살표 좌표 계산 (모든 값을 정수로 변환)
        arrow_points = [
            QPoint(center.x(), center.y() - int(size // 3)),      # 화살표 끝
            QPoint(center.x() - int(size // 4), center.y()),      # 왼쪽 날개
            QPoint(center.x() - int(size // 6), center.y()),      # 왼쪽 몸통
            QPoint(center.x() - int(size // 6), center.y() + int(size // 3)),  # 왼쪽 꼬리
            QPoint(center.x() + int(size // 6), center.y() + int(size // 3)),  # 오른쪽 꼬리
            QPoint(center.x() + int(size // 6), center.y()),      # 오른쪽 몸통
            QPoint(center.x() + int(size // 4), center.y()),      # 오른쪽 날개
        ]
        
        # 회전 변환
        painter.translate(center)
        painter.rotate(angle)
        painter.translate(-center)
        
        # 화살표 그리기
        polygon = QPolygon(arrow_points)
        painter.drawPolygon(polygon)
        
        painter.end()
        return QIcon(pixmap)
    
    def _create_home_icon(self, size):
        """홈 아이콘 생성"""
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.setBrush(QBrush(QColor(255, 255, 255, 100)))
        
        # 집 모양 그리기 (모든 값을 정수로 변환)
        center_x = int(size // 2)
        center_y = int(size // 2)
        house_size = int(size * 0.7)
        
        # 지붕 (삼각형)
        roof_points = [
            QPoint(center_x, center_y - int(house_size // 3)),
            QPoint(center_x - int(house_size // 3), center_y),
            QPoint(center_x + int(house_size // 3), center_y)
        ]
        painter.drawPolygon(QPolygon(roof_points))
        
        # 벽 (사각형)
        wall_rect = QRect(center_x - int(house_size // 4), center_y, 
                         int(house_size // 2), int(house_size // 3))
        painter.drawRect(wall_rect)
        
        # 문 (작은 사각형)
        door_rect = QRect(center_x - int(house_size // 8), center_y + int(house_size // 6),
                         int(house_size // 4), int(house_size // 6))
        painter.fillRect(door_rect, QColor(0, 0, 0))
        
        painter.end()
        return QIcon(pixmap)
    
    def _create_zoom_icon(self, size, zoom_in=True):
        """줌 아이콘 생성"""
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        
        # 돋보기 원 (모든 값을 정수로 변환)
        center = QPoint(int(size // 2), int(size // 2))
        radius = int(size // 3)
        painter.drawEllipse(center, radius, radius)
        
        # 돋보기 손잡이
        handle_start = QPoint(center.x() + int(radius * 0.7), center.y() + int(radius * 0.7))
        handle_end = QPoint(center.x() + int(radius * 1.3), center.y() + int(radius * 1.3))
        painter.drawLine(handle_start, handle_end)
        
        # + 또는 - 기호
        symbol_size = int(radius * 0.6)
        if zoom_in:
            # + 기호
            painter.drawLine(center.x() - symbol_size // 2, center.y(),
                           center.x() + symbol_size // 2, center.y())
            painter.drawLine(center.x(), center.y() - symbol_size // 2,
                           center.x(), center.y() + symbol_size // 2)
        else:
            # - 기호
            painter.drawLine(center.x() - symbol_size // 2, center.y(),
                           center.x() + symbol_size // 2, center.y())
        
        painter.end()
        return QIcon(pixmap)
    
    def _load_external_icons(self):
        """외부 이미지 파일에서 아이콘 로드"""
        # 실제 아이콘 파일이 있을 때 사용
        icon_paths = {
            'up': 'icons/arrow_up.png',
            'down': 'icons/arrow_down.png',
            'left': 'icons/arrow_left.png',
            'right': 'icons/arrow_right.png',
            'up_left': 'icons/arrow_up_left.png',
            'up_right': 'icons/arrow_up_right.png',
            'down_left': 'icons/arrow_down_left.png',
            'down_right': 'icons/arrow_down_right.png',
            'center': 'icons/home.png',
            'zoom_in': 'icons/zoom_in.png',
            'zoom_out': 'icons/zoom_out.png'
        }
        
        self.icon_cache = {}
        for key, path in icon_paths.items():
            try:
                icon = QIcon(path)
                if not icon.isNull():
                    self.icon_cache[key] = icon
                else:
                    # 파일이 없으면 내장 아이콘 사용
                    self.icon_cache[key] = self._get_fallback_icon(key)
            except:
                self.icon_cache[key] = self._get_fallback_icon(key)
    
    # opencv_1_20250627_2.py
    def _get_fallback_icon(self, area_name):
        """대체 아이콘 (시스템 내장 아이콘 사용)"""
        # Qt 시스템 아이콘 사용
        style = self.style()
        
        icon_map = {
            'up': style.standardIcon(QStyle.SP_ArrowUp),
            'down': style.standardIcon(QStyle.SP_ArrowDown),
            'left': style.standardIcon(QStyle.SP_ArrowLeft),
            'right': style.standardIcon(QStyle.SP_ArrowRight),
            'center': style.standardIcon(QStyle.SP_ComputerIcon),
            'zoom_in': style.standardIcon(QStyle.SP_FileDialogDetailedView),
            'zoom_out': style.standardIcon(QStyle.SP_FileDialogListView)
        }
        
        return icon_map.get(area_name, style.standardIcon(QStyle.SP_ArrowUp))


    # opencv_1_20250627_2.py
    def _start_click_effect(self, area_name):
        """클릭 효과 시작"""
        self.clicked_area = area_name
        self.click_animation_step = 0
        
        # 애니메이션 타이머 시작
        self.click_animation_timer.start(self.click_animation_duration)  # 20ms 간격
        
        # 효과 지속 시간 타이머 시작
        self.click_effect_timer.start(self.click_effect_duration)
        
    # opencv_1_20250627_2.py
    def _animate_click_effect(self):
        """클릭 효과 애니메이션"""
        self.click_animation_step += 1
        if self.click_animation_step > 10:  # 10단계 애니메이션
            self.click_animation_timer.stop()
            self.click_animation_step = 0
        self.update()
        
    # opencv_1_20250627_2.py
    def _clear_click_effect(self):
        """클릭 효과 제거"""
        self.clicked_area = None
        self.click_animation_timer.stop()
        self.click_animation_step = 0
        self.update()


    # opencv_1_20250627_2.py
    def enable_ptz_control(self, ptz_enable=True, overlay_enable=False):
        """PTZ 제어 기능 활성화/비활성화
        
        Args:
            enabled (bool): PTZ 제어 활성화 여부
            show_overlay (bool): PTZ 오버레이 표시 여부
        """
        self.ptz_enabled = ptz_enable
        self.overlay_enabled = overlay_enable
        
        # if self.ptz_enabled:
        if self.overlay_enabled:
            self._calculate_ptz_areas()
        self.update()
    
    # opencv_1_20250627_2.py
    def _calculate_ptz_areas(self):
        """PTZ 제어 영역 계산"""
        
        # if not self.ptz_enabled:
        #     return
            
        width = self.width()
        height = self.height()
        
        # 3x3 그리드로 나누기
        grid_width = width // 3
        grid_height = height // 3
        
        self.ptz_areas = {
            'up_left': QRect(0, 0, grid_width, grid_height),
            'up': QRect(grid_width, 0, grid_width, grid_height),
            'up_right': QRect(grid_width * 2, 0, grid_width, grid_height),
            'left': QRect(0, grid_height, grid_width, grid_height),
            'center': QRect(grid_width, grid_height, grid_width, grid_height),
            'right': QRect(grid_width * 2, grid_height, grid_width, grid_height),
            'down_left': QRect(0, grid_height * 2, grid_width, grid_height),
            'down': QRect(grid_width, grid_height * 2, grid_width, grid_height),
            'down_right': QRect(grid_width * 2, grid_height * 2, grid_width, grid_height)
        }
        
    # opencv_1_20250627_2.py
    def _draw_ptz_overlay(self, painter):
        """PTZ 제어 오버레이 그리기"""
        
        if not self.ptz_areas:
            return
            
        # 기본 오버레이
        if self.overlay_enabled:
            overlay_color = QColor(255, 255, 255, 30)
            border_color = QColor(255, 255, 255, 80)
            
            painter.setBrush(QBrush(overlay_color))
            painter.setPen(QPen(border_color, 1))
            
            for area_name, rect in self.ptz_areas.items():
                painter.drawRect(rect)
                self._draw_area_icon(painter, area_name, rect, 0.7)
        
        # 호버 효과
        if self.hovered_area and self.hovered_area in self.ptz_areas:
            hover_rect = self.ptz_areas[self.hovered_area]
            hover_color = QColor(255, 255, 255, 60)
            hover_border = QColor(255, 255, 0, 150)  # 노란색 테두리
            
            painter.setBrush(QBrush(hover_color))
            painter.setPen(QPen(hover_border, 2))
            painter.drawRect(hover_rect)
            
            # 호버된 아이콘을 더 크게
            self._draw_area_icon(painter, self.hovered_area, hover_rect, 1.0)
        
        # 클릭 효과
        if self.clicked_area and self.clicked_area in self.ptz_areas:
            click_rect = self.ptz_areas[self.clicked_area]
            
            # 애니메이션 진행도
            progress = self.click_animation_step / 10.0
            
            # 클릭된 영역 하이라이트
            highlight_color = QColor(0, 255, 0, 100)
            painter.setBrush(QBrush(highlight_color))
            painter.setPen(QPen(QColor(0, 255, 0, 200), 3))
            painter.drawRect(click_rect)
            
            # 파동 효과
            center_x = click_rect.center().x()
            center_y = click_rect.center().y()
            ripple_radius = int(30 * progress)
            
            alpha = int(255 * (1.0 - progress))
            ripple_color = QColor(0, 255, 0, alpha)
            painter.setBrush(QBrush())
            painter.setPen(QPen(ripple_color, 2))
            painter.drawEllipse(center_x - ripple_radius, center_y - ripple_radius, 
                              ripple_radius * 2, ripple_radius * 2)
            
            # 클릭된 아이콘을 더 크게 + 녹색으로
            self._draw_area_icon(painter, self.clicked_area, click_rect, 1.2, 
                               QColor(0, 255, 0))
    
    # opencv_1_20250627_2.py
    def _draw_area_icon(self, painter, area_name, rect, scale=1.0, color_filter=None):
        """영역에 아이콘 그리기"""
        if area_name not in self.icon_cache:
            return
            
        icon = self.icon_cache[area_name]
        icon_size = int(self.icon_size * scale)
        
        # 아이콘 위치 계산 (중앙 정렬)
        icon_x = rect.center().x() - icon_size // 2
        icon_y = rect.center().y() - icon_size // 2
        icon_rect = QRect(icon_x, icon_y, icon_size, icon_size)
        
        # 아이콘 그리기
        if color_filter:
            # 색상 필터가 있으면 픽스맵을 수정해서 그리기
            try:
                pixmap = icon.pixmap(icon_size, icon_size)
                
                # 색상 오버레이 적용
                colored_pixmap = QPixmap(pixmap.size())
                colored_pixmap.fill(Qt.transparent)
                
                colored_painter = QPainter(colored_pixmap)
                colored_painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
                colored_painter.drawPixmap(0, 0, pixmap)
                colored_painter.setCompositionMode(QPainter.CompositionMode_SourceAtop)
                colored_painter.fillRect(colored_pixmap.rect(), color_filter)
                colored_painter.end()
                
                painter.drawPixmap(icon_rect, colored_pixmap)
            except Exception as e:
                print(f"색상 필터 아이콘 그리기 오류: {e}")
                # 오류 시 기본 아이콘 그리기
                icon.paint(painter, icon_rect)
        else:
            # 기본 아이콘 그리기
            try:
                icon.paint(painter, icon_rect)
            except Exception as e:
                print(f"아이콘 그리기 오류: {e}")
                # 대체 텍스트 그리기
                painter.setPen(QPen(QColor(255, 255, 255)))
                painter.setFont(QFont("Arial", 10))
                painter.drawText(icon_rect, Qt.AlignCenter, "?")

    # opencv_1_20250627_2.py
    def mousePressEvent(self, event):
        """마우스 클릭 이벤트"""
        if not self.ptz_enabled or not self.ptz_areas:
            return
            
        if event.button() == Qt.LeftButton:
            click_pos = event.pos()
            
            # 클릭된 영역 확인
            for area_name, rect in self.ptz_areas.items():
                if rect.contains(click_pos):
                    # 클릭 효과 시작
                    self._start_click_effect(area_name)
                    # PTZ 명령 처리
                    self._handle_ptz_click(area_name)
                    break
                    
        elif event.button() == Qt.RightButton:
            # 우클릭 = 줌 아웃
            self.ptz_zoom_out_clicked.emit()
            
        elif event.button() == Qt.MiddleButton:
            # 중간 버튼 = 줌 인
            self.ptz_zoom_in_clicked.emit()
    
    # opencv_1_20250627_2.py
    def mouseMoveEvent(self, event):
        """마우스 이동 이벤트 (호버 효과)"""
        if not self.ptz_enabled or not self.ptz_areas:
            return
            
        self.mouse_pos = event.pos()
        old_hovered = self.hovered_area
        self.hovered_area = None
        
        # 호버된 영역 찾기
        for area_name, rect in self.ptz_areas.items():
            if rect.contains(self.mouse_pos):
                self.hovered_area = area_name
                break
        
        # 호버 상태가 변경되면 업데이트
        if old_hovered != self.hovered_area:
            self.update()
    
    # opencv_1_20250627_2.py
    def leaveEvent(self, event):
        """마우스가 위젯을 벗어날 때"""
        self.hovered_area = None
        self.mouse_pos = None
        self.update()


    # opencv_1_20250627_2.py
    def wheelEvent(self, event):
        """마우스 휠 이벤트 (줌 제어)"""
        if not self.ptz_enabled:
            return
            
        if event.angleDelta().y() > 0:
            # 휠 업 = 줌 인
            self.ptz_zoom_in_clicked.emit()
        else:
            # 휠 다운 = 줌 아웃
            self.ptz_zoom_out_clicked.emit()
    
    # opencv_1_20250627_2.py
    def _handle_ptz_click(self, area_name):
        """PTZ 클릭 처리"""
        signal_map = {
            'up_left': self.ptz_up_left_clicked,
            'up': self.ptz_up_clicked,
            'up_right': self.ptz_up_right_clicked,
            'left': self.ptz_left_clicked,
            'center': self.ptz_center_clicked,
            'right': self.ptz_right_clicked,
            'down_left': self.ptz_down_left_clicked,
            'down': self.ptz_down_clicked,
            'down_right': self.ptz_down_right_clicked
        }
        
        if area_name in signal_map:
            signal_map[area_name].emit()
            print(f"PTZ {area_name} clicked")  # 디버그용
            
            # 클릭 피드백 텍스트 표시 (선택사항)
            direction_text = area_name.replace('_', ' ').upper()
            self.set_text("ptz_feedback", f"PTZ: {direction_text}", 
                         QColor(0, 255, 0), QFont("Arial", 10, QFont.Bold),
                         self.TEXT_CENTER)
            
            # 피드백 텍스트 자동 제거 타이머
            QTimer.singleShot(1000, lambda: self.clear_text("ptz_feedback"))

    # opencv_1_20250627_2.py
    def resizeEvent(self, event):
        """위젯 크기 변경 이벤트"""
        super().resizeEvent(event)
        
        # 변경 된 화면 사이즈에 맞게  overlay 사이즈도 변경!
        # if self.ptz_enabled:
        if self.overlay_enabled:
            self._calculate_ptz_areas()


    def clear_display(self):
        """화면 지우기"""
        self.image = None
        self.update()

    # opencv_1_20250627_1.py
    def clear_text(self, text_id=None):
        """텍스트 지우기
        
        Args:
            text_id (str, optional): 특정 텍스트 ID. None이면 모든 텍스트 삭제
        """
        if text_id is None:
            # 모든 텍스트 삭제
            self.texts.clear()
        else:
            # 특정 텍스트만 삭제
            if text_id in self.texts:
                del self.texts[text_id]
        self.update()
    
    # opencv_1_20250627_1.py - 다중 텍스트 지원으로 수정
    def set_text(self, text_id, text, color=None, font=None, position=None, margin=None):
        """텍스트 설정
        
        Args:
            text_id (str): 텍스트 식별자 (예: "rec", "camera_name", "time" 등)
            text (str): 표시할 텍스트
            color (QColor, optional): 텍스트 색상
            font (QFont, optional): 텍스트 폰트
            position (int, optional): 텍스트 위치 (TEXT_* 상수)
            margin (int, optional): 텍스트 여백
        """
        # 기본값 설정
        if color is None:
            color = self.default_color
        if font is None:
            font = self.default_font
        if position is None:
            position = self.TEXT_TOP_LEFT
        if margin is None:
            margin = self.default_margin
            
        # 텍스트 정보 저장
        self.texts[text_id] = {
            'text': text,
            'color': color,
            'font': font,
            'position': position,
            'margin': margin
        }
        
        self.update()  # 화면 다시 그리기

    # opencv_1_20250627_1.py
    def _draw_texts(self, painter):
        """모든 텍스트 그리기 (내부 메서드)
        
        Args:
            painter (QPainter): 페인터 객체
        """
        # 위치별로 텍스트들을 그룹화하여 겹치지 않도록 배치
        position_groups = {
            self.TEXT_TOP_LEFT: [],
            self.TEXT_TOP_RIGHT: [],
            self.TEXT_BOTTOM_LEFT: [],
            self.TEXT_BOTTOM_RIGHT: [],
            self.TEXT_CENTER: []
        }
        
        # 텍스트들을 위치별로 분류
        for text_id, text_info in self.texts.items():
            position_groups[text_info['position']].append(text_info)
        
        # 각 위치별로 텍스트 그리기
        for position, text_list in position_groups.items():
            if not text_list:
                continue
                
            self._draw_text_group(painter, text_list, position)
    
    def _draw_text_group(self, painter, text_list, position):
        """같은 위치의 텍스트 그룹 그리기
        
        Args:
            painter (QPainter): 페인터 객체
            text_list (list): 텍스트 정보 리스트
            position (int): 위치
        """
        y_offset = 0  # 세로 오프셋 (같은 위치에 여러 텍스트가 있을 때 사용)
        
        for text_info in text_list:
            # 폰트와 색상 설정
            painter.setFont(text_info['font'])
            painter.setPen(QPen(text_info['color']))
            
            # 텍스트 크기 계산
            font_metrics = painter.fontMetrics()
            text_width = font_metrics.width(text_info['text'])
            text_height = font_metrics.height()
            margin = text_info['margin']
            
            # 위치에 따른 좌표 계산
            if position == self.TEXT_TOP_LEFT:
                x = margin
                y = margin + text_height + y_offset
            elif position == self.TEXT_TOP_RIGHT:
                x = self.width() - text_width - margin
                y = margin + text_height + y_offset
            elif position == self.TEXT_BOTTOM_LEFT:
                x = margin
                y = self.height() - margin - y_offset
            elif position == self.TEXT_BOTTOM_RIGHT:
                x = self.width() - text_width - margin
                y = self.height() - margin - y_offset
            elif position == self.TEXT_CENTER:
                x = (self.width() - text_width) // 2
                y = (self.height() + text_height) // 2 + y_offset
            else:  # 기본값은 TEXT_TOP_LEFT
                x = margin
                y = margin + text_height + y_offset
            
            # 텍스트 그리기
            painter.drawText(x, y, text_info['text'])
            
            # 다음 텍스트를 위한 오프셋 조정
            if position in [self.TEXT_TOP_LEFT, self.TEXT_TOP_RIGHT, self.TEXT_CENTER]:
                y_offset += text_height + 5  # 아래로 이동
            else:  # BOTTOM 위치들
                y_offset += text_height + 5  # 위로 이동


    def update_frame(self, frame):
        """프레임 업데이트
    
        OpenCV 프레임(numpy 배열)을 QImage로 변환하여 표시합니다.
        
        Args:
            frame: OpenCV 프레임 (BGR 형식의 numpy 배열)
        """
        
        if frame is None:
            return

        try:
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                print(f"잘못된 프레임 형식: {frame.shape}")
                return

            height, width, channel = frame.shape        # 프레임 크기 정보
            if height <= 0 or width <= 0:
                print(f"잘못된 프레임 크기: {width}x{height}")
                return

            bytes_per_line = 3 * width                  # 한 줄의 바이트 수 (RGB = 3바이트)
            
            # OpenCV는 BGR, Qt는 RGB를 사용하므로 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # numpy 배열을 QImage로 변환
            q_image = QImage(
                rgb_frame.data,         # 이미지 데이터 포인터
                width,                  # 너비
                height,                 # 높이
                bytes_per_line,         # 한 줄의 바이트 수
                QImage.Format_RGB888    # 이미지 포맷(RGB 8비트 형식)
            )

            if q_image.isNull():
                print("QImage 생성 실패")
                return

            # 위젯 크기에 맞게 스케일링 (비율 유지, 부드러운 변환)
            self.image = q_image.scaled(
                self.size(), 
                Qt.KeepAspectRatio,         # 종횡비 유지
                Qt.SmoothTransformation     # 부드러운 변환
            )
            self.update()   # 위젯 다시 그리기 요청

        except Exception as e:
            print(f"Frame update error: {e}")
        
    def paintEvent(self, event):
        """페인트 이벤트
    
        QPainter를 사용하여 QImage를 위젯에 그리는 이벤트 핸들러입니다.
        """

        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)     # 배경을 검은색으로
        
        if self.image:
            # 중앙 정렬(이미지를 중앙에 배치)
            x = (self.width() - self.image.width()) // 2
            y = (self.height() - self.image.height()) // 2
            painter.drawImage(x, y, self.image)

        # opencv_1_20250627_1.py - 다중 텍스트 그리기
        if self.texts:
            self._draw_texts(painter)

        # opencv_1_20250627_2.py
        # PTZ 오버레이 그리기
        # if self.ptz_enabled:
        if self.overlay_enabled:
            self._draw_ptz_overlay(painter)


class CameraSettingsDialog(QDialog):
    """카메라 설정 다이얼로그
    
    RTSP URL, 해상도, FPS, 저장 경로를 설정합니다.
    """
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config                # 현재 설정
        
        self.setWindowTitle("카메라 설정")
        self.setModal(False)                # 비모달 (다른 창 사용 가능)
        self.setupUI()
        self.load_config()

        # opencv_1_20250702_2.py
        # 실행화면 사이즈 확인
        # 화면 크기 확인을 위해 약간의 지연 후 실행
        # from PyQt5.QtCore import QTimer
        # QTimer.singleShot(100, lambda: show_window_size(self, "화면 사이즈"))
        
    def setupUI(self):
        layout = QVBoxLayout()
        
        # RTSP URL 입력 필드
        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("RTSP URL:"))
        self.url_input = QLineEdit()

        # 2025.06.25_2
        self.url_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # 가로로 확장되도록 설정
        self.url_input.textChanged.connect(self.adjust_input_width)  # 텍스트 변경 시 이벤트 연결

        url_layout.addWidget(self.url_input)
        layout.addLayout(url_layout)
        
        # 해상도 선택 콤보박스 (1920x1080, 1280x720 등)
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("해상도:"))
        self.res_combo = QComboBox()
        self.res_combo.addItems([
            "1920x1080", "1280x720", "960x540", "640x480"
        ])
        res_layout.addWidget(self.res_combo)
        layout.addLayout(res_layout)
        
        # FPS 스핀박스 (1-60)
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        fps_layout.addWidget(self.fps_spin)
        layout.addLayout(fps_layout)
        
        # 저장 경로 입력 및 찾아보기 버튼
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("저장 경로:"))
        self.path_input = QLineEdit()
        path_layout.addWidget(self.path_input)
        self.path_btn = QPushButton("찾아보기")
        self.path_btn.clicked.connect(self.browse_path)
        path_layout.addWidget(self.path_btn)
        layout.addLayout(path_layout)


        # 공지 메시지(message1)
        message_layout = QHBoxLayout()
        message_layout.addWidget(QLabel("공지 메시지:"))
        self.message_input = QLineEdit()
        message_layout.addWidget(self.message_input)
        layout.addLayout(message_layout)
        
        # 🆕 자동 시작 옵션 그룹 추가
        autostart_group = QGroupBox("자동 시작 설정")
        autostart_layout = QVBoxLayout()

        # 스트리밍 자동 시작
        self.autostart_streaming_cb = QCheckBox("프로그램 시작 시 스트리밍 자동 시작")
        self.autostart_streaming_cb.setToolTip("RTSP URL이 설정된 경우에만 작동합니다.")
        autostart_layout.addWidget(self.autostart_streaming_cb)
        
        # 녹화 자동 시작
        self.autostart_recording_cb = QCheckBox("스트리밍 시작 후 녹화 자동 시작")
        self.autostart_recording_cb.setToolTip("스트리밍 자동 시작이 활성화된 경우에만 작동합니다.")
        autostart_layout.addWidget(self.autostart_recording_cb)
        
        # 자동 시작 관련 설명
        info_label = QLabel("💡 자동 시작 기능은 RTSP URL이 설정된 경우에만 작동합니다.")
        info_label.setStyleSheet("color: #666; font-size: 9pt; font-style: italic;")
        autostart_layout.addWidget(info_label)
        
        autostart_group.setLayout(autostart_layout)
        layout.addWidget(autostart_group)        

        # 저장소 관리 설정 추가
        storage_group = QGroupBox("저장소 관리")
        storage_layout = QVBoxLayout()
        
        # 최소 여유 공간 (GB)
        gb_layout = QHBoxLayout()
        gb_layout.addWidget(QLabel("최소 여유 공간 (GB):"))
        self.min_gb_spin = QSpinBox()
        self.min_gb_spin.setRange(1, 100)
        self.min_gb_spin.setValue(10)
        gb_layout.addWidget(self.min_gb_spin)
        storage_layout.addLayout(gb_layout)
        
        # 최소 여유 공간 (%)
        percent_layout = QHBoxLayout()
        percent_layout.addWidget(QLabel("최소 여유 공간 (%):"))
        self.min_percent_spin = QSpinBox()
        self.min_percent_spin.setRange(1, 50)
        self.min_percent_spin.setValue(10)
        percent_layout.addWidget(self.min_percent_spin)
        storage_layout.addLayout(percent_layout)
        
        # 자동 삭제 활성화
        self.auto_delete_cb = QCheckBox("용량 부족 시 오래된 파일 자동 삭제")
        self.auto_delete_cb.setChecked(True)
        storage_layout.addWidget(self.auto_delete_cb)
        
        # 삭제 배치 크기
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("한 번에 삭제할 파일 수:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 50)
        self.batch_spin.setValue(10)
        batch_layout.addWidget(self.batch_spin)
        storage_layout.addLayout(batch_layout)
        
        storage_group.setLayout(storage_layout)
        layout.addWidget(storage_group)

        # 설정 저장/취소 버튼
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.save_btn = QPushButton("저장")
        self.save_btn.setFixedSize(80, 30)
        self.save_btn.clicked.connect(self.save_config)
        self.cancel_btn = QPushButton("종료")
        self.cancel_btn.setFixedSize(80, 30)
        self.cancel_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        
    # 2025.06.25_2
    def adjust_input_width(self):
        """입력 필드의 너비를 텍스트 길이에 맞게 조정"""
        # 현재 텍스트의 너비 계산 (폰트 메트릭스 사용)
        font_metrics = self.url_input.fontMetrics()
        text_width = font_metrics.horizontalAdvance(self.url_input.text())
        
        # 여백 추가 (픽셀 단위)
        padding = 40
        new_width = text_width + padding
        
        # 최소 너비 설정 (선택사항)
        min_width = 200  # 원하는 최소 너비로 조정
        new_width = max(new_width, min_width)
        
        # 너비 업데이트
        self.url_input.setMinimumWidth(new_width)

    def load_config(self):
        """설정 불러오기"""
        camera = self.config.get("camera", {})

        storage = self.config.get("storage", DEFAULT_CONFIG["storage"])
        
        self.url_input.setText(camera.get("rtsp_url", ""))
        self.res_combo.setCurrentText(camera.get("resolution", "1920x1080"))
        self.fps_spin.setValue(camera.get("fps", 30))
        self.path_input.setText(camera.get("save_path", "./recordings"))

         # 공지 메시지(message1)
        self.message_input.setText(camera.get("message1", ""))

        # 🆕 자동 시작 설정 로드
        self.autostart_streaming_cb.setChecked(camera.get("autostart_streaming", False))
        self.autostart_recording_cb.setChecked(camera.get("autostart_recording", False))

        # 저장소 설정 로드
        self.min_gb_spin.setValue(storage.get("min_free_space_gb", 10))
        self.min_percent_spin.setValue(storage.get("min_free_space_percent", 10))
        self.auto_delete_cb.setChecked(storage.get("auto_delete", True))
        self.batch_spin.setValue(storage.get("delete_batch_size", 10))
        
    def browse_path(self):
        """저장 경로 선택"""
        path = QFileDialog.getExistingDirectory(self, "저장 경로 선택")
        if path:
            self.path_input.setText(path)
            
    def save_config(self):
        """설정 저장"""

        # 카메라 설정 저장
        self.config["camera"]["rtsp_url"] = self.url_input.text()
        self.config["camera"]["resolution"] = self.res_combo.currentText()
        self.config["camera"]["fps"] = self.fps_spin.value()
        self.config["camera"]["save_path"] = self.path_input.text()
        
        # 🆕 자동 시작 설정 저장
        self.config["camera"]["autostart_streaming"] = self.autostart_streaming_cb.isChecked()
        self.config["camera"]["autostart_recording"] = self.autostart_recording_cb.isChecked()

         # 공지 메시지(message1)
        self.config["camera"]["message1"] = self.message_input.text()

        # 저장소 설정 저장
        if "storage" not in self.config:
            self.config["storage"] = {}
        self.config["storage"]["min_free_space_gb"] = self.min_gb_spin.value()
        self.config["storage"]["min_free_space_percent"] = self.min_percent_spin.value()
        self.config["storage"]["auto_delete"] = self.auto_delete_cb.isChecked()
        self.config["storage"]["delete_batch_size"] = self.batch_spin.value()


        # 설정 파일로 저장
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=4)            
        
        # # 자동 시작 설정 관련 안내 메시지
        # autostart_msg = ""
        # if self.autostart_streaming_cb.isChecked():
        #     if self.autostart_recording_cb.isChecked():
        #         autostart_msg = "\n\n🚀 다음 프로그램 시작 시 스트리밍과 녹화가 자동으로 시작됩니다."
        #     else:
        #         autostart_msg = "\n\n🚀 다음 프로그램 시작 시 스트리밍이 자동으로 시작됩니다."

        QMessageBox.information(self, "알림", "설정이 저장되었습니다.")
        self.accept()


class KeySettingsDialog(QDialog):
    def __init__(self, parent=None, config_file="nvr_config.json"):
        super().__init__(parent)
        self.config_file = config_file
        self.config = self.load_config()
        self.init_ui()
        self.load_key_settings()

        # opencv_1_20250702_2.py
        # 실행화면 사이즈 확인
        # 화면 크기 확인을 위해 약간의 지연 후 실행
        # from PyQt5.QtCore import QTimer
        # QTimer.singleShot(100, lambda: show_window_size(self, "화면 사이즈"))
        
    def init_ui(self):
        self.setWindowTitle("메뉴 키 설정")
        # self.setFixedSize(600, 500)
        self.setModal(True)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout()
        
        # 키 설정 그룹박스
        key_group = QGroupBox("메뉴 키 설정")
        key_layout = QGridLayout()
        
        # 키 입력 필드들 생성
        self.key_inputs = {}
        
        # 첫 번째 열
        row = 0
        key_layout.addWidget(QLabel("카메라 연결:"), row, 0)
        self.key_inputs['camera_connect'] = QKeySequenceEdit()
        key_layout.addWidget(self.key_inputs['camera_connect'], row, 1)
        
        key_layout.addWidget(QLabel("카메라 종료:"), row, 2)
        self.key_inputs['camera_stop'] = QKeySequenceEdit()
        key_layout.addWidget(self.key_inputs['camera_stop'], row, 3)
        
        key_layout.addWidget(QLabel("이전 그룹:"), row, 4)
        self.key_inputs['prev_group'] = QKeySequenceEdit()
        key_layout.addWidget(self.key_inputs['prev_group'], row, 5)
        
        row += 1
        key_layout.addWidget(QLabel("카메라 전체 연결:"), row, 0)
        self.key_inputs['camera_connect_all'] = QKeySequenceEdit()
        key_layout.addWidget(self.key_inputs['camera_connect_all'], row, 1)
        
        key_layout.addWidget(QLabel("카메라 전체 종료:"), row, 2)
        self.key_inputs['camera_stop_all'] = QKeySequenceEdit()
        key_layout.addWidget(self.key_inputs['camera_stop_all'], row, 3)
        
        key_layout.addWidget(QLabel("다음 그룹:"), row, 4)
        self.key_inputs['next_group'] = QKeySequenceEdit()
        key_layout.addWidget(self.key_inputs['next_group'], row, 5)
        
        row += 1
        key_layout.addWidget(QLabel("이전 구성:"), row, 0)
        self.key_inputs['prev_config'] = QKeySequenceEdit()
        key_layout.addWidget(self.key_inputs['prev_config'], row, 1)
        
        key_layout.addWidget(QLabel("녹화 시작:"), row, 2)
        self.key_inputs['record_start'] = QKeySequenceEdit()
        key_layout.addWidget(self.key_inputs['record_start'], row, 3)
        
        key_layout.addWidget(QLabel("화면 회전:"), row, 4)
        self.key_inputs['screen_rotate'] = QKeySequenceEdit()
        key_layout.addWidget(self.key_inputs['screen_rotate'], row, 5)
        
        row += 1
        key_layout.addWidget(QLabel("다음 구성:"), row, 0)
        self.key_inputs['next_config'] = QKeySequenceEdit()
        key_layout.addWidget(self.key_inputs['next_config'], row, 1)
        
        key_layout.addWidget(QLabel("녹화 정지:"), row, 2)
        self.key_inputs['record_stop'] = QKeySequenceEdit()
        key_layout.addWidget(self.key_inputs['record_stop'], row, 3)
        
        key_layout.addWidget(QLabel("화면 반전:"), row, 4)
        self.key_inputs['screen_flip'] = QKeySequenceEdit()
        key_layout.addWidget(self.key_inputs['screen_flip'], row, 5)
        
        row += 1
        key_layout.addWidget(QLabel("화면 숨기기:"), row, 0)
        self.key_inputs['screen_hide'] = QKeySequenceEdit()
        key_layout.addWidget(self.key_inputs['screen_hide'], row, 1)
        
        key_layout.addWidget(QLabel("메뉴 열기:"), row, 2)
        self.key_inputs['menu_open'] = QKeySequenceEdit()
        key_layout.addWidget(self.key_inputs['menu_open'], row, 3)
        
        key_layout.addWidget(QLabel("프로그램 종료:"), row, 4)
        self.key_inputs['program_exit'] = QKeySequenceEdit()
        key_layout.addWidget(self.key_inputs['program_exit'], row, 5)
        
        key_group.setLayout(key_layout)
        main_layout.addWidget(key_group)
        
        # 버튼 레이아웃
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # 저장 버튼
        save_button = QPushButton("저장")
        save_button.setFixedSize(80, 30)
        save_button.clicked.connect(self.save_settings)
        button_layout.addWidget(save_button)
        
        # 종료 버튼
        exit_button = QPushButton("종료")
        exit_button.setFixedSize(80, 30)
        exit_button.clicked.connect(self.close)
        button_layout.addWidget(exit_button)
        
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)
        
    def load_config(self):
        """설정 파일 로드"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # 기본 설정 반환
            return {
                "camera": {
                    "rtsp_url": "rtsp://admin:smartitlog1~@192.168.0.115:554/Streaming/Channels/102",
                    "resolution": "640x480",
                    "fps": 20,
                    "save_path": "D:/",
                    "autostart_streaming": True,
                    "autostart_recording": True,
                    "message1": "테스트 중입니다. 화면 끄지 마세요!!"
                },
                "storage": {
                    "min_free_space_gb": 10,
                    "min_free_space_percent": 10,
                    "auto_delete": True,
                    "delete_batch_size": 10
                },
                "menu_keys": {},
                "ptz_keys": {},
                "backup": {
                    "src_path": "/media/pi/NVR_MAIN/camera_1/completed",
                    "dest_path": "D:/",
                    "ext": "mp4",
                    "delete_after_backup": True,
                    "verification": True,
                    "sync_interval": 5
                }
            }
    
    def load_key_settings(self):
        """저장된 키 설정 로드"""
        menu_keys = self.config.get("menu_keys", {})
        
        # 기본 키 설정
        default_keys = {
            'camera_connect': 'F1',
            'camera_stop': 'F2',
            'prev_group': 'N',
            'camera_connect_all': 'F3',
            'camera_stop_all': 'F4',
            'next_group': 'M',
            'prev_config': 'F5',
            'record_start': 'F7',
            'screen_rotate': 'F9',
            'next_config': 'F6',
            'record_stop': 'F8',
            'screen_flip': 'F10',
            'screen_hide': 'Esc',
            'menu_open': 'F11',
            'program_exit': 'F12'
        }
        
        # 키 입력 필드에 값 설정
        for key, input_field in self.key_inputs.items():
            key_value = menu_keys.get(key, default_keys.get(key, ''))
            if key_value:
                input_field.setKeySequence(QKeySequence(key_value))
    
    def save_settings(self):
        """키 설정 저장"""
        try:
            # 현재 키 설정 수집
            menu_keys = {}
            for key, input_field in self.key_inputs.items():
                key_sequence = input_field.keySequence()
                if not key_sequence.isEmpty():
                    menu_keys[key] = key_sequence.toString()
            
            # 중복 키 체크
            if self.check_duplicate_keys(menu_keys):
                QMessageBox.warning(self, "경고", "중복된 키가 있습니다. 다시 설정해주세요.")
                return
            
            # 설정 업데이트
            self.config["menu_keys"] = menu_keys
            
            # 파일에 저장
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=4)
            
            QMessageBox.information(self, "알림", "키 설정이 저장되었습니다.")
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"설정 저장 중 오류가 발생했습니다: {str(e)}")
    
    def check_duplicate_keys(self, menu_keys):
        """중복 키 체크"""
        key_values = list(menu_keys.values())
        return len(key_values) != len(set(key_values))

class PTZSettingsDialog(QDialog):
    def __init__(self, parent=None, config_file="nvr_config.json"):
        super().__init__(parent)
        self.config_file = config_file
        self.config = self.load_config()
        self.init_ui()
        self.load_ptz_settings()

        # opencv_1_20250702_2.py
        # 실행화면 사이즈 확인
        # 화면 크기 확인을 위해 약간의 지연 후 실행
        # from PyQt5.QtCore import QTimer
        # QTimer.singleShot(100, lambda: show_window_size(self, "화면 사이즈"))
        
    def init_ui(self):
        self.setWindowTitle("PTZ 키 설정")
        self.setModal(True)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout()
        
        # PTZ 키 설정 그룹박스
        ptz_group = QGroupBox("PTZ 키 설정")
        ptz_layout = QGridLayout()
        
        # 키 입력 필드들 생성
        self.key_inputs = {}
        
        # PTZ 설정 항목들 (첨부 이미지 기준으로 한국어 레이블 사용)
        row = 0
        
        # 첫 번째 행
        ptz_layout.addWidget(QLabel("팬 좌:"), row, 0)
        self.key_inputs['pan_left'] = QKeySequenceEdit()
        ptz_layout.addWidget(self.key_inputs['pan_left'], row, 1)
        
        ptz_layout.addWidget(QLabel("위:"), row, 2)
        self.key_inputs['up'] = QKeySequenceEdit()
        ptz_layout.addWidget(self.key_inputs['up'], row, 3)
        
        ptz_layout.addWidget(QLabel("오른쪽 위:"), row, 4)
        self.key_inputs['right_up'] = QKeySequenceEdit()
        ptz_layout.addWidget(self.key_inputs['right_up'], row, 5)
        
        row += 1
        
        # 두 번째 행
        ptz_layout.addWidget(QLabel("왼쪽:"), row, 0)
        self.key_inputs['left'] = QKeySequenceEdit()
        ptz_layout.addWidget(self.key_inputs['left'], row, 1)
        
        ptz_layout.addWidget(QLabel("정지:"), row, 2)
        self.key_inputs['stop'] = QKeySequenceEdit()
        ptz_layout.addWidget(self.key_inputs['stop'], row, 3)
        
        ptz_layout.addWidget(QLabel("오른쪽:"), row, 4)
        self.key_inputs['right'] = QKeySequenceEdit()
        ptz_layout.addWidget(self.key_inputs['right'], row, 5)
        
        row += 1
        
        # 세 번째 행
        ptz_layout.addWidget(QLabel("팬 아래:"), row, 0)
        self.key_inputs['pan_down'] = QKeySequenceEdit()
        ptz_layout.addWidget(self.key_inputs['pan_down'], row, 1)
        
        ptz_layout.addWidget(QLabel("아래:"), row, 2)
        self.key_inputs['down'] = QKeySequenceEdit()
        ptz_layout.addWidget(self.key_inputs['down'], row, 3)
        
        ptz_layout.addWidget(QLabel("오른쪽 아래:"), row, 4)
        self.key_inputs['right_down'] = QKeySequenceEdit()
        ptz_layout.addWidget(self.key_inputs['right_down'], row, 5)
        
        row += 1
        
        # 네 번째 행
        ptz_layout.addWidget(QLabel("줌 인:"), row, 0)
        self.key_inputs['zoom_in'] = QKeySequenceEdit()
        ptz_layout.addWidget(self.key_inputs['zoom_in'], row, 1)
        
        ptz_layout.addWidget(QLabel("줌 아웃:"), row, 2)
        self.key_inputs['zoom_out'] = QKeySequenceEdit()
        ptz_layout.addWidget(self.key_inputs['zoom_out'], row, 3)
        
        row += 1
        
        # 다섯 번째 행
        ptz_layout.addWidget(QLabel("PTZ 속도 증가:"), row, 0)
        self.key_inputs['ptz_speed_up'] = QKeySequenceEdit()
        ptz_layout.addWidget(self.key_inputs['ptz_speed_up'], row, 1)
        
        ptz_layout.addWidget(QLabel("PTZ 속도 감소:"), row, 2)
        self.key_inputs['ptz_speed_down'] = QKeySequenceEdit()
        ptz_layout.addWidget(self.key_inputs['ptz_speed_down'], row, 3)
        
        ptz_group.setLayout(ptz_layout)
        main_layout.addWidget(ptz_group)
        
        # 버튼 레이아웃
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # 저장 버튼
        save_button = QPushButton("저장")
        save_button.setFixedSize(80, 30)
        save_button.clicked.connect(self.save_settings)
        button_layout.addWidget(save_button)
        
        # 종료 버튼
        exit_button = QPushButton("종료")
        exit_button.setFixedSize(80, 30)
        exit_button.clicked.connect(self.close)
        button_layout.addWidget(exit_button)
        
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)
        
    def load_config(self):
        """설정 파일 로드"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # 기본 설정 반환
            return {
                "camera": {
                    "rtsp_url": "rtsp://admin:smartitlog1~@192.168.0.115:554/Streaming/Channels/102",
                    "resolution": "640x480",
                    "fps": 20,
                    "save_path": "D:/",
                    "autostart_streaming": True,
                    "autostart_recording": True,
                    "message1": "테스트 중입니다. 화면 끄지 마세요!!"
                },
                "storage": {
                    "min_free_space_gb": 10,
                    "min_free_space_percent": 10,
                    "auto_delete": True,
                    "delete_batch_size": 10
                },
                "menu_keys": {},
                "ptz_kyes": {},  # 원본 키명 유지
                "backup": {
                    "src_path": "/media/pi/NVR_MAIN/camera_1/completed",
                    "dest_path": "D:/",
                    "ext": "mp4",
                    "delete_after_backup": True,
                    "verification": True,
                    "sync_interval": 5
                }
            }
    
    def load_ptz_settings(self):
        """저장된 PTZ 키 설정 로드"""
        ptz_keys = self.config.get("ptz_kyes", {})  # 원본 키명 유지
        
        # 기본 PTZ 키 설정 (첨부 이미지의 기본값들)
        default_keys = {
            'pan_left': 'Q',        # 팬 좌
            'up': 'W',              # 위
            'right_up': 'E',        # 오른쪽 위
            'left': 'A',            # 왼쪽
            'stop': 'S',            # 정지
            'right': 'D',           # 오른쪽
            'pan_down': 'Z',        # 팬 아래
            'down': 'X',            # 아래
            'right_down': 'C',      # 오른쪽 아래
            'zoom_in': 'V',         # 줌 인
            'zoom_out': 'B',        # 줌 아웃
            'ptz_speed_up': 'R',    # PTZ 속도 증가
            'ptz_speed_down': 'T'   # PTZ 속도 감소
        }
        
        # 키 입력 필드에 값 설정
        for key, input_field in self.key_inputs.items():
            key_value = ptz_keys.get(key, default_keys.get(key, ''))
            if key_value:
                input_field.setKeySequence(QKeySequence(key_value))
    
    def save_settings(self):
        """PTZ 키 설정 저장"""
        try:
            # 현재 키 설정 수집
            ptz_keys = {}
            for key, input_field in self.key_inputs.items():
                key_sequence = input_field.keySequence()
                if not key_sequence.isEmpty():
                    ptz_keys[key] = key_sequence.toString()
            
            # 중복 키 체크
            if self.check_duplicate_keys(ptz_keys):
                QMessageBox.warning(self, "경고", "중복된 키가 있습니다. 다시 설정해주세요.")
                return
            
            # 설정 업데이트
            self.config["ptz_kyes"] = ptz_keys  # 원본 키명 유지
            
            # 파일에 저장
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=4)
            
            QMessageBox.information(self, "알림", "PTZ 키 설정이 저장되었습니다.")
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"설정 저장 중 오류가 발생했습니다: {str(e)}")
    
    def check_duplicate_keys(self, ptz_keys):
        """중복 키 체크"""
        key_values = list(ptz_keys.values())
        return len(key_values) != len(set(key_values))


class StorageManager:
    """저장소 관리 클래스 - USB 용량 관리 및 오래된 파일 자동 삭제"""
    
    def __init__(self, save_path: str, config: dict):
        """
        Args:
            save_path: 녹화 파일 저장 경로
            config: 저장소 관리 설정
        """
        self.save_path = save_path
        self.min_free_space_gb = config.get('min_free_space_gb', 10)
        self.min_free_space_percent = config.get('min_free_space_percent', 10)
        self.auto_delete = config.get('auto_delete', True)
        self.delete_batch_size = config.get('delete_batch_size', 10)
        
        # 로거 설정
        self.logger = logging.getLogger(__name__ + '.StorageManager')
        
    def is_mount(self, usb_path) -> bool:
        """usb가 연결되어있는지 확인
        
        Returns:
            bool: 연결되어 있으면 True
        """
        return os.path.exists(usb_path)

    def get_disk_usage(self) -> Tuple[float, float, float]:
        """디스크 사용량 정보 반환
        
        Returns:
            tuple: (전체 용량 GB, 사용된 용량 GB, 여유 공간 GB)
        """
        try:
            usage = psutil.disk_usage(self.save_path)
            total_gb = usage.total / (1024**3)
            used_gb = usage.used / (1024**3)
            free_gb = usage.free / (1024**3)
            return total_gb, used_gb, free_gb
        except Exception as e:
            self.logger.error(f"디스크 사용량 확인 실패: {e}")
            return 0, 0, 0
            
    def get_free_space_percent(self) -> float:
        """여유 공간 비율 반환 (%)"""
        try:
            usage = psutil.disk_usage(self.save_path)
            return (usage.free / usage.total) * 100
        except Exception as e:
            self.logger.error(f"여유 공간 비율 확인 실패: {e}")
            return 0
            
    def is_storage_low(self) -> bool:
        """저장 공간이 부족한지 확인
        
        Returns:
            bool: 저장 공간이 부족하면 True
        """
        _, _, free_gb = self.get_disk_usage()
        free_percent = self.get_free_space_percent()
        
        # GB 기준 또는 퍼센트 기준 중 하나라도 부족하면 True
        return (free_gb < self.min_free_space_gb or 
                free_percent < self.min_free_space_percent)
                
    def get_old_recordings(self, camera_id: int = None) -> List[Tuple[str, float, float]]:
        """오래된 녹화 파일 목록 반환
        
        Args:
            camera_id: 특정 카메라 ID (None이면 모든 카메라)
            
        Returns:
            list: [(파일경로, 수정시간, 파일크기MB), ...] 수정시간 기준 오름차순
        """
        recordings = []
        
        try:
            # 검색할 디렉토리 결정
            if camera_id:
                search_dirs = [os.path.join(self.save_path, f"camera_{camera_id}", "completed")]
            else:
                # 모든 카메라 디렉토리 검색
                search_dirs = []
                for item in os.listdir(self.save_path):
                    if item.startswith("camera_"):
                        completed_dir = os.path.join(self.save_path, item, "completed")
                        if os.path.exists(completed_dir):
                            search_dirs.append(completed_dir)
                            
            # 각 디렉토리에서 mp4 파일 검색
            for search_dir in search_dirs:
                if not os.path.exists(search_dir):
                    continue
                    
                for file in os.listdir(search_dir):
                    if file.lower().endswith('.mp4'):
                        file_path = os.path.join(search_dir, file)
                        try:
                            stat = os.stat(file_path)
                            recordings.append((
                                file_path,
                                stat.st_mtime,  # 수정 시간
                                stat.st_size / (1024**2)  # MB 단위
                            ))
                        except Exception as e:
                            self.logger.error(f"파일 정보 읽기 실패: {file_path}, {e}")
                            
            # 수정 시간 기준 오름차순 정렬 (오래된 파일이 앞에)
            recordings.sort(key=lambda x: x[1])
            
        except Exception as e:
            self.logger.error(f"녹화 파일 목록 조회 실패: {e}")
            
        return recordings
        
    def delete_old_recordings(self, required_space_mb: float = None) -> Tuple[int, float]:
        """오래된 녹화 파일 삭제
        
        Args:
            required_space_mb: 필요한 공간 (MB), None이면 설정된 최소 여유 공간까지 삭제
            
        Returns:
            tuple: (삭제된 파일 수, 확보된 공간 MB)
        """
        if not self.auto_delete:
            self.logger.info("자동 삭제가 비활성화되어 있습니다.")
            return 0, 0
            
        deleted_count = 0
        freed_space_mb = 0
        
        try:
            # 현재 여유 공간 확인
            _, _, free_gb = self.get_disk_usage()
            current_free_mb = free_gb * 1024
            
            # 목표 여유 공간 계산
            if required_space_mb:
                target_free_mb = current_free_mb + required_space_mb
            else:
                # 설정된 최소 여유 공간의 1.5배를 목표로 (여유를 두기 위해)
                target_free_mb = self.min_free_space_gb * 1024 * 1.5
                
            # 오래된 파일 목록 가져오기
            old_recordings = self.get_old_recordings()
            
            if not old_recordings:
                self.logger.warning("삭제할 녹화 파일이 없습니다.")
                return 0, 0
                
            # 배치 단위로 파일 삭제
            for i in range(0, len(old_recordings), self.delete_batch_size):
                if current_free_mb >= target_free_mb:
                    break
                    
                batch = old_recordings[i:i + self.delete_batch_size]
                
                for file_path, mtime, size_mb in batch:
                    try:
                        # 파일 삭제 전 로그
                        file_date = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                        self.logger.info(f"파일 삭제 중: {os.path.basename(file_path)} "
                                       f"(날짜: {file_date}, 크기: {size_mb:.1f}MB)")
                        
                        # 파일 삭제
                        os.remove(file_path)
                        
                        deleted_count += 1
                        freed_space_mb += size_mb
                        current_free_mb += size_mb
                        
                        # 목표 달성 확인
                        if current_free_mb >= target_free_mb:
                            break
                            
                    except Exception as e:
                        self.logger.error(f"파일 삭제 실패: {file_path}, {e}")
                        
                # 디스크 사용량 다시 확인 (실제 여유 공간 확인)
                _, _, free_gb = self.get_disk_usage()
                current_free_mb = free_gb * 1024
                
            self.logger.info(f"삭제 완료: {deleted_count}개 파일, {freed_space_mb:.1f}MB 확보")
            
        except Exception as e:
            self.logger.error(f"오래된 파일 삭제 중 오류: {e}")
            
        return deleted_count, freed_space_mb
        
    def ensure_free_space(self, required_space_mb: float = 500) -> bool:
        """필요한 여유 공간 확보
        
        Args:
            required_space_mb: 필요한 공간 (MB)
            
        Returns:
            bool: 공간 확보 성공 여부
        """
        try:
            # 현재 저장 공간 확인
            if not self.is_storage_low():
                return True
                
            self.logger.warning(f"저장 공간 부족 감지, {required_space_mb}MB 확보 시도")
            
            # 오래된 파일 삭제
            deleted_count, freed_space = self.delete_old_recordings(required_space_mb)
            
            # 다시 확인
            if not self.is_storage_low():
                self.logger.info("저장 공간 확보 성공")
                return True
            else:
                self.logger.error("저장 공간 확보 실패 - 더 이상 삭제할 파일이 없습니다.")
                return False
                
        except Exception as e:
            self.logger.error(f"저장 공간 확보 중 오류: {e}")
            return False


class CCTVBackupSettingsDialog(QDialog):
    """CCTV 백업 설정 다이얼로그"""
    
    def __init__(self, parent=None, config_path: str = "nvr_config.json"):
        super().__init__(parent)
        self.config_path = config_path
        self.init_ui()
        self.load_config()

        self.check_usb_status(self.src_path_edit.text(), self.usb_status_label_src)
        self.check_usb_status(self.dest_path_edit.text(), self.usb_status_label_dest)

        # opencv_1_20250702_2.py
        # 실행화면 사이즈 확인
        # 화면 크기 확인을 위해 약간의 지연 후 실행
        # from PyQt5.QtCore import QTimer
        # QTimer.singleShot(100, lambda: show_window_size(self, "화면 사이즈"))
        
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("CCTV 백업 설정")

        self.setFixedSize(600, 400)

        self.setModal(True)
        
        layout = QVBoxLayout()
        
        # 경로 설정 그룹
        self.create_path_group(layout)
        
        # USB 상태 그룹
        self.create_usb_status_group(layout)

        # 백업 옵션 그룹
        self.create_option_group(layout)
        
        # 버튼 그룹
        self.create_button_group(layout)
        
        self.setLayout(layout)
        
    def create_path_group(self, parent_layout):
        """경로 설정 그룹 생성"""
        path_group = QGroupBox("경로 설정")
        path_layout = QVBoxLayout()
        
        # 원본 경로
        src_layout = QHBoxLayout()
        src_layout.addWidget(QLabel("원본 경로:"))
        self.src_path_edit = QLineEdit()
        self.src_path_edit.setPlaceholderText("백업할 파일들이 있는 경로")
        src_layout.addWidget(self.src_path_edit)

        src_browse_btn = QPushButton("찾아보기")
        src_browse_btn.clicked.connect(self.browse_src_path)
        src_layout.addWidget(src_browse_btn)
        
        path_layout.addLayout(src_layout)
        
        # 대상 경로
        dest_layout = QHBoxLayout()
        dest_layout.addWidget(QLabel("대상 경로:"))
        self.dest_path_edit = QLineEdit()
        self.dest_path_edit.setPlaceholderText("백업 파일을 저장할 경로")
        dest_layout.addWidget(self.dest_path_edit)

        dest_browse_btn = QPushButton("찾아보기")
        dest_browse_btn.clicked.connect(self.browse_dest_path)
        dest_layout.addWidget(dest_browse_btn)
        
        path_layout.addLayout(dest_layout)
        
        # 파일 확장자
        ext_layout = QHBoxLayout()
        ext_layout.addWidget(QLabel("파일 확장자:"))
        self.ext_edit = QLineEdit("mp4")
        self.ext_edit.setMaximumWidth(100)
        self.ext_edit.setPlaceholderText("예: mp4")
        ext_layout.addWidget(self.ext_edit)
        ext_layout.addStretch()
        path_layout.addLayout(ext_layout)
        
        path_group.setLayout(path_layout)
        parent_layout.addWidget(path_group)
        
    def create_usb_status_group(self, parent_layout):
        """USB 상태 그룹"""
        usb_group = QGroupBox("USB 장치 상태")

        usb_layout = QVBoxLayout()      

        usb_layout_src = QHBoxLayout()
        self.usb_status_label_src2 = QLabel("원본 USB : ")
        self.usb_status_label_src2.setStyleSheet("font-weight: bold;")        
        self.usb_status_label_src2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)      # 라벨이 필요한 공간만 차지하도록 설정
        usb_layout_src.addWidget(self.usb_status_label_src2)
        self.usb_status_label_src = QLabel("...")
        self.usb_status_label_src.setStyleSheet("font-weight: bold;")        
        self.usb_status_label_src.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)   # 상태 라벨이 남은 공간을 채우도록 설정
        usb_layout_src.addWidget(self.usb_status_label_src)
        self.usb_check_btn_src = QPushButton("상태 확인")
        # self.usb_check_btn_src.clicked.connect(self.check_usb_status(self.dest_path_edit.text(), self.usb_status_label_src))
        self.usb_check_btn_src.clicked.connect(lambda: self.check_usb_status(self.src_path_edit.text(), self.usb_status_label_src))
        usb_layout_src.addWidget(self.usb_check_btn_src)

        usb_layout_dest = QHBoxLayout()
        self.usb_status_label_dest2 = QLabel("대상 USB : ")
        self.usb_status_label_dest2.setStyleSheet("font-weight: bold")
        usb_layout_dest.addWidget(self.usb_status_label_dest2)  
        self.usb_status_label_dest2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)      # 라벨이 필요한 공간만 차지하도록 설정
        self.usb_status_label_dest = QLabel("...")
        self.usb_status_label_dest.setStyleSheet("font-weight: bold;")
        self.usb_status_label_dest.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)   # 상태 라벨이 남은 공간을 채우도록 설정
        usb_layout_dest.addWidget(self.usb_status_label_dest)        
        self.usb_check_btn_dest = QPushButton("상태 확인")
        # self.usb_check_btn_dest.clicked.connect(self.check_usb_status(self.src_path_edit.text(), self.usb_status_label_dest))
        self.usb_check_btn_dest.clicked.connect(lambda: self.check_usb_status(self.dest_path_edit.text(), self.usb_status_label_dest))
        usb_layout_dest.addWidget(self.usb_check_btn_dest)
        
        # usb_group.setLayout(usb_layout_desc)
        usb_layout.addLayout(usb_layout_src)
        usb_layout.addLayout(usb_layout_dest) 

        usb_group.setLayout(usb_layout)

        parent_layout.addWidget(usb_group)

    def create_option_group(self, parent_layout):
        """백업 옵션 그룹 생성"""
        option_group = QGroupBox("백업 옵션")
        option_layout = QVBoxLayout()
        
        # 백업 후 삭제 옵션
        self.delete_after_backup_cb = QCheckBox("백업 완료 후 원본 파일 삭제")
        self.delete_after_backup_cb.setChecked(True)
        self.delete_after_backup_cb.setStyleSheet("font-weight: bold; color: #DC143C;")
        option_layout.addWidget(self.delete_after_backup_cb)
        
        # 파일 검증 옵션
        self.verification_cb = QCheckBox("백업 파일 무결성 검증 (권장)")
        self.verification_cb.setChecked(True)
        self.verification_cb.setStyleSheet("font-weight: bold; color: #2E8B57;")
        option_layout.addWidget(self.verification_cb)
        
        # 동기화 간격
        sync_layout = QHBoxLayout()
        sync_layout.addWidget(QLabel("동기화 간격:"))
        self.sync_interval_spin = QSpinBox()
        self.sync_interval_spin.setRange(5, 50)
        self.sync_interval_spin.setValue(10)
        self.sync_interval_spin.setSuffix(" 파일마다")
        sync_layout.addWidget(self.sync_interval_spin)
        sync_layout.addStretch()
        option_layout.addLayout(sync_layout)
        
        option_group.setLayout(option_layout)
        parent_layout.addWidget(option_group)
        
    def create_button_group(self, parent_layout):
        """버튼 그룹 생성"""
        button_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("저장")
        self.save_btn.clicked.connect(self.save_config)
        self.save_btn.setFixedSize(80, 30)
        
        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setFixedSize(80, 30)
        
        button_layout.addStretch()
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.cancel_btn)
        
        parent_layout.addLayout(button_layout)
        
    def browse_src_path(self):
        """원본 경로 선택"""
        path = QFileDialog.getExistingDirectory(self, "원본 경로 선택")
        if path:
            self.src_path_edit.setText(path)
            
    def browse_dest_path(self):
        """대상 경로 선택"""
        path = QFileDialog.getExistingDirectory(self, "대상 경로 선택")
        if path:
            self.dest_path_edit.setText(path)
    

    def check_usb_status(self, dest_path: str = None, usb_status_label: QLabel = None):
        """USB 상태 확인
        
        Args:
            dest_path: 확인할 경로 (None이면 self.dest_path_edit.text() 사용)
            usb_status_label: 상태를 표시할 라벨 (None이면 self.usb_status_label 사용)
        
        Returns:
            bool: USB 상태가 정상이면 True, 아니면 False
        """
        # 매개변수가 없으면 기본값 사용
        if dest_path is None:
            dest_path = self.dest_path_edit.text()
        
        if usb_status_label is None:
            usb_status_label = self.usb_status_label
        
        if not dest_path:
            usb_status_label.setText("❌ 대상 경로가 설정되지 않음")
            return False
            
        try:
            if not os.path.exists(dest_path):
                usb_status_label.setText("❌ 경로가 존재하지 않음")
                return False
                
            # 디스크 사용량 확인
            disk_usage = psutil.disk_usage(dest_path)
            free_gb = disk_usage.free / (1024**3)
            total_gb = disk_usage.total / (1024**3)
            
            # 쓰기 테스트
            test_file = os.path.join(dest_path, ".nvr_test_write")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                
                usb_status_label.setText(f"✅ USB 정상 ({free_gb:.1f}/{total_gb:.1f}GB)")
                usb_status_label.setStyleSheet("font-weight: bold; color: #2E8B57; padding: 5px;")
                return True
                
            except Exception as e:
                usb_status_label.setText(f"❌ 쓰기 오류: {str(e)}")
                usb_status_label.setStyleSheet("font-weight: bold; color: #DC143C; padding: 5px;")
                return False
                
        except Exception as e:
            usb_status_label.setText(f"❌ 상태 확인 실패: {str(e)}")
            usb_status_label.setStyleSheet("font-weight: bold; color: #DC143C; padding: 5px;")
            return False

    def load_config(self):
        """설정 파일 로드"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    backup_config = config.get('backup', DEFAULT_CONFIG['backup'])
            else:
                backup_config = DEFAULT_CONFIG['backup']
            
            self.src_path_edit.setText(backup_config.get('src_path', ''))
            self.dest_path_edit.setText(backup_config.get('dest_path', ''))
            self.ext_edit.setText(backup_config.get('ext', 'mp4'))
            self.delete_after_backup_cb.setChecked(backup_config.get('delete_after_backup', True))
            self.verification_cb.setChecked(backup_config.get('verification', True))
            self.sync_interval_spin.setValue(backup_config.get('sync_interval', 10))
            
        except Exception as e:
            QMessageBox.warning(self, "경고", f"설정 파일 로드 실패: {str(e)}")
            
    def save_config(self):
        """설정 저장"""
        try:
            # 기존 설정 파일이 있으면 로드, 없으면 기본 구조 생성
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                config = DEFAULT_CONFIG.copy()
            
            # backup 섹션이 없으면 생성
            if 'backup' not in config:
                config['backup'] = {}
            
            # 설정 저장
            config['backup']['src_path'] = self.src_path_edit.text()
            config['backup']['dest_path'] = self.dest_path_edit.text()
            config['backup']['ext'] = self.ext_edit.text()
            config['backup']['delete_after_backup'] = self.delete_after_backup_cb.isChecked()
            config['backup']['verification'] = self.verification_cb.isChecked()
            config['backup']['sync_interval'] = self.sync_interval_spin.value()
            
            # 설정 파일로 저장
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            
            QMessageBox.information(self, "알림", "설정이 저장되었습니다.")
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"설정 파일 저장 실패: {str(e)}")

class BackupExecuteWorker(QThread):
    """백업 실행 워커 스레드"""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    file_completed = pyqtSignal(str, bool, str)  # filename, success, error_message
    backup_completed = pyqtSignal(int, int)  # success_count, total_count
    usb_health_updated = pyqtSignal(str)
    
    def __init__(self, files_to_backup: List[Dict], dest_path: str, backup_config: dict):
        super().__init__()
        self.files_to_backup = files_to_backup
        self.dest_path = dest_path
        self.backup_config = backup_config
        self.is_cancelled = False
        self.success_count = 0
        self.sync_counter = 0
        
    def cancel(self):
        """백업 취소"""
        self.is_cancelled = True
        
    def run(self):
        """백업 실행"""
        total_files = len(self.files_to_backup)
        self.success_count = 0
        
        try:
            # USB 상태 확인
            if not self._check_usb_health(self.dest_path):
                self.status_updated.emit("USB 장치 상태가 불안정합니다. 백업을 중단합니다.")
                self.backup_completed.emit(0, total_files)
                return
                
            self.status_updated.emit(f"총 {total_files}개 파일 백업 시작...")
            
            # 초기 sync
            self._safe_sync()
            
            # 각 파일 백업
            for idx, file_info in enumerate(self.files_to_backup):
                if self.is_cancelled:
                    self._safe_sync()
                    self.status_updated.emit("사용자가 백업을 취소했습니다.")
                    break
                    
                try:
                    # 주기적 USB 상태 확인
                    if idx % 5 == 0:
                        if not self._check_usb_health(self.dest_path):
                            self.status_updated.emit("USB 장치에 문제가 발생했습니다. 백업을 중단합니다.")
                            break
                    
                    # 파일 백업
                    self._backup_file(file_info)
                    self.success_count += 1
                    self.file_completed.emit(file_info['name'], True, "")
                    
                    # 주기적 sync
                    self.sync_counter += 1
                    if self.sync_counter >= self.backup_config.get('sync_interval', 10):
                        self._safe_sync()
                        self.sync_counter = 0
                    
                    # 진행률 업데이트
                    progress = int(((idx + 1) / total_files) * 100)
                    self.progress_updated.emit(progress)
                    
                    # CPU 부하 감소
                    time.sleep(0.01)
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"파일 백업 실패: {file_info['name']}, 오류: {error_msg}")
                    self.file_completed.emit(file_info['name'], False, error_msg)
                    
            # 최종 sync
            self._safe_sync()
            self.status_updated.emit("백업 완료, 파일시스템 동기화 중...")
            time.sleep(2)
            
            # 완료 신호
            self.backup_completed.emit(self.success_count, total_files)
            
        except Exception as e:
            self._safe_sync()
            error_msg = f"백업 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            self.status_updated.emit(error_msg)
            self.backup_completed.emit(self.success_count, total_files)
    
    def _check_usb_health(self, chk_path: str) -> bool:
        """USB 장치 상태 확인"""
        try:
            if not os.path.exists(chk_path):
                self.usb_health_updated.emit("❌ 대상 경로가 존재하지 않음")
                return False
                
            # 디스크 사용량 확인
            disk_usage = psutil.disk_usage(chk_path)
            free_gb = disk_usage.free / (1024**3)
            
            if free_gb < 1.0:
                self.usb_health_updated.emit(f"⚠️ 여유 공간 부족: {free_gb:.1f}GB")
                return False
            
            # 쓰기 테스트
            test_file = os.path.join(chk_path, ".nvr_test_write")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                self.usb_health_updated.emit(f"✅ USB 정상 (여유공간: {free_gb:.1f}GB)")
                return True
            except:
                self.usb_health_updated.emit("❌ USB 쓰기 오류")
                return False
                
        except Exception as e:
            self.usb_health_updated.emit(f"❌ USB 상태 확인 실패: {str(e)}")
            return False
    
    def _safe_sync(self):
        """파일시스템 동기화"""
        try:
            if IS_WINDOWS:
                return
                
            os.sync()
            subprocess.run(['sync'], check=False, timeout=10)
            logger.info("파일시스템 동기화 완료")
        except Exception as e:
            logger.warning(f"동기화 실행 중 오류: {str(e)}")
    
    def _backup_file(self, file_info: Dict):
        """개별 파일 백업"""
        src_file = file_info['path']
        
        # 대상 파일 경로 생성
        rel_path = os.path.relpath(src_file, os.path.dirname(src_file))
        dest_file = os.path.join(self.dest_path, rel_path)
        
        # 대상 디렉토리 생성
        dest_dir = os.path.dirname(dest_file)
        os.makedirs(dest_dir, exist_ok=True)
        
        # 파일 복사
        if not os.path.exists(dest_file):
            self._copy_file_with_verification(src_file, dest_file)
            logger.info(f"파일 백업 완료: {src_file} -> {dest_file}")
        else:
            # 이미 존재하는 경우
            if self.backup_config.get('verification', True):
                if self._verify_files_identical(src_file, dest_file):
                    logger.info(f"파일 이미 존재하고 동일함: {dest_file}")
                else:
                    # 다른 파일인 경우 새 이름으로 저장
                    new_dest_file = self._get_unique_filename(dest_file)
                    self._copy_file_with_verification(src_file, new_dest_file)
                    logger.info(f"파일 백업 완료 (중복 회피): {src_file} -> {new_dest_file}")
            else:
                logger.info(f"파일 이미 존재 (건너뜀): {dest_file}")
        
        # 백업 후 원본 삭제 (설정된 경우)
        if self.backup_config.get('delete_after_backup', False):
            try:
                if self.backup_config.get('verification', True):
                    if self._verify_backup_integrity(src_file, dest_file):
                        os.remove(src_file)
                        logger.info(f"원본 파일 삭제 완료: {src_file}")
                    else:
                        logger.warning(f"백업 파일 검증 실패, 원본 파일 보존: {src_file}")
                else:
                    os.remove(src_file)
                    logger.info(f"원본 파일 삭제 완료 (검증 생략): {src_file}")
            except Exception as e:
                logger.error(f"원본 파일 삭제 실패: {src_file}, 오류: {str(e)}")
    
    def _copy_file_with_verification(self, src_file: str, dest_file: str):
        """파일 복사"""
        buffer_size = 64 * 1024  # 64KB
        
        with open(src_file, 'rb') as src, open(dest_file, 'wb') as dst:
            while True:
                chunk = src.read(buffer_size)
                if not chunk:
                    break
                dst.write(chunk)
                dst.flush()
                os.fsync(dst.fileno())
                
                if self.is_cancelled:
                    break
        
        # 메타데이터 복사
        shutil.copystat(src_file, dest_file)
    
    def _verify_files_identical(self, file1: str, file2: str) -> bool:
        """파일 동일성 확인"""
        try:
            stat1 = os.stat(file1)
            stat2 = os.stat(file2)
            
            if stat1.st_size != stat2.st_size:
                return False
                
            if abs(stat1.st_mtime - stat2.st_mtime) < 2:
                return True
                
            return False
        except:
            return False
    
    def _verify_backup_integrity(self, src_file: str, dest_file: str) -> bool:
        """백업 파일 무결성 검증"""
        try:
            if not os.path.exists(dest_file):
                return False
                
            if os.path.getsize(src_file) != os.path.getsize(dest_file):
                return False
                
            return True
        except:
            return False
    
    def _get_unique_filename(self, filepath: str) -> str:
        """중복되지 않는 파일명 생성"""
        base_name, ext = os.path.splitext(filepath)
        counter = 1
        while os.path.exists(f"{base_name}_{counter}{ext}"):
            counter += 1
        return f"{base_name}_{counter}{ext}"


class CCTVBackupExecuteDialog(QDialog):
    """CCTV 백업 실행 다이얼로그"""
    
    def __init__(self, parent=None, config_path: str = "nvr_config.json", ui_idx = 0):
        super().__init__(parent)
        self.config_path = config_path
        self.backup_config = {}
        self.backup_worker = None
        self.file_list = []
        
        # opencv_1_20250702_2.py
        # ui_idx = 0 전체 보여주기
        #        = 1 ui 간소화
        self.ui_idx = ui_idx


        # opencv_1_20250702_2.py
        # 축소모드 시, 높이 480으로 고정
        # self.ui_idx = 1         # 축소모드 디버깅
        if self.ui_idx == 1:
            self.setFixedHeight(480)  # 세로 480픽셀로 고정

        self.init_ui()
        self.load_config()        
      
        # opencv_1_20250702_2.py
        # 실행화면 사이즈 확인
        # 화면 크기 확인을 위해 약간의 지연 후 실행
        # from PyQt5.QtCore import QTimer
        # QTimer.singleShot(100, lambda: show_window_size(self, "화면 사이즈"))


    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("CCTV 백업 실행")

        # self.setFixedSize(900, 700)
        # self.setFixedSize(800, 650)
        # opencv_1_20250702_2.py
        self.setFixedWidth(800)  # 가로만 800픽셀로 고정


        self.setModal(True)
        
        layout = QVBoxLayout()        
        
        # 파일 조회 그룹
        self.create_search_group(layout)
        
        # 파일 목록 그룹
        self.create_file_list_group(layout)
        
        # USB 상태 그룹
        if self.ui_idx == 0:
            self.create_usb_status_group(layout)
        
        # 진행 상태 그룹
        self.create_progress_group(layout)
        
        # 로그 그룹
        if self.ui_idx == 0:
            self.create_log_group(layout)
        
        # 버튼 그룹
        self.create_button_group(layout)
        
        self.setLayout(layout)        
        
    def create_search_group(self, parent_layout):
        """파일 조회 그룹"""
        search_group = QGroupBox("파일 조회")
        search_layout = QHBoxLayout()
        
        # 시작일자
        search_layout.addWidget(QLabel("시작일자:"))
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDate(QDate.currentDate().addDays(-7))       # 일주일 전 - 날짜
        search_layout.addWidget(self.start_date_edit)
        
        search_layout.addWidget(QLabel(" ~ "))

        # 종료일자
        search_layout.addWidget(QLabel("종료일자:"))
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDate(QDate.currentDate())                     # 현재 - 날짜
        search_layout.addWidget(self.end_date_edit)
        
        # 여백 추가 (픽셀 단위)
        search_layout.addSpacing(20)  # 20픽셀 여백

        # 전체 조회 체크박스
        self.search_all_cb = QCheckBox("전체 조회")
        self.search_all_cb.stateChanged.connect(self.on_search_all_changed)
        search_layout.addWidget(self.search_all_cb)
        
        search_layout.addStretch()
        
        # 조회 버튼
        self.search_btn = QPushButton("조회")
        self.search_btn.clicked.connect(self.search_files)
        self.search_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        search_layout.addWidget(self.search_btn)
        
        search_group.setLayout(search_layout)
        parent_layout.addWidget(search_group)
        
    def create_file_list_group(self, parent_layout):
        """파일 목록 그룹"""
        file_group = QGroupBox("파일 목록")
        file_layout = QVBoxLayout()
        
        # 선택 정보 표시
        info_layout = QHBoxLayout()
        self.file_info_label = QLabel("파일을 조회해주세요.")
        info_layout.addWidget(self.file_info_label)
        
        # 전체 선택/해제 버튼
        self.select_all_btn = QPushButton("전체 선택")
        self.select_all_btn.clicked.connect(self.select_all_files)
        self.select_all_btn.setEnabled(False)
        info_layout.addWidget(self.select_all_btn)
        
        self.deselect_all_btn = QPushButton("전체 해제")
        self.deselect_all_btn.clicked.connect(self.deselect_all_files)
        self.deselect_all_btn.setEnabled(False)
        info_layout.addWidget(self.deselect_all_btn)
        
        file_layout.addLayout(info_layout)
        
        # 파일 목록 테이블
        self.file_table = QTableWidget()
        self.file_table.setColumnCount(6)
        self.file_table.setHorizontalHeaderLabels(['선택', '파일명', '크기', '날짜', '상태', '경로'])
        self.file_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.file_table.setAlternatingRowColors(True)
        
        # 컬럼 너비 설정
        header = self.file_table.horizontalHeader()
        header.resizeSection(0, 50)   # 선택
        header.resizeSection(1, 250)  # 파일명
        header.resizeSection(2, 80)   # 크기
        header.resizeSection(3, 120)  # 날짜
        header.resizeSection(4, 80)   # 상태
        header.setStretchLastSection(True)  # 경로
        
        # # 체크박스 클릭 시 용량 계산
        # self.file_table.itemChanged.connect(self.calculate_selected_size)
        
        file_layout.addWidget(self.file_table)
        
        file_group.setLayout(file_layout)
        parent_layout.addWidget(file_group)
        
    def create_usb_status_group(self, parent_layout):
        """USB 상태 그룹"""
        usb_group = QGroupBox("USB 장치 상태")
        usb_layout = QHBoxLayout()
        
        self.usb_status_label = QLabel("상태 확인 중...")
        self.usb_status_label.setStyleSheet("font-weight: bold; padding: 5px;")
        usb_layout.addWidget(self.usb_status_label)
        
        self.usb_check_btn = QPushButton("상태 확인")
        self.usb_check_btn.clicked.connect(self.check_usb_status)
        usb_layout.addWidget(self.usb_check_btn)
        
        usb_group.setLayout(usb_layout)
        parent_layout.addWidget(usb_group)
        
    def create_progress_group(self, parent_layout):
        """진행 상태 그룹"""
        progress_group = QGroupBox("백업 진행 상태")
        progress_layout = QVBoxLayout()
        
        # 상태 라벨
        self.status_label = QLabel("대기 중...")
        self.status_label.setStyleSheet("font-weight: bold; color: #2E8B57;")
        progress_layout.addWidget(self.status_label)
        
        # 진행률 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        # 현재 처리 상태
        self.current_status_label = QLabel("")
        self.current_status_label.setStyleSheet("color: #4169E1; font-size: 9pt;")
        progress_layout.addWidget(self.current_status_label)
        
        progress_group.setLayout(progress_layout)
        parent_layout.addWidget(progress_group)
        
    def create_log_group(self, parent_layout):
        """로그 그룹"""
        log_group = QGroupBox("작업 로그")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(100)
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                font-family: 'Courier New', monospace;
                font-size: 9pt;
                color: #000000;
            }
        """)
        
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        parent_layout.addWidget(log_group)
        
    def create_button_group(self, parent_layout):
        """버튼 그룹"""
        button_layout = QHBoxLayout()
        
        self.config_btn = QPushButton("설정 변경")
        self.config_btn.clicked.connect(self.open_settings)        

        self.backup_btn = QPushButton("백업 시작")
        self.backup_btn.clicked.connect(self.start_backup)
        self.backup_btn.setEnabled(False)
        self.backup_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.clicked.connect(self.cancel_backup)
        self.cancel_btn.setEnabled(False)
        
        self.close_btn = QPushButton("닫기")
        self.close_btn.clicked.connect(self.close)
        
        button_layout.addWidget(self.config_btn)

        button_layout.addStretch()

        button_layout.addWidget(self.backup_btn)
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.close_btn)
        
        parent_layout.addLayout(button_layout)
        
    def load_config(self):
        """설정 파일 로드"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.backup_config = config.get('backup', {})
                                        
                    # USB 상태 확인
                    self.check_usb_status()                    
            else:
                self.add_log("설정 파일이 없습니다.")
                
        except Exception as e:
            self.add_log(f"설정 로드 실패: {str(e)}")
    
    def open_settings(self):
        """설정 다이얼로그 열기"""
        # from CCTVBackupSettingsDialog import CCTVBackupSettingsDialog
        settings_dialog = CCTVBackupSettingsDialog(self, self.config_path)
        if settings_dialog.exec_():
            # 설정이 저장되면 다시 로드
            self.load_config()
            self.file_list.clear()
            self.file_table.setRowCount(0)
            self.file_info_label.setText("설정이 변경되었습니다. 파일을 다시 조회해주세요.")
    
    def on_search_all_changed(self, state):
        """전체 조회 체크박스 상태 변경"""
        if state == Qt.Checked:
            self.start_date_edit.setEnabled(False)
            self.end_date_edit.setEnabled(False)
        else:
            self.start_date_edit.setEnabled(True)
            self.end_date_edit.setEnabled(True)
    
    def search_files(self):
        """파일 조회"""
        src_path = self.backup_config.get('src_path', '')
        if not src_path or not os.path.exists(src_path):
            QMessageBox.warning(self, "경고", "원본 경로가 설정되지 않았거나 존재하지 않습니다.")
            return
        
        self.add_log("파일 조회 시작...")
        self.file_list.clear()
        self.file_table.setRowCount(0)
        
        # 파일 확장자
        file_ext = self.backup_config.get('ext', 'mp4').lower()
        
        # 날짜 범위 설정
        if self.search_all_cb.isChecked():
            start_date = None
            end_date = None
        else:
            # 2025.07.01 duzin
            # start_date = self.start_date_edit.date().toPyPython()
            # end_date = self.end_date_edit.date().toPyPython()
            # PyQt5에서 QDate를 Python date 객체로 변환하는 올바른 방법
            qdate_start = self.start_date_edit.date()
            qdate_end = self.end_date_edit.date()

            start_date = datetime(qdate_start.year(), qdate_start.month(), qdate_start.day()).date()
            end_date = datetime(qdate_end.year(), qdate_end.month(), qdate_end.day()).date()
        
        # 파일 검색
        try:
            for root, dirs, files in os.walk(src_path):
                for file in files:
                    if file.lower().endswith(f'.{file_ext}'):
                        file_path = os.path.join(root, file)
                        
                        # 날짜 필터링
                        if start_date and end_date:
                            # 파일명에서 날짜 추출 (cam1_recording_20250630_100339.mp4)
                            try:
                                date_part = file.split('_')[2]  # 20250630
                                file_date = datetime.strptime(date_part, '%Y%m%d').date()
                                
                                if not (start_date <= file_date <= end_date):
                                    continue
                            except:
                                # 날짜 추출 실패 시 수정 시간으로 확인
                                file_mtime = os.path.getmtime(file_path)
                                file_date = datetime.fromtimestamp(file_mtime).date()
                                
                                if not (start_date <= file_date <= end_date):
                                    continue
                        
                        # 파일 정보 수집
                        file_size = os.path.getsize(file_path)
                        file_mtime = os.path.getmtime(file_path)
                        
                        file_info = {
                            'name': file,
                            'path': file_path,
                            'size': file_size,
                            'date': datetime.fromtimestamp(file_mtime),
                            'selected': False,
                            'status': '대기'
                        }
                        
                        self.file_list.append(file_info)
            
            # 날짜순 정렬
            self.file_list.sort(key=lambda x: x['date'], reverse=True)
            
            # 테이블에 표시
            self.display_files()
            
            total_size = sum(f['size'] for f in self.file_list)
            self.file_info_label.setText(f"총 {len(self.file_list)}개 파일 ({self.format_size(total_size)})")
            
            if self.file_list:
                self.select_all_btn.setEnabled(True)
                self.deselect_all_btn.setEnabled(True)
                self.add_log(f"총 {len(self.file_list)}개 파일 조회 완료")
            else:
                self.add_log("조회된 파일이 없습니다.")
                
        except Exception as e:
            self.add_log(f"파일 조회 실패: {str(e)}")
            QMessageBox.critical(self, "오류", f"파일 조회 실패: {str(e)}")
    
    def display_files(self):
        """파일 목록을 테이블에 표시"""
        
        # 시그널 임시 차단
        self.file_table.blockSignals(True)
        
        self.file_table.setRowCount(len(self.file_list))
        
        for row, file_info in enumerate(self.file_list):
            # 체크박스 - 더 안전한 방식으로 생성
            checkbox = QTableWidgetItem()

            # 상태가 "완료"인 경우 체크박스 비활성화
            if file_info['status'] == '완료':
                checkbox.setFlags(Qt.ItemIsEnabled)  # UserCheckable 플래그 제거
                checkbox.setCheckState(Qt.Unchecked)
                # 시각적으로 구분하기 위해 배경색 변경
                from PyQt5.QtGui import QColor
                checkbox.setBackground(QColor(240, 240, 240))
            else:
                checkbox.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                checkbox.setCheckState(Qt.Unchecked)

            self.file_table.setItem(row, 0, checkbox)
            
            # 파일명
            name_item = QTableWidgetItem(file_info['name'])
            name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.file_table.setItem(row, 1, name_item)
            
            # 크기
            size_item = QTableWidgetItem(self.format_size(file_info['size']))
            size_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            size_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.file_table.setItem(row, 2, size_item)
            
            # 날짜
            date_str = file_info['date'].strftime('%Y-%m-%d %H:%M')
            date_item = QTableWidgetItem(date_str)
            date_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.file_table.setItem(row, 3, date_item)
            
            # 상태
            status_item = QTableWidgetItem(file_info['status'])
            status_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.file_table.setItem(row, 4, status_item)
            
            # 경로
            path_item = QTableWidgetItem(file_info['path'])
            path_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.file_table.setItem(row, 5, path_item)
        
        # 시그널 차단 해제
        self.file_table.blockSignals(False)
        
        # 이벤트 핸들러 연결 (한 번만)
        try:
            self.file_table.itemChanged.disconnect()
        except TypeError:
            pass
        self.file_table.itemChanged.connect(self.calculate_selected_size)
        
        # 초기 계산 수행
        self.calculate_selected_size()
    
    def select_all_files(self):
        """전체 파일 선택"""

        # 시그널 차단
        self.file_table.blockSignals(True)
        
        for row in range(self.file_table.rowCount()):
            checkbox_item = self.file_table.item(row, 0)
            if checkbox_item is not None:
                # 체크 가능한 상태인지 확인 (완료된 파일은 제외)
                if checkbox_item.flags() & Qt.ItemIsUserCheckable:
                    checkbox_item.setCheckState(Qt.Checked)
        
        # 시그널 차단 해제
        self.file_table.blockSignals(False)
        
        # 수동으로 계산 실행
        self.calculate_selected_size()
    
    def deselect_all_files(self):
        """전체 파일 선택 해제"""
        
        # 시그널 차단
        self.file_table.blockSignals(True)
        
        for row in range(self.file_table.rowCount()):
            checkbox_item = self.file_table.item(row, 0)
            if checkbox_item is not None:
                checkbox_item.setCheckState(Qt.Unchecked)
        
        # 시그널 차단 해제
        self.file_table.blockSignals(False)
        
        # 수동으로 계산 실행
        self.calculate_selected_size()
    
    def calculate_selected_size(self):
        """선택된 파일들의 총 용량 계산"""
        total_size = 0
        selected_count = 0
        
        # 테이블 행 수와 파일 리스트 길이 확인
        if self.file_table.rowCount() != len(self.file_list):
            return
        
        for row in range(self.file_table.rowCount()):
            # 체크박스 아이템 존재 여부 확인
            checkbox_item = self.file_table.item(row, 0)
            if checkbox_item is None:
                continue
                
            try:
                if checkbox_item.checkState() == Qt.Checked:
                    if row < len(self.file_list):  # 인덱스 범위 확인
                        self.file_list[row]['selected'] = True
                        total_size += self.file_list[row]['size']
                        selected_count += 1
                else:
                    if row < len(self.file_list):  # 인덱스 범위 확인
                        self.file_list[row]['selected'] = False
            except Exception as e:
                # 예외 발생 시 로그만 남기고 계속 진행
                logger.warning(f"체크박스 상태 확인 중 오류 (행 {row}): {str(e)}")
                continue
        
        # 선택 정보 업데이트
        if selected_count > 0:
            self.file_info_label.setText(
                f"총 {len(self.file_list)}개 중 {selected_count}개 선택 ({self.format_size(total_size)})"
            )
            self.backup_btn.setEnabled(True)
        else:
            total_size = sum(f['size'] for f in self.file_list)
            self.file_info_label.setText(f"총 {len(self.file_list)}개 파일 ({self.format_size(total_size)})")
            self.backup_btn.setEnabled(False)
    
    def check_usb_status(self):
        """USB 상태 확인"""
        dest_path = self.backup_config.get('dest_path', '')
        if not dest_path:
            self.usb_status_label.setText("❌ 대상 경로가 설정되지 않음")
            return False
            
        try:
            if not os.path.exists(dest_path):
                self.usb_status_label.setText("❌ 경로가 존재하지 않음")
                return False
                
            # 디스크 사용량 확인
            disk_usage = psutil.disk_usage(dest_path)
            free_gb = disk_usage.free / (1024**3)
            total_gb = disk_usage.total / (1024**3)
            
            # 쓰기 테스트
            test_file = os.path.join(dest_path, ".nvr_test_write")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                
                self.usb_status_label.setText(f"✅ USB 정상 ({free_gb:.1f}/{total_gb:.1f}GB)")
                self.usb_status_label.setStyleSheet("font-weight: bold; color: #2E8B57; padding: 5px;")
                return True
                
            except Exception as e:
                self.usb_status_label.setText(f"❌ 쓰기 오류: {str(e)}")
                self.usb_status_label.setStyleSheet("font-weight: bold; color: #DC143C; padding: 5px;")
                return False
                
        except Exception as e:
            self.usb_status_label.setText(f"❌ 상태 확인 실패: {str(e)}")
            self.usb_status_label.setStyleSheet("font-weight: bold; color: #DC143C; padding: 5px;")
            return False
    
    def start_backup(self):
        """백업 시작"""
        # 선택된 파일 확인
        selected_files = [f for f in self.file_list if f['selected']]
        if not selected_files:
            QMessageBox.warning(self, "경고", "백업할 파일을 선택해주세요.")
            return
        
        # 용량 확인
        total_size = sum(f['size'] for f in selected_files)
        dest_path = self.backup_config.get('dest_path', '')
        
        try:
            disk_usage = psutil.disk_usage(dest_path)
            free_space = disk_usage.free
            
            if total_size > free_space:
                QMessageBox.critical(
                    self, "용량 부족",
                    f"선택한 파일의 총 용량({self.format_size(total_size)})이\n"
                    f"대상 경로의 여유 공간({self.format_size(free_space)})보다 큽니다.\n\n"
                    "일부 파일의 선택을 해제하거나 대상 경로의 공간을 확보해주세요."
                )
                return
        except Exception as e:
            QMessageBox.warning(self, "경고", f"여유 공간 확인 실패: {str(e)}\n계속 진행하시겠습니까?")
        
        # 백업 확인
        delete_mode = self.backup_config.get('delete_after_backup', False)
        message = f"{len(selected_files)}개 파일을 백업하시겠습니까?\n\n"
        message += f"총 용량: {self.format_size(total_size)}\n"
        if delete_mode:
            message += "\n⚠️ 백업 완료 후 원본 파일이 삭제됩니다!"
        
        reply = QMessageBox.question(
            self, "백업 확인", message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # UI 상태 변경
        self.backup_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.close_btn.setEnabled(False)
        self.search_btn.setEnabled(False)
        self.select_all_btn.setEnabled(False)
        self.deselect_all_btn.setEnabled(False)
        
        # 진행 상태 초기화
        self.progress_bar.setValue(0)
        self.status_label.setText("백업 준비 중...")
        self.current_status_label.setText("")
        
        # 상태 초기화
        for file_info in self.file_list:
            if file_info['selected']:
                file_info['status'] = '대기'
        self.display_files()
        
        # 백업 워커 시작
        self.backup_worker = BackupExecuteWorker(selected_files, dest_path, self.backup_config)
        self.backup_worker.progress_updated.connect(self.update_progress)
        self.backup_worker.status_updated.connect(self.update_status)
        self.backup_worker.file_completed.connect(self.update_file_status)
        self.backup_worker.backup_completed.connect(self.backup_finished)
        self.backup_worker.usb_health_updated.connect(self.update_usb_status)
        
        self.backup_worker.start()
        
        self.add_log(f"백업 시작: {len(selected_files)}개 파일")
    
    def cancel_backup(self):
        """백업 취소"""
        if self.backup_worker and self.backup_worker.isRunning():
            reply = QMessageBox.question(
                self, "확인",
                "백업을 취소하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.backup_worker.cancel()
                self.add_log("사용자가 백업을 취소했습니다.")
    
    def update_progress(self, value):
        """진행률 업데이트"""
        self.progress_bar.setValue(value)
    
    def update_status(self, status):
        """상태 업데이트"""
        self.status_label.setText(status)
        self.add_log(status)
    
    def update_file_status(self, filename, success, error_msg):
        """파일 상태 업데이트"""
        for row in range(self.file_table.rowCount()):
            if self.file_table.item(row, 1).text() == filename:
                if success:
                    status = "완료"
                    color = QColor(0, 128, 0)  # 녹색

                    # 완료된 파일의 체크박스 비활성화
                    checkbox_item = self.file_table.item(row, 0)
                    if checkbox_item is not None:
                        checkbox_item.setFlags(Qt.ItemIsEnabled)  # UserCheckable 제거
                        checkbox_item.setCheckState(Qt.Unchecked)  # 체크 해제
                        checkbox_item.setBackground(QColor(240, 240, 240))  # 회색 배경
                else:
                    status = f"실패: {error_msg}"
                    color = QColor(255, 0, 0)  # 빨간색
                
                status_item = self.file_table.item(row, 4)
                status_item.setText(status)
                status_item.setForeground(color)
                
                # 파일 정보도 업데이트
                for file_info in self.file_list:
                    if file_info['name'] == filename:
                        file_info['status'] = status
                        break
                
                self.current_status_label.setText(f"처리 중: {filename}")
                break
    
    def update_usb_status(self, status):
        """USB 상태 업데이트"""
        self.usb_status_label.setText(status)
        if "✅" in status:
            self.usb_status_label.setStyleSheet("font-weight: bold; color: #2E8B57; padding: 5px;")
        elif "⚠️" in status:
            self.usb_status_label.setStyleSheet("font-weight: bold; color: #FF8C00; padding: 5px;")
        else:
            self.usb_status_label.setStyleSheet("font-weight: bold; color: #DC143C; padding: 5px;")
    
    def backup_finished(self, success_count, total_count):
        """백업 완료"""
        # UI 상태 복원
        self.backup_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        self.search_btn.setEnabled(True)
        self.select_all_btn.setEnabled(True)
        self.deselect_all_btn.setEnabled(True)
        
        # 결과 표시
        if success_count == total_count:
            self.status_label.setText(f"백업 완료! (전체 {total_count}건 성공)")
            self.status_label.setStyleSheet("font-weight: bold; color: #2E8B57;")
            QMessageBox.information(
                self, "백업 완료",
                f"백업이 완료되었습니다.\n\n"
                f"전체 {total_count}건 중 {success_count}건 성공"
            )
        else:
            self.status_label.setText(f"백업 완료 (전체 {total_count}건 중 {success_count}건 성공)")
            self.status_label.setStyleSheet("font-weight: bold; color: #FF8C00;")
            QMessageBox.warning(
                self, "백업 완료",
                f"백업이 완료되었습니다.\n\n"
                f"전체 {total_count}건 중 {success_count}건 성공\n"
                f"실패: {total_count - success_count}건"
            )
        
        self.current_status_label.setText("")
        self.add_log(f"백업 완료: 전체 {total_count}건 중 {success_count}건 성공")
        
        # 워커 정리
        if self.backup_worker:
            self.backup_worker.quit()
            self.backup_worker.wait()
            self.backup_worker = None
    
    def add_log(self, message):
        """로그 추가"""

        # opencv_1_20250702_2.py
        if self.ui_idx != 0:
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_text.append(log_entry)
        "파일 조회 실패: disconnect() failed between 'itemChanged' and all its connections"

        # 스크롤을 맨 아래로
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def format_size(self, size_bytes):
        """파일 크기를 읽기 쉬운 형식으로 변환"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    def closeEvent(self, event):
        """다이얼로그 닫기 이벤트"""
        if self.backup_worker and self.backup_worker.isRunning():
            reply = QMessageBox.question(
                self, "확인",
                "백업이 진행 중입니다. 정말 닫으시겠습니까?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.backup_worker.cancel()
                self.backup_worker.quit()
                self.backup_worker.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

class SplashScreen_1(QWidget):
    """애니메이션이 있는 Splash 화면"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowFlags(
            Qt.Dialog | 
            Qt.WindowStaysOnTopHint | 
            Qt.FramelessWindowHint |
            Qt.WindowSystemMenuHint
        )
        
        self.setWindowModality(Qt.ApplicationModal)
        self.setWindowTitle("연결 중...")
        self.setFixedSize(400, 180)
        
        self.setup_ui()
        self.setup_animation()
        self.center_on_parent()
        
    def setup_ui(self):
        """UI 설정"""
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # 메인 메시지
        self.main_label = QLabel("RTSP 카메라 연결 중...")
        self.main_label.setAlignment(Qt.AlignCenter)
        self.main_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.main_label.setStyleSheet("color: #333; margin-bottom: 10px;")
        
        # 상세 상태 메시지
        self.status_label = QLabel("연결 준비 중...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Arial", 10))
        self.status_label.setStyleSheet("color: #666;")
        
        # 프로그레스 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 무한 애니메이션
        self.progress_bar.setFixedHeight(25)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #ccc;
                border-radius: 8px;
                background-color: #f0f0f0;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 6px;
            }
        """)
        
        # 애니메이션 점들
        self.dots_label = QLabel("●○○")
        self.dots_label.setAlignment(Qt.AlignCenter)
        self.dots_label.setFont(QFont("Arial", 16))
        self.dots_label.setStyleSheet("color: #4CAF50;")
        
        layout.addWidget(self.main_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.dots_label)
        
        self.setLayout(layout)
        
        # 배경 스타일
        self.setStyleSheet("""
            SplashScreen {
                background-color: white;
                border: 3px solid #4CAF50;
                border-radius: 15px;
            }
        """)
    
    def setup_animation(self):
        """애니메이션 설정"""
        # 상태 메시지 애니메이션
        self.message_timer = QTimer()
        self.message_timer.timeout.connect(self.update_status_message)
        
        # 점 애니메이션
        self.dots_timer = QTimer()
        self.dots_timer.timeout.connect(self.update_dots)
        
        self.message_index = 0
        self.dots_index = 0
        
        # 상태 메시지들
        self.status_messages = [
            "네트워크 상태 확인 중...",
            "RTSP 서버에 연결 중...",
            "카메라 응답 대기 중...",
            "스트림 설정 중...",
            "연결 품질 확인 중..."
        ]
        
        # 점 패턴들
        self.dots_patterns = ["●○○", "○●○", "○○●", "○●○"]
    
    def start_animation(self):
        """애니메이션 시작"""
        self.message_timer.start(2000)  # 2초마다 메시지 변경
        self.dots_timer.start(500)      # 0.5초마다 점 변경
    
    def stop_animation(self):
        """애니메이션 중지"""
        self.message_timer.stop()
        self.dots_timer.stop()
    
    def update_status_message(self):
        """상태 메시지 업데이트"""
        self.status_label.setText(self.status_messages[self.message_index])
        self.message_index = (self.message_index + 1) % len(self.status_messages)
    
    def update_dots(self):
        """점 애니메이션 업데이트"""
        self.dots_label.setText(self.dots_patterns[self.dots_index])
        self.dots_index = (self.dots_index + 1) % len(self.dots_patterns)
    
    def set_status(self, message):
        """상태 메시지 직접 설정"""
        self.status_label.setText(message)
    
    def set_progress_text(self, text):
        """프로그레스 바 텍스트 설정"""
        self.progress_bar.setFormat(text)
    
    def show(self):
        """표시"""
        super().show()
        self.start_animation()
        self.raise_()
        self.activateWindow()
    
    def close(self):
        """닫기"""
        self.stop_animation()
        super().close()
    
    def center_on_parent(self):
        """부모 창 중앙에 배치"""
        if self.parent():
            parent_geometry = self.parent().geometry()
            x = parent_geometry.x() + (parent_geometry.width() - self.width()) // 2
            y = parent_geometry.y() + (parent_geometry.height() - self.height()) // 2
            self.move(x, y)

class SplashScreen_2(QWidget):
    """현대적이고 세련된 애니메이션 Splash 화면"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowFlags(
            Qt.Dialog | 
            Qt.WindowStaysOnTopHint | 
            Qt.FramelessWindowHint |
            Qt.WindowSystemMenuHint
        )
        
        self.setWindowModality(Qt.ApplicationModal)
        self.setWindowTitle("연결 중...")
        self.setFixedSize(450, 200)  # 조금 더 넓게
        
        # 애니메이션 상태
        self.fade_value = 0
        self.pulse_direction = 1
        
        self.setup_ui()
        self.setup_animation()
        self.center_on_parent()
        
    def setup_ui(self):
        """현대적인 UI 설정"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)  # 외부 여백 제거
        main_layout.setSpacing(0)
        
        # 메인 컨테이너 (그라데이션 배경용)
        self.container = QWidget()
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(30, 25, 30, 25)
        container_layout.setSpacing(20)
        
        # 헤더 영역
        header_layout = QHBoxLayout()
        
        # 로고/아이콘 영역 (원형 아이콘)
        self.icon_label = QLabel("📹")  # 이모지 또는 실제 아이콘으로 교체 가능
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setFont(QFont("Arial", 24))
        self.icon_label.setFixedSize(50, 50)
        self.icon_label.setStyleSheet("""
            QLabel {
                background-color: rgba(76, 175, 80, 0.2);
                border: 2px solid #4CAF50;
                border-radius: 25px;
                color: #4CAF50;
            }
        """)
        
        # 타이틀 영역
        title_layout = QVBoxLayout()
        
        self.main_label = QLabel("RTSP 카메라 연결")
        self.main_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.main_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.main_label.setStyleSheet("""
            color: #2E3440;
            margin: 0px;
            padding: 0px;
        """)
        
        self.subtitle_label = QLabel("네트워크 연결을 확인하고 있습니다")
        self.subtitle_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.subtitle_label.setFont(QFont("Arial", 10))
        self.subtitle_label.setStyleSheet("""
            color: #5E81AC;
            margin: 0px;
            padding: 0px;
        """)
        
        title_layout.addWidget(self.main_label)
        title_layout.addWidget(self.subtitle_label)
        title_layout.setSpacing(2)
        
        header_layout.addWidget(self.icon_label)
        header_layout.addSpacing(15)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        # 상태 메시지 영역
        self.status_label = QLabel("연결 준비 중...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Arial", 11))
        self.status_label.setStyleSheet("""
            color: #4C566A;
            background-color: rgba(136, 192, 208, 0.1);
            padding: 8px 15px;
            border-radius: 12px;
            border: 1px solid rgba(136, 192, 208, 0.3);
        """)
        
        # 프로그레스 영역
        progress_container = QWidget()
        progress_layout = QVBoxLayout(progress_container)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(8)
        
        # 커스텀 프로그레스 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 무한 애니메이션
        self.progress_bar.setFixedHeight(6)  # 얇게
        self.progress_bar.setTextVisible(False)  # 텍스트 숨김
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: rgba(76, 175, 80, 0.2);
                border: none;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4CAF50, stop:0.5 #66BB6A, stop:1 #81C784);
                border-radius: 3px;
            }
        """)
        
        # 애니메이션 점들 (더 세련된 스타일)
        dots_container = QWidget()
        dots_layout = QHBoxLayout(dots_container)
        dots_layout.setContentsMargins(0, 0, 0, 0)
        dots_layout.setSpacing(8)
        dots_layout.setAlignment(Qt.AlignCenter)
        
        # 3개의 점을 개별 라벨로 생성
        self.dot_labels = []
        for i in range(3):
            dot = QLabel("●")
            dot.setAlignment(Qt.AlignCenter)
            dot.setFont(QFont("Arial", 12))
            dot.setFixedSize(20, 20)
            dot.setStyleSheet("""
                color: rgba(76, 175, 80, 0.3);
                background-color: transparent;
            """)
            self.dot_labels.append(dot)
            dots_layout.addWidget(dot)
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(dots_container)
        
        # 레이아웃 조립
        container_layout.addLayout(header_layout)
        container_layout.addWidget(self.status_label)
        container_layout.addWidget(progress_container)
        
        main_layout.addWidget(self.container)
        self.setLayout(main_layout)
        
        # 전체 배경 스타일 (글라스모피즘 효과)
        self.setStyleSheet("""
            SplashScreen {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 255, 255, 0.95),
                    stop:1 rgba(248, 249, 250, 0.95));
                border: 2px solid rgba(76, 175, 80, 0.3);
                border-radius: 20px;
            }
        """)
        
        # 드롭 쉐도우 효과
        self.shadow_effect = QGraphicsDropShadowEffect()
        self.shadow_effect.setBlurRadius(30)
        self.shadow_effect.setColor(QColor(0, 0, 0, 60))
        self.shadow_effect.setOffset(0, 10)
        self.setGraphicsEffect(self.shadow_effect)
    
    def setup_animation(self):
        """향상된 애니메이션 설정"""
        # 상태 메시지 애니메이션
        self.message_timer = QTimer()
        self.message_timer.timeout.connect(self.update_status_message)
        
        # 점 애니메이션 (순차적으로 켜짐)
        self.dots_timer = QTimer()
        self.dots_timer.timeout.connect(self.update_dots)
        
        # 페이드 효과
        self.fade_timer = QTimer()
        self.fade_timer.timeout.connect(self.update_fade)
        
        # 아이콘 펄스 효과
        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(self.update_icon_pulse)
        
        self.message_index = 0
        self.dots_index = 0
        
        # 더 다양하고 구체적인 상태 메시지들
        self.status_messages = [
            "네트워크 인터페이스 확인 중...",
            "RTSP 프로토콜 초기화 중...",
            "카메라 서버에 연결 시도 중...",
            "인증 정보 확인 중...",
            "비디오 스트림 협상 중...",
            "해상도 및 코덱 설정 중...",
            "버퍼링 설정 최적화 중...",
            "연결 품질 테스트 중...",
            "스트리밍 준비 완료 중..."
        ]
    
    def start_animation(self):
        """모든 애니메이션 시작"""
        self.message_timer.start(1000)   # 1.8초마다 메시지 변경
        self.dots_timer.start(400)       # 0.4초마다 점 변경
        self.fade_timer.start(16)        # 부드러운 페이드
        self.pulse_timer.start(100)      # 아이콘 펄스
    
    def stop_animation(self):
        """모든 애니메이션 중지"""
        timers = ['message_timer', 'dots_timer', 'fade_timer', 'pulse_timer']
        for timer_name in timers:
            if hasattr(self, timer_name):
                getattr(self, timer_name).stop()
    
    def update_status_message(self):
        """상태 메시지 페이드 인/아웃과 함께 업데이트"""
        self.status_label.setText(self.status_messages[self.message_index])
        self.message_index = (self.message_index + 1) % len(self.status_messages)
        
        # 메시지 변경 시 살짝 페이드 효과
        self.animate_message_change()
    
    def update_dots(self):
        """점들을 순차적으로 활성화"""
        # 모든 점을 비활성화
        for dot in self.dot_labels:
            dot.setStyleSheet("""
                color: rgba(76, 175, 80, 0.3);
                background-color: transparent;
            """)
        
        # 현재 점을 활성화
        if self.dots_index < len(self.dot_labels):
            active_dot = self.dot_labels[self.dots_index]
            active_dot.setStyleSheet("""
                color: #4CAF50;
                background-color: rgba(76, 175, 80, 0.2);
                border-radius: 10px;
            """)
        
        self.dots_index = (self.dots_index + 1) % (len(self.dot_labels) + 1)
    
    def update_fade(self):
        """서브타이틀 페이드 효과"""
        self.fade_value += self.pulse_direction * 3
        if self.fade_value >= 100:
            self.fade_value = 100
            self.pulse_direction = -1
        elif self.fade_value <= 30:
            self.fade_value = 30
            self.pulse_direction = 1
        
        alpha = self.fade_value / 100.0
        self.subtitle_label.setStyleSheet(f"""
            color: rgba(94, 129, 172, {alpha});
            margin: 0px;
            padding: 0px;
        """)
    
    def update_icon_pulse(self):
        """아이콘 펄스 효과"""
        # 현재 스타일을 기반으로 펄스 효과 구현
        scale = 1.0 + (self.fade_value / 1000.0)  # 미세한 크기 변화
        opacity = 0.7 + (self.fade_value / 500.0)  # 투명도 변화
        
        self.icon_label.setStyleSheet(f"""
            QLabel {{
                background-color: rgba(76, 175, 80, {opacity * 0.3});
                border: 2px solid rgba(76, 175, 80, {opacity});
                border-radius: 25px;
                color: rgba(76, 175, 80, {opacity});
            }}
        """)
    
    def animate_message_change(self):
        """메시지 변경 시 애니메이션"""
        # 간단한 투명도 변화로 부드러운 전환 효과
        effect = QGraphicsOpacityEffect()
        effect.setOpacity(0.7)
        self.status_label.setGraphicsEffect(effect)
        
        # 0.2초 후 원래대로
        QTimer.singleShot(200, lambda: self.status_label.setGraphicsEffect(None))
    
    def set_status(self, message):
        """상태 메시지 직접 설정 (외부에서 호출)"""
        self.status_label.setText(message)
        self.animate_message_change()
    
    def set_progress_text(self, text):
        """진행률 텍스트 설정"""
        # 프로그레스 바 위에 별도 라벨로 표시
        if not hasattr(self, 'progress_text_label'):
            self.progress_text_label = QLabel()
            self.progress_text_label.setAlignment(Qt.AlignCenter)
            self.progress_text_label.setFont(QFont("Arial", 9))
            self.progress_text_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            # 프로그레스 바 아래에 추가
            self.layout().itemAt(0).widget().layout().insertWidget(-1, self.progress_text_label)
        
        self.progress_text_label.setText(text)
    
    def show(self):
        """표시 - 페이드 인 효과와 함께"""
        super().show()
        self.start_animation()
        self.raise_()
        self.activateWindow()
        
        # 부드러운 페이드 인 효과
        self.fade_in_effect = QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.fade_in_effect)
        
        self.fade_in_animation = QPropertyAnimation(self.fade_in_effect, b"opacity")
        self.fade_in_animation.setDuration(1000)
        self.fade_in_animation.setStartValue(0.0)
        self.fade_in_animation.setEndValue(1.0)
        self.fade_in_animation.setEasingCurve(QEasingCurve.OutCubic)
        self.fade_in_animation.start()
    
    def close(self):
        """닫기 - 모든 효과 완전 정리"""
        self.stop_animation()
        
        try:
            # 모든 그래픽 효과 제거
            self.setGraphicsEffect(None)
            if hasattr(self, 'shadow_effect'):
                self.shadow_effect = None
            if hasattr(self, 'fade_in_effect'):
                self.fade_in_effect = None
            
            # 모든 자식 위젯의 효과도 제거
            for child in self.findChildren(QWidget):
                child.setGraphicsEffect(None)
            
            # 페이드 아웃 없이 즉시 닫기 (문제 방지)
            super().close()
            
            # 명시적으로 삭제
            self.deleteLater()
            
        except Exception as e:
            print(f"Splash close error: {e}")
            super().close()

    def center_on_parent(self):
        """부모 창 중앙에 배치"""
        if self.parent():
            parent_geometry = self.parent().geometry()
            x = parent_geometry.x() + (parent_geometry.width() - self.width()) // 2
            y = parent_geometry.y() + (parent_geometry.height() - self.height()) // 2
            self.move(x, y)
        else:
            # 부모가 없으면 화면 중앙에
            screen = QDesktopWidget().screenGeometry()
            x = (screen.width() - self.width()) // 2
            y = (screen.height() - self.height()) // 2
            self.move(x, y)

class SplashScreen(QWidget):
    """자연스러운 페이드 효과가 있는 Splash 화면"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowFlags(
            Qt.Dialog | 
            Qt.WindowStaysOnTopHint | 
            Qt.FramelessWindowHint |
            Qt.WindowSystemMenuHint
        )
        
        self.setWindowModality(Qt.ApplicationModal)
        self.setWindowTitle("연결 중...")
        self.setFixedSize(450, 200)
        
        # 투명도 관련 변수
        self.current_opacity = 0.0
        self.target_opacity = 1.0
        self.is_closing = False
        
        self.setup_ui()
        self.setup_animation()
        self.center_on_parent()
        
        # 초기에는 완전 투명
        self.setWindowOpacity(0.0)
        
    def setup_ui(self):
        """UI 설정 - 배경을 더 부드럽게"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 메인 컨테이너
        self.container = QWidget()
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(30, 25, 30, 25)
        container_layout.setSpacing(20)
        
        # 헤더 영역
        header_layout = QHBoxLayout()
        
        # 아이콘
        self.icon_label = QLabel("📹")
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setFont(QFont("Arial", 24))
        self.icon_label.setFixedSize(50, 50)
        self.icon_label.setStyleSheet("""
            QLabel {
                background-color: rgba(76, 175, 80, 0.15);
                border: 2px solid rgba(76, 175, 80, 0.8);
                border-radius: 25px;
                color: #4CAF50;
            }
        """)
        
        # 타이틀 영역
        title_layout = QVBoxLayout()
        
        self.main_label = QLabel("RTSP 카메라 연결")
        self.main_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.main_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.main_label.setStyleSheet("color: #2E3440; margin: 0px; padding: 0px;")
        
        self.subtitle_label = QLabel("네트워크 연결을 확인하고 있습니다")
        self.subtitle_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.subtitle_label.setFont(QFont("Arial", 10))
        self.subtitle_label.setStyleSheet("color: #5E81AC; margin: 0px; padding: 0px;")
        
        title_layout.addWidget(self.main_label)
        title_layout.addWidget(self.subtitle_label)
        title_layout.setSpacing(2)
        
        header_layout.addWidget(self.icon_label)
        header_layout.addSpacing(15)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        # 상태 메시지
        self.status_label = QLabel("연결 준비 중...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Arial", 11))
        self.status_label.setStyleSheet("""
            color: #4C566A;
            background-color: rgba(136, 192, 208, 0.08);
            padding: 8px 15px;
            border-radius: 12px;
            border: 1px solid rgba(136, 192, 208, 0.2);
        """)
        
        # 프로그레스 영역
        progress_container = QWidget()
        progress_layout = QVBoxLayout(progress_container)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(8)
        
        # 프로그레스 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: rgba(76, 175, 80, 0.15);
                border: none;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(76, 175, 80, 0.8), 
                    stop:0.5 rgba(102, 187, 106, 0.8), 
                    stop:1 rgba(129, 199, 132, 0.8));
                border-radius: 3px;
            }
        """)
        
        # 점 애니메이션
        dots_container = QWidget()
        dots_layout = QHBoxLayout(dots_container)
        dots_layout.setContentsMargins(0, 0, 0, 0)
        dots_layout.setSpacing(8)
        dots_layout.setAlignment(Qt.AlignCenter)
        
        self.dot_labels = []
        for i in range(3):
            dot = QLabel("●")
            dot.setAlignment(Qt.AlignCenter)
            dot.setFont(QFont("Arial", 12))
            dot.setFixedSize(20, 20)
            dot.setStyleSheet("color: rgba(76, 175, 80, 0.3); background-color: transparent;")
            self.dot_labels.append(dot)
            dots_layout.addWidget(dot)
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(dots_container)
        
        # 레이아웃 조립
        container_layout.addLayout(header_layout)
        container_layout.addWidget(self.status_label)
        container_layout.addWidget(progress_container)
        
        main_layout.addWidget(self.container)
        self.setLayout(main_layout)
        
        # 부드러운 배경 (덜 강한 색상)
        self.setStyleSheet("""
            SplashScreen {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 255, 255, 0.98),
                    stop:1 rgba(248, 249, 250, 0.95));
                border: 1px solid rgba(76, 175, 80, 0.2);
                border-radius: 20px;
            }
        """)
    
    def setup_animation(self):
        """애니메이션 설정"""
        # 기존 애니메이션들
        self.message_timer = QTimer()
        self.message_timer.timeout.connect(self.update_status_message)
        
        self.dots_timer = QTimer()
        self.dots_timer.timeout.connect(self.update_dots)
        
        # 부드러운 페이드 애니메이션
        self.fade_timer = QTimer()
        self.fade_timer.timeout.connect(self.update_fade)
        
        self.message_index = 0
        self.dots_index = 0
        
        self.status_messages = [
            "네트워크 인터페이스 확인 중...",
            "RTSP 프로토콜 초기화 중...",
            "카메라 서버에 연결 시도 중...",
            "인증 정보 확인 중...",
            "비디오 스트림 협상 중...",
            "해상도 및 코덱 설정 중...",
            "연결 품질 테스트 중...",
            "스트리밍 준비 완료 중..."
        ]
    
    def start_animation(self):
        """애니메이션 시작"""
        self.message_timer.start(2000)
        self.dots_timer.start(400)
        self.fade_timer.start(16)  # 60fps로 부드러운 페이드
    
    def stop_animation(self):
        """애니메이션 중지"""
        timers = ['message_timer', 'dots_timer', 'fade_timer']
        for timer_name in timers:
            if hasattr(self, timer_name):
                getattr(self, timer_name).stop()
    
    def update_fade(self):
        """부드러운 페이드 업데이트"""
        if self.is_closing:
            # 페이드 아웃
            if self.current_opacity > 0:
                self.current_opacity -= 0.05  # 부드러운 페이드 아웃
                if self.current_opacity <= 0:
                    self.current_opacity = 0
                    self.fade_timer.stop()
                    super().close()  # 완전히 투명해지면 닫기
                self.setWindowOpacity(self.current_opacity)
        else:
            # 페이드 인            
            if self.current_opacity < self.target_opacity:
                self.current_opacity += 0.03  # 부드러운 페이드 인
                if self.current_opacity >= self.target_opacity:
                    self.current_opacity = self.target_opacity
                self.setWindowOpacity(self.current_opacity)
    
    def update_status_message(self):
        """상태 메시지 업데이트"""
        self.status_label.setText(self.status_messages[self.message_index])
        self.message_index = (self.message_index + 1) % len(self.status_messages)
    
    def update_dots(self):
        """점 애니메이션 업데이트"""
        # 모든 점 비활성화
        for dot in self.dot_labels:
            dot.setStyleSheet("color: rgba(76, 175, 80, 0.3); background-color: transparent;")
        
        # 현재 점 활성화
        if self.dots_index < len(self.dot_labels):
            active_dot = self.dot_labels[self.dots_index]
            active_dot.setStyleSheet("""
                color: #4CAF50;
                background-color: rgba(76, 175, 80, 0.1);
                border-radius: 10px;
            """)
        
        self.dots_index = (self.dots_index + 1) % (len(self.dot_labels) + 1)
    
    def show(self):
        """부드러운 페이드 인으로 표시"""
        super().show()
        self.start_animation()
        self.raise_()
        self.activateWindow()
        
        # 즉시 페이드 인 시작
        self.current_opacity = 0.0
        self.target_opacity = 1.0
        self.setWindowOpacity(0.0)
    
    def close(self):
        """닫기 - 모든 효과 완전 정리"""
        self.stop_animation()
        
        try:
            # 모든 그래픽 효과 제거
            self.setGraphicsEffect(None)
            if hasattr(self, 'shadow_effect'):
                self.shadow_effect = None
            if hasattr(self, 'fade_in_effect'):
                self.fade_in_effect = None
            
            # 모든 자식 위젯의 효과도 제거
            for child in self.findChildren(QWidget):
                child.setGraphicsEffect(None)
            
            # 페이드 아웃 없이 즉시 닫기 (문제 방지)
            super().close()
            
            # 명시적으로 삭제
            self.deleteLater()
            
        except Exception as e:
            print(f"Splash close error: {e}")
            super().close()

    
    def set_status(self, message):
        """상태 메시지 설정"""
        self.status_label.setText(message)
    
    def center_on_parent(self):
        """부모 창 중앙에 배치"""
        if self.parent():
            parent_geometry = self.parent().geometry()
            x = parent_geometry.x() + (parent_geometry.width() - self.width()) // 2
            y = parent_geometry.y() + (parent_geometry.height() - self.height()) // 2
            self.move(x, y)


class MainWindow(QMainWindow):
    """메인 윈도우
    
    애플리케이션의 메인 윈도우입니다.
    """

    def __init__(self):
        super().__init__()

        # 특정 버튼만 비활성화
        self.setWindowFlags(
            Qt.Window                           # 기본 윈도우
            | Qt.CustomizeWindowHint            # 사용자 정의 윈도우 힌트 활성화
            # | Qt.WindowTitleHint                # 타이틀바 표시
            # | Qt.WindowSystemMenuHint           # 시스템 메뉴만 표시 (아이콘 더블클릭 메뉴)

            # |Qt.WindowMinimizeButtonHint |    # 최소화 버튼 (주석 처리로 숨김)
            # Qt.WindowMaximizeButtonHint |     # 최대화 버튼 (주석 처리로 숨김)
            # Qt.WindowCloseButtonHint          # 닫기 버튼 (주석 처리로 숨김)
        )


        # =================================================
        # 전체화면 설정 & 화면 중앙에 위치시키기
        screen = QDesktopWidget().screenGeometry()
        self.screen_width = screen.width()
        self.screen_height = screen.height()
        # self.resize(screen.width(), screen.height())
        # window = self.geometry()
        
        # x = (screen.width() - window.width()) // 2
        # y = (screen.height() - window.height()) // 2
        
        # self.move(x, y)
        # =================================================
        

        # 종료 처리를 위한 플래그
        self.is_closing = False

        # 설정 파일 로드
        self.config = self.load_config()
        
        # 쓰레드 초기화
        self.video_thread = None                            # 비디오 스레드
        self.monitor_thread = SystemMonitorThread()         # 시스템 모니터 스레드
        
        # 마지막 네트워크/디스크 사용량 초기화 (네트워크/디스크 속도 계산용 이전 값)
        self.last_net_sent = 0
        self.last_net_recv = 0
        self.last_disk_read = 0
        self.last_disk_write = 0
        
        self.setupUI()
        self.setup_connections()
        
        # 시스템 모니터 시작
        self.monitor_thread.start()
        
        # 🔄 기존 자동 시작 로직 수정
        # RTSP URL과 자동 시작 설정을 모두 확인
        camera_config = self.config.get("camera", {})
        rtsp_url = camera_config.get("rtsp_url", "")
        autostart_streaming = camera_config.get("autostart_streaming", False)
        autostart_recording = camera_config.get("autostart_recording", False)
        self.message1 = camera_config.get("message1", False)

        # 🆕 자동 시작 로직
        if rtsp_url and autostart_streaming:
            print("자동 스트리밍 시작 설정이 활성화됨")
            # QTimer를 사용해서 UI 초기화 완료 후 자동 시작
            QTimer.singleShot(1000, self.auto_start_streaming)
        else:
            print(f"자동 스트리밍 시작 비활성화 - RTSP URL: {bool(rtsp_url)}, 자동시작: {autostart_streaming}")
            self.update_status(f"자동 스트리밍 시작 비활성화 - RTSP URL: {bool(rtsp_url)}, 자동시작: {autostart_streaming}")

        # 종료 시 자동 정리를 위한 등록
        atexit.register(self.cleanup_application)

        atexit.register(self.emergency_cleanup)

    
    def auto_start_streaming(self):
        """자동 스트리밍 시작"""
        try:
            print("자동 스트리밍 시작 중...")
            logger.info("자동 스트리밍 시작 중...")

            camera_config = self.config.get("camera", {})
            autostart_recording = camera_config.get("autostart_recording", False)
            rtsp_url = camera_config.get("rtsp_url", "")

            if rtsp_url == "":
                print("❌[연결실패] - CCTV 주소가 설정되지 않았습니다.")
                self.update_status("❌[연결실패] - CCTV 주소가 설정되지 않았습니다.")
                return

            # 스트리밍 시작
            if self.start_streaming():
                print("자동 스트리밍 시작 성공")
                
                # 스트리밍이 성공적으로 시작되면 녹화 자동 시작 확인
                if autostart_recording:
                    print("자동 녹화 시작 설정이 활성화됨")
                    # 스트리밍이 완전히 시작될 때까지 잠시 대기
                    QTimer.singleShot(1000, self.auto_start_recording)
                else:
                    print("자동 녹화 시작 비활성화")
            else:
                print("자동 스트리밍 시작 실패")
                self.update_status("❌ 자동 스트리밍 시작 실패")
                
        except Exception as e:
            print(f"자동 스트리밍 시작 중 오류: {e}")
            self.update_status(f"자동 스트리밍 시작 중 오류: {e}")
    

    def auto_start_recording(self, attempt=0, max_attempts=10):
        """자동 녹화 시작"""
        try:

            # 디버깅을 위한 로그 추가
            print(f"[DEBUG] video_thread: {self.video_thread}")
            if self.video_thread is not None:
                print(f"[DEBUG] video_thread.isRunning(): {self.video_thread.isRunning()}")

            if self.video_thread and self.video_thread.isRunning() and self.video_thread.is_running:
                print(f"자동 녹화 시작 중...")
                
                if self.video_thread.start_recording():
                    self.record_btn.setText("녹화 종료")
                    self.update_status("자동 녹화 시작됨")

                    # opencv_1_20250627_1.py
                    self.video_widget.set_text(
                        "rec_status",
                        "● REC", 
                        QColor(255, 0, 0),  # 빨간색
                        QFont("Arial", 16, QFont.Bold),
                        VideoWidget.TEXT_TOP_RIGHT
                    )

                    print("자동 녹화 시작 성공")
                else:
                    print("자동 녹화 시작 실패")
                    self.update_status("자동 녹화 시작 실패")
            else:
                print(f"[DEBUG] auto_start_recording(재시도): {attempt}")

                 # 조건이 아직 만족하지 않으면 재시도
                if attempt < max_attempts - 1:
                    # 0.5초 후에 다시 시도
                    QTimer.singleShot(500, lambda: self.auto_start_recording(attempt + 1, max_attempts))
                else:
                    print("자동 녹화 시작 실패 - 스트리밍이 실행되지 않음")
                    self.update_status("자동 녹화 시작 실패 - 스트리밍이 실행되지 않음")
                
        except Exception as e:
            print(f"자동 녹화 시작 중 오류: {e}")
            self.update_status(f"자동 녹화 오류: {str(e)}")


    def cleanup_application(self):
        """애플리케이션 정리 - 안전한 종료 처리"""
        if self.is_closing:
            return
            
        self.is_closing = True
        
        try:
            # 스레드들 강제 종료 설정
            if hasattr(self, 'video_thread') and self.video_thread:
                self.video_thread.force_stop = True
                self.video_thread.is_running = False
                
            if hasattr(self, 'monitor_thread') and self.monitor_thread:
                self.monitor_thread.force_stop = True
                self.monitor_thread.is_running = False
        except:
            pass

    def emergency_cleanup(self):
        """비상 정리 - 최소한의 정리만 수행"""
        try:
            if hasattr(self, 'video_thread') and self.video_thread:
                if self.video_thread.isRunning():
                    self.video_thread.terminate()  # 강제 종료
                    
            if hasattr(self, 'monitor_thread') and self.monitor_thread:
                if self.monitor_thread.isRunning():
                    self.monitor_thread.terminate()  # 강제 종료
        except:
            pass  # 모든 예외 무시
    

    def setupUI(self):
        """UI 설정"""
        self.setWindowTitle("iTlOG NVR System")
        # self.setGeometry(100, 100, 1024, 768)
        # self.setGeometry(100, 100, 800, 600)
        # self.setGeometry(100, 100, 1920, 1080)
        
        # 윈도우 아이콘 설정
        self.setWindowIcon(QIcon('./icons/itlog_ci.png'))  # 파일 경로
        
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 메뉴바
        self.create_menu()
        
        # 비디오 위젯
        self.video_widget = VideoWidget()
        main_layout.addWidget(self.video_widget, 1)
        
        # 컨트롤 패널
        control_panel = QWidget()
        control_layout = QHBoxLayout()
        control_panel.setLayout(control_layout)
        
        # opencv_1_20250707_1.py / 주석처리
        # # FPS 표시
        # self.fps_label = QLabel("FPS: 0.0")
        # control_layout.addWidget(self.fps_label)
        # self.fps_label.setFont(QFont("Arial", 9))

        # 상태 표시
        self.status_label = QLabel("대기 중")
        control_layout.addWidget(self.status_label)
        self.status_label.setFont(QFont("Arial", 9))

        control_layout.addStretch()
        
        # 버튼들
        self.stream_btn = QPushButton("스트리밍 시작")
        self.stream_btn.clicked.connect(self.toggle_streaming)
        control_layout.addWidget(self.stream_btn)
        self.stream_btn.setFont(QFont("Arial", 9))

        self.record_btn = QPushButton("녹화 시작")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setEnabled(False)
        control_layout.addWidget(self.record_btn)
        self.record_btn.setFont(QFont("Arial", 9))

        self.reload_btn = QPushButton("설정 다시 불러오기")
        self.reload_btn.clicked.connect(self.reload_config)
        control_layout.addWidget(self.reload_btn)
        self.reload_btn.setFont(QFont("Arial", 9))
        
        # opencv_1_20250627_2.py
        # PTZ 활성화 체크박스
        self.ptz_checkbox = QCheckBox("PTZ 제어 활성화")
        self.ptz_checkbox.toggled.connect(self.toggle_ptz)
        control_layout.addWidget(self.ptz_checkbox)
        self.ptz_checkbox.setFont(QFont("Arial", 9))

        # opencv_1_20250627_2.py
        # PTZ 오버레이 표시 체크박스
        self.overlay_checkbox = QCheckBox("PTZ 오버레이 표시")
        self.overlay_checkbox.toggled.connect(self.toggle_overlay)
        control_layout.addWidget(self.overlay_checkbox)
        self.overlay_checkbox.setFont(QFont("Arial", 9))

        main_layout.addWidget(control_panel)
        
        # 상태바
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # 시스템 정보 라벨들
        self.cpu_label = QLabel("CPU: 0%")
        self.mem_label = QLabel("MEM: 0%")
        self.temp_label = QLabel("TEMP: 0°C")
        self.io_label = QLabel("IO: R:0MB/s W:0MB/s")
        self.net_label = QLabel("NET: ↓0MB/s ↑0MB/s")

        self.statusBar.addPermanentWidget(self.cpu_label)
        self.statusBar.addPermanentWidget(self.mem_label)
        self.statusBar.addPermanentWidget(self.temp_label)
        self.statusBar.addPermanentWidget(self.io_label)
        self.statusBar.addPermanentWidget(self.net_label)

        # 상태바에 저장소 상태 라벨 추가
        self.storage_label = QLabel("저장소: - GB")
        self.statusBar.addPermanentWidget(self.storage_label)
        
        # 폰트 설정
        self.cpu_label.setFont(QFont("Arial", 9))
        self.mem_label.setFont(QFont("Arial", 9))
        self.temp_label.setFont(QFont("Arial", 9))
        self.io_label.setFont(QFont("Arial", 9))
        self.net_label.setFont(QFont("Arial", 9))
        self.storage_label.setFont(QFont("Arial", 9))
        # self.storage_label.setStyleSheet("color: #2e7d32;")  # 녹색

        # opencv_1_20250627_2.py
        # PTZ 신호 연결
        self.connect_ptz_signals()

    # opencv_1_20250627_2.py
    def toggle_ptz(self, checked):
        """PTZ 제어 토글"""
        self.video_widget.enable_ptz_control(checked, self.overlay_checkbox.isChecked())
        status = "PTZ 제어 활성화" if checked else "PTZ 제어 비활성화"
        self.status_label.setText(status)
        
    # opencv_1_20250627_2.py
    def toggle_overlay(self, checked):
        """PTZ 오버레이 토글"""
        # if self.ptz_checkbox.isChecked():
        self.video_widget.enable_ptz_control(self.ptz_checkbox.isChecked(), checked)
    
    # opencv_1_20250627_2.py
    def connect_ptz_signals(self):
        """PTZ 신호 연결"""
        self.video_widget.ptz_up_clicked.connect(lambda: self.ptz_command("UP"))
        self.video_widget.ptz_down_clicked.connect(lambda: self.ptz_command("DOWN"))
        self.video_widget.ptz_left_clicked.connect(lambda: self.ptz_command("LEFT"))
        self.video_widget.ptz_right_clicked.connect(lambda: self.ptz_command("RIGHT"))
        self.video_widget.ptz_up_left_clicked.connect(lambda: self.ptz_command("UP_LEFT"))
        self.video_widget.ptz_up_right_clicked.connect(lambda: self.ptz_command("UP_RIGHT"))
        self.video_widget.ptz_down_left_clicked.connect(lambda: self.ptz_command("DOWN_LEFT"))
        self.video_widget.ptz_down_right_clicked.connect(lambda: self.ptz_command("DOWN_RIGHT"))
        self.video_widget.ptz_center_clicked.connect(lambda: self.ptz_command("HOME"))
        self.video_widget.ptz_zoom_in_clicked.connect(lambda: self.ptz_command("ZOOM_IN"))
        self.video_widget.ptz_zoom_out_clicked.connect(lambda: self.ptz_command("ZOOM_OUT"))
    
    # opencv_1_20250627_2.py
    def ptz_command(self, command):
        """PTZ 명령 처리"""
        self.status_label.setText(f"PTZ 명령: {command}")
        print(f"PTZ Command: {command}")

        # 실제 PTZ 제어 코드를 여기에 추가
        # 예: self.send_ptz_command_to_camera(command)


    def create_menu(self):
        """메뉴 생성"""
        menubar = self.menuBar()
        
        # 스타일시트 적용
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #2b2b2b;
                color: white;
                border: none;
                padding: 5px 0px;  /* 상하 여백 추가 */
                font-size: 13px;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 4px 8px;  /* 메뉴 항목 패딩 */
                margin: 0px 1px;
                border-radius: 2px;
            }
            QMenuBar::item:selected {
                background-color: #404040;
            }
            QMenuBar::item:pressed {
                background-color: #555555;
            }
        """)

        # # 실시간 메뉴
        # realtime_menu = menubar.addMenu("실시간")
        # # 실시간 메뉴 - 실시간 CCTV
        # realtime_action = QAction("실시간 CCTV", self)
        # realtime_action.triggered.connect(self.show_realtime)
        # realtime_menu.addAction(realtime_action)
        
        # 설정 메뉴
        settings_menu = menubar.addMenu("설정")
        # 설정 메뉴 - 카메라 설정
        camera_action = QAction("카메라 설정", self)
        camera_action.triggered.connect(self.show_camera_settings)
        settings_menu.addAction(camera_action)
        # 설정 메뉴 - 메뉴 키 설정
        key_action = QAction("메뉴 키 설정", self)
        key_action.triggered.connect(self.show_key_settings)
        settings_menu.addAction(key_action)
        # 설정 메뉴 - PTZ 키 설정
        ptz_action = QAction("PTZ 키 설정", self)
        ptz_action.triggered.connect(self.show_ptz_settings)
        settings_menu.addAction(ptz_action)

        # 2025.06.25_2
        # 백업 메뉴
        settings_menu = menubar.addMenu("백업")
        # 백업 메뉴 - CCTV 데이터백업
        databackup_set_action = QAction("데이터백업 설정", self)
        databackup_set_action.triggered.connect(self.show_backup_set_dialog)
        settings_menu.addAction(databackup_set_action)
        # 백업 메뉴 - CCTV 데이터백업
        databackup_action = QAction("CCTV 데이터백업", self)
        databackup_action.triggered.connect(self.show_backup_dialog)
        settings_menu.addAction(databackup_action)

        # opencv_1_20250630_1.py
        # 정보 메뉴
        settings_menu = menubar.addMenu("정보")
        # 프로그램 정보
        about_action = QAction("프로그램 정보", self)
        about_action.triggered.connect(self.show_about)
        settings_menu.addAction(about_action)
        # 정보 메뉴 - 종료
        exit_action = QAction("프로그램 종료", self)
        exit_action.triggered.connect(self.exit_application)
        settings_menu.addAction(exit_action)


    def show_about(self):
        """프로그램 정보 표시 - 크기 조절 가능한 버전"""
        
        # 방법 1: QMessageBox 객체 직접 생성 후 크기 설정
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("itlog NVR 시스템 정보")
        msg_box.setText(
            "itlog NVR 시스템\n"
            "버전: 1.0.0\n"
            "개발: itlog\n"
            "날짜: 2025.06.30"
        )
        msg_box.setIcon(QMessageBox.Information)
        
        # 크기 설정
        # msg_box.resize(400, 200)  # 너비 400, 높이 200
        # 또는 최소 크기 설정
        msg_box.setMinimumSize(400, 200)
        
        msg_box.exec_()


    def exit_application(self):
        """메뉴에서 프로그램 종료"""
        # 사용자에게 종료 확인
        reply = QMessageBox.question(
            self, 
            "프로그램 종료", 
            "정말로 프로그램을 종료하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # closeEvent()와 동일한 방식으로 종료 처리
            self.close()  # 이렇게 하면 closeEvent()가 호출됩니다


    def setup_connections(self):
        """시그널 연결
    
        스레드에서 발생하는 시그널을 메인 윈도우의 슬롯에 연결합니다.
        """

        # 모니터링 스레드 시그널만 연결 (video_thread는 별도 처리)
        self.monitor_thread.update_signal.connect(self.update_system_info)

        
    def load_config(self):
        """설정 불러오기"""
        
        # print(f"Looking for config file at: {os.path.abspath(CONFIG_FILE)}")
        # print(f"Current working directory: {os.getcwd()}")
        # print(f"File exists: {os.path.exists(CONFIG_FILE)}")
        # print(f"File attributes: {os.stat(CONFIG_FILE) if os.path.exists(CONFIG_FILE) else 'File not found'}")

        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return DEFAULT_CONFIG.copy()
        
    
    def show_realtime(self):
        """실시간 화면 표시 (현재 화면)"""
        pass
        
    def show_camera_settings(self):
        """카메라 설정 표시"""
        dialog = CameraSettingsDialog(self.config, self)
        if dialog.exec_():
            self.config = self.load_config()

            self.stop_streaming()
            self.auto_start_streaming()

            QMessageBox.information(self, "알림", "설정을 다시 불러왔습니다.")
            
    def show_key_settings(self):
        """키 설정 표시"""
        dialog = KeySettingsDialog(self)
        dialog.show()
        
    def show_ptz_settings(self):
        """PTZ 설정 표시"""
        dialog = PTZSettingsDialog(self)
        dialog.show()
        
    # opencv_1_20250701_1.py
    def show_backup_set_dialog(self):
        """백업 다이얼로그 표시 (메인 NVR 프로그램에서 호출)"""
        try:
            dialog = CCTVBackupSettingsDialog(self)

            return dialog.exec_()
        except Exception as e:
            logger.error(f"백업 다이얼로그 실행 오류: {str(e)}")
            QMessageBox.critical(self, "오류", f"백업 기능을 실행할 수 없습니다:\n{str(e)}")
            
            return False
    # 2025.06.25_2
    def show_backup_dialog(self):
        """백업 다이얼로그 표시 (메인 NVR 프로그램에서 호출)"""
        try:
            # opencv_1_20250702_2.py
            # 라즈베리파이 터치 LCD(800x480) 경우, 백업화면 축소관련
            if (self.screen_height < 600):
                dialog = CCTVBackupExecuteDialog(self, ui_idx = 1)           # 축소표시
            else:
                dialog = CCTVBackupExecuteDialog(self, ui_idx = 0)           # 전체 표시

            return dialog.exec_()
        except Exception as e:
            logger.error(f"백업 다이얼로그 실행 오류: {str(e)}")
            QMessageBox.critical(self, "오류", f"백업 기능을 실행할 수 없습니다:\n{str(e)}")
            
            return False
            

    def create_video_thread(self):
        """새로운 비디오 스레드 생성"""
        try:
            if self.video_thread is not None:
                # 기존 스레드가 있다면 완전히 정리
                self.cleanup_video_thread()
                
            # 새로운 스레드 생성
            self.video_thread = VideoThread()

            # 시그널 연결
            self.setup_video_connections()
            print("새로운 VideoThread 생성 완료")
            return True
            
        except Exception as e:
            print(f"VideoThread 생성 실패: {e}")
            self.video_thread = None
            return False
        
    def setup_video_connections(self):
        """비디오 스레드 시그널 연결"""
        if self.video_thread is None:  # None 체크 추가
            print("video_thread가 None입니다. 시그널 연결을 건너뜁니다.")
            return
            
        try:
            # 기존 연결이 있다면 먼저 해제
            self.disconnect_video_signals()
            
            # 새로운 연결 설정
            self.video_thread.frame_ready.connect(self.video_widget.update_frame)
            self.video_thread.fps_update.connect(self.update_fps)
            self.video_thread.status_update.connect(self.update_status)
            self.video_thread.storage_warning.connect(self.update_storage_status)
            
            print("비디오 시그널 연결 완료")
            
        except Exception as e:
            print(f"비디오 시그널 연결 실패: {e}")
            
    def disconnect_video_signals(self):
        """비디오 스레드 시그널 연결 해제"""
        if self.video_thread is None:
            return
            
        try:
            self.video_thread.frame_ready.disconnect()
            self.video_thread.fps_update.disconnect()
            self.video_thread.status_update.disconnect()
            self.video_thread.storage_warning.disconnect()
        except:
            pass  # 연결이 없는 경우 예외 무시

    def cleanup_video_thread(self):
        """비디오 스레드 완전 정리"""
        if self.video_thread is None:
            return
            
        try:
            print("비디오 스레드 정리 시작")
            
            # 시그널 연결 해제
            self.disconnect_video_signals()
                
            # 스레드 종료
            if self.video_thread.isRunning():
                self.video_thread.stop()
                self.video_thread.wait(5000)  # 5초 대기
                
            # 강제 종료
            if self.video_thread.isRunning():
                self.video_thread.terminate()
                self.video_thread.wait(2000)
                
            self.video_thread = None
            print("비디오 스레드 정리 완료")
            
        except Exception as e:
            print(f"비디오 스레드 정리 중 오류: {e}")
            self.video_thread = None

    def toggle_streaming(self):
        """스트리밍 토글"""

        if self.is_closing:
            return

        try:
            if self.video_thread and self.video_thread.isRunning():
                self.stop_streaming()
            else:
                self.start_streaming()
        except Exception as e:
            print(f"스트리밍 토글 오류: {e}")
            QMessageBox.critical(self, "오류", f"스트리밍 상태를 변경할 수 없습니다: {e}")

    def start_streaming(self):
        """스트리밍 시작"""

        if self.is_closing:
            return False

        # 1. Splash 화면 표시
        splash = SplashScreen(self)
        splash.show()        
        
        # UI 업데이트
        QApplication.processEvents()

        try:
            camera_config = self.config.get("camera", {})
            rtsp_url = camera_config.get("rtsp_url", "")
            self.message1 = camera_config.get("message1", False)

            # 1. RTSP URL 유무 확인
            if not rtsp_url:
                # QMessageBox.warning(self, "오류", "[❌[연결실패] - CCTV 주소가 설정되지 않았습니다.")
                print("[❌[연결실패] - CCTV 주소가 설정되지 않았습니다")
                self.update_status("[❌[연결실패] - CCTV 주소가 설정되지 않았습니다")
                return False

            # 2. RTSP URL 유효성 검사
            if not (rtsp_url.startswith('rtsp://') or rtsp_url.startswith('rtsps://')):
                # QMessageBox.warning(self, f"❌[연결실패] - 잘못된 RTSP URL 형식입니다: {rtsp_url}")
                print(f"❌[연결실패] - 잘못된 RTSP URL 형식입니다: {rtsp_url}")
                self.update_status(f"❌[연결실패] - 잘못된 RTSP URL 형식입니다: {rtsp_url}")
                return False

            # 3. 네트워크 인터페이스 상태 확인
            try:
                # Windows인 경우
                if platform.system().lower() == "windows":
                    result = subprocess.run(
                        ["ipconfig"], 
                        capture_output=True, 
                        text=True, 
                        timeout=5
                    )
                    if "Media disconnected" in result.stdout:
                        # QMessageBox.warning(self, "오류", "❌네트워크 케이블이 연결되어 있지 않습니다.")
                        print("❌네트워크 케이블이 연결되어 있지 않습니다.")
                        self.update_status("❌네트워크 케이블이 연결되어 있지 않습니다.")
                        return False
                # # Linux/Mac인 경우
                # else:
                #     result = subprocess.run(
                #         ["ip", "addr", "show"], 
                #         capture_output=True, 
                #         text=True, 
                #         timeout=5
                #     )
                #     if "state DOWN" in result.stdout:
                #         # QMessageBox.warning(self, "오류", "❌네트워크 인터페이스가 비활성화되었습니다.")
                #         print("❌네트워크 인터페이스가 비활성화되었습니다.")
                #         self.update_status("❌네트워크 인터페이스가 비활성화되었습니다.")
                #         return False
            except Exception as e:
                print(f"네트워크 인터페이스 확인 중 오류: {e}")

            # 3. RTSP 카메라 연결 테스트 (비동기로 처리)
            try:
                # 5초 타임아웃으로 연결 테스트
                success, message = self.check_camera_connection(rtsp_url, timeout=5)
                
                if not success:
                    # UI 업데이트는 메인 스레드에서 처리
                    QMetaObject.invokeMethod(
                        self,
                        "show_error_message",
                        Qt.QueuedConnection,
                        Q_ARG(str, "오류"),
                        Q_ARG(str, f"❌[연결실패] - {message}")
                    )
                    print(f"❌[연결실패] - {message}")
                    return False
                    
                print("✅ 카메라 연결 테스트 성공")
                
            except Exception as e:
                QMetaObject.invokeMethod(
                    self,
                    "show_error_message",
                    Qt.QueuedConnection,
                    Q_ARG(str, "오류"),
                    Q_ARG(str, f"❌[연결 오류] - {str(e)}")
                )
                print(f"❌[연결 오류] - {str(e)}")
                return False


            # 카메라 메시지 표시
            self.video_widget.set_text(
                "test_alert",
                self.message1, 
                QColor(255, 0, 0),  # 빨간색
                QFont("Arial", 40, QFont.Bold),
                VideoWidget.TEXT_CENTER )

            print("스트리밍 시작 요청")

            # 새로운 스레드 생성
            if not self.create_video_thread():
                # QMessageBox.warning(self, "오류", "비디오 스레드를 생성할 수 없습니다.")
                print("비디오 스레드를 생성할 수 없습니다.")
                self.update_status("비디오 스레드를 생성할 수 없습니다.")
                return False
                
            # 설정 적용
            self.video_thread.set_config(self.config)

            # 스레드 시작
            self.video_thread.start()
            
            # UI 업데이트
            self.stream_btn.setText("스트리밍 종료")
            self.record_btn.setEnabled(True)
            
            # opencv_1_20250627_1.py
            self.video_widget.set_text(
                "camera_status",
                "● LIVE", 
                QColor(0, 255, 0),  # 녹색
                QFont("Arial", 16, QFont.Bold),
                VideoWidget.TEXT_TOP_LEFT
            )

            print("스트리밍 시작 완료")
            return True
            
        except Exception as e:
            print(f"스트리밍 시작 실패: {e}")
            # QMessageBox.critical(self, "오류", f"스트리밍을 시작할 수 없습니다: {e}")
            self.update_status(f"스트리밍을 시작할 수 없습니다: {e}")
            return False
        finally:
            splash.close()
        
    @pyqtSlot(str, str)
    def show_error_message(self, title, message):
        """메인 스레드에서 에러 메시지 표시"""
        QMessageBox.warning(self, title, message)

    def check_camera_connection(self, rtsp_url, timeout=10):
        """UI 업데이트를 유지하면서 카메라 연결 확인"""
        
        import queue
        from threading import Thread
        
        result_queue = queue.Queue()
        
        def _check():
            """백그라운드에서 실제 연결 테스트"""
            try:
                import cv2
                
                cap = cv2.VideoCapture(rtsp_url)
                
                if not cap.isOpened():
                    result_queue.put((False, "카메라에 연결할 수 없습니다."))
                    return
                
                # 프레임 읽기 시도
                ret, frame = cap.read()
                cap.release()
                
                if not ret or frame is None:
                    result_queue.put((False, "카메라에서 프레임을 받아올 수 없습니다."))
                else:
                    result_queue.put((True, "카메라 연결 테스트 성공"))
                    
            except Exception as e:
                result_queue.put((False, f"카메라 연결 오류: {str(e)}"))
        
        # 백그라운드 스레드 시작
        thread = Thread(target=_check, daemon=True)
        thread.start()
        
        # 결과를 기다리면서 UI 업데이트 유지
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # 0.05초마다 결과 확인
                result = result_queue.get(timeout=0.05)
                return result
            except queue.Empty:
                # 결과가 아직 없으면 UI 업데이트
                QApplication.processEvents()
                continue
        
        # 타임아웃
        return (False, "카메라 연결 시간 초과")


    def stop_streaming(self):
        """스트리밍 종료"""
        
        try:
            if self.video_thread and self.video_thread.isRunning():
                # 녹화 중이면 먼저 중지
                if self.video_thread.is_recording:
                    self.video_thread.stop_recording()
                    
                # 스레드 종료
                self.video_thread.stop()
                
                # 완전 정리
                self.cleanup_video_thread()
            
            # UI 업데이트
            self.stream_btn.setText("스트리밍 시작")
            self.record_btn.setEnabled(False)
            self.record_btn.setText("녹화 시작")
            
            # 화면 초기화
            self.video_widget.image = None
            self.video_widget.update()
            
            # opencv_1_20250627_1.py
            self.video_widget.set_text(
                "camera_status",
                "", 
                QColor(0, 255, 0),  # 녹색
                QFont("Arial", 16, QFont.Bold),
                VideoWidget.TEXT_TOP_LEFT
            )

            # opencv_1_20250627_1.py
            self.video_widget.set_text(
                "rec_status",
                "", 
                QColor(255, 0, 0),  # 빨간색
                QFont("Arial", 16, QFont.Bold),
                VideoWidget.TEXT_TOP_RIGHT
            )

            print("스트리밍 종료 완료")
            
        except Exception as e:
            print(f"스트리밍 종료 실패: {e}")
        
    def toggle_recording(self):
        """녹화 토글"""

        if self.is_closing:
            return

        # video_thread가 None인지 확인
        if not self.video_thread or not self.video_thread.isRunning():
            QMessageBox.warning(self, "경고", "스트리밍이 시작되지 않았습니다.")
            return

        try:
            if self.video_thread.is_recording:
                self.video_thread.stop_recording()
                self.record_btn.setText("녹화 시작")

                # opencv_1_20250627_1.py
                self.video_widget.set_text(
                    "rec_status",
                    "", 
                    QColor(255, 0, 0),  # 빨간색
                    QFont("Arial", 16, QFont.Bold),
                    VideoWidget.TEXT_TOP_RIGHT
                )

                self.update_status("녹화 종료")
            else:
                if self.video_thread.start_recording():
                    self.record_btn.setText("녹화 종료")

                    # opencv_1_20250627_1.py
                    self.video_widget.set_text(
                        "rec_status",
                        "● REC", 
                        QColor(255, 0, 0),  # 빨간색
                        QFont("Arial", 16, QFont.Bold),
                        VideoWidget.TEXT_TOP_RIGHT
                    )

                    self.update_status("녹화 중")
                else:
                    QMessageBox.warning(self, "경고", "녹화를 시작할 수 없습니다.")
        except Exception as e:
            print(f"녹화 토글 오류: {e}")
            QMessageBox.critical(self, "오류", f"녹화 상태를 변경할 수 없습니다: {e}")
                
    
    def reload_config(self):
        """설정 다시 불러오기"""

        # if self.is_closing:
        #     return

        # self.config = self.load_config()
        # if self.video_thread.isRunning():
        #     self.stop_streaming()

        #     # opencv_1_20250630_1.py
        #     # self.start_streaming()
        #     self.auto_start_streaming()
        # else:
        #     self.auto_start_streaming()
        self.stop_streaming()
        self.auto_start_streaming()

        QMessageBox.information(self, "알림", "설정을 다시 불러왔습니다.")
        
    
    def update_fps(self, fps):
        """FPS 업데이트"""

        if not self.is_closing:
            # opencv_1_20250707_1.py
            # self.fps_label.setText(f"FPS: {fps:.1f}")
            self.video_widget.set_text(
                "fps",
                f"FPS: {fps:.1f}", 
                QColor(255, 0, 0),  # 빨간색
                QFont("Arial", 16, QFont.Bold),
                VideoWidget.TEXT_BOTTOM_LEFT
            )
        
    def update_status(self, status):
        """상태 업데이트"""

        if not self.is_closing:
            self.status_label.setText(status)
        
    def update_system_info(self, info):
        """시스템 정보 업데이트"""

        if self.is_closing:
            return

        try:
            self.cpu_label.setText(f"CPU: {info['cpu']:.1f}%")
            self.mem_label.setText(f"MEM: {info['memory']:.1f}%")
            self.temp_label.setText(f"TEMP: {info['temperature']:.1f}°C")
            
            # 네트워크 속도 계산 (MB/s)
            if self.last_net_sent > 0:
                net_sent_speed = (info['net_sent'] - self.last_net_sent) / 1024 / 1024
                net_recv_speed = (info['net_recv'] - self.last_net_recv) / 1024 / 1024
                self.net_label.setText(f"NET: ↓{net_recv_speed:.1f}MB/s ↑{net_sent_speed:.1f}MB/s")
                
            # 디스크 I/O 속도 계산 (MB/s)
            if self.last_disk_read > 0:
                disk_read_speed = (info['disk_read'] - self.last_disk_read) / 1024 / 1024
                disk_write_speed = (info['disk_write'] - self.last_disk_write) / 1024 / 1024
                self.io_label.setText(f"IO: R:{disk_read_speed:.1f}MB/s W:{disk_write_speed:.1f}MB/s")
                
            self.last_net_sent = info['net_sent']
            self.last_net_recv = info['net_recv']
            self.last_disk_read = info['disk_read']
            self.last_disk_write = info['disk_write']
        except Exception as e:
            if not self.is_closing:
                print(f"System info update error: {e}")

    def update_storage_status(self, message):
        """저장소 상태 업데이트"""
        if not self.is_closing:
            self.storage_label.setText(message)
            
            # 경고나 오류 메시지인 경우 색상 변경
            if "⚠️" in message:
                self.storage_label.setStyleSheet("color: orange; font-weight: bold;")
            elif "❌" in message:
                self.storage_label.setStyleSheet("color: red; font-weight: bold;")
            elif "✓" in message:
                self.storage_label.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.storage_label.setStyleSheet("")  # 기본 스타일

    def closeEvent(self, event):
        """종료 이벤트 - 개선된 종료 처리"""
        if self.is_closing:
            event.accept()
            return
            
        self.is_closing = True
        
        # 진행 다이얼로그 표시
        progress = QProgressDialog("프로그램을 종료하는 중...", None, 0, 100, self)
        progress.setWindowTitle("종료 중")
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)  # 취소 버튼 제거
        progress.show()
        
        try:
            # 1단계: 비디오 스레드 종료
            progress.setValue(20)
            progress.setLabelText("비디오 스트리밍 종료 중...")
            QApplication.processEvents()
            
            if hasattr(self, 'video_thread') and self.video_thread:
                self.video_thread.stop()
                
            # 2단계: 모니터 스레드 종료
            progress.setValue(40)
            progress.setLabelText("시스템 모니터링 종료 중...")
            QApplication.processEvents()
            
            if hasattr(self, 'monitor_thread') and self.monitor_thread:
                self.monitor_thread.stop()
                
            # 3단계: 리소스 정리
            progress.setValue(60)
            progress.setLabelText("리소스 정리 중...")
            QApplication.processEvents()
            
            self.cleanup_application()
            
            # 4단계: 스레드 종료 대기
            progress.setValue(80)
            progress.setLabelText("스레드 종료 대기 중...")
            QApplication.processEvents()
            
            # 추가 대기 시간
            time.sleep(0.5)
            
            progress.setValue(100)
            progress.setLabelText("종료 완료")
            QApplication.processEvents()
            
        except Exception as e:
            # 종료 시점의 예외는 로그만 남기고 무시
            print(f"Close event warning: {e}")
        finally:
            try:
                progress.close()
            except:
                pass
            event.accept()

def setup_dark_theme(app):
    """어플리케이션에 다크 테마를 적용하는 함수
    
    Args:
        app (QApplication): 테마를 적용할 QApplication 인스턴스
    """
    # Fusion 스타일 사용 (크로스 플랫폼에서 일관된 모양 제공)
    app.setStyle('Fusion')
    
    # 다크 팔레트 생성
    dark_palette = QPalette()
    
    # 기본 색상 설정
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))          # 창 배경색
    dark_palette.setColor(QPalette.WindowText, Qt.white)                # 창 텍스트 색상
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))            # 입력 위젯 배경색
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))   # 테이블 행 교대 색상
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)               # 툴팁 배경색
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)               # 툴팁 텍스트 색상
    dark_palette.setColor(QPalette.Text, Qt.white)                      # 텍스트 색상
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))          # 버튼 배경색
    dark_palette.setColor(QPalette.ButtonText, Qt.white)                # 버튼 텍스트 색상
    dark_palette.setColor(QPalette.BrightText, Qt.red)                  # 강조 텍스트 색상
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))          # 링크 색상
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))     # 선택 항목 배경색
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)           # 선택 항목 텍스트 색상
    
    # 팔레트 적용
    app.setPalette(dark_palette)

def main():
    """메인 함수"""

    app = None
    window = None
    
    try:
        app = QApplication(sys.argv)            # Qt 애플리케이션 생성
        
        # width, height = show_quick_resolution()
        # print(f"해상도: {width}x{height}")

        # 다크 테마 적용
        setup_dark_theme(app)
        
        # 2025.06.24 Err_close_threading
        # 전역 예외 핸들러 설정 (Qt 예외 처리)
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            

            # 종료 관련 예외들을 더 포괄적으로 무시
            ignore_messages = [
                "DeleteDummyThreadOnDel",
                "NoneType",
                "context manager protocol",
                "_shutdown",
                "weakref"
            ]
            
            error_str = str(exc_value)
            if any(msg in error_str for msg in ignore_messages):
                return  # 무시
                
            # 실제 중요한 예외만 출력
            print(f"Uncaught exception: {exc_type.__name__}: {exc_value}")

            
        sys.excepthook = handle_exception


        window = MainWindow()       # 메인 윈도우 생성

        # 최대화된 상태로 시작
        # window.showMaximized()
        window.showFullScreen()   # 풀스크린 모드
        #window.show()               # 윈도우 표시
        
        # 이 시점부터 프로그램은 이벤트 기반으로 동작합니다
        # 애플리케이션 실행
        exit_code = app.exec_()
        
        # 명시적 정리
        if window:
            window.cleanup_application()
            
        # Qt 애플리케이션 종료
        if app:
            app.quit()
            
        return exit_code

    except Exception as e:
        print(f"Application error: {e}")
        return 1
    finally:
        # 최종 정리
        try:
            if window and hasattr(window, 'cleanup_application'):
                window.cleanup_application()
        except:
            pass
            
        try:
            if app:
                app.quit()
        except:
            pass

if __name__ == "__main__":
    # 시그널 핸들러 설정 (Ctrl+C 처리)
    import signal                           # 시그널 처리를 위한 표준 라이브러리 임포트
    
    def signal_handler(sig, frame):
        """SIGINT 시그널(주로 Ctrl+C)을 처리하는 핸들러 함수"""
        print("\n프로그램을 종료합니다...")     # 사용자에게 종료 메시지 출력

        # 전역 변수로 선언된 'app' 객체가 존재하면
        if 'app' in globals():
            QApplication.quit()             # Qt 애플리케이션 종료 요청
        
        sys.exit(0)                         # 프로그램 안전 종료 (0은 정상 종료를 의미)
    

    # SIGINT 시그널(키보드 인터럽트)에 대한 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    
    # 메인 함수 실행
    exit_code = main()
    sys.exit(exit_code)  # 종료 시에는 모든 예외 무시