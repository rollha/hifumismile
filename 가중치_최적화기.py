#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
가중치 최적화기 v1.0
Weight Optimizer

이 애플리케이션은 이미지 생성 AI의 태그 가중치를
시각적으로 비교하고 최적화하는 데 도움을 주는 도구입니다.

기능:
1. 범위 설정: 여러 태그와 가중치 범위를 설정하여 모든 조합을 텍스트 파일로 생성합니다.
2. 이미지 태깅: 생성된 조합 파일을 기반으로, 폴더 내의 모든 이미지에 대해 태그 파일을 자동으로 생성합니다.
3. 2D 이미지 뷰어: 두 개의 태그를 X, Y축으로 설정하고, 가중치에 따라 이미지를 2D 그리드에 배열하여 시각적으로 비교합니다.

License: MIT License
Version: 1.0
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from decimal import Decimal, getcontext, InvalidOperation
import itertools
import json
import os
import re
import sys
from typing import List, Dict, Tuple, Optional, Set
from PIL import Image, ImageTk

# --- 상수 정의 ---
APP_VERSION = "v1.0"
SETTINGS_FILE = "settings.json"
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
PRECISION = 10
MAX_COMBINATIONS_WARNING = 1000000  # 100만개 조합 경고 임계값
MAX_THUMBNAIL_SIZE = 512
MIN_THUMBNAIL_SIZE = 20
DEFAULT_THUMBNAIL_SIZE = 128

# 부동소수점 정밀도 설정
getcontext().prec = PRECISION


# --- 유틸리티 함수들 ---
def parse_tags(content: str) -> Dict[str, Decimal]:
    """텍스트 파일 내용에서 태그와 가중치를 파싱합니다.
    
    Args:
        content: 파싱할 텍스트 내용
        
    Returns:
        태그명을 키로, 가중치를 값으로 하는 딕셔너리
    """
    tags = {}
    # 새로운 형식: :seq, 가중치::태그 ::, 가중치::태그 ::
    pattern = re.compile(r"(?:seq,\s*)?([\d.-]+)::([\w\s():-]*?)\s*::")
    matches = pattern.findall(content)
    
    for weight_str, tag_str in matches:
        try:
            weight = Decimal(weight_str.strip())
            tag = tag_str.strip()
            if tag:  # 빈 태그는 제외
                tags[tag] = weight
        except InvalidOperation:
            continue
    return tags


def validate_numeric_input(value_str: str, field_name: str = "값") -> Tuple[Optional[Decimal], Optional[str]]:
    """숫자 입력값의 유효성을 검사합니다.
    
    Args:
        value_str: 검사할 문자열 값
        field_name: 필드 이름 (오류 메시지용)
        
    Returns:
        (값, 오류메시지) 튜플. 오류가 없으면 (값, None)
    """
    if not value_str.strip():
        return None, f"{field_name}을 입력해주세요."
    
    try:
        value = Decimal(value_str.strip())
        return value, None
    except InvalidOperation:
        return None, f"{field_name}이 숫자 형식이 아닙니다."


def get_image_files(folder_path: str) -> List[str]:
    """폴더에서 이미지 파일들을 찾아 반환합니다.
    
    Args:
        folder_path: 검색할 폴더 경로
        
    Returns:
        이미지 파일 경로 리스트 (생성 시간순 정렬)
        
    Raises:
        IOError: 폴더 읽기 실패 시
    """
    try:
        if not os.path.exists(folder_path):
            raise IOError("폴더가 존재하지 않습니다.")
        
        all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
        image_files = [f for f in all_files if os.path.isfile(f) and f.lower().endswith(IMAGE_EXTENSIONS)]
        
        # 파일 크기와 생성 시간을 고려한 정렬
        image_files.sort(key=lambda x: (os.path.getctime(x), os.path.getsize(x)))
        return image_files
    except (IOError, OSError) as e:
        raise IOError(f"폴더 읽기 오류: {e}")


def open_image_cross_platform(path: str) -> None:
    """크로스 플랫폼 이미지 열기 함수 - 보안 강화 버전
    
    Args:
        path: 열 이미지 파일 경로
        
    Raises:
        IOError: 이미지 열기 실패 시
    """
    try:
        # 경로 유효성 검사
        if not os.path.exists(path):
            raise IOError("파일이 존재하지 않습니다.")
        
        # 파일 확장자 검사
        if not path.lower().endswith(IMAGE_EXTENSIONS):
            raise IOError("지원되지 않는 파일 형식입니다.")
        
        # 절대 경로로 변환하여 보안 강화
        abs_path = os.path.abspath(path)
        
        if os.name == 'nt':  # Windows
            # Windows에서는 os.startfile 대신 더 안전한 방법 사용
            try:
                import webbrowser
                # 로컬 파일을 file:// 프로토콜로 열기
                webbrowser.open(f"file://{abs_path}")
            except:
                # webbrowser가 실패하면 기본 방법 사용
                os.startfile(abs_path)
        elif os.name == 'posix':  # macOS, Linux
            import subprocess
            import shutil
            
            # 명령어 존재 여부 확인
            if os.uname().sysname == 'Darwin':  # macOS
                cmd = 'open'
            else:  # Linux
                cmd = 'xdg-open'
                if not shutil.which(cmd):
                    raise OSError(f"{cmd} 명령어를 찾을 수 없습니다.")
            
            # 안전한 subprocess 실행
            result = subprocess.run([cmd, abs_path], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            
            if result.returncode != 0:
                raise IOError(f"이미지 열기 실패: {result.stderr}")
        else:
            raise OSError("지원되지 않는 운영체제입니다.")
            
    except subprocess.TimeoutExpired:
        raise IOError("이미지 열기 시간 초과")
    except Exception as e:
        raise IOError(f"이미지를 여는 데 실패했습니다: {e}")


def calculate_total_combinations(steps_list: List[List[Decimal]]) -> int:
    """총 조합 수를 계산합니다.
    
    Args:
        steps_list: 각 태그별 스텝 값들의 리스트
        
    Returns:
        총 조합 수
    """
    total = 1
    for steps in steps_list:
        total *= len(steps)
    return total


class CombinatorTab(ttk.Frame):
    """첫 번째 탭: 태그와 가중치 범위를 설정하여 조합 파일을 생성합니다."""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.pack(expand=True, fill="both")
        self.word_rows: List[Dict] = []
        self._build_ui()
        self.load_settings()

    def _build_ui(self):
        """UI 위젯을 생성하고 배치합니다."""
        # --- 설명 ---
        desc_frame = ttk.Frame(self)
        desc_frame.pack(fill="x", padx=10, pady=(10, 5))
        desc_label = ttk.Label(desc_frame,
                               text="입력한 태그와 범위대로 한줄씩 조합을 생성하여 txt파일로 저장합니다.",
                               wraplength=700,
                               justify="center")
        desc_label.pack(fill="x")

        # --- 단어 목록 프레임 (스크롤 가능) ---
        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=5)
        canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 헤더 추가
        header_frame = ttk.Frame(self.scrollable_frame)
        header_frame.pack(fill="x", padx=5, pady=(0,5))
        ttk.Label(header_frame, text="태그", width=15).pack(side="left", padx=5)
        ttk.Label(header_frame, text="최솟값", width=10).pack(side="left", padx=5)
        ttk.Label(header_frame, text="최댓값", width=10).pack(side="left", padx=5)
        ttk.Label(header_frame, text="스텝", width=10).pack(side="left", padx=5)
        ttk.Label(header_frame, text="계산 결과", width=25).pack(side="left", padx=5)
        ttk.Label(header_frame, text="삭제", width=5).pack(side="left", padx=5)

        # --- 하단 컨트롤 프레임 ---
        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(fill="x", padx=10, pady=10)
        ttk.Label(bottom_frame, text="태그 추가:").pack(side="left", padx=(0, 5))
        self.new_word_entry = ttk.Entry(bottom_frame)
        self.new_word_entry.pack(side="left", fill="x", expand=True)
        self.new_word_entry.bind("<Return>", self.add_word_event)
        
        # 우클릭 메뉴 추가
        self.context_menu = tk.Menu(self, tearoff=0)
        self.context_menu.add_command(label="모든 태그 삭제", command=self.clear_all_tags)
        self.context_menu.add_command(label="설정 저장", command=self.save_settings)
        self.context_menu.add_command(label="설정 불러오기", command=self.load_settings)
        
        ttk.Button(bottom_frame, text="조합 txt 파일 생성", command=self.generate_file).pack(side="right")

    def clear_all_tags(self):
        """모든 태그를 삭제합니다."""
        if not self.word_rows:
            return
            
        if messagebox.askyesno("확인", "모든 태그를 삭제하시겠습니까?"):
            self.word_rows.clear()
            for child in self.scrollable_frame.winfo_children():
                if isinstance(child, ttk.Frame) and child != self.scrollable_frame.winfo_children()[0]:  # 헤더 제외
                    child.destroy()

    def load_settings(self):
        """`settings.json` 파일에서 단어 목록과 설정을 불러옵니다."""
        if not os.path.exists(SETTINGS_FILE):
            return
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = json.load(f)
            
            # 기존 태그들 삭제
            self.clear_all_tags()
            
            for setting in settings:
                step_val = setting.get("step_val", "0.1")  # 기본값 0.1
                self.add_word(setting["word"], setting["min_val"], setting["max_val"], step_val)
                
            messagebox.showinfo("성공", f"{len(settings)}개의 태그 설정을 불러왔습니다.")
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            messagebox.showwarning("설정 로드 오류", f"설정 파일을 불러오는 데 실패했습니다.\n{e}")

    def save_settings(self):
        """현재 단어 목록과 설정을 `settings.json` 파일에 저장합니다."""
        if not self.word_rows:
            messagebox.showwarning("경고", "저장할 태그가 없습니다.")
            return
            
        settings = []
        for row in self.word_rows:
            setting = {
                "word": row["word"],
                "min_val": row["min_var"].get(),
                "max_val": row["max_var"].get(),
                "step_val": row["step_var"].get()
            }
            settings.append(setting)

        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=4, ensure_ascii=False)
            messagebox.showinfo("성공", f"{len(settings)}개의 태그 설정을 저장했습니다.")
        except IOError as e:
            messagebox.showerror("설정 저장 오류", f"설정 파일을 저장하는 데 실패했습니다.\n{e}")

    def add_word_event(self, event=None):
        """'추가' 버튼이나 엔터 키로 새 단어를 추가하는 이벤트 핸들러입니다."""
        word = self.new_word_entry.get().strip()
        if not word:
            messagebox.showwarning("경고", "태그를 입력해주세요.")
            return

        if any(row['word'] == word for row in self.word_rows):
            messagebox.showwarning("경고", f"이미 존재하는 태그입니다: {word}")
            return

        self.add_word(word)
        self.new_word_entry.delete(0, "end")

    def add_word(self, word: str, min_val: str = "0.1", max_val: str = "2.0", step_val: str = "0.1"):
        """UI에 새 단어 행을 추가하고 관련 데이터를 저장합니다."""
        row_frame = ttk.Frame(self.scrollable_frame)
        row_frame.pack(fill="x", padx=5, pady=2)

        min_var = tk.StringVar(value=min_val)
        max_var = tk.StringVar(value=max_val)
        step_var = tk.StringVar(value=step_val)

        min_var.trace_add("write", lambda *args, w=word: self.on_value_change(w))
        max_var.trace_add("write", lambda *args, w=word: self.on_value_change(w))
        step_var.trace_add("write", lambda *args, w=word: self.on_value_change(w))

        ttk.Label(row_frame, text=word, width=15).pack(side="left", padx=5)
        ttk.Entry(row_frame, textvariable=min_var, width=10).pack(side="left", padx=5)
        ttk.Entry(row_frame, textvariable=max_var, width=10).pack(side="left", padx=5)
        ttk.Entry(row_frame, textvariable=step_var, width=10).pack(side="left", padx=5)
        result_label = ttk.Label(row_frame, text="", width=25)
        result_label.pack(side="left", padx=5)

        remove_button = ttk.Button(row_frame, text="X", width=3, command=lambda w=word: self.remove_word(w))
        remove_button.pack(side="left", padx=5)

        row_data = {
            "word": word, "frame": row_frame, "min_var": min_var,
            "max_var": max_var, "step_var": step_var, "result_label": result_label
        }
        self.word_rows.append(row_data)
        self.on_value_change(word)

    def remove_word(self, word_to_remove):
        """단어 목록에서 특정 단어를 제거합니다."""
        self.word_rows = [row for row in self.word_rows if row["word"] != word_to_remove]
        # UI에서 해당 프레임을 찾아 삭제
        for child in self.scrollable_frame.winfo_children():
            # 첫 번째 라벨(단어 라벨)을 기준으로 해당 행을 식별
            if isinstance(child, ttk.Frame) and child.winfo_children():
                label = child.winfo_children()[0]
                if isinstance(label, ttk.Label) and label.cget("text") == word_to_remove:
                    child.destroy()
                    break

    def on_value_change(self, word=None, *args):
        """값(최소, 최대, 스텝)이 변경될 때 계산을 다시 실행합니다."""
        if word is None: # 전역 값 변경 (더 이상 사용되지 않음)
            return
        else: # 특정 단어 값 변경
            for row in self.word_rows:
                if row["word"] == word:
                    self._update_calculation(row)
                    break

    def _update_calculation(self, row_data):
        """특정 단어 행의 스텝 수와 최댓값을 계산하여 라벨에 표시합니다."""
        min_val, min_error = validate_numeric_input(row_data["min_var"].get(), "최솟값")
        max_val, max_error = validate_numeric_input(row_data["max_var"].get(), "최댓값")
        step_val, step_error = validate_numeric_input(row_data["step_var"].get(), "스텝")

        if min_error or max_error or step_error:
            error_msg = ", ".join(filter(None, [min_error, max_error, step_error]))
            row_data["result_label"].config(text=error_msg)
            return

        # 이 시점에서 모든 값이 None이 아님을 보장
        if min_val is None or max_val is None or step_val is None:
            row_data["result_label"].config(text="입력값 오류")
            return

        if step_val <= 0:
            result_text = "스텝은 0보다 커야 합니다."
        elif min_val > max_val:
            result_text = "최솟값이 최댓값보다 큽니다."
        else:
            count = int((max_val - min_val) / step_val) + 1
            last_valid_step = min_val + (count - 1) * step_val
            result_text = f"스텝 수: {count}, 스텝 최댓값: {last_valid_step}"

        row_data["result_label"].config(text=result_text)

    def generate_file(self):
        """설정된 모든 조합을 텍스트 파일로 생성합니다."""
        if not self.word_rows:
            messagebox.showwarning("경고", "파일을 생성하려면 먼저 태그를 추가해야 합니다.")
            return

        try:
            words, all_steps = self._prepare_generation_data()
        except (ValueError, InvalidOperation) as e:
            messagebox.showerror("오류", str(e))
            return

        # 총 조합 수 계산 및 경고
        total_combinations = calculate_total_combinations(all_steps)
        if total_combinations > MAX_COMBINATIONS_WARNING:
            result = messagebox.askyesno("경고", 
                f"총 {total_combinations:,}개의 조합이 생성됩니다.\n"
                f"이 작업은 시간이 오래 걸릴 수 있습니다.\n"
                f"계속하시겠습니까?")
            if not result:
                return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt", 
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="조합 파일을 저장할 위치를 선택하세요"
        )
        if not file_path:
            return

        self._write_combinations_to_file(file_path, words, all_steps, total_combinations)

    def _prepare_generation_data(self):
        """파일 생성을 위한 데이터를 준비하고 유효성을 검사합니다."""
        words, all_steps = [], []
        for row in self.word_rows:
            word = row["word"]
            min_val_str, max_val_str = row["min_var"].get(), row["max_var"].get()
            step_val_str = row["step_var"].get()

            try:
                min_val, max_val, step_val = Decimal(min_val_str), Decimal(max_val_str), Decimal(step_val_str)
            except InvalidOperation:
                raise InvalidOperation(f"'{word}' 태그의 파라미터가 숫자 형식이 아닙니다.")

            if step_val <= 0 or min_val > max_val:
                raise ValueError(f"'{word}' 태그의 파라미터가 잘못되었습니다. 확인해주세요.")

            steps = []
            current_val = min_val
            while current_val <= max_val:
                steps.append(current_val)
                current_val += step_val
            words.append(word)
            all_steps.append(steps)
        return words, all_steps

    def _write_combinations_to_file(self, file_path: str, words: List[str], all_steps: List[List[Decimal]], total_combinations: int):
        """조합을 계산하고 파일에 씁니다."""
        # itertools.product는 마지막 리스트부터 순회하므로, 순서를 뒤집어 첫 번째 단어부터 바뀌도록 함
        reversed_words = words[::-1]
        reversed_steps = all_steps[::-1]

        try:
            # 진행 상황 표시를 위한 대화상자
            progress_window = tk.Toplevel(self.master)
            progress_window.title("파일 생성 중...")
            progress_window.geometry("400x150")
            progress_window.resizable(False, False)
            progress_window.grab_set()
            
            # 진행 상황 표시
            progress_label = ttk.Label(progress_window, text="조합 파일을 생성하고 있습니다...")
            progress_label.pack(pady=20)
            
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=total_combinations)
            progress_bar.pack(fill="x", padx=20, pady=10)
            
            progress_window.update()

            with open(file_path, 'w', encoding='utf-8') as f:
                combo_count = 0
                for combo_values in itertools.product(*reversed_steps):
                    original_order_values = combo_values[::-1]
                    parts = [f":seq, {original_order_values[0]}::{words[0]} ::"]
                    parts.extend(f", {original_order_values[i]}::{words[i]} ::" for i in range(1, len(words)))
                    f.write(''.join(parts) + ',\n')
                    combo_count += 1
                    
                    # 진행 상황 업데이트 (1000개마다)
                    if combo_count % 1000 == 0:
                        progress_var.set(combo_count)
                        progress_window.update()
                        
                        # 취소 확인
                        try:
                            progress_window.update()
                        except tk.TclError:
                            # 창이 닫힌 경우
                            break

            progress_window.destroy()
            messagebox.showinfo("성공", f"총 {combo_count:,}개의 조합을 파일에 저장했습니다.\n파일 위치: {file_path}")
            
        except IOError as e:
            messagebox.showerror("파일 저장 오류", f"파일을 저장하는 중 오류가 발생했습니다:\n{e}")
        except Exception as e:
            messagebox.showerror("예상치 못한 오류가 발생했습니다:\n{e}")
        finally:
            try:
                progress_window.destroy()
            except:
                pass

class TaggerTab(ttk.Frame):
    """두 번째 탭: 조합 파일을 기반으로 이미지에 태그 파일을 생성합니다."""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.pack(expand=True, fill="both")
        self.image_folder_path = tk.StringVar()
        self.combo_file_path = tk.StringVar()
        self._build_ui()

    def _build_ui(self):
        """UI 위젯을 생성하고 배치합니다."""
        # --- 설명 ---
        desc_frame = ttk.Frame(self)
        desc_frame.pack(fill="x", padx=10, pady=(10, 5))
        desc_label = ttk.Label(desc_frame,
                               text="조합 파일에 기반하여 폴더 내 모든 이미지에 대해 텍스트 파일을 만듭니다. 오래된 이미지부터 한줄씩 부여됩니다.",
                               wraplength=700,
                               justify="center")
        desc_label.pack(fill="x")

        # --- 입력 프레임 ---
        input_frame = ttk.Frame(self)
        input_frame.pack(fill="x", padx=10, pady=10)
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="이미지 폴더:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(input_frame, textvariable=self.image_folder_path, width=70, state="readonly").grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(input_frame, text="폴더 선택", command=self.select_folder).grid(row=0, column=2, padx=5)

        ttk.Label(input_frame, text="조합 파일:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(input_frame, textvariable=self.combo_file_path, width=70, state="readonly").grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Button(input_frame, text="파일 선택", command=self.select_combo_file).grid(row=1, column=2, padx=5)

        # --- 실행 프레임 ---
        action_frame = ttk.Frame(self)
        action_frame.pack(fill="x", padx=10, pady=10)
        ttk.Button(action_frame, text="태그 파일 생성 시작", command=self.start_tagging).pack(pady=10)

    def select_folder(self):
        """'폴더 선택' 대화상자를 엽니다."""
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.image_folder_path.set(folder_selected)

    def select_combo_file(self):
        """'파일 선택' 대화상자를 엽니다."""
        file_selected = filedialog.askopenfilename(
            title="조합 파일을 선택하세요", filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
        )
        if file_selected:
            self.combo_file_path.set(file_selected)

    def start_tagging(self):
        """태그 파일 생성을 시작합니다."""
        folder_path = self.image_folder_path.get()
        combo_path = self.combo_file_path.get()
        if not folder_path or not combo_path:
            messagebox.showwarning("경고", "이미지 폴더와 조합 파일을 모두 선택해야 합니다.")
            return

        try:
            # 조합 파일 읽기
            with open(combo_path, 'r', encoding='utf-8') as f:
                tag_lines = [line.strip() for line in f if line.strip()]

            if not tag_lines:
                messagebox.showwarning("경고", "조합 파일에 내용이 없습니다.")
                return

            # 이미지 파일 찾기
            image_files = get_image_files(folder_path)

            if not image_files:
                messagebox.showwarning("경고", "폴더에 이미지 파일이 없습니다.")
                return

            self._create_tag_files(image_files, tag_lines)

        except IOError as e:
            messagebox.showerror("파일 오류", f"파일 처리 중 오류가 발생했습니다:\n{e}")
        except UnicodeDecodeError as e:
            messagebox.showerror("인코딩 오류", f"파일 인코딩 오류가 발생했습니다:\n{e}")
        except Exception as e:
            messagebox.showerror("알 수 없는 오류", f"작업 중 예상치 못한 오류가 발생했습니다:\n{e}")

    def _create_tag_files(self, image_files: List[str], tag_lines: List[str]):
        """이미지 목록과 태그 목록을 기반으로 텍스트 파일을 생성합니다."""
        total_images = len(image_files)
        total_tags = len(tag_lines)
        
        if total_images > total_tags:
            result = messagebox.askyesno("경고", 
                f"이미지 수({total_images})가 조합 파일의 줄 수({total_tags})보다 많습니다.\n"
                f"일부 이미지는 태그되지 않을 수 있습니다.\n"
                f"계속하시겠습니까?")
            if not result:
                return

        # 진행 상황 표시
        progress_window = tk.Toplevel(self.master)
        progress_window.title("태그 파일 생성 중...")
        progress_window.geometry("400x150")
        progress_window.resizable(False, False)
        progress_window.grab_set()
        
        progress_label = ttk.Label(progress_window, text="태그 파일을 생성하고 있습니다...")
        progress_label.pack(pady=20)
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=min(total_images, total_tags))
        progress_bar.pack(fill="x", padx=20, pady=10)
        
        status_label = ttk.Label(progress_window, text=f"0 / {min(total_images, total_tags)} 파일 처리됨")
        status_label.pack(pady=10)
        
        progress_window.update()

        try:
            tagged_count = 0
            for i, img_path in enumerate(image_files):
                if i >= len(tag_lines):
                    break

                txt_path = os.path.splitext(img_path)[0] + ".txt"
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(tag_lines[i])
                tagged_count += 1
                
                # 진행 상황 업데이트
                if i % 10 == 0 or i == min(total_images, total_tags) - 1:
                    progress_var.set(i + 1)
                    status_label.config(text=f"{i + 1} / {min(total_images, total_tags)} 파일 처리됨")
                    progress_window.update()

            progress_window.destroy()
            messagebox.showinfo("성공", f"총 {tagged_count}개의 이미지에 대한 태그 파일을 생성했습니다.")
            
        except Exception as e:
            progress_window.destroy()
            messagebox.showerror("오류", f"태그 파일 생성 중 오류가 발생했습니다:\n{e}")
        finally:
            try:
                progress_window.destroy()
            except:
                pass

class ViewerTab(ttk.Frame):
    """세 번째 탭: 2D 그리드 뷰어로 태그 가중치를 시각적으로 비교합니다."""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.pack(expand=True, fill="both")

        self.image_data: List[Dict] = []
        self.image_folder = tk.StringVar()
        self.x_axis_tag = tk.StringVar()
        self.y_axis_tag = tk.StringVar()
        self.thumbnail_references: List[ImageTk.PhotoImage] = []
        self.thumbnail_size = DEFAULT_THUMBNAIL_SIZE
        self.filter_vars: Dict[str, tk.StringVar] = {}
        self._build_ui()

    def _build_ui(self):
        """UI 위젯을 생성하고 배치합니다."""
        # --- 상단 컨트롤 프레임 ---
        top_container = ttk.Frame(self)
        top_container.pack(fill="x", padx=10, pady=(5,0))

        self._build_top_controls(top_container)
        self._build_filter_controls(top_container)

        # --- 뷰어 프레임 (고정 헤더 포함) ---
        self._build_viewer_grid()

    def _build_top_controls(self, parent):
        top_row_frame = ttk.Frame(parent)
        top_row_frame.pack(fill="x", expand=True)

        folder_frame = ttk.Frame(top_row_frame)
        folder_frame.pack(side="left", fill="x", expand=True)
        ttk.Button(folder_frame, text="이미지 폴더 선택", command=self.load_folder_data).pack(side="left", padx=(0, 5))
        ttk.Label(folder_frame, textvariable=self.image_folder).pack(side="left", padx=5)

        zoom_frame = ttk.Frame(top_row_frame)
        zoom_frame.pack(side="right")
        ttk.Button(zoom_frame, text="+", command=lambda: self.zoom(1.25), width=3).pack(side="left")
        ttk.Button(zoom_frame, text="-", command=lambda: self.zoom(0.8), width=3).pack(side="left")
        ttk.Button(zoom_frame, text="초기화", command=self.reset_zoom, width=5).pack(side="left", padx=(5, 0))

        axis_frame = ttk.Frame(parent)
        axis_frame.pack(fill="x", pady=2)

        ttk.Label(axis_frame, text="X축:").pack(side="left", padx=(0, 5))
        self.x_axis_combo = ttk.Combobox(axis_frame, textvariable=self.x_axis_tag, state="readonly", width=20)
        self.x_axis_combo.pack(side="left", padx=(0, 10))

        ttk.Label(axis_frame, text="Y축:").pack(side="left", padx=(0, 5))
        self.y_axis_combo = ttk.Combobox(axis_frame, textvariable=self.y_axis_tag, state="readonly", width=20)
        self.y_axis_combo.pack(side="left")

        self.x_axis_combo.bind("<<ComboboxSelected>>", self.on_axis_change)
        self.y_axis_combo.bind("<<ComboboxSelected>>", self.on_axis_change)

    def _build_filter_controls(self, parent):
        filter_container = ttk.LabelFrame(parent, text="고정 태그 필터")
        filter_container.pack(fill="x", pady=(2,5), expand=True)

        filter_canvas = tk.Canvas(filter_container, height=45)
        filter_scrollbar = ttk.Scrollbar(filter_container, orient="horizontal", command=filter_canvas.xview)
        self.filter_frame = ttk.Frame(filter_canvas)

        filter_canvas.create_window((0, 0), window=self.filter_frame, anchor="nw")
        filter_canvas.configure(xscrollcommand=filter_scrollbar.set)

        filter_scrollbar.pack(side="bottom", fill="x")
        filter_canvas.pack(side="left", fill="x", expand=True)
        self.filter_frame.bind("<Configure>", lambda e: filter_canvas.configure(scrollregion=filter_canvas.bbox("all")))

    def _build_viewer_grid(self):
        grid_container = ttk.Frame(self)
        grid_container.pack(fill="both", expand=True, padx=10, pady=5)
        grid_container.grid_rowconfigure(1, weight=1)
        grid_container.grid_columnconfigure(1, weight=1)

        self.x_header_canvas = tk.Canvas(grid_container, height=25)
        self.y_header_canvas = tk.Canvas(grid_container, width=50)
        self.grid_canvas = tk.Canvas(grid_container)

        self.x_header_canvas.grid(row=0, column=1, sticky="ew")
        self.y_header_canvas.grid(row=1, column=0, sticky="ns")
        self.grid_canvas.grid(row=1, column=1, sticky="nsew")

        self.x_header_frame = ttk.Frame(self.x_header_canvas)
        self.y_header_frame = ttk.Frame(self.y_header_canvas)
        self.grid_frame = ttk.Frame(self.grid_canvas)

        self.x_header_canvas.create_window((0, 0), window=self.x_header_frame, anchor="nw")
        self.y_header_canvas.create_window((0, 0), window=self.y_header_frame, anchor="nw")
        self.grid_canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")

        v_scroll = ttk.Scrollbar(grid_container, orient="vertical", command=self.scroll_y)
        h_scroll = ttk.Scrollbar(grid_container, orient="horizontal", command=self.scroll_x)
        v_scroll.grid(row=1, column=2, sticky="ns")
        h_scroll.grid(row=2, column=1, sticky="ew")

        self.grid_canvas.config(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        for canvas, frame in [(self.grid_canvas, self.grid_frame),
                              (self.x_header_canvas, self.x_header_frame),
                              (self.y_header_canvas, self.y_header_frame)]:
            frame.bind("<Configure>", lambda e, c=canvas: c.configure(scrollregion=c.bbox("all")))

    def load_folder_data(self):
        """폴더를 로드하고 이미지와 태그 데이터를 파싱합니다."""
        folder = filedialog.askdirectory()
        if not folder: 
            return

        self.image_folder.set(os.path.basename(folder))
        self.image_data = []
        all_tags = set()

        try:
            # 진행 상황 표시
            progress_window = tk.Toplevel(self.master)
            progress_window.title("폴더 로딩 중...")
            progress_window.geometry("300x100")
            progress_window.resizable(False, False)
            progress_window.grab_set()
            
            progress_label = ttk.Label(progress_window, text="이미지와 태그 파일을 로딩하고 있습니다...")
            progress_label.pack(pady=20)
            progress_window.update()

            files_in_folder = os.listdir(folder)
            image_files = [f for f in files_in_folder if f.lower().endswith(IMAGE_EXTENSIONS)]
            
            for i, filename in enumerate(image_files):
                img_path = os.path.join(folder, filename)
                txt_path = os.path.splitext(img_path)[0] + ".txt"

                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        parsed_tags = parse_tags(content)
                        if parsed_tags:
                            self.image_data.append({"path": img_path, "tags": parsed_tags})
                            all_tags.update(parsed_tags.keys())
                    except (IOError, UnicodeDecodeError):
                        # 개별 파일 오류는 무시하고 계속 진행
                        continue
                
                # 진행 상황 업데이트
                if i % 10 == 0:
                    progress_label.config(text=f"로딩 중... ({i+1}/{len(image_files)})")
                    progress_window.update()

            progress_window.destroy()
            
        except (IOError, OSError) as e:
            try:
                progress_window.destroy()
            except:
                pass
            messagebox.showerror("폴더 읽기 오류", f"폴더를 읽는 중 오류가 발생했습니다.\n{e}")
            return

        if not self.image_data:
            messagebox.showinfo("정보", "텍스트 파일과 매칭되는 이미지를 찾지 못했습니다.")
            return

        sorted_tags = sorted(list(all_tags))
        self.x_axis_combo['values'] = sorted_tags
        self.y_axis_combo['values'] = sorted_tags
        self.x_axis_combo.set('')
        self.y_axis_combo.set('')
        self.on_axis_change()
        
        messagebox.showinfo("성공", f"{len(self.image_data)}개의 이미지를 로드했습니다.")

    def on_axis_change(self, event=None):
        """X/Y축 선택이 변경되면 필터 UI를 업데이트하고 그리드를 다시 그립니다."""
        self._update_filters_ui()
        self.display_grid()

    def _update_filters_ui(self):
        """X/Y축을 제외한 나머지 태그들로 필터 UI를 생성합니다."""
        for widget in self.filter_frame.winfo_children(): 
            widget.destroy()
        self.filter_vars = {}

        x_tag, y_tag = self.x_axis_tag.get(), self.y_axis_tag.get()
        all_tags = set(key for item in self.image_data for key in item['tags'])
        filter_tags = sorted(list(all_tags - {x_tag, y_tag, ''}))

        for tag in filter_tags:
            frame = ttk.Frame(self.filter_frame)
            frame.pack(side="left", padx=10, pady=2)

            ttk.Label(frame, text=f"{tag}:", anchor="w").pack(fill="x")

            values = sorted(list(set(item['tags'][tag] for item in self.image_data if tag in item['tags'])))
            str_values = ["Any"] + [str(v) for v in values]

            var = tk.StringVar(value="Any")
            combo = ttk.Combobox(frame, textvariable=var, values=str_values, state="readonly", width=8)
            combo.pack(fill="x")
            combo.bind("<<ComboboxSelected>>", self.display_grid)
            self.filter_vars[tag] = var

    def display_grid(self, event=None):
        """필터링된 데이터를 기반으로 2D 이미지 그리드를 표시합니다."""
        x_tag, y_tag = self.x_axis_tag.get(), self.y_axis_tag.get()
        
        # Clear previous widgets and references
        for frame in [self.grid_frame, self.x_header_frame, self.y_header_frame]:
            for widget in frame.winfo_children():
                widget.destroy()
        
        # 메모리에서 이전 썸네일 참조 제거
        self.thumbnail_references.clear()

        if not x_tag or not y_tag: 
            return

        filtered_data = self._get_filtered_data(x_tag, y_tag)
        if not filtered_data: 
            return

        x_weights = sorted(list(set(d['tags'][x_tag] for d in filtered_data)))
        y_weights = sorted(list(set(d['tags'][y_tag] for d in filtered_data)))
        grid_data = self._map_images_to_grid(filtered_data, x_tag, y_tag)

        self._populate_grid_and_headers(grid_data, x_weights, y_weights)

    def _get_filtered_data(self, x_tag: str, y_tag: str) -> List[Dict]:
        """활성 필터에 따라 이미지 데이터를 필터링합니다."""
        active_filters = {tag: var.get() for tag, var in self.filter_vars.items() if var.get() != "Any"}

        def item_matches(item: Dict) -> bool:
            if not (x_tag in item['tags'] and y_tag in item['tags']): 
                return False
            for tag, value_str in active_filters.items():
                try:
                    if tag not in item['tags'] or item['tags'][tag] != Decimal(value_str):
                        return False
                except InvalidOperation: 
                    return False
            return True

        return [item for item in self.image_data if item_matches(item)]

    def _map_images_to_grid(self, data: List[Dict], x_tag: str, y_tag: str) -> Dict[Tuple[Decimal, Decimal], List[str]]:
        """이미지를 (x_weight, y_weight) 키에 매핑합니다."""
        grid_map = {}
        for item in data:
            key = (item['tags'][x_tag], item['tags'][y_tag])
            if key not in grid_map: 
                grid_map[key] = []
            grid_map[key].append(item['path'])
        return grid_map

    def _populate_grid_and_headers(self, grid_data: Dict[Tuple[Decimal, Decimal], List[str]], x_weights: List[Decimal], y_weights: List[Decimal]):
        """그리드와 헤더에 썸네일과 라벨을 채웁니다."""
        # 헤더 프레임의 column/row 구성을 설정하여 셀 크기를 균일하게 만듭니다.
        for c in range(len(x_weights)):
            self.x_header_frame.grid_columnconfigure(c, minsize=self.thumbnail_size + 4)
        for r in range(len(y_weights)):
            self.y_header_frame.grid_rowconfigure(r, minsize=self.thumbnail_size + 4)

        # 헤더 채우기
        for c, weight in enumerate(x_weights):
            ttk.Label(self.x_header_frame, text=str(weight), anchor="center").grid(row=0, column=c, sticky="nsew")
        for r, weight in enumerate(y_weights):
            ttk.Label(self.y_header_frame, text=str(weight), anchor="e", padding=(0,0,5,0)).grid(row=r, column=0, sticky="nsew")

        # 그리드 채우기
        for r, y_w in enumerate(y_weights):
            for c, x_w in enumerate(x_weights):
                placeholder = ttk.Frame(self.grid_frame, width=self.thumbnail_size, height=self.thumbnail_size, relief="groove", borderwidth=1)
                placeholder.grid(row=r, column=c, padx=2, pady=2)
                placeholder.grid_propagate(False)

                if (x_w, y_w) in grid_data:
                    self._create_thumbnail(placeholder, grid_data[(x_w, y_w)][0])

    def _create_thumbnail(self, parent: ttk.Frame, img_path: str):
        """지정된 부모 위젯 안에 썸네일을 생성합니다."""
        try:
            img = Image.open(img_path)
            img.thumbnail((self.thumbnail_size, self.thumbnail_size), Image.Resampling.LANCZOS)
            thumb = ImageTk.PhotoImage(img)

            label = ttk.Label(parent, image=thumb)
            self.thumbnail_references.append(thumb)
            label.pack(expand=True)
            label.bind("<Double-1>", lambda e, p=img_path: self.open_image(p))
            
            # 툴팁 추가
            self._create_tooltip(label, f"파일: {os.path.basename(img_path)}\n크기: {img.size[0]}x{img.size[1]}")
            
        except (IOError, OSError) as e:
            # 파일을 찾을 수 없거나 읽을 수 없는 경우
            label = ttk.Label(parent, text="파일\n오류")
            label.pack(expand=True)
        except Exception as e:
            # 기타 이미지 처리 오류
            label = ttk.Label(parent, text="이미지\n오류")
            label.pack(expand=True)

    def _create_tooltip(self, widget: ttk.Label, text: str):
        """위젯에 툴팁을 추가합니다."""
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            label = ttk.Label(tooltip, text=text, justify="left", background="#ffffe0", relief="solid", borderwidth=1)
            label.pack()
            
            def hide_tooltip(event):
                tooltip.destroy()
            
            widget.bind("<Leave>", hide_tooltip)
            tooltip.bind("<Leave>", hide_tooltip)
        
        widget.bind("<Enter>", show_tooltip)

    def open_image(self, path: str):
        """기본 이미지 뷰어로 이미지를 엽니다."""
        try:
            open_image_cross_platform(path)
        except IOError as e:
            messagebox.showerror("이미지 열기 오류", str(e))
        except Exception as e:
            messagebox.showerror("이미지 열기 오류", f"이미지를 여는 데 실패했습니다:\n{e}")

    def zoom(self, factor: float):
        """그리드의 썸네일 크기를 조절합니다."""
        new_size = self.thumbnail_size * factor
        if MIN_THUMBNAIL_SIZE < new_size < MAX_THUMBNAIL_SIZE:
            self.thumbnail_size = int(new_size)
            self.display_grid()

    def reset_zoom(self):
        """썸네일 크기를 기본값으로 초기화합니다."""
        self.thumbnail_size = DEFAULT_THUMBNAIL_SIZE
        self.display_grid()

    def scroll_x(self, *args):
        """가로 스크롤을 동기화합니다."""
        self.grid_canvas.xview(*args)
        self.x_header_canvas.xview(*args)

    def scroll_y(self, *args):
        """세로 스크롤을 동기화합니다."""
        self.grid_canvas.yview(*args)
        self.y_header_canvas.yview(*args)

class MainApplication(tk.Tk):
    """메인 애플리케이션 클래스."""
    
    def __init__(self):
        super().__init__()
        self.title(f"가중치 최적화기 {APP_VERSION}")
        self.geometry("900x700")
        
        # 창을 화면 중앙에 배치
        self.center_window()
        
        # 아이콘 설정 (있는 경우)
        try:
            if os.name == 'nt':  # Windows
                self.iconbitmap(default='icon.ico')
        except:
            pass

        notebook = ttk.Notebook(self)
        notebook.pack(expand=True, fill="both", padx=10, pady=10)

        self.combinator_tab = CombinatorTab(notebook)
        notebook.add(self.combinator_tab, text="범위 설정")

        tagger_tab = TaggerTab(notebook)
        notebook.add(tagger_tab, text="이미지 태깅")

        viewer_tab = ViewerTab(notebook)
        notebook.add(viewer_tab, text="2D 이미지 뷰어")

        # 메뉴바 추가
        self._create_menu()
        
        # 상태바 추가
        self._create_status_bar()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 시작 메시지
        self.status_var.set("애플리케이션이 준비되었습니다.")

    def center_window(self):
        """창을 화면 중앙에 배치합니다."""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _create_menu(self):
        """메뉴바를 생성합니다."""
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        # 파일 메뉴
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="파일", menu=file_menu)
        file_menu.add_command(label="설정 저장", command=self.save_settings)
        file_menu.add_command(label="설정 불러오기", command=self.load_settings)
        file_menu.add_separator()
        file_menu.add_command(label="종료", command=self.on_closing)
        
        # 도움말 메뉴
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="도움말", menu=help_menu)
        help_menu.add_command(label="사용법", command=self.show_help)
        help_menu.add_command(label="정보", command=self.show_about)

    def _create_status_bar(self):
        """상태바를 생성합니다."""
        self.status_var = tk.StringVar()
        self.status_var.set("준비")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.pack(side="bottom", fill="x")

    def save_settings(self):
        """설정을 저장합니다."""
        try:
            self.combinator_tab.save_settings()
            self.status_var.set("설정이 저장되었습니다.")
        except Exception as e:
            messagebox.showerror("오류", f"설정 저장 중 오류가 발생했습니다:\n{e}")

    def load_settings(self):
        """설정을 불러옵니다."""
        try:
            self.combinator_tab.load_settings()
            self.status_var.set("설정을 불러왔습니다.")
        except Exception as e:
            messagebox.showerror("오류", f"설정 불러오기 중 오류가 발생했습니다:\n{e}")

    def show_help(self):
        """사용법을 표시합니다."""
        help_text = """
가중치 최적화기 사용법

1. 범위 설정 탭:
   - 태그를 추가하고 최솟값, 최댓값, 스텝을 설정합니다
   - 조합 파일을 생성하여 모든 가능한 조합을 텍스트 파일로 저장합니다

2. 이미지 태깅 탭:
   - 이미지 폴더와 조합 파일을 선택합니다
   - 폴더 내 이미지에 자동으로 태그 파일을 생성합니다

3. 2D 이미지 뷰어 탭:
   - 이미지 폴더를 선택하여 로드합니다
   - X축과 Y축에 태그를 선택하여 2D 그리드로 시각화합니다
   - 필터를 사용하여 특정 가중치 값만 표시할 수 있습니다
   - 썸네일을 더블클릭하여 원본 이미지를 볼 수 있습니다

단축키:
- Enter: 태그 추가 (범위 설정 탭)
- 더블클릭: 이미지 열기 (2D 뷰어 탭)
        """
        messagebox.showinfo("사용법", help_text)

    def show_about(self):
        """정보를 표시합니다."""
        about_text = f"""
가중치 최적화기 {APP_VERSION}

이미지 생성 AI의 태그 가중치를
시각적으로 비교하고 최적화하는 도구입니다.

기능:
• 태그 가중치 조합 생성
• 이미지 자동 태깅
• 2D 그리드 뷰어

Author: Weight Optimizer Team
License: MIT License
        """
        messagebox.showinfo("정보", about_text)

    def on_closing(self):
        """창을 닫을 때 설정을 저장합니다."""
        if messagebox.askokcancel("종료", "정말로 종료하시겠습니까?"):
            try:
                self.combinator_tab.save_settings()
                self.status_var.set("설정을 저장하고 종료합니다...")
                self.update()
            except Exception as e:
                messagebox.showerror("오류", f"설정 저장 중 오류가 발생했습니다:\n{e}")
            finally:
                self.destroy()

if __name__ == "__main__":
    try:
        # 시스템 호환성 검사
        if sys.version_info < (3, 7):
            print("오류: Python 3.7 이상이 필요합니다.")
            print(f"현재 버전: {sys.version}")
            input("엔터 키를 눌러 종료하세요...")
            sys.exit(1)
        
        # 필요한 모듈 확인
        try:
            import tkinter
            import PIL
        except ImportError as e:
            print(f"오류: 필요한 모듈이 설치되지 않았습니다: {e}")
            print("다음 명령어로 설치하세요: pip install -r requirements.txt")
            input("엔터 키를 눌러 종료하세요...")
            sys.exit(1)
        
        # 애플리케이션 시작
        print(f"가중치 최적화기 {APP_VERSION} 시작 중...")
        app = MainApplication()
        
        # 메인 루프 시작
        app.mainloop()
        
    except KeyboardInterrupt:
        print("\n프로그램이 사용자에 의해 중단되었습니다.")
    except tk.TclError as e:
        print(f"GUI 오류가 발생했습니다: {e}")
        print("디스플레이 설정을 확인하거나 다른 환경에서 실행해보세요.")
        input("엔터 키를 눌러 종료하세요...")
    except Exception as e:
        print(f"프로그램 실행 중 예상치 못한 오류가 발생했습니다: {e}")
        print("오류 상세 정보:")
        import traceback
        traceback.print_exc()
        input("엔터 키를 눌러 종료하세요...")
    finally:
        print("프로그램을 종료합니다.") 