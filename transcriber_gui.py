import ctypes

import warnings
import sys
import time
import threading
import numpy as np

warnings.filterwarnings("ignore")

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QProgressBar, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt5.QtGui import QFont, QColor, QPalette, QTextCursor

import soundcard as sc
import whisper
import torch


SAMPLE_RATE = 16000
MODEL_NAME = "large-v3"


class Signals(QObject):
    transcription_done = pyqtSignal(str)
    transcription_started = pyqtSignal(float)
    transcription_error = pyqtSignal(str)


class TranscribeWorker(QThread):
    def __init__(self, audio, model, signals):
        super().__init__()
        self.audio = audio
        self.model = model
        self.signals = signals

    def run(self):
        duration = len(self.audio) / SAMPLE_RATE
        self.signals.transcription_started.emit(duration)
        try:
            result = self.model.transcribe(
                self.audio,
                language="ru",
                condition_on_previous_text=False,
                temperature=0.0
            )
            text = result["text"].strip()
            self.signals.transcription_done.emit(text)
        except Exception as e:
            self.signals.transcription_error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper Транскрипция")
        self.setMinimumSize(720, 540)
        self.resize(820, 620)

        self.recording = False
        self.buffer = []
        self.lock = threading.Lock()
        self.capture_thread = None
        self.worker = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.elapsed = 0
        self.estimated = 0
        self.signals = Signals()
        self.signals.transcription_done.connect(self._on_done)
        self.signals.transcription_started.connect(self._on_started)
        self.signals.transcription_error.connect(self._on_error)

        self._load_model()
        self._build_ui()

    def _load_model(self):
        self._model_loaded = False
        self._model = None

        def load():
            self._model = whisper.load_model(MODEL_NAME)
            self._model_loaded = True

        t = threading.Thread(target=load, daemon=True)
        t.start()

    def _build_ui(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f0f0f;
            }
            QWidget#central {
                background-color: #0f0f0f;
            }
            QLabel#title {
                color: #f0f0f0;
                font-size: 22px;
                font-weight: 600;
                letter-spacing: 1px;
            }
            QLabel#subtitle {
                color: #555;
                font-size: 12px;
                letter-spacing: 2px;
            }
            QLabel#status {
                color: #888;
                font-size: 12px;
            }
            QLabel#timer_label {
                color: #444;
                font-size: 11px;
                letter-spacing: 1px;
            }
            QPushButton#btn_record {
                background-color: #1a1a1a;
                color: #f0f0f0;
                border: 1px solid #333;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 500;
                padding: 10px 28px;
                letter-spacing: 0.5px;
            }
            QPushButton#btn_record:hover {
                background-color: #222;
                border-color: #555;
            }
            QPushButton#btn_record:pressed {
                background-color: #111;
            }
            QPushButton#btn_transcribe {
                background-color: #e8e8e8;
                color: #0f0f0f;
                border: none;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 600;
                padding: 10px 28px;
                letter-spacing: 0.5px;
            }
            QPushButton#btn_transcribe:hover {
                background-color: #ffffff;
            }
            QPushButton#btn_transcribe:pressed {
                background-color: #cccccc;
            }
            QPushButton#btn_transcribe:disabled {
                background-color: #222;
                color: #444;
            }
            QPushButton#btn_clear {
                background-color: #2a2a2a;
                color: #c0c0c0;
                border: 1px solid #444;
                border-radius: 8px;
                font-size: 12px;
                padding: 10px 20px;
            }
            QPushButton#btn_clear:hover {
                background-color: #333;
                color: #f0f0f0;
                border-color: #666;
            }
            QPushButton#btn_copy {
                background-color: #2a2a2a;
                color: #c0c0c0;
                border: 1px solid #444;
                border-radius: 8px;
                font-size: 12px;
                padding: 10px 20px;
            }
            QPushButton#btn_copy:hover {
                background-color: #333;
                color: #f0f0f0;
                border-color: #666;
            }
            QTextEdit#output {
                background-color: #141414;
                color: #d0d0d0;
                border: 1px solid #222;
                border-radius: 10px;
                font-size: 14px;
                line-height: 1.6;
                padding: 16px;
                selection-background-color: #333;
            }
            QProgressBar {
                background-color: #1a1a1a;
                border: none;
                border-radius: 3px;
                height: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #e8e8e8;
                border-radius: 3px;
            }
            QFrame#divider {
                color: #1e1e1e;
            }
        """)

        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(32, 28, 32, 28)
        main_layout.setSpacing(0)

        # Header
        header = QVBoxLayout()
        header.setSpacing(4)
        title = QLabel("WHISPER")
        title.setObjectName("title")
        subtitle = QLabel("AUDIO TRANSCRIPTION")
        subtitle.setObjectName("subtitle")
        header.addWidget(title)
        header.addWidget(subtitle)
        main_layout.addLayout(header)
        main_layout.addSpacing(24)

        # Status row
        status_row = QHBoxLayout()
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color: #333; font-size: 10px;")
        self.status_label = QLabel("Загрузка модели...")
        self.status_label.setObjectName("status")
        status_row.addWidget(self.status_dot)
        status_row.addSpacing(6)
        status_row.addWidget(self.status_label)
        status_row.addStretch()

        self.rec_indicator = QLabel("● REC")
        self.rec_indicator.setStyleSheet("color: #c0392b; font-size: 11px; font-weight: 600; letter-spacing: 1px;")
        self.rec_indicator.setVisible(False)
        status_row.addWidget(self.rec_indicator)

        main_layout.addLayout(status_row)
        main_layout.addSpacing(20)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        self.timer_label = QLabel("")
        self.timer_label.setObjectName("timer_label")
        self.timer_label.setAlignment(Qt.AlignRight)
        self.timer_label.setVisible(False)
        main_layout.addWidget(self.timer_label)
        main_layout.addSpacing(16)

        # Text output
        self.output = QTextEdit()
        self.output.setObjectName("output")
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("Транскрибированный текст появится здесь...")
        self.output.setFont(QFont("Consolas", 13))
        main_layout.addWidget(self.output, 1)
        main_layout.addSpacing(20)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.btn_record = QPushButton("▶  Начать запись")
        self.btn_record.setObjectName("btn_record")
        self.btn_record.setFixedHeight(44)
        self.btn_record.clicked.connect(self._toggle_record)

        self.btn_transcribe = QPushButton("Транскрибировать")
        self.btn_transcribe.setObjectName("btn_transcribe")
        self.btn_transcribe.setFixedHeight(44)
        self.btn_transcribe.setEnabled(False)
        self.btn_transcribe.clicked.connect(self._transcribe)

        self.btn_clear = QPushButton("Очистить")
        self.btn_clear.setObjectName("btn_clear")
        self.btn_clear.setFixedHeight(44)
        self.btn_clear.clicked.connect(self._clear)

        self.btn_copy = QPushButton("Скопировать")
        self.btn_copy.setObjectName("btn_copy")
        self.btn_copy.setFixedHeight(44)
        self.btn_copy.clicked.connect(self._copy)

        btn_row.addWidget(self.btn_record)
        btn_row.addWidget(self.btn_transcribe)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_copy)
        btn_row.addWidget(self.btn_clear)
        main_layout.addLayout(btn_row)

        # Poll for model load
        self.model_timer = QTimer()
        self.model_timer.timeout.connect(self._check_model)
        self.model_timer.start(500)

        # Rec blink timer
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self._blink_rec)
        self._blink_state = True

    def _check_model(self):
        if self._model_loaded:
            self.model_timer.stop()
            self.status_dot.setStyleSheet("color: #27ae60; font-size: 10px;")
            self.status_label.setText("Готово к записи")
            self.btn_record.setEnabled(True)

    def _toggle_record(self):
        if not self.recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        self.recording = True
        self.buffer = []
        self.btn_record.setText("■  Остановить запись")
        self.btn_record.setStyleSheet("""
            QPushButton#btn_record {
                background-color: #1e0a0a;
                color: #e74c3c;
                border: 1px solid #5a1a1a;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 500;
                padding: 10px 28px;
            }
            QPushButton#btn_record:hover { background-color: #2a0d0d; border-color: #7a2a2a; }
        """)
        self.btn_transcribe.setEnabled(False)
        self.rec_indicator.setVisible(True)
        self.blink_timer.start(600)
        self.status_dot.setStyleSheet("color: #c0392b; font-size: 10px;")
        self.status_label.setText("Запись идёт...")

        self.capture_thread = threading.Thread(target=self._capture, daemon=True)
        self.capture_thread.start()

    def _stop_recording(self):
        self.recording = False
        self.blink_timer.stop()
        self.rec_indicator.setVisible(False)
        self.btn_record.setText("▶  Начать запись")
        self.btn_record.setStyleSheet("")
        self.status_dot.setStyleSheet("color: #f39c12; font-size: 10px;")

        buf_len = len(self.buffer)
        if buf_len > 0:
            duration = buf_len / SAMPLE_RATE
            self.status_label.setText(f"Записано {duration:.0f} сек — нажми «Транскрибировать»")
            self.btn_transcribe.setEnabled(True)
        else:
            self.status_label.setText("Буфер пуст")

    def _blink_rec(self):
        self._blink_state = not self._blink_state
        self.rec_indicator.setVisible(self._blink_state)

    def _capture(self):
        ctypes.windll.ole32.CoInitialize(None)
        try:
            speaker = sc.default_speaker()
            with sc.get_microphone(
                id=str(speaker.name), include_loopback=True
            ).recorder(samplerate=SAMPLE_RATE) as mic:
                while self.recording:
                    data = mic.record(numframes=SAMPLE_RATE)
                    mono = data[:, 0].astype(np.float32)
                    with self.lock:
                        self.buffer.extend(mono)
        finally:
            ctypes.windll.ole32.CoUninitialize()

    def _transcribe(self):
        with self.lock:
            if not self.buffer:
                return
            audio = np.array(self.buffer, dtype=np.float32)

        self.btn_transcribe.setEnabled(False)
        self.btn_record.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.timer_label.setVisible(True)
        self.progress_bar.setValue(0)

        self.worker = TranscribeWorker(audio, self._model, self.signals)
        self.worker.start()

    def _on_started(self, duration):
        self.estimated = duration
        self.elapsed = 0
        self.status_dot.setStyleSheet("color: #3498db; font-size: 10px;")
        self.status_label.setText(f"Транскрибирую {duration:.0f} сек аудио...")
        self.timer.start(100)

    def _tick(self):
        self.elapsed += 0.1
        pct = min(int((self.elapsed / max(self.estimated, 1)) * 100), 95)
        self.progress_bar.setValue(pct)
        remaining = max(0, self.estimated - self.elapsed)
        self.timer_label.setText(f"{self.elapsed:.0f}с прошло  •  ~{remaining:.0f}с осталось")

    def _on_done(self, text):
        self.timer.stop()
        self.progress_bar.setValue(100)

        if text:
            with open("output.txt", "a", encoding="utf-8") as f:
                f.write(text + "\n\n")

            if self.output.toPlainText():
                self.output.append("\n")
            self.output.append(text)
            cursor = self.output.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.output.setTextCursor(cursor)

        self.status_dot.setStyleSheet("color: #27ae60; font-size: 10px;")
        self.status_label.setText("Готово! Текст сохранён в output.txt")
        self.btn_record.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.timer_label.setVisible(False)
        self.buffer = []

    def _on_error(self, error):
        self.timer.stop()
        self.status_dot.setStyleSheet("color: #e74c3c; font-size: 10px;")
        self.status_label.setText(f"Ошибка: {error}")
        self.btn_record.setEnabled(True)
        self.btn_transcribe.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.timer_label.setVisible(False)

    def _copy(self):
        text = self.output.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self.btn_copy.setText("Скопировано ✓")
            QTimer.singleShot(2000, lambda: self.btn_copy.setText("Скопировать"))

    def _clear(self):
        self.output.clear()

    def closeEvent(self, event):
        self.recording = False
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
