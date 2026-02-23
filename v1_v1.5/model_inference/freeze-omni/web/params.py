import argparse
import os
import json
import queue
import torch
import yaml
import threading
import struct
import time
import torchaudio
import datetime
import builtins

import numpy as np

from copy import deepcopy
from threading import Timer
from flask import Flask, render_template, request
from flask_socketio import SocketIO, disconnect, emit

from models.pipeline import inferencePipeline

from web.queue import PCMQueue, ThreadSafeQueue
from web.vad import VAD


class GlobalParams:
    def __init__(self, tts_pool, pipeline_pool):
        """
        Initialize the GlobalParams class with necessary components for managing global parameters and states.

        Parameters:
        - tts_pool: Pool of speech decoder.
        - pipeline_pool: Pool of inference pipeline.

        Returns:
        - None
        """
        self.tts_pool = tts_pool
        self.pipeline_pool = pipeline_pool

        self.tts_obj = self.tts_pool.acquire()
        self.pipeline_obj = self.pipeline_pool.acquire()
        # init default prompt
        init_outputs = self.pipeline_obj.pipeline_proc.speech_dialogue(
            None,
            stat="pre",
            role="You are a helpful voice assistant.\
                                                                             Your answer should be coherent, natural, simple, complete.\
                                                                             Your name is Xiao Yun.\
                                                                             Your inventor is Tencent.",
        )
        self.system_role = deepcopy(init_outputs)

        self.wakeup_and_vad = VAD()
        self.reset()

    def set_prompt(self, prompt):
        self.system_role = self.pipeline_obj.pipeline_proc.speech_dialogue(
            None, stat="pre", role=prompt
        )

    def reset(self):
        self.stop_generate = False
        self.is_generate = False
        self.wakeup_and_vad.in_dialog = False
        self.generate_outputs = deepcopy(self.system_role)
        self.whole_text = ""

        self.tts_over = False
        self.tts_over_time = 0
        self.tts_data = ThreadSafeQueue()
        self.pcm_fifo_queue = PCMQueue()

        self.stop_tts = False
        self.stop_pcm = False

    def interrupt(self):
        self.stop_generate = True
        self.tts_over = True
        while True:
            time.sleep(0.01)
            if self.is_generate == False:
                self.stop_generate = False
                while True:
                    time.sleep(0.01)
                    if self.tts_data.is_empty():
                        self.whole_text = ""
                        self.tts_over = False
                        self.tts_over_time += 1
                        break
                break

    def release(self):
        self.tts_pool.release(self.tts_obj)
        self.pipeline_pool.release(self.pipeline_obj)

    def print(self):
        print("stop_generate:", self.stop_generate)
        print("is_generate:", self.is_generate)
        print("whole_text:", self.whole_text)
        print("tts_over:", self.tts_over)
        print("tts_over_time:", self.tts_over_time)
