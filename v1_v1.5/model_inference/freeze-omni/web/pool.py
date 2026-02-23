import os
import sys
import copy
import json
import torch
import random
import argparse
import subprocess
import numpy as np
import soundfile as sf
import subprocess
import concurrent.futures

from models.decoder.llm2tts import llm2TTS
from models.pipeline import inferencePipeline


class PooledCodecTTSObject:
    def __init__(self, model_path):
        self.in_use = False
        self.tts_proc = llm2TTS(model_path)


class TTSObjectPool:
    def __init__(self, size=10, model_path=""):
        """
        Initialize the TTSObjectPool with a specified size and model path.

        Parameters:
        - size (int): The number of TTS objects to initialize in the pool. Default is 10.
        - model_path (str): The path to the model file used by the TTS objects. Default is an empty string.

        Returns:
        - None
        """
        self.pool = self._initialize_pool(size, model_path)

    def _initialize_pool(self, size, model_path):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(PooledCodecTTSObject, model_path) for _ in range(size)
            ]
            return [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

    def acquire(self):
        for obj in self.pool:
            if not obj.in_use:
                obj.in_use = True
                return obj
        raise Exception("No available objects in the pool")

    def release(self, obj):
        obj.in_use = False

    def print_info(self):
        for i in range(len(self.pool)):
            print(f"TTS Object {i} is in use: {self.pool[i].in_use}")


class inferencePipelineObject:
    def __init__(self, configs):
        self.user_count = 0
        self.pipeline_proc = inferencePipeline(configs)


class pipelineObjectPool:
    def __init__(self, size, configs):
        """
        Initialize the pipelineObjectPool with a specified size and configs.

        Parameters:
        - size (int): The number of TTS objects to initialize in the pool.
        - configs : The config of pipeline object.

        Returns:
        - None
        """
        self.pool = self._initialize_pool(size, configs)

    def _initialize_pool(self, size, configs):
        pool = [inferencePipelineObject(configs) for _ in range(size)]
        return pool

    def acquire(self):
        # Find the object with the minimum user count
        min_user_obj = min(self.pool, key=lambda obj: obj.user_count)
        min_user_obj.user_count += 1
        return min_user_obj

    def release(self, obj):
        if obj.user_count > 0:
            obj.user_count -= 1

    def print_info(self):
        for i, obj in enumerate(self.pool):
            print(f"Pipeline Object {i} user count: {obj.user_count}")
