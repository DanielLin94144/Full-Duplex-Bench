import sys
import os

# Use current working directory as a fallback
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, parent_dir)
from glob import glob

import numpy as np

from copy import deepcopy
from threading import Timer
from web.pool import TTSObjectPool, pipelineObjectPool

import argparse
import os
import json
import torch
import threading
import time
import torchaudio
import math
import soundfile as sf

import datetime
from datetime import datetime

###### hyperparameters setting ######
# create empty configs for argument passing
configs = {}

configs["model_path"] = "./models/Freeze-Omni/checkpoints/checkpoints"
configs["llm_path"] = "./models/Freeze-Omni/Qwen2-7B-Instruct"
configs["top_p"] = 0.8
configs["top_k"] = 20
configs["temperature"] = 0.8
configs["llm_exec_nums"] = 1
configs = argparse.Namespace(**configs)

sid = "1"  # random session id

root_file_dir = (
    "/./evaluation/data/candor_pause_handling/candor_pause_handling/*/input.wav"
)
output_path = "./evaluation/exp_results/pause_handling"

TIMEOUT = 600
PIPELINE_NUMS = configs.llm_exec_nums
MAX_USERS = 1

####################################

# init inference pipelines pool
pipeline_pool = pipelineObjectPool(size=PIPELINE_NUMS, configs=configs)
# inint speech decoder pool
tts_pool = TTSObjectPool(size=MAX_USERS, model_path=configs.model_path)

server_configs = json.loads(open(configs.model_path + "/server.json").read())


connected_users = {}


def decoder(
    cur_hidden_state,
    cur_text,
    outputs,
    connected_users,
    sid,
    generate_num,
    last_text,
    is_last_chunk=False,
):
    """
    Decodes the current hidden state and text to generate audio segments using speech decoder.

    Parameters:
    - cur_hidden_state (list of torch.Tensor): The current hidden state of the language model.
    - cur_text (str): The current text to be synthesized.
    - connected_users (dict): A dictionary containing information about connected users.
    - sid (str): The session ID of the user.
    - is_last_chunk (bool, optional): Indicates if the current text is the last chunk of the input

    Returns:
    - int: The updated number of audio segments generated.
    """
    hidden_state_output = torch.cat(cur_hidden_state).squeeze(1)
    cur_text_procced = connected_users[sid][1].pipeline_obj.pipeline_proc.post_process(
        cur_text
    )
    print("Synthesis: ", [cur_text_procced])
    embeddings = connected_users[sid][
        1
    ].pipeline_obj.pipeline_proc.model.llm_decoder.model.embed_tokens(
        torch.tensor(
            connected_users[sid][1].pipeline_obj.pipeline_proc.model.tokenizer.encode(
                cur_text_procced
            )
        ).cuda()
    )
    codec_chunk_size = server_configs["decoder_first_chunk_size"]
    codec_padding_size = server_configs["decoder_chunk_overlap_size"]
    seg_threshold = server_configs["decoder_seg_threshold_first_pack"]
    if generate_num != 0:
        codec_chunk_size = server_configs["decoder_chunk_size"]
        seg_threshold = server_configs["decoder_seg_threshold"]
    for seg in connected_users[sid][1].tts_obj.tts_proc.run(
        embeddings.reshape(-1, 896).unsqueeze(0),
        server_configs["decoder_top_k"],
        hidden_state_output.reshape(-1, 896).unsqueeze(0),
        codec_chunk_size,
        codec_padding_size,
        server_configs["decoder_penalty_window_size"],
        server_configs["decoder_penalty"],
        server_configs["decoder_N"],
        seg_threshold,
    ):
        if generate_num == 0:
            try:
                split_idx = torch.nonzero(seg.abs() > 0.03, as_tuple=True)[-1][0]
                seg = seg[:, :, split_idx:]
            except:
                print("Do not need to split")
                pass
        generate_num += 1
        if connected_users[sid][1].tts_over:
            print("_________________________")
            print("tts_over")
            print("_________________________")
            connected_users[sid][1].tts_data.clear()
            connected_users[sid][1].whole_text = ""
            break
        connected_users[sid][1].tts_data.put(
            (seg.squeeze().float().cpu().numpy() * 32768).astype(np.int16)
        )
        print("Generate: ", generate_num)
    return generate_num


def generate(outputs, sid):
    """
    Generates speech dialogue output based on the current state and user session ID.

    Parameters:
    - outputs (dict): A dictionary containing the current state of the dialogue system.
    - sid (str): The session ID of the user.

    Returns:
    - None
    """
    # Stage3: start speak
    connected_users[sid][1].is_generate = True

    outputs = connected_users[sid][1].pipeline_obj.pipeline_proc.speech_dialogue(
        None, **outputs
    )
    connected_users[sid][1].generate_outputs = deepcopy(outputs)

    cur_hidden_state = []
    cur_hidden_state.append(outputs["hidden_state"])

    connected_users[sid][1].whole_text = ""
    # Stage4: contiune speak until stat is set to 'sl'
    # use 'stop' to interrupt generation, stat need to be manually set as 'sl'
    stop = False
    cur_text = ""
    last_text = ""
    generate_num = 0
    while True:
        if connected_users[sid][1].stop_generate:
            break
        if len(outputs["past_tokens"]) > 100:
            stop = True
        if stop:
            break
        del outputs["text"]
        del outputs["hidden_state"]
        outputs = connected_users[sid][1].pipeline_obj.pipeline_proc.speech_dialogue(
            None, **outputs
        )
        connected_users[sid][1].generate_outputs = deepcopy(outputs)
        if outputs["stat"] == "cs":
            cur_hidden_state.append(outputs["hidden_state"])
            if "�" in outputs["text"][len(last_text) :]:
                continue
            connected_users[sid][1].whole_text += outputs["text"][len(last_text) :]
            cur_text += outputs["text"][len(last_text) :]
            # print([connected_users[sid][1].whole_text])
            if generate_num == 0 or (len(cur_hidden_state) >= 20):
                suffix_list = [
                    ",",
                    "，",
                    "。",
                    "：",
                    "？",
                    "！",
                    ".",
                    ":",
                    "?",
                    "!",
                    "\n",
                ]
            else:
                suffix_list = ["。", "：", "？", "！", ".", "?", "!", "\n"]
            if outputs["text"][len(last_text) :].endswith(tuple(suffix_list)) and (
                len(cur_hidden_state) >= 4
            ):
                if (
                    outputs["text"][len(last_text) :].endswith(".")
                    and last_text[-1].isdigit()
                ):
                    pass
                else:
                    if not connected_users[sid][1].tts_over:
                        if len(cur_hidden_state) > 0:
                            generate_num = decoder(
                                cur_hidden_state,
                                cur_text,
                                outputs,
                                connected_users,
                                sid,
                                generate_num,
                                last_text,
                            )
                            cur_text = ""
                            cur_hidden_state = []
            last_text = outputs["text"]
        else:
            break
    if not connected_users[sid][1].tts_over:
        if len(cur_hidden_state) != 0:
            generate_num = decoder(
                cur_hidden_state,
                cur_text,
                outputs,
                connected_users,
                sid,
                generate_num,
                last_text,
                is_last_chunk=True,
            )
            cur_text = ""
    connected_users[sid][1].is_generate = False


def llm_prefill(data, outputs, sid, is_first_pack=False):
    """
    Prefills the LLM of speech dialogue system using speech.

    Parameters:
    - data (dict): A dictionary containing the current state of the user's input,
                   including features and status.
    - outputs (dict): A dictionary containing the current state of the dialogue system.
    - sid (str): The session ID of the user.
    - is_first_pack (bool, optional): Indicates if the current input packet is the first one in a new conversation
    """

    if data["status"] == "sl":
        # Satge1: start listen
        # stat will be auto set to 'cl' after Stage1
        outputs = connected_users[sid][1].pipeline_obj.pipeline_proc.speech_dialogue(
            torch.tensor(data["feature"]), **outputs
        )

    if data["status"] == "el":
        connected_users[sid][1].wakeup_and_vad.in_dialog = False
        print("Sid: ", sid, " Detect vad time out")

    if data["status"] == "cl":
        if outputs["stat"] == "cl":
            # Stage2: continue listen
            # stat will be auto set to 'ss' when endpoint is detected
            outputs = connected_users[sid][
                1
            ].pipeline_obj.pipeline_proc.speech_dialogue(
                torch.tensor(data["feature"]), **outputs
            )

            print("predict stat:", outputs["stat"])
        if is_first_pack:
            outputs["stat"] = "cl"
        if outputs["stat"] == "el":
            connected_users[sid][1].wakeup_and_vad.in_dialog = False
            print("Sid: ", sid, " Detect invalid break")
        if outputs["stat"] == "ss":
            connected_users[sid][1].interrupt()

            print("Sid: ", sid, " Detect break")
            connected_users[sid][1].wakeup_and_vad.in_dialog = False
            generate_thread = threading.Thread(
                target=generate, args=(deepcopy(outputs), sid)
            )
            generate_thread.start()
    return outputs


def disconnect_user(sid):
    if sid in connected_users:
        print(f"Disconnecting user {sid} due to time out")
        # socketio.emit('out_time', to=sid)
        connected_users[sid][0].cancel()
        connected_users[sid][1].interrupt()
        connected_users[sid][1].stop_pcm = True
        connected_users[sid][1].release()
        time.sleep(3)
        del connected_users[sid]


from web.vad import VAD
from copy import deepcopy
from web.queue import PCMQueue, ThreadSafeQueue


class MyGlobalParams:
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
                                                                             Do not answer too long.",
        )
        #  \
        #  Your name is Xiao Yun.\
        #  Your inventor is Tencent.')
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
        # self.pcm_fifo_queue = PCMQueue()

        self.stop_tts = False
        self.stop_pcm = False

    # def interrupt(self):
    #     self.stop_generate = True
    #     self.tts_over = True

    #     cnt = 0
    #     while(True):
    #         time.sleep(0.01)
    #         if(self.is_generate == False):
    #             self.stop_generate = False
    #             # while True:
    #                 # time.sleep(0.01)
    #                 # if self.tts_data.is_empty():
    #             self.whole_text = ""
    #             self.tts_over = False
    #             self.tts_over_time += 1
    #             # break

    #             break

    def interrupt(self, timeout=5.0):
        """
        中断生成过程，等待生成线程退出和 tts_data 队列清空，增加了超时保护以防止无限等待。
        """
        self.stop_generate = True
        self.tts_over = True

        # 等待生成线程退出（is_generate 变为 False），超时后跳出循环
        start_time = time.time()
        while self.is_generate:
            time.sleep(0.01)
            if time.time() - start_time > timeout:
                print(
                    "Warning: Generation did not stop within {} seconds.".format(
                        timeout
                    )
                )
                break
        self.stop_generate = False

        # 等待 tts_data 队列清空，超时后跳出循环
        start_time = time.time()
        while not self.tts_data.is_empty():
            time.sleep(0.01)
            if time.time() - start_time > timeout:
                print("Warning: tts_data not empty after {} seconds.".format(timeout))
                break

        self.whole_text = ""
        self.tts_over = False
        self.tts_over_time += 1

    def release(self):
        self.tts_pool.release(self.tts_obj)
        self.pipeline_pool.release(self.pipeline_obj)

    def print(self):
        print("stop_generate:", self.stop_generate)
        print("is_generate:", self.is_generate)
        print("whole_text:", self.whole_text)
        print("tts_over:", self.tts_over)
        print("tts_over_time:", self.tts_over_time)


connected_users = {}
connected_users[sid] = []
connected_users[sid].append(Timer(TIMEOUT, disconnect_user, [sid]))
connected_users[sid].append(MyGlobalParams(tts_pool, pipeline_pool))

tts_pool.print_info()
pipeline_pool.print_info()


def send_pcm(sid):
    """
    Sends PCM audio data to the dialogue system for processing.

    Parameters:
    - sid (str): The session ID of the user.
    """

    chunk_size = connected_users[sid][1].wakeup_and_vad.get_chunk_size()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    wav_files = sorted(glob(root_file_dir))

    # iterate through all the wav files
    for input_wav in wav_files:

        wav, fs = sf.read(input_wav)

        # file_name = input_wav.split("/")[-2]
        file_name = f"{input_wav.split('/')[4]}/{input_wav.split('/')[-2]}"

        print("Processing: ", input_wav)

        # # read the targeted time interval json file
        # with open(input_wav.replace("audio.wav", "not_interrupt_time.json"), 'r') as f:
        #     time_intervals = json.load(f)

        # start_time = time_intervals[0]
        # end_time = time_intervals[1]

        wav = torch.tensor(wav)
        if fs != 16000:
            wav = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)(
                wav.float()
            )
            fs = 16000

        wav_input = torch.zeros(math.ceil(wav.shape[0] / chunk_size) * chunk_size)
        wav_input[: wav.shape[0]] = wav

        # interrupt_time_interval = []

        chunked_inputs = []
        for i in range(0, wav_input.shape[0], chunk_size):
            chunked_inputs.append(wav_input[i : i + chunk_size])

        entire_output_audio = None
        time_aligned_output_audio = None

        # # save the concat_wav as audio file
        sf.write(f"candor_pause_temp.wav", wav_input.numpy(), 16000)

        cnt = 0
        idx = 0

        while True:
            if connected_users[sid][1].stop_pcm:
                print("Sid: ", sid, " Stop pcm")
                connected_users[sid][1].stop_generate = True
                connected_users[sid][1].stop_tts = True
                break

            # e = connected_users[sid][1].pcm_fifo_queue.get(chunk_size)

            if cnt >= len(chunked_inputs):
                print("Sid: ", sid, " Finish pcm")
                break

            time.sleep(0.16)
            # Get current date and time
            current_time = datetime.now()
            print("Real Time: ", current_time.strftime("%H:%M:%S"))

            e = chunked_inputs[cnt]

            cnt += 1

            print("Sid: ", sid, " Time: ", cnt * chunk_size / fs)

            res = connected_users[sid][1].wakeup_and_vad.predict(np.float32(e))
            print(res["status"])

            force_tts_over = False

            chunk_start_time = cnt * chunk_size / fs
            chunk_end_time = (cnt + 1) * chunk_size / fs

            # if chunk_start_time >= start_time and chunk_end_time <= end_time:

            if res["status"] == "sl":
                print("Sid: ", sid, " Vad start")
                force_tts_over = True

                outputs = deepcopy(connected_users[sid][1].generate_outputs)
                outputs["adapter_cache"] = None
                outputs["encoder_cache"] = None
                outputs["pe_index"] = 0
                outputs["stat"] = "sl"
                outputs["last_id"] = None
                if "text" in outputs:
                    del outputs["text"]
                if "hidden_state" in outputs:
                    del outputs["hidden_state"]

                send_dict = {}
                for i in range(len(res["feature_last_chunk"])):
                    if i == 0:
                        send_dict["status"] = "sl"
                    else:
                        send_dict["status"] = "cl"
                    send_dict["feature"] = res["feature_last_chunk"][i]
                    outputs = llm_prefill(send_dict, outputs, sid, is_first_pack=True)
                send_dict["status"] = "cl"
                send_dict["feature"] = res["feature"]
                outputs = llm_prefill(send_dict, outputs, sid)

            elif res["status"] == "cl" or res["status"] == "el":
                send_dict = {}
                send_dict["status"] = res["status"]
                send_dict["feature"] = res["feature"]
                outputs = llm_prefill(send_dict, outputs, sid)

            final_output_audio = None
            if not connected_users[sid][1].tts_data.is_empty():
                output_data = connected_users[sid][1].tts_data.get()

                final_output_audio = output_data.astype(np.float32) / 32768.0
                print(final_output_audio.shape)

                if final_output_audio is not None:
                    if connected_users[sid][1].tts_over_time > 0:
                        connected_users[sid][1].tts_over_time = 0

                    if entire_output_audio is None:
                        entire_output_audio = final_output_audio
                    else:
                        entire_output_audio = np.concatenate(
                            (entire_output_audio, final_output_audio)
                        )

            curr_chunk_output = None

            if force_tts_over:
                curr_chunk_output = np.zeros(3840)
                entire_output_audio = None
            else:
                if (
                    entire_output_audio is not None
                    and idx < len(entire_output_audio) // 3840
                ):
                    curr_chunk_output = entire_output_audio[
                        idx * 3840 : (idx + 1) * 3840
                    ]
                    idx += 1

                else:
                    curr_chunk_output = np.zeros(3840)

            if time_aligned_output_audio is None:
                time_aligned_output_audio = curr_chunk_output
            else:
                time_aligned_output_audio = np.concatenate(
                    (time_aligned_output_audio, curr_chunk_output)
                )

        # read the input audio file
        input_audio, fs = sf.read("candor_pause_temp.wav")
        # resample to 24000 Hz
        input_audio = torchaudio.transforms.Resample(orig_freq=fs, new_freq=24000)(
            torch.tensor(input_audio).float()
        )

        # save the input audio and output audio as two-channel audio file
        # if the length of input audio and output audio are not equal, pad the shorter one with zeros
        if input_audio.shape[0] > time_aligned_output_audio.shape[0]:
            time_aligned_output_audio = np.concatenate(
                (
                    time_aligned_output_audio,
                    np.zeros(input_audio.shape[0] - time_aligned_output_audio.shape[0]),
                )
            )
        elif input_audio.shape[0] < time_aligned_output_audio.shape[0]:
            input_audio = np.concatenate(
                (
                    input_audio,
                    np.zeros(time_aligned_output_audio.shape[0] - input_audio.shape[0]),
                )
            )

        if not os.path.exists(os.path.join(output_path, file_name)):
            os.makedirs(os.path.join(output_path, file_name))

        # save the input audio file
        sf.write(os.path.join(output_path, file_name, "input.wav"), input_audio, 24000)
        # save the output audio file
        sf.write(
            os.path.join(output_path, file_name, "output.wav"),
            time_aligned_output_audio,
            24000,
        )

        # save the two-channel audio file
        sf.write(
            os.path.join(output_path, file_name, "two_channel.wav"),
            np.stack([input_audio, time_aligned_output_audio], axis=1),
            24000,
        )

        connected_users[sid][1].interrupt()
        connected_users[sid][1].reset()
        connected_users[sid][1].wakeup_and_vad.reset_vad()
        # connected_users[sid][1].wakeup_and_vad.in_dialog = True


if __name__ == "__main__":
    print("Start Freeze-Omni sever")
    pcm_thread = threading.Thread(target=send_pcm, args=(sid,))
    pcm_thread.start()
