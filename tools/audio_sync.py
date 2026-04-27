#!/usr/bin/env python3
"""
音声同期ツール: α7C映像の内蔵マイク音声とK688別録音声を
相互相関で同期し、高音質音声に差し替えてmergeする。
"""

import argparse
import subprocess
import tempfile
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate


SYNC_SAMPLE_RATE = 16000  # 同期検出用のサンプルレート（低めで高速化）


def extract_audio_as_wav(input_path: str, output_wav: str) -> None:
    """ffmpegで音声をモノラル16kHz wavに変換抽出する。"""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-ac", "1", "-ar", str(SYNC_SAMPLE_RATE),
        "-sample_fmt", "s16", output_wav,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def find_offset(ref_wav: str, target_wav: str) -> float:
    """相互相関でtargetのrefに対する時間オフセット(秒)を返す。
    正の値 = targetがrefより遅く開始。
    """
    _, ref = wavfile.read(ref_wav)
    _, target = wavfile.read(target_wav)

    # float32に正規化
    ref = ref.astype(np.float32)
    target = target.astype(np.float32)

    # 相互相関
    corr = correlate(ref, target, mode="full")
    peak = np.argmax(corr)

    # オフセット計算: mode="full"のとき、peak = len(target) - 1 がオフセット0
    offset_samples = peak - (len(target) - 1)
    offset_sec = offset_samples / SYNC_SAMPLE_RATE

    return offset_sec


def normalize_audio(input_path: str, output_path: str) -> None:
    """ノイズ除去 + コンプレッション + ラウドネス正規化(-14 LUFS)を適用する。"""
    audio_filter = (
        "afftdn=nf=-25:nr=10:nt=w,"           # ホワイトノイズ除去（換気扇等の定常ノイズ）
        "acompressor=threshold=-25dB:ratio=4"   # ダイナミックレンジ圧縮
        ":attack=5:release=50:makeup=8,"        #   （拍手等のピークを抑え、声を持ち上げる）
        "loudnorm=I=-14:TP=-1.5:LRA=11"        # YouTube基準(-14 LUFS)にラウドネス正規化
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-af", audio_filter,
        "-c:v", "copy",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def merge(video_path: str, audio_path: str, offset_sec: float, output_path: str) -> None:
    """映像と音声をオフセット付きでmergeする。
    offset_sec: 負=外部音声が動画より先に録音開始、正=動画の方が先に開始。
    音声を実際にトリム/パディングして結合（メタデータに依存しない）。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        aligned_audio = f"{tmpdir}/aligned.m4a"

        if offset_sec < 0:
            # 外部音声が先に開始 → 先頭をトリム
            trim_sec = abs(offset_sec)
            cmd_audio = [
                "ffmpeg", "-y",
                "-ss", f"{trim_sec:.3f}", "-i", audio_path,
                "-c:a", "aac", "-b:a", "192k",
                aligned_audio,
            ]
        else:
            # 動画が先に開始 → 音声の先頭に無音を追加
            delay_ms = int(offset_sec * 1000)
            cmd_audio = [
                "ffmpeg", "-y",
                "-i", audio_path,
                "-af", f"adelay={delay_ms}|{delay_ms}",
                "-c:a", "aac", "-b:a", "192k",
                aligned_audio,
            ]

        subprocess.run(cmd_audio, capture_output=True, check=True)

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", aligned_audio,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "copy",
            "-shortest",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)


def main():
    parser = argparse.ArgumentParser(description="音声同期 + 音声最適化ツール")
    parser.add_argument("video", help="α7C動画ファイル (.mp4)")
    parser.add_argument("audio", help="別録音声ファイル (.m4a/.wav)")
    parser.add_argument("-o", "--output", help="出力ファイル (デフォルト: <元ファイル名>_synced.mp4)")
    parser.add_argument("--skip-normalize", action="store_true",
                        help="音声最適化（ノイズ除去・コンプレッション・ラウドネス正規化）をスキップ")
    args = parser.parse_args()

    video_path = args.video
    audio_path = args.audio
    output_path = args.output or str(
        Path(video_path).parent / f"{Path(video_path).stem}_synced.mp4"
    )

    steps = 3 if args.skip_normalize else 4

    print(f"[1/{steps}] 音声抽出中...")
    with tempfile.TemporaryDirectory() as tmpdir:
        ref_wav = f"{tmpdir}/ref.wav"
        target_wav = f"{tmpdir}/target.wav"

        extract_audio_as_wav(video_path, ref_wav)
        extract_audio_as_wav(audio_path, target_wav)

        print(f"[2/{steps}] オフセット検出中...")
        offset = find_offset(ref_wav, target_wav)
        print(f"       検出オフセット: {offset:+.3f} 秒")

    if args.skip_normalize:
        print(f"[3/{steps}] 映像+音声 merge中...")
        merge(video_path, audio_path, offset, output_path)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            synced_tmp = f"{tmpdir}/synced_tmp.mp4"
            print(f"[3/{steps}] 映像+音声 merge中...")
            merge(video_path, audio_path, offset, synced_tmp)
            print(f"[4/{steps}] 音声最適化中（ノイズ除去 → コンプレッション → ラウドネス正規化）...")
            normalize_audio(synced_tmp, output_path)

    print(f"完了: {output_path}")


if __name__ == "__main__":
    main()
