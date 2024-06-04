from openai import OpenAI
from pydub import AudioSegment

openai = OpenAI(api_key='sk-proj-yo5CH4QlmaITlcM8mKZ3T3BlbkFJsR1Rs3ZhFq6CPzhb8Bw7')


def split_audio(audio_file):
    file = AudioSegment.from_file(audio_file)

    ten_minutes = 10 * 60 * 1000
    chunks = []

    for i in range(0, len(file), ten_minutes):
        chunk = file[i:i+ten_minutes].export()
        chunks.append(chunk)

    return chunks


def transcribe_chunk(chunk, prompt):
    transcription = openai.audio.transcriptions.create(
        model="whisper-1",
        file=chunk,
        language="en",
        prompt=prompt,
        response_format="text",
    )

    return transcription.text


def transcribe_audio(audio_file):
    chunks = split_audio(audio_file)

    transcription = ""

    for chunk in chunks:
        transcription += transcribe_chunk(chunk, transcription)

    return transcription


if __name__ == '__main__':
    transcription = transcribe_audio('/Users/lara/Desktop/1535. Cd. 5 21.m4a')
    print(transcription)
