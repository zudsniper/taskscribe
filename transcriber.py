#! /usr/bin/python3

from openai import OpenAI
import os
import glob
import yaml
import mimetypes
import av
from colorama import Fore, Back, Style
from markdownify import markdownify as md
from loguru import logger
from dotenv import load_dotenv
from pydub import AudioSegment

# Load environment variables from .env file
load_dotenv()

# Load OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    logger.error("OPENAI_API_KEY is not set in the .env file. Please set it before running the script.")
    exit(1)

client = OpenAI(api_key=openai_api_key)
logger.info(Fore.GREEN + "Initialized OpenAI client." + Style.RESET_ALL)


# set the API key
# client.api_key = openai_api_key
# NOTE  -- this is AUTOMATICALLY SET for the environment by the env variable.

def is_audio_file(filename):
    """Check if the file is a supported audio format."""
    return any(filename.endswith(ext) for ext in ['.mp3', '.m4a', '.aac', '.wav'])


def get_audio_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    return audio.duration_seconds


# def split_audio(audio_file, chunk_duration):
#     """Split audio file into chunks of specified duration."""
#     audio = AudioSegment.from_file(audio_file, format=None)
#     chunks = []
#     current_start = 0
#
#     while current_start < len(audio):
#         current_chunk = audio[current_start:current_start + chunk_duration * 1000]
#         if format is None:
#             chunks.append(current_chunk)
#         else:
#             chunks.append(current_chunk.export(format=format).read())
#         current_start += chunk_duration * 1000
#
#     return chunks

# def split_audio(audio_file, chunk_duration, formato=None):
#     """Split audio file into chunks of specified duration."""
#
#     # Guess the file format based on its extension
#     mimetype = mimetypes.guess_type(audio_file)[0]
#     ext_format = mimetype.split('/')[-1] if mimetype else None
#
#     # Handle .m4a files
#     # if formato == 'mp4a-latm':
#     #     formato = 'aac'
#     #
#     # logger.debug("processing audio file: " + audio_file)
#     # logger.debug("guessed format: " + ext_format)
#     # if formato is None:
#     #     formato = ext_format
#
#     audio = AudioSegment.from_file(audio_file, "aac") # TODO: dont fix this
#     chunks = []
#     current_start = 0
#
#     while current_start < len(audio):
#         current_chunk = audio[current_start:current_start + chunk_duration * 1000]
#         if formato is None:
#             chunks.append(current_chunk)
#         else:
#             chunks.append(current_chunk.export(format=formato).read())
#         current_start += chunk_duration * 1000
#
#     return chunks


def split_audio(audio_file, chunk_duration):
    """Split audio file into chunks of specified duration."""
    # Open the input file
    input_container = av.open(audio_file)

    # Select the audio stream
    audio_stream = next(s for s in input_container.streams if s.type == 'audio')

    # Calculate the number of frames per chunk
    frames_per_chunk = chunk_duration * audio_stream.rate

    chunks = []
    current_chunk = b''
    current_frame = 0

    # Iterate over the audio frames
    for frame in input_container.decode(audio_stream):
        # Add the frame to the current chunk
        current_chunk += frame.planes[0]

        # If the current chunk has enough frames, add it to the chunks list
        if current_frame + frame.samples >= frames_per_chunk:
            chunks.append(current_chunk)
            current_chunk = b''
            current_frame = 0

        current_frame += frame.samples

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# TODO:
#   - Utilize previously transcribed chunks as context for each new chunk in some AI powered way

def transcribe_chunk(chunk):
    # Convert chunk to bytes
    chunk_bytes = chunk.export("temp.wav", format="wav").read()

    # Make API request
    with open("temp.wav", "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
            prompt="" # TODO: this is 244 char max, use for acronyms, punctuation, and limited context.
                      #   ...figure out how the hell to add context to the request.
        )
    return response.text


def transcribe_audio(file_path):
    "combine chunk transcriptions for final string"
    chunks = split_audio(file_path, chunk_duration=get_audio_duration(file_path))
    transcriptions = [transcribe_chunk(chunk) for chunk in chunks]
    full_transcription = " ".join(transcriptions)
    return full_transcription



# def transcribe_audio(file_path):
#     """Transcribe audio file using OpenAI Whisper model."""
#     logger.info(f"Transcribing audio file: {file_path}")
#     model = whisper.load_model("base")
#     result = model.transcribe(file_path)
#     return result['text']

# def transcribe_audio(file_path):
#     """Transcribe audio file using OpenAI Whisper ASR API."""
#     logger.info(f"Transcribing audio file: {file_path}")
#
#     # Open the audio file
#     with open(file_path, "rb") as audio_file:
#         while True:
#             # Read audio data in chunks
#             audio_data = audio_file.read(1024)
#             if not audio_data:
#                 break
#
#             # Send the API request and get the response
#             response = openai.Whisper.read(audio=audio_data)
#
#             # Check the response status
#             if response['status'] != 'completed':
#                 logger.error(f"Failed to transcribe audio file: {file_path}. Response status: {response['status']}")
#                 return None
#
#             # Get the transcription
#             result = response['choices'][0]['text']
#             return result

def convert_to_yaml(transcription_text):
    """Convert transcription text to YAML format."""
    # Here we assume the transcription text follows a certain structure
    # This function should be adapted based on the specific structure of the transcription text
    data = yaml.safe_load(transcription_text)
    return data


def convert_yaml_to_markdown(yaml_data):
    """Convert YAML data to markdown formatted checklist using OpenAI API."""
    logger.info("Converting YAML to Markdown format")
    markdown_text = md(yaml.dump(yaml_data))
    return markdown_text


def process_file(file_path):
    """Process a single audio file to generate markdown checklist."""
    transcription = transcribe_audio(file_path)
    yaml_data = convert_to_yaml(transcription)
    markdown_content = convert_yaml_to_markdown(yaml_data)
    output_file = file_path.rsplit('.', 1)[0] + '.md'
    with open(output_file, 'w') as f:
        f.write(markdown_content)
    logger.info(f"Markdown checklist generated: {output_file}")
    return output_file


def process_directory(directory_path):
    """Process all audio files in a directory (non-recursively) to generate a combined markdown checklist."""
    audio_files = [f for f in glob.glob(os.path.join(directory_path, '*')) if is_audio_file(f)]
    all_transcriptions = []
    for audio_file in audio_files:
        transcription = transcribe_audio(audio_file)
        yaml_data = convert_to_yaml(transcription)
        all_transcriptions.append(yaml_data)

    combined_yaml = {'list': all_transcriptions}
    markdown_content = convert_yaml_to_markdown(combined_yaml)
    output_file = os.path.join(directory_path, os.path.basename(directory_path).upper() + '.md')
    with open(output_file, 'w') as f:
        f.write(markdown_content)
    logger.info(f"Combined markdown checklist generated: {output_file}")
    return output_file


def main(input_path):
    """Main function to process input file or directory."""
    if os.path.isfile(input_path):
        output_file = process_file(input_path)
        logger.success(f"Markdown checklist generated: {output_file}")
    elif os.path.isdir(input_path):
        output_file = process_directory(input_path)
        logger.success(f"Combined markdown checklist generated: {output_file}")
    else:
        logger.error("Invalid input path. Please provide a valid file or directory.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        logger.error("Usage: python transcriber.py <file_or_directory_path>")
    else:
        main(sys.argv[1])
