import tempfile

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

# --- HELPER FUNCTIONS --- #
def str_to_bool(value):
    """Receives all sorts of values and interprets as yes or no"""
    if value is None:
        return False
    str_to_bool_map = {'true': True, 'yes': True, 'y': True, '0': False, 'false': False, 'no': False, 'n': False, '1': True}  # TODO: this is sort of haphazard, should be more robust I think
    try:
        return str_to_bool_map.get(value.lower(), bool(int(value)))
    except ValueError:
        return False


# Populate environment variables from .env file
load_dotenv()

# Check dev env vars / special behavior
skip_whisper = not str_to_bool(os.getenv("DEV_LOAD_TRANSCRIPT"))  # no idea why I flipped it like this, but it's what I did
transcript_path = os.getenv("DEV_TRANSCRIPT_PATH")  # should be to a text file that is the transcript in plaintext

if skip_whisper:
    logger.warning("[DEV_LOAD_TRANSCRIPT] Skipping OpenAI transcription of audio files & loading transcript from file instead.")
    logger.info(f"Loading transcript from {transcript_path}...")
    with open(transcript_path, 'r') as f:
        dev_transcript = f.read()


# get OpenAI API key from env var
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    logger.error("OPENAI_API_KEY is not set in the .env file. Please set it before running the script.")
    exit(1)

client = OpenAI(api_key=openai_api_key)
logger.info(Fore.GREEN + "Initialized OpenAI client." + Style.RESET_ALL)

def is_audio_file(filename):
    """Check if the file is a supported audio format."""
    return any(filename.endswith(ext) for ext in ['.mp3', '.m4a', '.aac', '.wav'])

def split_audio(file):
    chunk_size_ms = 9 * 60 * 1000  # 9 minutes in milliseconds

    chunk_files = []

    for i in range(0, len(file), chunk_size_ms):
        chunk = file[i:i+chunk_size_ms]
        # Create a temporary file for each chunk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            chunk.export(f.name, format='mp3')
            chunk_files.append(f.name)

    return chunk_files


# TODO:
#   - Use the prompt field to provide context from the previous chunks (which must be length minimized as the max length is 241 characters)
def transcribe_chunk(chunk, prompt):
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=chunk,
        language="en",
        prompt=prompt,
        response_format="text",
    )

    return transcription

def transcribe_audio(file):
    audio_f = AudioSegment.from_file(file)
    if audio_f.duration_seconds < 10 * 60:  # 10 minutes
        return transcribe_chunk(audio_f, "")  # no prompt is correct when there is only 1 chunk
    chunk_files = split_audio(audio_f)

    transcription = ""

    index = 0
    for chunk_file in chunk_files:
        with open(chunk_file, 'rb') as f:
            logger.info(f"Transcribing chunk: {chunk_file}" + Fore.WHITE + f" ({index + 1}/{len(chunk_files)})" + Style.RESET_ALL)
            logger.debug(Fore.LIGHTWHITE_EX + f"chunk size: {os.path.getsize(chunk_file)} bytes" + Style.RESET_ALL)
            transcription += transcribe_chunk(f, transcription)
        os.remove(chunk_file)  # Delete the temporary file
        index += 1

    return transcription

# TODO: FINISH THIS FUNCTION (it's like the entire value-add)
def convert_to_yaml(transcription_text):
    """Convert transcription text to YAML format."""
    # TODO:
    #  - Use GPT-4 or GPT-4o to convert the transcription into a nested list in YAML format
    #  - This is the highest priority task since transcription works.
    #  - Testing this will also be expensive, so it needs to be done carefully.
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
    if skip_whisper:
        transcription = dev_transcript
    else:
        transcription = transcribe_audio(file_path)
    yaml_data = convert_to_yaml(transcription)
    markdown_content = convert_yaml_to_markdown(yaml_data)
    output_file = file_path.rsplit('.', 1)[0] + '.md'
    with open(output_file, 'w') as f:
        f.write(markdown_content)
    logger.info(f"Markdown checklist generated: {output_file}")
    return output_file


# TODO:
#   - Handle parse order of audio files: in example, they were parsed in an order which is not the order they were recorded in;
#   - This may end up being inconsequential, but it's worth noting
def process_directory(directory_path):
    """Process all audio files in a directory (non-recursively) to generate a combined markdown checklist."""
    if skip_whisper:
        combined_yaml = convert_to_yaml(dev_transcript)
    else:
        audio_files = [f for f in glob.glob(os.path.join(directory_path, '*')) if is_audio_file(f)]
        all_transcriptions = []
        for audio_file in audio_files:
            logger.info(f"Processing audio file: {audio_file}")
            transcription = transcribe_audio(audio_file)
            logger.info(f"Transcription: {transcription}")
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
