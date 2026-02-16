

import requests
import glob
import gzip
import shutil
import os
import sys
import csv



def download_file(url, filename):
    """Download a file"""
    
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_length = int(response.headers.get('content-length', 0))
        downloaded = 0
        with open(filename, 'wb') as file:
            print(f"Downloading {filename}...")
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                downloaded += size
                done = int(50 * downloaded / total_length)
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded}B/{total_length}B")
                sys.stdout.flush()
        print("\nDownload complete.")
    else:
        print(f"Failed to download {filename}")



def save_all_text_lines_to_file(input_filename, output_filename):
    """Save all text lines from a CSV file to a text file, filtering by predicted_language."""
    
    with open(input_filename, 'r', encoding='utf-8') as input_file, \
         open(output_filename, 'w', encoding='utf-8') as output_file:
        line_count = 0
        for line in input_file:
            line_count += 1
            if 'kaz' in line:
                text = line.split(',')[0].strip('"')
                output_file.write(text + '\n')
        print(f"Processed {line_count} lines and saved to {output_filename}")



def merge_csv_files(input_pattern, output_filename):
    """Merge multiple CSV files into one large CSV file with low memory usage."""
    
    print(f"Starting to merge files matching {input_pattern} into {output_filename}")
    
    with open(output_filename, 
                'w', 
                newline='', 
                encoding='utf-8') as output_file:
        
        writer = None
        
        for input_filename in glob.glob(input_pattern):
            print(f"Processing file: {input_filename}")
            with open(input_filename, 'r', newline='', encoding='utf-8') as input_file:
                reader = csv.reader(input_file)
                if writer is None:
                    writer = csv.writer(output_file)
                    headers = next(reader)
                    writer.writerow(headers)
                for row in reader:
                    writer.writerow(row)
    
    print(f"Merging complete. Output saved to {output_filename}")



def filter_and_write_lines(input_filename, output_filename):
    line_count = 0
    with open(input_filename, 'r', encoding='utf-8') as input_file, \
         open(output_filename, 'w', encoding='utf-8') as output_file:
        csv_reader = csv.reader(input_file)
        for row in csv_reader:
            if row[1] != 'rus':
                text = '"' + row[0].strip('"') + '"'
                output_file.write(text + '\n')
                line_count += 1
    print(f"Filtered and wrote {line_count} lines excluding Russian language entries.")



def show_unique_characters_in_file(filename):
    """Print the sorted list of unique characters in a file."""
    with open(filename, 'r', encoding='utf-8') as file:
        contents = file.read()
    unique_characters = set(contents)
    print(sorted(unique_characters))



def clean_file_contents(input_filename, output_filename):
    """Clean the file contents by allowing only specified characters."""
    
    allowed_chars = set('\n !(),-.0123456789?АаӘәБбВвГгҒғДдЕеЁёЖжЗзИиЙйКкҚқЛлМмНнҢңОоӨөПпРрСсТтУуҰұҮүФфХхҺһЦцЧчШшЩщЪъЫыІіЬьЭэЮюЯя')
    print(f"Cleaning the contents of {input_filename} and saving to {output_filename}")
    
    with open(input_filename, 'r', encoding='utf-8') as input_file, \
         open(output_filename, 'w', encoding='utf-8') as output_file:
        line_count = 0
        char_count = 0
        for line in input_file:
            cleaned_line = ''.join(filter(allowed_chars.__contains__, line))
            output_file.write(cleaned_line)
            line_count += 1
            char_count += len(cleaned_line)
            print(f"Processed {line_count} lines and {char_count} characters...", end='\r')
    print("\nCleaning complete. The cleaned contents have been saved.")


def compress_file(input_filename, output_filename):
    """Compress a file using gzip with logging and simple text progress status."""
    
    print(f"Compressing {input_filename} to {output_filename}")
    total_size = os.path.getsize(input_filename)
    processed_size = 0
    with open(input_filename, 'rb') as file_in:
        with gzip.open(output_filename, 'wb') as file_out:
            while True:
                chunk = file_in.read(1024)
                if not chunk:
                    break
                file_out.write(chunk)
                processed_size += len(chunk)
                progress = (processed_size / total_size) * 100
                print(f"Progress: {progress:.2f}%", end='\r')
    print("\nCompression complete. The file {output_filename} is ready.")


def copy_compressed_file_to_content_folder(src, dst_folder):
    """Copy a compressed file to the specified content folder with logging and simple text progress status."""
    
    print(f"Copying {src} to {dst_folder}")
    total_size = os.path.getsize(src)
    processed_size = 0
    with open(src, 'rb') as file_in:
        with open(os.path.join(dst_folder, os.path.basename(src)), 'wb') as file_out:
            while True:
                chunk = file_in.read(1024)
                if not chunk:
                    break
                file_out.write(chunk)
                processed_size += len(chunk)
                progress = (processed_size / total_size) * 100
                print(f"Progress: {progress:.2f}%", end='\r')
    print(f"\nCopy complete. The file {src} has been copied to {dst_folder}")

# Display the first 10 lines of a file
# show_first_10_lines('leipzig.csv')

# Display the first 10 text lines from a CSV file
# show_first_10_text_lines('leipzig.csv')

# Show unique characters in a file
# show_unique_characters_in_file('text.txt')
# show_unique_characters_in_file('cleaned_text.txt')



# Start the download process
download_file('https://huggingface.co/datasets/kz-transformers/multidomain-kazakh-dataset/resolve/main/', 'kazakhNews.csv')
download_file('https://huggingface.co/datasets/kz-transformers/multidomain-kazakh-dataset/resolve/main/', 'kazakhBooks.csv')
download_file('https://huggingface.co/datasets/kz-transformers/multidomain-kazakh-dataset/resolve/main/', 'leipzig.csv')
download_file('https://huggingface.co/datasets/kz-transformers/multidomain-kazakh-dataset/resolve/main/', 'oscar.csv')
download_file('https://huggingface.co/datasets/kz-transformers/multidomain-kazakh-dataset/resolve/main/', 'cc100-monolingual-crawled-data.csv')


# Combine CSV files into a single dataset
merge_csv_files('*.csv', 'multidomain-kazakh-dataset.csv')

# Filter and write lines to a file
# filter_and_write_lines('leipzig.csv', 'filtered_leipzig.csv')


# Save all text lines from a CSV file to a text file
# save_all_text_lines_to_file('filtered_leipzig.csv', 'filtered_leipzig.txt')

# Clean the file contents
clean_file_contents('filtered_leipzig.csv', 'cleaned_leipzig.txt')

# Compress the cleaned file
# compress_file('cleaned_text.txt', 'cleaned_text.txt.gz')

# Copy the compressed file to the content folder
# copy_compressed_file_to_content_folder('cleaned_text.txt.gz', 'content/')


