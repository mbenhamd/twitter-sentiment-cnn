import requests, zipfile
import io
import os
import shutil
import tempfile
import zipfile
import urllib
from tqdm import tqdm

def download_from_url(url, dst):
    """
    @param: url to download file
    @param: dst place to put the file
    """
    file_size = int(urllib.request.urlopen(url).info().get('Content-Length', -1))
    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)
    else:
        first_byte = 0
    if first_byte >= file_size:
        return file_size
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
        total=file_size, initial=first_byte,
        unit='B', unit_scale=True, desc=url.split('/')[-1])
    req = requests.get(url, headers=header, stream=True)
    with(open(dst, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return file_size

DATASET_FOLDER = 'twitter-sentiment-dataset/'
URL = 'http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip'

if not os.path.exists(DATASET_FOLDER):
    os.mkdir(DATASET_FOLDER)
zip_file = ''.join([DATASET_FOLDER, 'sentiment-dataset.csv.zip'])

print('Downloading dataset...')
fsize = download_from_url(URL, zip_file)

print('Extracting data...')
z = zipfile.ZipFile(zip_file)
z.extractall(DATASET_FOLDER)
z.close()
old_file = ''.join([DATASET_FOLDER, 'Sentiment Analysis Dataset.csv'])
new_file = ''.join([DATASET_FOLDER, 'sentiment-dataset.csv'])
os.rename(old_file, new_file)
os.remove(zip_file)
print('Done. Exiting...')
