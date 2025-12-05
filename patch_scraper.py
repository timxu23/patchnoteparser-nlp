'''
simple patch note scraper
this will help us scrape a bunch of testing texts to test the rest of our parsing
takes in args -- the first n args are the patchnote url to be scraped, and
the last arg is directory to be saved in. default: public/patch-notes-test/

USAGE: python3 patch_scraper.py <your-urls> -o custom/save-path
'''

# referenced bs4 samplecode:
# Source - https://stackoverflow.com/a
# Posted by PeYoTlL, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-20, License - CC BY-SA 4.0

from urllib.request import urlopen
from bs4 import BeautifulSoup
import sys
import os
import re
import argparse
from urllib.parse import urlparse

                            # === parser ===
def parse_arguments():
    '''parses with provided patch note url arguments'''

    parser = argparse.ArgumentParser(description="A script that processes files.")

    parser.add_argument("urls", metavar='URL', type=str, nargs='+', help="Need url")

    parser.add_argument('-o', '--output-dir', dest='dir_path', 
                        type=str, default='public/patch-notes-test/', 
                        help='directory where everything is saved at. Default: public/patch-notes-test/')
    return parser.parse_args()


            # ================ [ main scraping logic] ================
def run_scraper():
    args = parse_arguments()
    urls = args.urls
    dir_path = args.dir_path

    # validity checks
    if not urls: 
        print('[SCRAPER] NO FILES PROVIDED. EXITING.')
        sys.exit(1)

    print(f"[SCRAPER] PROCESSING {len(sys.argv)} FILES")
    print(f"[SCRAPER] OUTPUT DIRECTORY SET TO: {dir_path}")

    try:
        os.makedirs(dir_path, exist_ok=True)
        print(f'[SCRAPER] directory setup complete: {dir_path}')
    except OSError as e:
        print(f'[SCRAPER] ERROR: could not create a directory {dir_path}. \n {e}')
        sys.exit(1)

    # scraping 
    for url in urls:
        try:
            html = urlopen(url).read()
        except Exception as e:
            print(f"[SCRAPER] ERROR opening URL {url}: {e}")
            continue # skip to the next URL
        soup = BeautifulSoup(html, features="html.parser")

        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        # ------------ file saving -------------
        
        # Create a clean filename from the URL path
        try:
            parsed_url = urlparse(url)
            base_filename = parsed_url.path.strip('/').split('/')[-1]
            if not base_filename:
                base_filename = parsed_url.netloc
        except:
            base_filename = 'scraped_content'

        filename_slug = slugify(base_filename)
        filename = f"{filename_slug or 'unknown-url'}.txt"
        output_path = os.path.join(dir_path, filename)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"[SCRAPER] SAVED content from {url} to {output_path}")
        except Exception as e:
            print(f"[SCRAPER] ERROR writing file {output_path}: {e}")


                # ============ [HELPER FNS] =============

def slugify(value):
    value = str(value).lower()
    value = re.sub(r'[^\w\s-]', '', value)
    value = re.sub(r'[-\s]+', '-', value).strip('-')
    
    return value


if __name__ == "__main__":
    run_scraper()