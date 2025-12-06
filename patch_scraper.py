'''
simple patch note scraper
this will help us scrape a bunch of testing texts to test the rest of our parsing
takes in args -- the first n args are the patchnote url to be scraped, and
the last arg is directory to be saved in. default: public/patch-notes-html/

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
                        type=str, default='public/patch-notes-html/', 
                        help='directory where everything is saved at. Default: public/patch-notes-html/')
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
    text_dir = 'public/patch-notes-test/'

    for target_dir in (dir_path, text_dir):
        try:
            os.makedirs(target_dir, exist_ok=True)
            print(f'[SCRAPER] directory setup complete: {target_dir}')
        except OSError as e:
            print(f'[SCRAPER] ERROR: could not create a directory {target_dir}. \n {e}')
            sys.exit(1)

    # scraping 
    for url in urls:
        try:
            html = urlopen(url).read()
        except Exception as e:
            print(f"[SCRAPER] ERROR opening URL {url}: {e}")
            continue
        soup = BeautifulSoup(html, features="html.parser")

        for script in soup(["script", "style"]):
            script.extract()
        content_root = soup.body or soup
        html_output = content_root.prettify()
        text_output = html_output

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
        filename = f"{filename_slug or 'unknown-url'}.html"
        output_path = os.path.join(dir_path, filename)
        text_path = os.path.join(text_dir, f"{filename_slug or 'unknown-url'}.txt")

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_output)
            print(f"[SCRAPER] SAVED content from {url} to {output_path}")
        except Exception as e:
            print(f"[SCRAPER] ERROR writing file {output_path}: {e}")
        try:
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text_output)
            print(f"[SCRAPER] SAVED text dump from {url} to {text_path}")
        except Exception as e:
            print(f"[SCRAPER] ERROR writing text file {text_path}: {e}")


                # ============ [HELPER FNS] =============

def slugify(value):
    value = str(value).lower()
    value = re.sub(r'[^\w\s-]', '', value)
    value = re.sub(r'[-\s]+', '-', value).strip('-')
    
    return value


if __name__ == "__main__":
    run_scraper()
