import os
import argparse
import zipfile
import wget


def download(url, targetdir):
    """
    Download a file and save it in some target directory.
    Args:
        url: The url from which the file must be downloaded.
        targetdir: The path to the directory where the file must be saved.
    Returns:
        The path to the downloaded file.
    """
    print("* Downloading data from {}...".format(url))
    filepath = os.path.join(targetdir, url.split('/')[-1])
    wget.download(url, filepath)
    return filepath


def unzip(filepath):
    """
    Extract the data from a zipped file and delete the archive.
    Args:
        filepath: The path to the zipped file.
    """
    print("\n* Extracting: {}...".format(filepath))
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        for name in zf.namelist():
            # Ignore useless files in archives.
            if "__MACOSX" in name or\
               ".DS_Store" in name or\
               "Icon" in name:
                continue
            zf.extract(name, dirpath)
    # Delete the archive once the data has been extracted.
    os.remove(filepath)


def download_unzip(url, targetdir):
    """
    Download and unzip data from some url and save it in a target directory.
    Args:
        url: The url to download the data from.
        targetdir: The target directory in which to download and unzip the
                   data.
    """
    filepath = os.path.join(targetdir, url.split('/')[-1])
    target = os.path.join(targetdir,
                          ".".join((url.split('/')[-1]).split('.')[:-1]))

    if not os.path.exists(targetdir):
        print("* Creating target directory {}...".format(targetdir))
        os.makedirs(targetdir)

    # Skip download and unzipping if the unzipped data is already available.
    if os.path.exists(target) or os.path.exists(target + ".txt"):
        print("* Found unzipped data in {}, skipping download and unzip..."
              .format(targetdir))
    # Skip downloading if the zipped data is already available.
    elif os.path.exists(filepath):
        print("* Found zipped data in {} - skipping download..."
              .format(targetdir))
        unzip(filepath)
    # Download and unzip otherwise.
    else:
        unzip(download(url, targetdir))


if __name__ == "__main__":
    # Default data.
    snli_url = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
    mnli_url = "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"
 

    parser = argparse.ArgumentParser(description='Download the SNLI dataset')
    parser.add_argument("--dataset_url",
                        default=snli_url,
                        help="URL of the dataset to download")
    parser.add_argument("--target_dir",
                        default=os.path.join("../datasets"),
                        help="Path to a directory where data must be saved")
    args = parser.parse_args()
    print(args.dataset_url)
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    print(20*"=", "Fetching the dataset:", 20*'=')
    download_unzip(args.dataset_url, args.target_dir)
