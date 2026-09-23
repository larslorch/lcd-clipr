from pathlib import Path
import shutil
import zipfile
import argparse
import fnmatch
from datetime import datetime

def collect_and_zip_csv_files(path, target_folder, *, crawl="df__*.csv", exclude=None):
    base_path = Path(path)
    if not base_path.is_absolute():
        base_path = Path.cwd() / base_path
    base_path = base_path.resolve()

    timestamp = datetime.now().strftime("--%Y-%m-%d--%H-%M")

    target_folder = base_path / (target_folder + timestamp)
    target_folder.mkdir(parents=True, exist_ok=True)

    files_found = []
    for file_path in base_path.rglob(crawl):
        # avoid crawling the target folder
        if target_folder in file_path.parents:
            continue

        # Skip files inside excluded folders matching glob
        if exclude is not None:
            if any(fnmatch.fnmatch(parent.name, exclude) for parent in file_path.parents):
                continue

        files_found.append(file_path)

        print(file_path)

        # make flattened path the new file name in case names are not unique
        # excludes all folders that are standard in a shallow experiment folder setup
        relative_path = file_path.relative_to(base_path)
        flattened_name = "_".join(relative_path.parts[:-3] + relative_path.parts[-1:])
        shutil.copy2(file_path, target_folder / flattened_name)

    zip_path = base_path / f"{target_folder}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in target_folder.iterdir():
            zipf.write(file, arcname=file.name)

    shutil.rmtree(target_folder)

    print(f"Found {len(files_found)} files matching '{crawl}'")
    if exclude:
        print(f"Excluded files matching '{exclude}'")
    print(f"Zipped all matching files into: {zip_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("target_folder", type=str)
    parser.add_argument("--crawl", type=str, default="df__*.csv")
    parser.add_argument("--exclude", type=str, default="2025*")
    kwargs = parser.parse_args()

    collect_and_zip_csv_files(kwargs.path, kwargs.target_folder, crawl=kwargs.crawl, exclude=kwargs.exclude)