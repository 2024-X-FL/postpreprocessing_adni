import os
import subprocess
import math
import nibabel as nib

# 상위 디렉터리 설정
base_dicom_folder = "/mnt/d/ADNI/extracted_files/ADNI"
output_folder = "/mnt/d/ADNI/nifti_files"
processed_log = "/mnt/d/ADNI/scripts/processed_folders.log"
error_log = "/mnt/d/ADNI/scripts/error_folders.log"

# 출력 폴더 생성
os.makedirs(output_folder, exist_ok=True)

# 이미 처리된 폴더 목록 가져오기
if os.path.exists(processed_log):
    with open(processed_log, 'r') as f:
        processed_dirs = set(f.read().splitlines())
else:
    processed_dirs = set()

# 디렉터리 목록 가져오기
all_dirs = [os.path.join(base_dicom_folder, d) for d in os.listdir(base_dicom_folder) if
            os.path.isdir(os.path.join(base_dicom_folder, d))]
dirs_to_process = []


# 변환된 파일 확인 함수
def is_converted(dicom_dir):
    # 변환된 NIfTI 파일이 있는지 확인
    converted_files = [f for f in os.listdir(output_folder) if f.startswith(os.path.basename(dicom_dir))]
    for f in converted_files:
        if f.endswith('.nii.gz'):
            nifti_file_path = os.path.join(output_folder, f)
            img = nib.load(nifti_file_path)
            if len(img.shape) == 4:  # Check if the NIfTI file is 4D
                return True
    return False


for dir in all_dirs:
    if dir not in processed_dirs and not is_converted(dir):
        dirs_to_process.append(dir)

# 디렉터리를 60개씩 나누기
chunk_size = 60
num_chunks = math.ceil(len(dirs_to_process) / chunk_size)

for i in range(num_chunks):
    chunk_dirs = dirs_to_process[i * chunk_size:(i + 1) * chunk_size]
    print(f"Processing chunk {i + 1}/{num_chunks}: {len(chunk_dirs)} directories")

    for dir in chunk_dirs:
        print(f"  Processing directory: {dir}")
        try:
            result = subprocess.run(["dcm2niix", "-z", "y", "-v", "1", "-o", output_folder, dir], check=True,
                                    capture_output=True, text=True)
            print(result.stdout)

            # 변환된 NIfTI 파일 확인
            converted_files = [f for f in os.listdir(output_folder) if f.startswith(os.path.basename(dir))]
            nifti_files = [f for f in converted_files if f.endswith('.nii.gz')]
            nifti_files_4d = []
            for nifti_file in nifti_files:
                nifti_file_path = os.path.join(output_folder, nifti_file)
                img = nib.load(nifti_file_path)
                if len(img.shape) == 4:  # 4D NIfTI 파일인지 확인
                    print(f"  Successfully processed 4D NIfTI: {nifti_file_path}")
                    nifti_files_4d.append(nifti_file_path)
                else:
                    print(f"  NIfTI file is not 4D: {nifti_file_path}")
                    os.remove(nifti_file_path)
                    print(f"  Non-4D NIfTI file removed: {nifti_file_path}")

            # 4D NIfTI 파일이 하나 이상 있는 경우에만 디렉토리를 처리된 것으로 기록
            if nifti_files_4d:
                with open(processed_log, 'a') as f:
                    f.write(dir + '\n')
            else:
                with open(error_log, 'a') as f:
                    f.write(f"Non-4D files in directory {dir}\n")

        except subprocess.CalledProcessError as e:
            print(f"  Error processing directory {dir}: {e.stderr}")
            # 오류 디렉터리 기록
            with open(error_log, 'a') as f:
                f.write(f"{dir}: {e.stderr}\n")
        except Exception as e:
            print(f"  Unexpected error processing directory {dir}: {str(e)}")
            # 오류 디렉터리 기록
            with open(error_log, 'a') as f:
                f.write(f"{dir}: {str(e)}\n")

print("Selected directories processed.")
