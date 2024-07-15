import os
import subprocess
import math

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
all_dirs = [os.path.join(base_dicom_folder, d) for d in os.listdir(base_dicom_folder) if os.path.isdir(os.path.join(base_dicom_folder, d))]
dirs_to_process = []

# 변환된 파일 확인 함수
def is_converted(dicom_dir):
    # 변환된 NIfTI 파일이 있는지 확인 (디렉토리 이름으로 확인)
    expected_nifti_file = os.path.join(output_folder, os.path.basename(dicom_dir) + '.nii.gz')
    return os.path.exists(expected_nifti_file)

for dir in all_dirs:
    if dir not in processed_dirs and not is_converted(dir):
        dirs_to_process.append(dir)

# 디렉터리를 60개씩 나누기
chunk_size = 60
num_chunks = math.ceil(len(dirs_to_process) / chunk_size)

for i in range(num_chunks):
    chunk_dirs = dirs_to_process[i*chunk_size:(i+1)*chunk_size]
    print(f"Processing chunk {i+1}/{num_chunks}: {len(chunk_dirs)} directories")

    for dir in chunk_dirs:
        print(f"  Processing directory: {dir}")
        try:
            result = subprocess.run(["dcm2niix", "-z", "y", "-v", "1", "-o", output_folder, dir], check=True, capture_output=True, text=True)
            print(result.stdout)  # 변환 작업의 출력 내용을 표시
            print(f"  Successfully processed: {dir}")
            # 변환된 디렉터리 기록
            with open(processed_log, 'a') as f:
                f.write(dir + '\n')
        except subprocess.CalledProcessError as e:
            print(f"  Error processing directory {dir}: {e.stderr}")
            # 오류 디렉터리 기록
            with open(error_log, 'a') as f:
                f.write(dir + '\n')
        except Exception as e:
            print(f"  Unexpected error processing directory {dir}: {str(e)}")
            # 오류 디렉터리 기록
            with open(error_log, 'a') as f:
                f.write(dir + '\n')

print("Selected directories processed.")
