import os
import nibabel as nib
import numpy as np
from nilearn import image, masking

# 경로 설정
nifti_folder = "/mnt/d/ADNI/nifti_files"
processed_folder = "/mnt/d/ADNI/processed_niftyreg"
log_file_path = "/mnt/d/ADNI/scripts/processed_nifti_files.log"
template_path = "/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz"

# 로그 파일 읽기
processed_files = set()
if os.path.exists(log_file_path):
    with open(log_file_path, 'r') as log_file:
        processed_files = set(line.strip() for line in log_file)

# 출력 폴더 생성
os.makedirs(processed_folder, exist_ok=True)

# NIfTI 파일 목록 가져오기
nifti_files = [f for f in os.listdir(nifti_folder) if f.endswith('.nii.gz') and f not in processed_files]

# NIfTI 파일 전처리
for nifti_file in nifti_files:
    nifti_file_path = os.path.join(nifti_folder, nifti_file)
    print(f"Processing {nifti_file_path}")

    # 파일 로드
    img = nib.load(nifti_file_path)
    img_data = img.get_fdata()

    # MNI152 템플릿 로드
    mni_template = nib.load(template_path)

    # 슬라이스별로 처리
    processed_slices = []
    for i in range(img_data.shape[3]):
        print(f"Processing slice {i + 1}/{img_data.shape[3]}...")
        img_slice = nib.Nifti1Image(img_data[..., i], img.affine)

        # MNI 공간으로 매핑
        img_mni_slice = image.resample_to_img(img_slice, mni_template, interpolation='continuous')

        # 비뇌 영역 제거
        brain_mask_slice = masking.compute_brain_mask(img_mni_slice)
        resampled_brain_mask_slice = image.resample_to_img(brain_mask_slice, img_mni_slice, interpolation='nearest')
        brain_img_slice = image.math_img("img * mask", img=img_mni_slice, mask=resampled_brain_mask_slice)

        # 슬라이스 추가
        processed_slices.append(brain_img_slice.get_fdata())

    # 모든 슬라이스를 다시 4D로 결합
    processed_data = np.stack(processed_slices, axis=-1)
    processed_img = nib.Nifti1Image(processed_data, img.affine, img.header)

    # 결과 저장
    output_file = os.path.join(processed_folder, f"{os.path.basename(nifti_file).replace('.nii.gz', '')}_processed.nii.gz")
    nib.save(processed_img, output_file)
    print(f"Processed file saved: {output_file}")

    # 로그 파일에 기록
    with open(log_file_path, 'a') as log_file:
        log_file.write(nifti_file + '\n')

print("All NIfTI files processed.")
