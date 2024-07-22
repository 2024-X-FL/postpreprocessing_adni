import os
import nibabel as nib
import numpy as np
from nilearn import image, masking
import subprocess

# 경로 설정
nifti_folder = "/mnt/d/ADNI/nifti_files"
processed_folder = "/mnt/d/ADNI/processed_niftyreg"
log_file_path = "/mnt/d/ADNI/scripts/processed_nifti_files.log"
error_log_file_path = "/mnt/d/ADNI/scripts/error_nifti_files.log"
template_path = "/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz"

# BET에 사용할 데이터 선택
use_first_volume = False
use_mean_volume = False
use_each_frame = True  # 세 개의 옵션 중 하나만 True로 설정

# 로그 파일 읽기
processed_files = set()
if os.path.exists(log_file_path):
    with open(log_file_path, 'r') as log_file:
        processed_files = set(line.strip() for line in log_file)

error_files = set()
if os.path.exists(error_log_file_path):
    with open(error_log_file_path, 'r') as log_file:
        error_files = set(line.strip() for line in log_file)

# NIfTI 파일 목록 가져오기
nifti_files = [os.path.join(nifti_folder, f) for f in os.listdir(nifti_folder) if f.endswith('.nii.gz')]

# 출력 폴더 생성
os.makedirs(processed_folder, exist_ok=True)

# NIfTI 파일 전처리
for nifti_file_path in nifti_files:
    if nifti_file_path in processed_files:
        print(f"{nifti_file_path} is already processed. Skipping...")
        continue

    if nifti_file_path in error_files:
        print(f"{nifti_file_path} previously encountered an error. Skipping...")
        continue

    try:
        print(f"Processing {nifti_file_path}")

        # Motion correction (MCFLIRT from FSL)
        mcflirt_output = nifti_file_path.replace('.nii.gz', '_mc.nii.gz')
        subprocess.run(["mcflirt", "-in", nifti_file_path, "-out", mcflirt_output], check=True)

        # 파일 로드
        img = nib.load(mcflirt_output)
        img_data = img.get_fdata()
        affine = img.affine
        header = img.header

        # 환자 폴더 생성
        file_name = os.path.basename(nifti_file_path).replace('.nii.gz', '')
        patient_id = file_name.split('_')[0] + '_' + file_name.split('_')[1]
        session_id = file_name.replace(patient_id + '_', '')
        patient_folder = os.path.join(processed_folder, patient_id)
        session_folder = os.path.join(patient_folder, session_id)
        os.makedirs(session_folder, exist_ok=True)

        # 시간 프레임 결과 폴더 생성
        frames_folder = os.path.join(session_folder, "frames")
        os.makedirs(frames_folder, exist_ok=True)

        processed_slices = []
        num_frames = img_data.shape[-1]

        # BET 방법 선택
        if use_first_volume:
            print("Using the first volume for BET")
            bet_input_img = nib.Nifti1Image(img_data[..., 0], affine, header)
            bet_input_path = os.path.join(session_folder, f"{file_name}_first_volume.nii.gz")
            nib.save(bet_input_img, bet_input_path)
            bet_output_path = os.path.join(session_folder, f"{file_name}_brain.nii.gz")
            subprocess.run(["bet", bet_input_path, bet_output_path, "-f", "0.5", "-g", "0"], check=True)
            if not os.path.exists(bet_output_path):
                raise Exception(f"Brain extraction failed for first volume")

        elif use_mean_volume:
            print("Using the mean volume for BET")
            mean_data = np.mean(img_data, axis=-1)
            bet_input_img = nib.Nifti1Image(mean_data, affine, header)
            bet_input_path = os.path.join(session_folder, f"{file_name}_mean_volume.nii.gz")
            nib.save(bet_input_img, bet_input_path)
            bet_output_path = os.path.join(session_folder, f"{file_name}_brain.nii.gz")
            subprocess.run(["bet", bet_input_path, bet_output_path, "-f", "0.5", "-g", "0"], check=True)
            if not os.path.exists(bet_output_path):
                raise Exception(f"Brain extraction failed for mean volume")

        
        elif use_each_frame:
            for frame in range(num_frames):
                print(f"Processing frame {frame + 1}/{num_frames}...")
                frame_data = img_data[..., frame]
                frame_img = nib.Nifti1Image(frame_data, affine, header)
        
                # 프레임 데이터 저장
                frame_output = os.path.join(frames_folder, f"{file_name}_frame_{frame}.nii.gz")
                nib.save(frame_img, frame_output)
        
                # 뇌 추출 (BET from FSL)
                brain_output = os.path.join(frames_folder, f"{file_name}_brain_{frame}.nii.gz")
                subprocess.run(["bet", frame_output, brain_output, "-f", "0.5", "-g", "0"], check=True)
                if not os.path.exists(brain_output):
                    raise Exception(f"Brain extraction failed for time frame {frame + 1}")

        for frame in range(num_frames):
            print(f"Processing frame {frame + 1}/{num_frames}...")
            frame_data = img_data[..., frame]
            frame_img = nib.Nifti1Image(frame_data, affine, header)

            # 프레임 데이터 저장
            frame_output = os.path.join(frames_folder, f"{file_name}_frame_{frame}.nii.gz")
            nib.save(frame_img, frame_output)

            # 뇌 추출 (BET from FSL)
            if use_each_frame:
                brain_output = os.path.join(frames_folder, f"{file_name}_brain_{frame}.nii.gz")
                subprocess.run(["bet", frame_output, brain_output, "-f", "0.5", "-g", "0"], check=True)
                if not os.path.exists(brain_output):
                    raise Exception(f"Brain extraction failed for time frame {frame + 1}")

            # Affine 정렬 (reg_aladin from NiftyReg)
            affine_output = os.path.join(frames_folder, f"{file_name}_affine_{frame}.nii.gz")
            subprocess.run(["reg_aladin", "-ref", template_path, "-flo", brain_output if use_each_frame else bet_output_path, "-res", affine_output], check=True)

            if not os.path.exists(affine_output):
                raise Exception(f"Affine registration failed for time frame {frame + 1}")

            # 비선형 정렬 (reg_f3d from NiftyReg)
            nonlinear_output = os.path.join(frames_folder, f"{file_name}_nonlinear_{frame}.nii.gz")
            subprocess.run(["reg_f3d", "-ref", template_path, "-flo", affine_output, "-res", nonlinear_output], check=True)

            if not os.path.exists(nonlinear_output):
                raise Exception(f"Nonlinear registration failed for time frame {frame + 1}")

            # 비뇌 영역 제거
            brain_mask_output = os.path.join(frames_folder, f"{file_name}_mask_{frame}.nii.gz")
            resampled_brain_img = image.resample_to_img(brain_output if use_each_frame else bet_output_path, nonlinear_output)
            brain_mask = masking.compute_brain_mask(resampled_brain_img)
            brain_img_slice = image.math_img("img * mask", img=resampled_brain_img, mask=brain_mask)
            nib.save(brain_img_slice, brain_mask_output)

            # 처리된 슬라이스 추가
            processed_slices.append(nib.load(brain_mask_output).get_fdata())

            # 임시 파일 삭제
            os.remove(frame_output)
            if use_each_frame:
                os.remove(brain_output)
            os.remove(affine_output)
            os.remove(nonlinear_output)

        # 4D NIfTI 파일로 결합
        processed_data = np.stack(processed_slices, axis=-1)
        processed_img = nib.Nifti1Image(processed_data, affine, header)

        # 결과 저장
        output_file = os.path.join(session_folder, f"{file_name}_processed.nii.gz")
        nib.save(processed_img, output_file)
        print(f"Processed file saved: {output_file}")

        # 로그 파일에 기록
        with open(log_file_path, 'a') as log_file:
            log_file.write(nifti_file_path + '\n')

    except Exception as e:
        print(f"Error processing {nifti_file_path}: {e}")
        with open(error_log_file_path, 'a') as log_file:
            log_file.write(nifti_file_path + '\n')

print("All NIfTI files processed.")
