import nibabel.gifti as gi
from nibabel.gifti import GiftiDataArray, GiftiImage
from nibabel.loadsave import save
from numpy import load
from glob import glob
from os.path import join, split, exists
from os import makedirs
from tqdm import tqdm

contrast_info = [["LANGUAGE", 1, "MATH"],
             ["LANGUAGE", 2, "STORY"],
             ["LANGUAGE", 3, "MATH-STORY"],
             ["RELATIONAL", 1, "MATCH"],
             ["RELATIONAL", 2, "REL"],
             ["RELATIONAL", 3, "MATCH-REL"],
             ["SOCIAL", 1, "RANDOM"],
             ["SOCIAL", 2, "TOM"],
             ["SOCIAL", 6, "TOM-RANDOM"],
             ["EMOTION", 1, "FACES"],
             ["EMOTION", 2, "SHAPES"],
             ["EMOTION", 3, "FACES-SHAPES"],
             ["WM", 1, "2BK_BODY"],
             ["WM", 2, "2BK_FACE"],
             ["WM", 3, "2BK_PLACE"],
             ["WM", 4, "2BK_TOOL"],
             ["WM", 5, "0BK_BODY"],
             ["WM", 6, "0BK_FACE"],
             ["WM", 7, "0BK_PLACE"],
             ["WM", 8, "0BK_TOOL"],
             ["WM", 9, "2BK"],
             ["WM", 10, "0BK"],
             ["WM", 11, "2BK-0BK"],
             ["WM", 15, "BODY"],
             ["WM", 16, "FACE"],
             ["WM", 17, "PLACE"],
             ["WM", 18, "TOOL"],
             ["WM", 19, "BODY-AVG"],
             ["WM", 20, "FACE-AVG"],
             ["WM", 21, "PLACE-AVG"],
             ["WM", 22, "TOOL-AVG"],
             ["MOTOR", 1, "CUE"],
             ["MOTOR", 2, "LF"],
             ["MOTOR", 3, "LH"],
             ["MOTOR", 4, "RF"],
             ["MOTOR", 5, "RH"],
             ["MOTOR", 6, "T"],
             ["MOTOR", 7, "AVG"],
             ["MOTOR", 8, "CUE-AVG"],
             ["MOTOR", 9, "LF-AVG"],
             ["MOTOR", 10, "LH-AVG"],
             ["MOTOR", 11, "RF-AVG"],
             ["MOTOR", 12, "RH-AVG"],
             ["MOTOR", 13, "T-AVG"],
             ["GAMBLING", 1, "PUNISH"],
             ["GAMBLING", 2, "REWARD"],
             ["GAMBLING", 3, "PUNISH-REWARD"]]

# target_directory = '/Users/sorenmadsen/Documents/brainsurf_model/HCP_feat64_s8_c25_lr0.01_seed28_epochs50/finetuned_feat64_s8_c25_lr0.01_seed28/predict_on_test_subj/best_corr'

def generate_gifti_files(target_directory):
    print(f'Working on {target_directory}\n...')
    for f in tqdm(glob(join(target_directory, '*.npy'))):
        dir, fname = split(f)
        output_dir = join(dir, 'gifti')
        # output_path = join(output_path, fname)
        data = load(f).mean(0) # average over trials -> [tasks, 32k]
        filename = fname.split('.')[0]
        left_fname =  f'{filename}_L.func.gii'
        right_fname = f'{filename}_R.func.gii'
        if not exists(output_dir):
            makedirs(output_dir)

        assert data.shape[0] % 2 == 0

        for i in range(data.shape[0] // 2):
            task = f'{contrast_info[i][0]}_{contrast_info[i][2]}'
            output_path = join(output_dir, task)
            if not exists(output_path):
                makedirs(output_path)

            # Save left hemi
            temp_gifti = GiftiImage()
            contrast = data[2*i]
            gifti = GiftiDataArray(contrast)
            temp_gifti.add_gifti_data_array(gifti)
            temp_gifti._meta['AnatomicalStructurePrimary'] = 'CortexLeft'
            save(temp_gifti, join(output_path, left_fname))

            # Save right hemi
            temp_gifti = GiftiImage()
            contrast = data[2*i + 1]
            gifti = GiftiDataArray(contrast)
            temp_gifti.add_gifti_data_array(gifti)
            temp_gifti._meta['AnatomicalStructurePrimary'] = 'CortexRight'
            save(temp_gifti, join(output_path, right_fname))