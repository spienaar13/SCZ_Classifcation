import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from glob import glob
import torch
from skimage.measure import label, regionprops
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage import zoom

# set random seed
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Dataset
class MRIDataset(Dataset):
    def __init__(self, DataDir, mode, input_T1, input_DeepC, double_vgg, DeepC_isotropic, DeepC_isotropic_crop, transform=None, \
        T1_normalization_method = 'max', DeepC_normalization_method = 'max',fold_list=None):
        print('***************')
        print('MRIDataset')
        self.DataDir = DataDir
        self.input_T1 = input_T1
        self.input_DeepC = input_DeepC
        self.double_vgg = double_vgg
        self.DeepC_isotropic = DeepC_isotropic
        self.DeepC_isotropic_crop = DeepC_isotropic_crop
        self.fold_list=fold_list
        
        if mode=='train':
            if input_T1:
                temp=[]            
                for i in self.fold_list:
                    sub_fold1=glob(DataDir+str(i)+'/MNI152_affine_WB_iso1mm/*/*COBRE*.nii.gz')
                    sub_fold2=glob(DataDir+str(i)+'/MNI152_affine_WB_iso1mm/*/*BrainGlu*.nii.gz')
                    sub_fold3=glob(DataDir+str(i)+'/MNI152_affine_WB_iso1mm/*/*NMorph*.nii.gz')
                    sub_fold4=glob(DataDir+str(i)+'/MNI152_affine_WB_iso1mm/*/*MCIC*.nii.gz')
                    sub_fold=sub_fold1+sub_fold2+sub_fold3+sub_fold4

                    temp.extend(sub_fold)
                self.T1_img_files = sorted(temp)
            # self.T1_img_files = sorted(glob(DataDir + 'MNI152_affine_WB_iso1mm/*/*.nii.gz'))
            # DHW_CU_affine_CUReso; Step4a_MNI152_WB_affine_iso1mm; T1_WH_MNI152_affine_iso1mm
                print(f'T1 path: {DataDir} / {self.fold_list} MNI152_affine_WB_iso1mm')
                print(f'Load T1. Total T1 {mode} number is: ' + str(len(self.T1_img_files)))
            if input_T1:# and DeepC_isotropic:
                temp=[]
                for i in self.fold_list:
                    sub_fold1=glob(DataDir+str(i)+'/MNI152_affine_WB_iso1mm_apar_aseg_deep_transform/*/*COBRE*.mgz')
                    sub_fold2=glob(DataDir+str(i)+'/MNI152_affine_WB_iso1mm_apar_aseg_deep_transform/*/*BrainGlu*.mgz')
                    sub_fold3=glob(DataDir+str(i)+'/MNI152_affine_WB_iso1mm_apar_aseg_deep_transform/*/*NMorph*.mgz')
                    sub_fold4=glob(DataDir+str(i)+'/MNI152_affine_WB_iso1mm_apar_aseg_deep_transform/*/*MCIC*.mgz')
                    sub_fold=sub_fold1+sub_fold2+sub_fold3+sub_fold4
                    temp.extend(sub_fold)
                self.DeepC_img_files = sorted(temp)

                temp=[]
            # self.DeepC_img_files = sorted(glob(DataDir + 'CU_WB_affine_ACBV_iso1mm/*/*.nii.gz'))
                print(f'FastSurfer path:{DataDir} / {self.fold_list} MNI152_affine_WB_iso1mm_apar_aseg')
                print(f'Total FastSurfer {mode} number is: ' + str(len(self.DeepC_img_files)))

        if mode=='validation': 
            if input_T1:
                temp=[]            
                for i in self.fold_list:
                    sub_fold1=glob(DataDir+str(i)+'/MNI152_affine_WB_iso1mm/*/*COBRE*.nii.gz')
                    sub_fold2=glob(DataDir+str(i)+'/MNI152_affine_WB_iso1mm/*/*BrainGlu*.nii.gz')
                    sub_fold3=glob(DataDir+str(i)+'/MNI152_affine_WB_iso1mm/*/*NMorph*.nii.gz')
                    sub_fold4=glob(DataDir+str(i)+'/MNI152_affine_WB_iso1mm/*/*MCIC*.nii.gz')
                    sub_fold=sub_fold1+sub_fold2+sub_fold3+sub_fold4

                    temp.extend(sub_fold)
                self.T1_img_files = sorted(temp)
            # self.T1_img_files = sorted(glob(DataDir + 'MNI152_affine_WB_iso1mm/*/*.nii.gz'))
            # DHW_CU_affine_CUReso; Step4a_MNI152_WB_affine_iso1mm; T1_WH_MNI152_affine_iso1mm
                print(f'T1 path: {DataDir} / {self.fold_list} MNI152_affine_WB_iso1mm')
                print(f'Load T1. COBRE and BrainGluSchi and NMorph and T1 {mode} number is: ' + str(len(self.T1_img_files)))
            if input_T1:# and DeepC_isotropic:
                temp=[]
                for i in self.fold_list:
                    sub_fold1=glob(DataDir+str(i)+'/MNI152_affine_WB_iso1mm_apar_aseg_deep_transform/*/*COBRE*.mgz')
                    sub_fold2=glob(DataDir+str(i)+'/MNI152_affine_WB_iso1mm_apar_aseg_deep_transform/*/*BrainGlu*.mgz')
                    sub_fold3=glob(DataDir+str(i)+'/MNI152_affine_WB_iso1mm_apar_aseg_deep_transform/*/*NMorph*.mgz')
                    sub_fold4=glob(DataDir+str(i)+'/MNI152_affine_WB_iso1mm_apar_aseg_deep_transform/*/*MCIC*.mgz')
                    sub_fold=sub_fold1+sub_fold2+sub_fold3+sub_fold4
                    temp.extend(sub_fold)
                self.DeepC_img_files = sorted(temp)

                temp=[]

            # self.DeepC_img_files = sorted(glob(DataDir + 'CU_WB_affine_ACBV_iso1mm/*/*.nii.gz'))
                print(f'FastSurfer path:{DataDir} / {self.fold_list} MNI152_affine_WB_iso1mm_apar_aseg')
                print(f'Total FastSurfer {mode} number is: ' + str(len(self.DeepC_img_files)))

        if mode=='test':
            self.fold_list=['fold1','fold2','fold3','fold4','fold5','fold6','fold7','fold8','fold9','fold10']
            if input_T1:
                temp=[]            
                for i in self.fold_list:
                    sub_fold4=glob(DataDir+str(i)+'/MNI152_affine_WB_iso1mm/*/*MCIC*.nii.gz')
                    sub_fold=sub_fold4
                    temp.extend(sub_fold)
                self.T1_img_files = sorted(temp)
            # self.T1_img_files = sorted(glob(DataDir + 'MNI152_affine_WB_iso1mm/*/*.nii.gz'))
            # DHW_CU_affine_CUReso; Step4a_MNI152_WB_affine_iso1mm; T1_WH_MNI152_affine_iso1mm
                print(f'T1 path: {DataDir} / {self.fold_list} MNI152_affine_WB_iso1mm')
                print(f'Load T1.test set {mode} number is: ' + str(len(self.T1_img_files)))
            if input_T1:# and DeepC_isotropic:
                temp=[]
                for i in self.fold_list:
                    sub_fold4=glob(DataDir+str(i)+'/MNI152_affine_WB_iso1mm_apar_aseg_deep_transform/*/*MCIC*.mgz')
                    sub_fold = sub_fold4
                    temp.extend(sub_fold)
                self.DeepC_img_files = sorted(temp)

                temp=[]

            # self.DeepC_img_files = sorted(glob(DataDir + 'CU_WB_affine_ACBV_iso1mm/*/*.nii.gz'))
                print(f'FastSurfer path:{DataDir} / {self.fold_list} MNI152_affine_WB_iso1mm_apar_aseg')
                print(f'Total FastSurfer {mode} number is: ' + str(len(self.DeepC_img_files)))

        self.transform = transform
        self.T1_normalization_method = T1_normalization_method
        self.DeepC_normalization_method = DeepC_normalization_method    

        
    def center_crop_or_pad(self, input_scan, desired_dimension):
        input_dimension = input_scan.shape
        #print('Input dimension: ', input_dimension, '\ndesired dimension: ', desired_dimension)

        x_lowerbound_target = int(np.floor((desired_dimension[0] - input_dimension[0]) / 2)) if desired_dimension[0] >= input_dimension[0] else 0
        y_lowerbound_target = int(np.floor((desired_dimension[1] - input_dimension[1]) / 2)) if desired_dimension[1] >= input_dimension[1] else 0
        z_lowerbound_target = int(np.floor((desired_dimension[2] - input_dimension[2]) / 2)) if desired_dimension[2] >= input_dimension[2] else 0
        x_upperbound_target = x_lowerbound_target + input_dimension[0] if desired_dimension[0] >= input_dimension[0] else None
        y_upperbound_target = y_lowerbound_target + input_dimension[1] if desired_dimension[1] >= input_dimension[1] else None
        z_upperbound_target = z_lowerbound_target + input_dimension[2] if desired_dimension[2] >= input_dimension[2] else None

        x_lowerbound_input = 0 if desired_dimension[0] >= input_dimension[0] else int(np.floor((input_dimension[0] - desired_dimension[0]) / 2))
        y_lowerbound_input = 0 if desired_dimension[1] >= input_dimension[1] else int(np.floor((input_dimension[1] - desired_dimension[1]) / 2))
        z_lowerbound_input = 0 if desired_dimension[2] >= input_dimension[2] else int(np.floor((input_dimension[2] - desired_dimension[2]) / 2))
        x_upperbound_input = None if desired_dimension[0] >= input_dimension[0] else x_lowerbound_input + desired_dimension[0]
        y_upperbound_input = None if desired_dimension[1] >= input_dimension[1] else y_lowerbound_input + desired_dimension[1]
        z_upperbound_input = None if desired_dimension[2] >= input_dimension[2] else z_lowerbound_input + desired_dimension[2]

        output_scan = np.zeros(desired_dimension).astype(np.float32)  

        output_scan[x_lowerbound_target : x_upperbound_target, \
                    y_lowerbound_target : y_upperbound_target, \
                    z_lowerbound_target : z_upperbound_target] = \
        input_scan[x_lowerbound_input: x_upperbound_input, \
                   y_lowerbound_input: y_upperbound_input, \
                   z_lowerbound_input: z_upperbound_input]

        return output_scan

    def WB_to_brain_mask(self, WB_scan, threshold = 5):
        binary_scan = WB_scan.copy()
        binary_scan[WB_scan <= np.float32(threshold)] = 0
        binary_scan[WB_scan > np.float32(threshold)] = 1

        # Binary hole filling.
        filled_scan = np.float32(binary_fill_holes(binary_scan))

        # Discard everything except for the biggest 3D connected component.
        label_map = label(filled_scan)
        region_area = []; region_label = []
        for region in regionprops(label_map):
            region_area.append(region.area)
            region_label.append(region.label)
        assert len(region_area) == len(region_label)
       # assert len(region_area) > 0
        if len(region_area) <= 0:
            print('len(region_area)',len(region_area))
       
        region_label_biggest_component = region_label[np.argmax(region_area)]
        
        new_scan = np.float32(np.zeros(filled_scan.shape))
        new_scan[label_map == region_label_biggest_component] = 1
        
        return new_scan

    def WB_to_brain_mask_4ACBV(self, WB_scan, threshold = 0):
        binary_scan = WB_scan.copy()
        binary_scan[WB_scan <= np.float32(threshold)] = 0
        binary_scan[WB_scan > np.float32(threshold)] = 1

        # Binary hole filling.
        filled_scan = np.float32(binary_fill_holes(binary_scan))

        # Discard everything except for the biggest 3D connected component.
        label_map = label(filled_scan)
        region_area = []; region_label = []
        for region in regionprops(label_map):
            region_area.append(region.area)
            region_label.append(region.label)
        assert len(region_area) == len(region_label)
        assert len(region_area) > 0
       
        region_label_biggest_component = region_label[np.argmax(region_area)]
        
        new_scan = np.float32(np.zeros(filled_scan.shape))
        new_scan[label_map == region_label_biggest_component] = 1
        
        return new_scan


    def __len__(self):
        if self.input_T1:
            return len(self.T1_img_files)
        else:
            return len(self.DeepC_img_files)
    
    def __getitem__(self, idx):
        if self.input_T1:
            label = self.T1_img_files[idx].split('/')[-2]
        else:
            label = self.DeepC_img_files[idx].split('/')[-2]
        label = (1 if label == 'schiz' else 0)

        current_T1 = nib.load(self.T1_img_files[idx]).get_fdata().astype(np.float32)
        current_DeepC = nib.load(self.DeepC_img_files[idx]).get_fdata().astype(np.float32)
        
        # normalization (important !!!!)
        assert self.T1_normalization_method in ['NA', 'max', 'WBtop10PercentMean','mean','std']
        assert self.DeepC_normalization_method in ['NA', 'max', 'WBtop10PercentMean','mean','std']


        #T1_dimension = (192, 192, 192)
        T1_dimension_original = (193,229,193)
        T1_dimension = (192, 192, 192)

        DS_ratio1 = 0.5

        if self.input_T1:
            current_T1 = self.center_crop_or_pad(current_T1, T1_dimension)
            current_T1 = zoom(current_T1,(DS_ratio1,DS_ratio1,DS_ratio1),  order=0) 
            current_T1 = current_T1 

            current_DeepC = self.center_crop_or_pad(current_DeepC, T1_dimension)
            current_DeepC = zoom(current_DeepC,(DS_ratio1, DS_ratio1,DS_ratio1), order=0) 
            current_DeepC = current_DeepC

            #Shape (96, 96, 96)

        
        if current_T1 is not None:
            if self.T1_normalization_method == 'max':
                current_T1 = current_T1 / current_T1.max()
            # elif self.T1_normalization_method == 'WBtop10PercentMean':
            #     #current_T1_BM = self.WB_to_brain_mask(current_T1)
            #     normalization_factor = \
            #     np.mean(current_T1[np.logical_and(current_T1 >= \
            #         np.percentile(current_DeepC[current_DeepC_BM == 1], 90, interpolation = 'nearest'), \
            #         current_T1_BM == 1)])
            #     assert normalization_factor > 0
            #     current_T1 = current_T1 / normalization_factor
            # elif self.T1_normalization_method == 'mean':
            #     #current_T1_BM = self.WB_to_brain_mask(current_T1)
            #     current_T1 = current_T1 / np.mean(current_DeepC[current_DeepC_BM == 1])
            # elif self.T1_normalization_method == 'std':
            #     T1wMRI_BrainMask = self.WB_to_brain_mask(WB_scan = current_T1)
            #     current_T1 = (current_T1 - np.asarray(current_T1[T1wMRI_BrainMask== 1]).mean().astype(np.float32))/np.asarray(current_T1[T1wMRI_BrainMask == 1]).std().astype(np.float32)
            #     current_T1 = current_T1 * T1wMRI_BrainMask

        # Normailzing DeepC same as fastsurfer loader
        if current_T1 is not None:
            if self.DeepC_normalization_method == 'max':
                current_DeepC = current_DeepC / current_DeepC.max()
            # elif self.DeepC_normalization_method == 'WBtop10PercentMean':
            #     current_DeepC_BM = self.WB_to_brain_mask(current_DeepC)
            #     normalization_factor = \
            #     np.mean(current_DeepC[np.logical_and(current_DeepC >= \
            #         np.percentile(current_DeepC[current_DeepC_BM == 1], 90, interpolation = 'nearest'), \
            #         current_DeepC_BM == 1)])
            #     assert normalization_factor > 0
            #     current_DeepC = current_DeepC / normalization_factor
            # elif self.DeepC_normalization_method == 'mean':
            #     current_DeepC_BM = self.WB_to_brain_mask(current_DeepC)
            #     current_DeepC = current_DeepC / np.mean(current_DeepC[current_DeepC_BM == 1])
            # elif self.DeepC_normalization_method == 'NA':
            #     current_DeepC = current_DeepC    
 
        if self.input_T1 and self.input_DeepC:
            sample = {'T1': current_T1,
                      'DeepC': current_DeepC,
                      'label': label,
                      }
        elif self.input_T1:
            # Stack the outputs into two channels 
            current_T1 = np.stack((current_T1, current_DeepC), axis=0)

            sample = {'T1': current_T1,
                      'label': label,
                      }
        elif self.input_DeepC:
            sample = {'DeepC': current_DeepC,
                      'label': label,
                      }

        if self.transform:
            sample = self.transform(sample)

        return sample
