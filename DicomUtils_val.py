
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom as dicom
import os
import scipy.ndimage
#import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.draw import polygon
import os, sys, glob
import skimage.io

# Some constants 
INPUT_FOLDER = 'data_val/1/'
OUTPUT_FOLDER = 'data_val/train/'

#INPUT_FOLDER = '/usr2/Data/VIP Cup 2018 ICIP/TestData/'
#OUTPUT_FOLDER = '/usr2/Data/VIP Cup 2018 ICIP/Test/'

patients = os.listdir(INPUT_FOLDER)
patients.sort()


def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing

def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image


def normalize_array(image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


def zero_center(image):
    image = image - image.mean()
    return image




def read_structure(structure):
    contours = []
    for i in range(len(structure.ROIContourSequence)):
        for j in range(len(structure.ROIContourSequence[i].ContourSequence)):
            #print i,structure.ROIContourSequence[i].ContourSequence[j].ContourGeometricType," ",
            if structure.ROIContourSequence[i].ContourSequence[j].ContourGeometricType == "POINT":
            	continue
            for k in range(len(structure.ROIContourSequence[i].ContourSequence[j].ContourImageSequence)):
                contour = {}
                contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
                contour['number'] = structure.ROIContourSequence[i].ContourSequence[j].ContourNumber
                contour['name'] = structure.ROIContourSequence[i].ContourSequence[j].ContourImageSequence[0].ReferencedSOPInstanceUID
                contour['contours'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
                contours.append(contour)
    return contours

def get_mask(contours, slices):
    z = [s.ImagePositionPatient[2] for s in slices]
    pos_r = slices[0].ImagePositionPatient[1]
    spacing_r = slices[0].PixelSpacing[1]
    pos_c = slices[0].ImagePositionPatient[0]
    spacing_c = slices[0].PixelSpacing[0]
    #print z
    #print pos_r, pos_c, spacing_r, spacing_c
    #label = np.zeros_like(image, dtype=np.uint8)
    label = np.zeros((slices[0].Columns, slices[0].Rows, len(slices) ), dtype=np.uint8)
    for con in contours:# For each contour get the corresponding matching slice
        num = int(con['number'])
        ref_sopuid = con['name']
        for s in slices:
            if(s.SOPInstanceUID == ref_sopuid): # Match contour with slice
                for c in con['contours']:
                    nodes = np.array(c).reshape((-1, 3))
                    assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
                    #print nodes
                    try:
                        z_index = z.index(nodes[0, 2])
                    except ValueError:
                        print "Boundary case, skipping...."
                        continue
                        #z_index = zNew.index(nodes[0, 2])
                    #z_index = z.index(nodes[0, 2])
                    r = (nodes[:, 1] - pos_r) / spacing_r
                    c = (nodes[:, 0] - pos_c) / spacing_c
                    #r = s.Rows + np.floor(s.Rows * s.PixelSpacing[1]) * nodes[:,1]
                    #c = s.Columns + np.floor(s.Columns * s.PixelSpacing[0]) * nodes[:,0]
                    #print r,c
                    rr, cc = polygon(r, c)
                    #print rr,cc
                    #label[rr, cc, s.InstanceNumber] = num
                    label[rr, cc, z_index] = num

    colors = tuple(np.array([con['color'] for con in contours]) / 255.0)
    return label, colors

# patients = []
# for lung in os.listdir(INPUT_FOLDER):
#     for name in os.listdir(os.path.join(INPUT_FOLDER,lung)):
#         print(name)
#         pass


patients = [os.path.join(INPUT_FOLDER, name) for name in os.listdir(INPUT_FOLDER) if os.path.isdir(os.path.join(INPUT_FOLDER, name))]
patients.sort()
#print patients

def test_single_patient():
    patient = patients[89] # Just get the first patient for demo 
    print patient
    for subdir, dirs, files in os.walk(patient):
        #print subdir
        #print dirs
        #print files
        dcms = glob.glob(os.path.join(subdir, "*.dcm"))
        #print dcms
        if len(dcms) == 1:
            print("GT path",os.path.join(subdir, files[0]))
            structure = dicom.read_file(os.path.join(subdir, files[0]))
            contours = read_structure(structure)
        elif len(dcms) > 1:
            slices = [dicom.read_file(dcm) for dcm in dcms]
    return
    pat_id = slices[0].PatientID        
    print "Patient ID : ",pat_id

    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    image = np.stack([s.pixel_array for s in slices], axis=-1)

    print image.dtype
    image = image.astype(np.int16)
    image = image* slices[0].RescaleSlope + slices[0].RescaleIntercept

    #print contours
    if(len(contours) !=0 ):
        label, colors = get_mask(contours, slices)

    
    image = np.swapaxes(image,0,2)
    image = np.swapaxes(image,1,2)
    skimage.io.imsave(INPUT_FOLDER+pat_id+'.tiff', image.astype(np.int16), plugin='tifffile', compress = 1)
    if(len(contours) ==0 ):
        sys.exit(0)
    label = (label > 0).astype(np.uint8)*255
    label = np.swapaxes(label,0,2)
    label = np.swapaxes(label,1,2)
    skimage.io.imsave(INPUT_FOLDER+pat_id+'_GT.tiff', label.astype(np.uint8), plugin='tifffile', compress = 1)

    print "Img shape and # contours :: ",image.shape, len(contours)
    # Plot to check slices, for example 50 to 59
    plt.figure(figsize=(15, 15))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(image[ i + 64, ...], cmap="gray",label="%d"%(i+64) )
        plt.contour(label[ i + 64, ...], levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors=colors)
        #plt.imshow(label[..., i + 64], cmap="gray")
    plt.show()

    labels_img2 = np.zeros((image.shape[0], image.shape[1], image.shape[2], 3)) 
    print image.min(), image.max()
    img = 255*normalize_array(image.astype(np.float32), -500,300)
    print img.min(), img.max()
    skimage.io.imsave(INPUT_FOLDER+pat_id+'_uc.tiff', img.astype(np.uint8), plugin='tifffile', compress = 1)
    labels_img2[ ... , 0 ] = img; labels_img2[ ... , 1 ] = img; labels_img2[ ... , 2 ] = img
    tmp = np.where( label > 0, 1, 0)
    opacity = 0.5
    print tmp.shape
    
    labels_img2[ tmp==1  , 2 ] = opacity * img[tmp==1]
    labels_img2[ tmp==1  , 1 ] = opacity * img[tmp==1]
    labels_img2[ tmp==1  , 0 ] = opacity * img[tmp==1] + (1 - opacity) * 255
    
    skimage.io.imsave(INPUT_FOLDER+pat_id+'_Labels.tiff', labels_img2.astype(np.uint8), plugin='tifffile', compress = 1)

#test_single_patient()
#sys.exit(0)

for patient in patients[:100]:
    contours = {}
    print("Patient")
    for subdir, dirs, files in os.walk(patient):
        print("Path: ",subdir)
        #print subdir
        #print dirs
        #print files
        dcms = glob.glob(os.path.join(subdir, "*.dcm"))
        #print dcms
        if len(dcms) == 1:
            # continue
            print("File: ",os.path.join(subdir, files[0]))
            structure = dicom.read_file(os.path.join(subdir, files[0]))
            contours = read_structure(structure)
        elif len(dcms) > 1:
            slices = [dicom.read_file(dcm) for dcm in dcms]
    
    pat_id = slices[0].PatientID     
    print "Patient ID : ",pat_id,

    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    image = np.stack([s.pixel_array for s in slices], axis=-1)
    #print contours
    if(len(contours) !=0 ):
        label, colors = get_mask(contours, slices)
    
    print image.dtype
    image = image.astype(np.int16)
    image = image* slices[0].RescaleSlope + slices[0].RescaleIntercept
    #image = get_pixels_hu(slices)

    image = np.swapaxes(image,0,2)
    image = np.swapaxes(image,1,2)
    skimage.io.imsave(OUTPUT_FOLDER+pat_id+'.tiff', image.astype(np.int16), plugin='tifffile', compress = 1)
    if(len(contours) ==0 ):
        continue
    label = (label > 0).astype(np.uint8)*255
    label = np.swapaxes(label,0,2)
    label = np.swapaxes(label,1,2)
    skimage.io.imsave(OUTPUT_FOLDER+pat_id+'_GT.tiff', label.astype(np.uint8), plugin='tifffile', compress = 1)

    print "Img shape and # contours :: ",image.shape, len(contours)

    labels_img2 = np.zeros((image.shape[0], image.shape[1], image.shape[2], 3)) 
    img = 255*normalize_array(image.astype(np.float32), -500,300)
    labels_img2[ ... , 0 ] = img; labels_img2[ ... , 1 ] = img; labels_img2[ ... , 2 ] = img
    tmp = np.where( label > 0, 1, 0)
    opacity = 0.5
    
    labels_img2[ tmp==1  , 2 ] = opacity * img[tmp==1]
    labels_img2[ tmp==1  , 1 ] = opacity * img[tmp==1]
    labels_img2[ tmp==1  , 0 ] = opacity * img[tmp==1] + (1 - opacity) * 255
    
    skimage.io.imsave(OUTPUT_FOLDER+pat_id+'_Labels.tiff', labels_img2.astype(np.uint8), plugin='tifffile', compress = 1)




