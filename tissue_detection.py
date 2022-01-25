import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2

def extract_stains(img, saveFile=None, Io=240, alpha=1, beta=0.15):
    ''' Normalize staining appearence of H&E stained images

    Example use:
        see test.py

    Input:
        I: RGB input image
        Io: (optional) transmitted light intensity

    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image

    Reference:
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''

    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])

    maxCRef = np.array([1.9705, 1.0308])

    # define height and width of image
    h, w, c = img.shape

    # remove black background pixel
    black_px = np.where((img[:, :, 0] <= 5) & (img[:, :, 1] <= 5) & (img[:, :, 2] <= 5))
    img[black_px] = [255, 255, 255]

    plt.imshow(img)
    plt.title("Black removed")
    plt.show()

    # reshape image
    img = img.reshape((-1, 3))

    # calculate optical density
    OD = -np.log((img.astype(np.float) + 1) / Io)

    # remove transparent pixels
    ODhat = OD[~np.any(OD < beta, axis=1)]

    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    # eigvecs *= -1

    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = ODhat.dot(eigvecs[:, 1:3])

    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])

    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    # unmix hematoxylin and eosin
    H = np.multiply(Io, np.exp(np.expand_dims(-HE[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
    H[H > 255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

    E = np.multiply(Io, np.exp(np.expand_dims(-HE[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
    E[E > 255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)

    #if saveFile is not None:
    #    Image.fromarray(Inorm).save(saveFile + '.png')
    #    Image.fromarray(H).save(saveFile + '_H.png')
    #    Image.fromarray(E).save(saveFile + '_E.png')

    return Inorm, H, E


def tissue_detection(img):

    kernel_size = 3

    # remove alpha channel
    img = img[:, :, 0:3]

    print("Removing ")
    top_border = int(len(img)/5)
    # hack for removing border artifacts
    img[0:top_border, :, :] = [0, 0, 0]

    # remove black background pixel
    black_px = np.where((img[:, :, 0] <= 5) & (img[:, :, 1] <= 5) & (img[:, :, 2] <= 5))
    img[black_px] = [255, 255, 255]

    # apply median filter to remove artifacts created by transitions to background pixels
    median_filtered_img = cv2.medianBlur(img, 11)

    # convert to HSV color space
    hsv_image = cv2.cvtColor(median_filtered_img, cv2.COLOR_RGB2HSV)

    # get saturation channel
    saturation = hsv_image[:, :, 1]

    # Otsu's thresholding
    _, threshold_image = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # apply dilation to image to close spots inside mask regions
    kernel = np.ones(shape=(kernel_size, kernel_size))
    tissue_mask = cv2.dilate(threshold_image, kernel, iterations=1)
    #tissue_mask = cv2.erode(tissue_mask, kernel)

    return tissue_mask