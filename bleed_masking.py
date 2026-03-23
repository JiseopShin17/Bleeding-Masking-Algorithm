def bleed_masking(datapath, saturate=50000, BI_thres=500, detect_thres=0.4, CL=6,flagval=1):
    """
    Generating a bleeding mask for a KMTNet single-epoch FITS image.

    This function identifies saturated pixels and traces vertical bleed trails
    along the detector column direction. For each saturated pixel, it first
    evaluates whether the source is likely to produce a bleed trail using a
    bleeding index (BI), defined from the summed pixel values in a fixed 
    offset range from the saturated pixel. If the BI exceeds the given
    threshold, the algorithm scans along the column and marks pixels as
    bleeding until it encounters `CL` consecutive pixels below the detection
    threshold. For more description for each parameter, see Shin et al. (2026, in prep).

    The masking direction depends on the chip name in the input filename:
    chips containing "kk" or "nn" are masked downward, while chips containing
    "mm" or "tt" are masked upward.

    Parameters
    ----------
    datapath : str
        Path to the input FITS image.
    saturate : float, optional
        Saturation threshold in ADU used to identify candidate saturated
        pixels. Default is 50000.
    detect_thres : float, optional
        Multiplicative factor applied to the background sigma to define the
        detection threshold:
        `detection_thres = median_background + detect_thres * sigma_background`.
        Default is 0.4.
    BI_thres : float, optional
        Threshold for the bleeding index (BI). Only saturated pixels with
        BI larger than this value are treated as bleeding sources.
        Default is 500.
    CL : int, optional
        Continuity length. Masking stops when `CL` consecutive pixels fall
        below the signal threshold. Default is 6.
    flagval : int or float, optional
        Value assigned to pixels flagged as bleeding in the output mask.
        Default is 1.

    Returns
    -------
    bpMask : ndarray
        2D mask array with the same shape as the input image, where bleeding
        pixels are marked with `flagval` and all other pixels are 0.

    Notes
    -----
    This algorithm is designed for KMTNet single-epoch images and assumes
    that bleeding trails are aligned with the detector columns. The chip-based
    masking direction reflects the readout-dependent orientation of the bleed
    trails in KMTNet CCD images.
    """

    import os
    import numpy as np
    from astropy.stats import sigma_clipped_stats
    from astropy.io import fits
    
    hdul    = fits.open(datapath)
    data    = hdul[0].data
    hdr     = hdul[0].header
    leny, lenx = data.shape
    bpMask = np.zeros_like(data)
    _, med, sig = sigma_clipped_stats(data)
    signal_thres = med + detect_thres * sig

    chip    = os.path.basename(datapath)
    # Determine the direction of masking based on the chip type
    if ("kk" in chip) or ("nn" in chip):
        direction = 'downward'
    elif ("mm" in chip) or ("tt" in chip):
        direction = 'upward'
    else:
        raise ValueError("Unknown chip type")

    for i in range(lenx):
        col = data[:, i]
        sat_indices = np.where(col > saturate)[0]

        for y_idx in sat_indices:
            if bpMask[y_idx, i] == flagval:
                continue  # Skip already marked pixels

            # Calculate the sum of pixel values in the vicinity to determine if there is actual bleeding
            if direction == 'downward':
                ystart = max(y_idx - 40, 0)
                yend = max(y_idx - 20, 0)
            else:  # 'upward'
                ystart = min(y_idx + 20, leny - 1)
                yend = min(y_idx + 40, leny - 1)

            if ystart < yend:
                bpidx = np.sum(data[ystart:yend+1, i])
                ylength = yend - ystart + 1
                if bpidx - med * ylength > BI_thres:
                    bpMask[y_idx, i] = flagval  # Set mask only if the condition is met
                    revert_pixel = 0

                    # Define scanning range based on direction
                    range_start, range_end, step = (y_idx, -1, -1) if direction == 'downward' else (y_idx, leny, 1)

                    # Scan through the column in the specified direction
                    for j in range(range_start, range_end, step):
                        if col[j] > signal_thres:
                            bpMask[j, i] = flagval
                            revert_pixel = 0
                        else:
                            bpMask[j, i] = flagval
                            revert_pixel += 1

                        # Stop marking when enough consecutive small values are found
                        if revert_pixel >= CL:
                            if direction == 'downward':
                                end_idx = max(j - CL, 0)
                            else:
                                end_idx = min(j + CL, leny)
                            bpMask[j:end_idx, i] = 0
                            break
    return bpMask