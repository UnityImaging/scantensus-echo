import pydicom

def get_xy_from_dicom(dcm: pydicom.Dataset):
    valid_region = []
    size_region = []
    for i, region in enumerate(dcm.SequenceOfUltrasoundRegions):
        if region.RegionSpatialFormat == 1:
            valid_region.append(i)
            size = (region.RegionLocationMaxX1 - region.RegionLocationMinX0) * (
                    region.RegionLocationMaxY1 - region.RegionLocationMinY0)
            size_region.append(size)

    selected_region = valid_region[size_region.index(max(size_region))]

    PhysicalDeltaX = dcm.SequenceOfUltrasoundRegions[selected_region].PhysicalDeltaX
    PhysicalDeltaY = dcm.SequenceOfUltrasoundRegions[selected_region].PhysicalDeltaY

    return PhysicalDeltaX, PhysicalDeltaY