import openslide
from pathlib import Path
import re

import xml.dom.minidom as minidom
#
#  adapted from: https://github.com/KatherLab/STAMP/blob/main/src/stamp/preprocessing/tiling.py#L379
def get_slide_mpp_(
    slide: openslide.AbstractSlide | Path, *, default_mpp: float | None
) -> float | None:
    """
    Retrieve the microns per pixel (MPP) value from a slide.
    This function attempts to extract the MPP value from the given slide. If the slide
    is provided as a file path, it will be opened using OpenSlide. The function first
    checks for the MPP value in the slide's properties. If not found, it attempts to
    extract the MPP value from the slide's comments and metadata. If all attempts fail
    and a default MPP value is provided, it will use the default value. If no MPP value
    can be determined and no default is provided, an MPPExtractionError is raised.
    Args:
        slide: The slide object or file path to the slide.
        default_mpp: The default MPP value to use if extraction fails.
    Returns:
        The extracted or default MPP value, or None if extraction fails and no default is provided.
    Raises:
        MPPExtractionError: If the MPP value cannot be determined and no default is provided.
    """

    if isinstance(slide, Path):
        slide = openslide.open_slide(slide)

    if openslide.PROPERTY_NAME_MPP_X in slide.properties:
        slide_mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    elif slide_mpp := _extract_mpp_from_comments(slide):
        pass
    elif slide_mpp := _extract_mpp_from_metadata(slide):
        pass

    if slide_mpp is None and default_mpp:
        print(
            f"could not infer slide MPP from metadata, using {default_mpp} instead."
        )
    elif slide_mpp is None and default_mpp is None:
        raise MPPExtractionError()

    return slide_mpp or default_mpp

def _extract_mpp_from_comments(slide: openslide.AbstractSlide) -> float | None:
    slide_properties = slide.properties.get("openslide.comment", "")
    pattern = r"<PixelSizeMicrons>(.*?)</PixelSizeMicrons>"
    match = re.search(pattern, slide_properties)
    if match is not None and (mpp := match.group(1)) is not None:
        return float(mpp)
    else:
        return None


def _extract_mpp_from_metadata(slide: openslide.AbstractSlide) -> float | None:
    try:
        xml_path = slide.properties.get("tiff.ImageDescription") or None
        if xml_path is None:
            return None
        doc = minidom.parseString(xml_path)
        collection = doc.documentElement
        images = collection.getElementsByTagName("Image")
        pixels = images[0].getElementsByTagName("Pixels")
        mpp = float(pixels[0].getAttribute("PhysicalSizeX"))
    except Exception:
        print("failed to extract MPP from image description")
        return None
    return mpp

class MPPExtractionError(Exception):
    """Raised when the Microns Per Pixel (MPP) extraction from the slide's metadata fails"""

    pass