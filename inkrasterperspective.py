#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inkscape extension to correct perspective of raster images.
Tested with Inkscape version 1.3.2

Author: Shrinivas Kulkarni (khemadeva@gmail.com)

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

import base64
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlparse, urlunparse
from urllib.request import urlopen

import cv2
import inkex
import numpy as np


def get_full_href(href: str, svg_file_path: str) -> str:
    """
    Get the full href for a given relative or absolute path.

    Args:
        href (str): The original href.
        svg_file_path (str): The path to the SVG file.

    Returns:
        str: The full href.
    """
    parsed_href = urlparse(href)

    if parsed_href.scheme == "file":
        path = Path(parsed_href.path)
        if path.is_absolute():
            return href
        else:
            new_path = Path(svg_file_path) / path
            new_href = urlunparse(parsed_href._replace(path=str(new_path)))
            return "file://" + str(new_href)
    elif parsed_href.scheme:
        return href
    else:
        path = Path(href)
        if path.is_absolute():
            return "file://" + str(path)
        else:
            new_path = Path(svg_file_path) / path
            return "file://" + str(new_path)


def get_cv_image(element: inkex.Image, svg_file_path: str) -> np.ndarray:
    """
    Get the OpenCV image from an embedded or linked image of Inkscape image element.

    Args:
        element (inkex.Image): The Inkscape image element.
        svg_file_path (str): The path to the SVG file.

    Returns:
        np.ndarray: The OpenCV image.
    """
    href = element.get("{http://www.w3.org/1999/xlink}href") or ""

    if href.startswith("data:image"):
        _, encoded = href.split(",", 1)
        img_data = base64.b64decode(encoded)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        cvImage = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        href = get_full_href(href, svg_file_path)
        resp = urlopen(href)
        img_array = np.asarray(bytearray(resp.read()), dtype="uint8")
        cvImage = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return cvImage


def put_cv_image(cvImage: np.ndarray, element: inkex.Image) -> None:
    """
    Update the Inkscape image element with the new OpenCV image.

    Args:
        cvImage (np.ndarray): The OpenCV image.
        element (inkex.Image): The Inkscape image element.
    """
    href = element.get("{http://www.w3.org/1999/xlink}href") or ""

    if not href.startswith("data:image"):
        inkex.errormsg(
            "New image is embedded. Use 'Extract Images' extension to export and link it."
        )
    _, buffer = cv2.imencode(".jpg", cvImage, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    img_bytes = buffer.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    data_uri = f"data:image/png;base64,{img_base64}"
    element.set("{http://www.w3.org/1999/xlink}href", data_uri)
    # else:
    #     image_path = get_full_href(href, svg_file_path)
    #     cv2.imwrite(image_path.replace("file://", ""), cvImage)


def get_elem_transform(
    elem: inkex.BaseElement, scale_x: float = 1, scale_y: float = 1
) -> inkex.Transform:
    """
    Get the transform for an Inkscape Image element.

    Args:
        elem (inkex.BaseElement): The Inkscape element.
        scale_x (float, optional): The x-scale factor. Defaults to 1.
        scale_y (float, optional): The y-scale factor. Defaults to 1.

    Returns:
        inkex.Transform: The resulting transform.
    """
    transform = inkex.Transform(elem.transform)
    img_translation = [float(elem.get("x") or "0"), float(elem.get("y") or "0")]
    transform = transform.add_translate(*img_translation)
    transform.add_scale(scale_x, scale_y)
    return transform


def apply_perspective(
    cvImage: np.ndarray, corners: List[inkex.Vector2d]
) -> Tuple[np.ndarray, int, int]:
    """
    Apply perspective transformation to the image.

    Args:
        cvImage (np.ndarray): The input OpenCV image.
        corners (List[inkex.Vector2d]): The four corners of the perspective rectangle.

    Returns:
        Tuple[np.ndarray, int, int]: The transformed image, new width, and new height.
    """

    def sorted_corners(corners: np.ndarray) -> np.ndarray:
        centroid = np.mean(corners, axis=0)
        # Calculate angles from centroid
        angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
        # Sort indices based on angles
        sorted_indices = np.argsort(angles)
        # Arrange corners counter-clockwise starting from top-right
        return corners[sorted_indices]

    corners_array = np.array(corners, dtype="float32")
    corners_array = sorted_corners(corners_array)

    # Calculate the new dimensions of the image based on the corners
    heightA = np.linalg.norm(corners_array[3] - corners_array[0])
    heightB = np.linalg.norm(corners_array[2] - corners_array[1])
    maxHeight = max(int(heightA), int(heightB))

    widthA = np.linalg.norm(corners_array[1] - corners_array[0])
    widthB = np.linalg.norm(corners_array[2] - corners_array[3])
    maxWidth = max(int(widthA), int(widthB))

    target_corners = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(corners_array, target_corners)
    new_image = cv2.warpPerspective(cvImage, matrix, (maxWidth, maxHeight))

    # cv2.imwrite("/tmp/persptest.jpg", cvImage)
    return new_image, maxWidth, maxHeight


class RasterPerspectiveEffect(inkex.Effect):
    """
    Inkscape extension to correct perspective of raster images.
    """

    def __init__(self):
        super().__init__()

    def effect(self):
        img_elem = path_elem = None
        for elem in self.svg.selection:
            if isinstance(elem, inkex.Image):
                if not img_elem:
                    img_elem = elem
            elif elem is not None and len(list(elem.path.end_points)) >= 4:
                path_elem = elem
            if path_elem is not None and img_elem is not None:
                break

        if img_elem is None or path_elem is None:
            inkex.errormsg("Please select one image and a shape with 4 points")
            return

        file_path = self.svg_path()
        if not file_path:
            return

        cvImage = get_cv_image(img_elem, file_path)

        scale_y, scale_x = [
            d1 / d0
            for (d0, d1) in zip(cvImage.shape[:2], [img_elem.height, img_elem.width])
        ]

        img_transform = get_elem_transform(img_elem, scale_x, scale_y)

        corners = [
            path_elem.composed_transform().apply_to_point(pt)
            for pt in list(path_elem.path.end_points)
        ][:4]
        local_corners = [(-img_transform).apply_to_point(pt) for pt in corners]

        cvImage, width, height = apply_perspective(cvImage, local_corners)
        width, height = img_transform.apply_to_point((width, height))
        img_elem.set("width", width)
        img_elem.set("height", height)
        img_elem.set("x", corners[0][0])
        img_elem.set("y", corners[0][1])

        put_cv_image(cvImage, img_elem)


if __name__ == "__main__":
    RasterPerspectiveEffect().run()
