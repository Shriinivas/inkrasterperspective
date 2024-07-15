# Inkscape Raster Perspective Correction Extension

<p align="center"><img src="https://github.com/Shriinivas/etc/blob/master/inkrasterperspective/inkpersp.jpg" alt="Draw Options"/></p><br/>
This Inkscape extension allows you to correct the perspective of raster images using OpenCV. It provides a simple way to fix distorted images directly within Inkscape.

## Features

- Correct perspective distortion in raster images
- Works with embedded and linked images

## Requirements

- Inkscape (tested with 1.3.2)
- Python 3.x
- OpenCV (cv2)
- NumPy

## Installation

1. **Download the Extension**: Download the files `inkrasterperspective.inx` and `inkrasterperspective.py` from this repository.
2. **Install the Extension**: Copy these files into your Inkscape extensions directory. You can find this in the 'User Extensions' option under the System section at Edit > Preferences in Inkscape.
3. **Restart Inkscape**: Close and reopen Inkscape to load the new extension.
4. **Install the required Python packages**:
   ```
   pip install opencv-python numpy
   ```
   Note: If you want to use Python from a managed environment, you can add the following under `<group id="extensions">` in `preferences.xml`:
   ```xml
   python-interpreter="<path_to_managed_python_binary>"
   ```
   Replace `<path_to_managed_python_binary>` with the actual path to your Python interpreter.

## Usage

1. Open your image in Inkscape.
2. Create a path with exactly 4 points that outline the area you want to correct.
3. Select both the image and the path.
4. Go to "Extensions" > "Raster" > "Perspective Fix".
5. Click "Apply" to correct the perspective.

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

```

```
