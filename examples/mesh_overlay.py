from surfer import Brain

"""
mesh_overlay.py
Creates a Brain object and visualizes an overlay file over it.

Source: https://pysurfer.github.io/auto_examples/plot_fmri_activation.html

Notes:

1) The script works only on Python 2 due to PySurfer constraints. Throughout the senior design project, we used
Enthought Canopy as the Python 2 package distribution. Other distributions/package managers are welcome for use.

2) The following dependencies should be installed: pysurfer, mayavi, nibabel, traitsui. TraitsUI uses QT visualization 
backend by default, which might require installation as well.

3) In the senior design project, the pysurfer API was significantly changed in the files viz.py and utils.py. This
was done to allow embedding of the visualization in our GUI as well as to allow visualization over time. At the moment,
such changes are not needed to produce example visualizations like below or visualizing brain mesh traversals etc.,
but depending on our requirements, it's something to keep in mind.

4) For visualizations, a subject directory is expected by pysurfer. This is a location that hosts the patient files.
An example could be found in the senior design project directory, under src/demo-gui/SUBJECTS_DIR. You can also
try out the GUI itself along with the visualizations by running gui.ipynb in src/demo-gui. However, this will require
the "altered" pysurfer files, so you are advised to reach out to me for further instructions if you want to try it out.
"""

print(__doc__)

"""
Bring up the visualization window.
"""
brain = Brain("fsaverage", "lh", "inflated")

"""
Get a path to the overlay file.
"""
overlay_file = "example_data/lh.sig.nii.gz"

"""
Display the overlay on the surface using the defaults to control thresholding
and colorbar saturation.  These can be set through your config file.
"""
brain.add_overlay(overlay_file)

"""
You can then turn the overlay off.
"""
brain.overlays["sig"].remove()

"""
Now add the overlay again, but this time with set threshold and showing only
the positive activations.
"""
brain.add_overlay(overlay_file, min=5, max=20, sign="pos")