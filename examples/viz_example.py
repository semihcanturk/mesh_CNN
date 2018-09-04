from surfer import Brain
"""
Creates a brain object and visualizes an activation over it.
This works only on Python 2 due to PySurfer constraints.

Source:
https://pysurfer.github.io/auto_examples/plot_fmri_activation.html
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