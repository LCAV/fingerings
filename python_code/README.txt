Semester project - Guessing the fingerings
EPFL - LCAV - 2017

AUTHOR: Raphaël Latty, raphael.latty@epfl.ch


INCLUDED FILES: The following files should be placed in the same folder
in order to run everything without problems.

    - FOLDERS: /data_dumps; /images_animations
    
    - NOTEBOOKS: 1_introduction.ipynb, 2_theory_and_background.ipynb, 3_a_simple_model.ipynb, 4_spectral_estimation.ipynb,
        5_inharmonicity.ipynb, 6_inharmonicity_based_classifier.ipynb, 7_conclusions
     
    - .PY: misc.py, MUSIC.py, partials_tracking.py, pitch_estimation.py, spectral_grid_search.py, string_model.py
    
    - OTHERS: README.txt (present file)


DATASET: The dataset (recordings) is located in the folder ../recordings


FILES DESCRIPTIONS:
    - The notebooks are numbered from 1 to 7, corresponding to the 
        different chapters/sections of the project.
    
    - The .py files contain functions and routines for the different
        chapters.
    
    - The /data_dumps folder contains data dumps in binary format,
        saved using the pickle module, and that can be loaded into 
        notebooks or .py files.
    
    - The /images_animations folder contains the images and animations
        used in the various notebooks.