## Quantum Reservoir Computing in PennyLane+PyTorch

### Aims
*Investigation of improvements to the implementation of quantum reservoir computing (QRC) models.*

### Current work
*Explore this repository and its files*
- Investigate the current implementation of the classical (RC) and quantum (QRC) versions of quantum reservoir computing in PennyLane.
- As we all may end up having slightly different projects (to be discussed),
  at this stage keep your draft work in your separate folders.

### Folders
*We'll need some common utilities, I suggest to keep them as .py files in a directory.*
- jacobs_examples: examples of QAE and supporting code from Jacob
- logs: this folder may be created to hold saved data, training history, plots, etc.
- qae_utils: which is a collection of Python utilities to include
  - Charts.py - functions plotting time-series data (fancy and flexible)
  - Files.py - functions saving time-series and support data to disk
  - Tools.py - some odd collection of utilities, including extras for PennyLane
  - Window.py - functions creating and managing sliding windows (making, splitting, etc.)

### Requirements
- Set up a virtual environment with **venv** or **anaconda** for Python 3.11 and activate it
- Then install all software using **requirements.txt** file (available here):
    - pip install -r \<place-you-saved-it\>/requirements.txt
- Or install by hand by following these instructions:
    - pip install pennylane==0.40.0 pennylane-lightning==0.40.0 (PennyLane for CPU)
    - pip install scikit-learn==1.6.1 pandas==2.2.3 (ML)
    - pip install matplotlib==3.10.1 plotly==6.0.0 seaborn==0.13.2 pillow==11.1.0 (plots and images)
    - pip install jupyter==1.1.1 jupyterlab==4.3.5 (running jupyter notebooks)
    - pip install kagglehub==0.3.10 ucimlrepo==0.0.7 (data access)
    - pip install pdflatex (optionally to plot and export some plots and tables to latex)
    - install [PyTorch](https://pytorch.org/get-started/locally/), as per web site instructions, also add:<br>
      pip install torchsummary torcheval torchmetrics

The **requirements.txt** file was tested for installation on 
Ubuntu 22.04-24.04, Windows 11 and MacOS Sequoia 15.3.1 (with M3 procesor).

### License
This project is licensed under the [GNU General Public License v3](./LICENSE).
The GPL v3 license requires attribution for modifications and derivatives, ensuring that users know which versions are changed and to protect the reputations of original authors.