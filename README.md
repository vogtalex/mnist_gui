Setup Environment:

    1. virtualenv myenv
    2. myenv\Scripts\activate
    3. pip install -r requirements.txt

How to run run_gui.py:

    1. Add neural network architecture to models folder
    2. Save your model weights from training into "lenet_mnist_model.pth" 
    -Developer Note: Change this into search for .pth file
    3. python tsne_setup.py
    4. python run_gui.py

Things to do:

    1. Modularization of adding images. Data loader from csv of {imageFile, label}
    2. Boxplots implementation
    3. Fixing setup:
        - Instead of multiple images take in a csv file of {imageFile, label}
        - Training, Testing images and labels
    4. Gui Bugs:
        - Text in entry needs to be bigger
        - Add title
        - Allignment of gui on different window size
        - Shaping images in grid. Something instead of: (fig.set_size_inches(6, 4))
        - Removal of true label on tsne plots
        - Search for .pth file in folder rather than just use lenet_mnist_model.pth
    5. Have user input their own neural net into model folder. Import python files grabbed from setup rather than just net.py
