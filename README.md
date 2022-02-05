Setup Environment:

    1. virtualenv myenv
    2. myenv\Scripts\activate
    3. pip install -r requirements.txt

How to run run_gui.py:

    1. Add neural network to models folder
    2. Save your model weights from training into "model_weights.pth"
    3. Save images in array as data.npy. This can be just done through numpy "save('data.npy', images)". 
    Images represents the following structure in an array [modelPrediction, trueValue, image].
    create_images.py is an example of saving images.
    4. python run_gui.py

P.S. Let me know if you need help with saving images. Let me know if there are any issues as well.
