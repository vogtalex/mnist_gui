Readme is currently out of date and will be updated later.

# MNIST and CIFAR GUI With Visualizations

Interactive GUI that displays visualizations to aid a user in making decisions on adversarial perturbed images.

### Installing

Setup Environment:

```
virtualenv myenv
myenv\Scripts\activate
pip install -r requirements.txt
```

Setup data:

- Download perturbed MNIST and CIFAR data from google drive: https://drive.google.com/drive/folders/1AZkgtlrVcRuVp7_hROSwT77ZAFPV4oZ-?usp=sharing
- Edit config.json to point to location of data

## Executing program

MNIST GUI:

```
$ python gui.py
```

CIFAR GUI:

```
$ python cifar_gui.py
```

## Authors

- Alex Vogt (vogtalex@oregonstate.edu)
- Sasha Rosenthal (rosenths@oregonstate.edu)
- Simon Millsap (millsaps@oregonstate.edu)
