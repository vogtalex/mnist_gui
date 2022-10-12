# MNIST GUI With Visualizations

Interactive GUI that displays visualizations to aid a user in making decisions on adversarial perturbed images.

### Installing

If you need to set up the project with a virtual environment due to module compatibility issues, not having global module install permissions, or for any other reasons, perform  virtual env install and then continue. Otherwise follow the regular install.

Virtual Environment install:
```
virtualenv myenv
myenv\Scripts\activate
pip install -r requirements.txt
```
Normal install:
```
pip install -r requirements.txt
```

Data setup:
- pregenerated data is included in [Data/madrynet_mnist_l2](https://github.com/vogtalex/mnist_gui/tree/main/Data/madrynet_mnist_l2).
- change `outputDir` in config to point to the location of the generated data.

## Executing program

```
$ python gui.py
```

## Generating attacked data:
- Change `modelDir` and `weightDir` in config to the location of your model and model weights respectively.
- Change `numSubsets` and `subsetSize` in config to the values you want.
- Change `outputDir` to where you want to save the newly generated datasets.
- run `python tsne_setup.py` to run the data generation.

## Additional notes
- The norm verifier can be run to verify the attack strength of the example from the display subset at the starting index, using `python norm_verifier.py`.
- The different visualizations can be easily enabled/disabled by swapping their respective true/false values in the config.
- The BoxPlot visualization won't visually update in the normal view after submitting, as it still needs to be rewritten to redraw like the other visualizations.

## Authors

- Alex Vogt (vogtalex@oregonstate.edu)
- Sasha Rosenthal (rosenths@oregonstate.edu)
- Simon Millsap (millsaps@oregonstate.edu)
