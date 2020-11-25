# Principal Feature Visualization (PFV)
Principal feature visualization is a visualization technique for convolutional
neural networks that highlights the contrasting features in a batch of images.
It produces one RGB heatmap per input image.

<img src="docs/overview_fig.png" width="480">

## Dependencies
* pytorch
* numpy

### Additional dependencies for the demo:
* torchvision
* matplotlib
* pillow

## Getting started
Install the dependencies listed above, and run the example in demo.py: `python demo.py`

## Example

A trained network shows good localization:

<img src="docs/trained_result.png" width="480">

But an untrained (re-initialized) network shows scrambled output, as expected:

<img src="docs/untrained_result.png" width="480">
