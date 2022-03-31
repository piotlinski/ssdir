# Supplementary material

The architecture details of the encoder and decoder networks are
described below and shown in figures. Figure below shows
visualization legend, describing each type of layer used in the model
and additional annotations for determining their shapes. Note that we
use the same notations as in the paper.

- _convolution_ - convolution layer with <img src="https://render.githubusercontent.com/render/math?math=16"> <img src="https://render.githubusercontent.com/render/math?math=3\times 3"> kernels without an activation function, outputting <img src="https://render.githubusercontent.com/render/math?math=32\times 32"> feature maps,
- _convolution with ReLU_ - convolution layer with ReLU activation function and <img src="https://render.githubusercontent.com/render/math?math=16"> <img src="https://render.githubusercontent.com/render/math?math=3\times 3"> kernels, outputting <img src="https://render.githubusercontent.com/render/math?math=32\times 32"> feature maps,
- _convolution with sigmoid_ - convolution layer with sigmoid activation function and <img src="https://render.githubusercontent.com/render/math?math=16"> <img src="https://render.githubusercontent.com/render/math?math=3\times 3"> kernels outputting <img src="https://render.githubusercontent.com/render/math?math=32\times 32"> feature maps,
- _transposed convolution with ReLU_ - transposed convolution layer with ReLU activation function and <img src="https://render.githubusercontent.com/render/math?math=16"> <img src="https://render.githubusercontent.com/render/math?math=3\times 3"> kernels, outputting <img src="https://render.githubusercontent.com/render/math?math=32\times 32"> feature maps,
- _maxpool_ - MaxPool2D layer,
- _feature map_ - feature map tensor of shape <img src="https://render.githubusercontent.com/render/math?math=16\times32\times32">,
- _flatten and stack_ - tensor created by flattening and stacking vectors assigned to each cell in each feature map, of shape <img src="https://render.githubusercontent.com/render/math?math=32\times2">.

![legend](assets/legend.png)

## Encoder network

Algorithm below presents SSDIR's encoder flow: for each cell in
feature pyramid's grids it creates _where_, _present_, _what_ and
_depth_ latent variables. The inference for each cell in the feature
pyramid, as well as generating latent representations, is conducted
parallelly.

| INPUT: normalized image <img src="https://render.githubusercontent.com/render/math?math=x"> objects' latent representations </br> OUTPUT: (<img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_\mathit{where}">, <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_\mathit{present}">, <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_\mathit{what}">, <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_\mathit{depth}">)|
|:-|
|<img src="https://render.githubusercontent.com/render/math?math=\mathit{features} \leftarrow"> Backbone(<img src="https://render.githubusercontent.com/render/math?math=x">);|
|<img src="https://render.githubusercontent.com/render/math?math=[cx_i, cy_i, w_i, h_i] \leftarrow"> WhereEncoder(<img src="https://render.githubusercontent.com/render/math?math=\mathit{features}">);|
|<img src="https://render.githubusercontent.com/render/math?math=\beta_i \leftarrow"> PresentEncoder(<img src="https://render.githubusercontent.com/render/math?math=\mathit{features}">);|
|<img src="https://render.githubusercontent.com/render/math?math=\mathbf{\mu}_\mathit{what}^i \leftarrow"> WhatEncoder(<img src="https://render.githubusercontent.com/render/math?math=\mathit{features}">);|
|<img src="https://render.githubusercontent.com/render/math?math=\mu_\mathit{depth}^i \leftarrow"> DepthEncoder(<img src="https://render.githubusercontent.com/render/math?math=\mathit{features}">);|
|<img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_\mathit{where} \leftarrow [\mathbf{cx}, \mathbf{cy}, \mathbf{w}, \mathbf{h}]">;|
|<img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_\mathit{present} \leftarrow \mathit{Bernoulli}(\mathbf{\beta})">;|
|<img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_\mathit{what} \leftarrow \mathcal{N}(\mathbf{\mu}_\mathit{what}, \mathbf{\sigma}_\mathit{what})">;|
|<img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_\mathit{depth} \leftarrow \mathcal{N}(\mathbf{\mu}_\mathit{depth}, \mathbf{\sigma}_\mathit{depth})">;|
|RETURN: <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_{where}">, <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_{present}">, <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_{what}">, <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_{depth}">|

#### Convolutional backbone

The backbone used in SSDIR is a standard VGG11 with batch normalization,
whose classification head was replaced with a feature pyramid, as shown
in figure below. The input images are normalized with mean
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{\mu}=\left\{0.485, 0.456, 0.406\right\}"> and standard deviation
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{\sigma}=\left\{0.229, 0.224, 0.225\right\}"> and resized to
<img src="https://render.githubusercontent.com/render/math?math=3\times300\times300">. As a result, the backbone outputs <img src="https://render.githubusercontent.com/render/math?math=5"> feature
maps of resolutions <img src="https://render.githubusercontent.com/render/math?math=(18\times18)">, <img src="https://render.githubusercontent.com/render/math?math=(9\times9)">, <img src="https://render.githubusercontent.com/render/math?math=(5\times5)">,
<img src="https://render.githubusercontent.com/render/math?math=(3\times3)"> and <img src="https://render.githubusercontent.com/render/math?math=(1\times1)">, which denote the sizes of grids in each
level of the feature pyramid. These features are passed to latent
vectors' encoders. During training, weights of the backbone are frozen.

![backbone](assets/backbone.png)

#### SSDIR _where_ and _present_ encoders

The architectures of _where_ and _present_ encoders are presented in
figures below. Both encoders are based on SSD
prediction heads and utilize one convolutional layer for each feature
map. Since the SSD model used in SSDIR assigns two predictions to each
cell, the output representation consists of <img src="https://render.githubusercontent.com/render/math?math=880"> vectors. As in the
backbone's case, the weights of encoders transferred from an SSD model
are frozen during training SSDIR.

| _where_ encoder | _present_ encoder |
|-|-|
| ![where_enc](assets/where_enc.png) | ![where_enc](assets/present_enc.png) |

#### SSDIR _depth_ and _what_ encoder

The _what_ encoder, shown in figure below, is slightly extended as compared to other
encoders. Each feature map can be processed by multiple convolutional
layers, each with the same number of kernels, equal to the size of
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_\mathit{what}"> latent vector. The output is a vector of means,
used to sample the <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_\mathit{what}"> latent vector.

The architecture of the _depth_ encoder is similar to the _present_
encoder (see figure below). As in the _what_ encoder, here the output
is used as mean for sampling <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_\mathit{depth}"> latent vector.
| _what_ encoder | _depth_ encoder |
|-|-|
| ![what_enc](assets/what_enc.png) | ![depth_enc](assets/depth_enc.png) |

These modules generate one latent vector for each cell; in order to
match the size of SSD's output, each latent vector is duplicated.

## Decoder network

Algorithm below shows the flow of the SSDIR decoder network.
First, all latent vectors are filtered according to
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_\mathit{present}">, producing <img src="https://render.githubusercontent.com/render/math?math=M"> present-only latent vectors.
For batched decoding and transforming reconstructions, all filtered
latent representations in a batch are stacked and forwarded at once
through _what_ decoder and the spatial transformer. The output image,
created by merging transformed reconstructions, is normalized to
increase the intensities for visual fidelity.

| INPUT: objects' latent representations (<img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_{where}">, <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_{present}">, <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_{what}">, <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_{depth}">) <br/>OUTPUT: reconstructed images <img src="https://render.githubusercontent.com/render/math?math=y">|
|:-|
|<img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_{what}^M \leftarrow"> Filter(<img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_{what}, \mathbf{z}_{present}">);
|<img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_{where}^M \leftarrow"> Filter(<img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_{where}, \mathbf{z}_{present}">);|
|<img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_{depth}^M \leftarrow"> Filter(<img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_{depth}, \mathbf{z}_{present}">);|
|<img src="https://render.githubusercontent.com/render/math?math=\mathit{recs} \leftarrow"> WhatDecoder(<img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_{what}^M">);|
|<img src="https://render.githubusercontent.com/render/math?math=\mathit{scaled\_recs} \leftarrow"> STN(<img src="https://render.githubusercontent.com/render/math?math=\mathit{recs}, \mathbf{z}_{where}^M">);|
|<img src="https://render.githubusercontent.com/render/math?math=\mathit{weights} \leftarrow"> SoftMax(<img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_{depth}^M">);|
|<img src="https://render.githubusercontent.com/render/math?math=y \leftarrow"> WeightedMerge(<img src="https://render.githubusercontent.com/render/math?math=\mathit{scaled\_recs}, \mathit{weights}">);|
|Normalize(<img src="https://render.githubusercontent.com/render/math?math=y">);|
|RETURN: <img src="https://render.githubusercontent.com/render/math?math=y">|

#### SSDIR _what_ decoder

The _what_ decoder consists of a sequence of convolutional layers. The
first one, containing <img src="https://render.githubusercontent.com/render/math?math=1024"> <img src="https://render.githubusercontent.com/render/math?math=1\times1"> kernels, prepares a larger
feature map for transposed convolution. Then, a series of transposed
convolutions with strides of size <img src="https://render.githubusercontent.com/render/math?math=2">, each with <img src="https://render.githubusercontent.com/render/math?math=(2\times2)">-sized
filters upscale the feature map to achieve <img src="https://render.githubusercontent.com/render/math?math=64\times64"> resolution.
Finally, the last convolutional layer with the sigmoid activation
function outputs 3 channels, creating <img src="https://render.githubusercontent.com/render/math?math=M"> objects' reconstructions.

![what_dec](assets/what_dec.png)

#### Spatial Transformer and merging

The filtered <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_\mathit{where}^M"> latent vectors are used to
transform decoded reconstructions to the inferred location on the image.
We use affine transformation to create <img src="https://render.githubusercontent.com/render/math?math=M"> <img src="https://render.githubusercontent.com/render/math?math=3\times300\times300"> images,
which are merged according to softmaxed <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_\mathit{depth}^M">.

## Other details

Standard deviations, used to sample latent representations of each
object, are treated as model hyperparameters. They are given for each
experiment in Section Training.

To increase the stability of training, the latent vectors of non-present
objects (those, whose <img src="https://render.githubusercontent.com/render/math?math=\beta"> probability is lower than <img src="https://render.githubusercontent.com/render/math?math=0.001">) can be
reset, in order to prevent their values from exploding, as noticed
during training when transferring a pre-trained backbone, _where_, and
_present_ encoders from SSD. In such a case, all non-present objects'
means were set to <img src="https://render.githubusercontent.com/render/math?math=0.0">, all standard deviations were set to <img src="https://render.githubusercontent.com/render/math?math=1.0">, and
all bounding box parameters were set to
<img src="https://render.githubusercontent.com/render/math?math=\left[0.0\ 0.0\ 0.0\ 0.0\right]">.

We noticed that training only _what_ and _depth_ encoders was not
sufficient to learn high-quality representations for more complex
datasets. In such a case, it is possible to clone the convolutional
backbone for learning <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_\mathit{what}"> and <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_\mathit{depth}">
and train it jointly, while preserving the originally learned weights
for inferring <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_\mathit{where}"> and <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_\mathit{present}">.

The weights of modules, which are not transferred from a trained SSD
model, are initialized using Xavier, with biases set to <img src="https://render.githubusercontent.com/render/math?math=0">.

# Datasets

#### [Multi-scale scattered MNIST](https://github.com/piotlinski/multiscalemnist)

We prepared the Multi-scale scattered MNIST dataset to test multi-object
representation learning using images with highly varying object sizes.
It can be treated as a benchmark dataset, and its construction procedure
contributes to this submission. The dataset was generated according to
algorithm below, with a given set of parameters:
 - <img src="https://render.githubusercontent.com/render/math?math=[s_x, s_y]">- output image size,
 - <img src="https://render.githubusercontent.com/render/math?math=grids"> - set of grid sizes used for placing digits,
 - <img src="https://render.githubusercontent.com/render/math?math={ds}_{min}"> - minimum size of a digit in the image,
 - <img src="https://render.githubusercontent.com/render/math?math={ds}_{max}"> - maximum size of a digit in the image,
 - <img src="https://render.githubusercontent.com/render/math?math=\Delta_{position}"> - the range in which the position of a digit may vary, given as the percentage of the cell size,
 - <img src="https://render.githubusercontent.com/render/math?math=\theta_{filled}"> - threshold for indicating if a cell in a grid is already filled, given as the percentage of the cell's area,
 - <img src="https://render.githubusercontent.com/render/math?math=n_{images}"> - number of images to generate.

|INPUTS: <img src="https://render.githubusercontent.com/render/math?math=\mathit{MNIST}"> dataset,  <img src="https://render.githubusercontent.com/render/math?math=[s_x, s_y]">, <img src="https://render.githubusercontent.com/render/math?math=\mathit{grids}">, <img src="https://render.githubusercontent.com/render/math?math=\mathit{ds}_\mathit{min}">, <img src="https://render.githubusercontent.com/render/math?math=\mathit{ds}_\mathit{max}">, <img src="https://render.githubusercontent.com/render/math?math=\Delta_\mathit{position}">, <img src="https://render.githubusercontent.com/render/math?math=\theta_\mathit{filled}">, <img src="https://render.githubusercontent.com/render/math?math=n_\mathit{images}"> </br>OUTPUTS: a generated dataset|
|:-|
|2<img src="https://render.githubusercontent.com/render/math?math=\mathbf{X}"> <img src="https://render.githubusercontent.com/render/math?math=\mathbf{X} \leftarrow [\ ]"> <img src="https://render.githubusercontent.com/render/math?math=x \leftarrow"> CreateEmptyImage(<img src="https://render.githubusercontent.com/render/math?math=s_x, s_y">);|
|<img src="https://render.githubusercontent.com/render/math?math=\mathit{bboxes} \leftarrow [\ ]">;|
|<img src="https://render.githubusercontent.com/render/math?math=\mathit{labels} \leftarrow [\ ]">;|
|<img src="https://render.githubusercontent.com/render/math?math=\mathit{grid\_size} \leftarrow"> DrawGridSize(<img src="https://render.githubusercontent.com/render/math?math=\mathit{grids}">);|
|<img src="https://render.githubusercontent.com/render/math?math=n_\mathit{cells} = \mathit{grid\_size}_x \cdot \mathit{grid\_size}_y">;|
|<img src="https://render.githubusercontent.com/render/math?math=n_\mathit{digits} \leftarrow"> randint(min=<img src="https://render.githubusercontent.com/render/math?math=n_\mathit{cells} / 2">, max=<img src="https://render.githubusercontent.com/render/math?math=n_\mathit{cells}">);|
|<img src="https://render.githubusercontent.com/render/math?math=\mathit{filled} \leftarrow [\ ]">;|
|<img src="https://render.githubusercontent.com/render/math?math=\mathit{cell} \leftarrow"> DrawGridCell(<img src="https://render.githubusercontent.com/render/math?math=\mathit{grid\_size}">, <img src="https://render.githubusercontent.com/render/math?math=\mathit{filled}">); |
|<img src="https://render.githubusercontent.com/render/math?math=\mathit{ds} \leftarrow"> RandomDigitSize(<img src="https://render.githubusercontent.com/render/math?math=\mathit{cell}_\mathit{size}">, <img src="https://render.githubusercontent.com/render/math?math=\mathit{ds}_\mathit{min}">, <img src="https://render.githubusercontent.com/render/math?math=\mathit{ds}_\mathit{max}">);|
|<img src="https://render.githubusercontent.com/render/math?math=\mathit{digit\_img}, \mathit{label} \leftarrow"> GetDigit(<img src="https://render.githubusercontent.com/render/math?math=\mathit{MNIST}">); |
|<img src="https://render.githubusercontent.com/render/math?math=\mathit{digit\_img} \leftarrow"> Resize(<img src="https://render.githubusercontent.com/render/math?math=\mathit{digit\_img}">, <img src="https://render.githubusercontent.com/render/math?math=ds">);|
|<img src="https://render.githubusercontent.com/render/math?math=\mathit{bbox} \leftarrow"> CalculateBboxCoords(<img src="https://render.githubusercontent.com/render/math?math=ds">, <img src="https://render.githubusercontent.com/render/math?math=cell">, <img src="https://render.githubusercontent.com/render/math?math=\Delta_\mathit{position}">);|
|<img src="https://render.githubusercontent.com/render/math?math=x \leftarrow"> AddDigitToImage(<img src="https://render.githubusercontent.com/render/math?math=\mathit{digit\_img}">, <img src="https://render.githubusercontent.com/render/math?math=\mathit{bbox}">, <img src="https://render.githubusercontent.com/render/math?math=x">); |
|<img src="https://render.githubusercontent.com/render/math?math=\mathit{bboxes} \leftarrow"> append(<img src="https://render.githubusercontent.com/render/math?math=\mathit{bbox}">);|
|<img src="https://render.githubusercontent.com/render/math?math=\mathit{labels} \leftarrow"> append(<img src="https://render.githubusercontent.com/render/math?math=\mathit{label}">);|
|<img src="https://render.githubusercontent.com/render/math?math=\mathit{filled} \leftarrow"> MarkFilled(<img src="https://render.githubusercontent.com/render/math?math=\mathit{bbox}">, <img src="https://render.githubusercontent.com/render/math?math=\theta_\mathit{filled}">);|
|<img src="https://render.githubusercontent.com/render/math?math=\mathbf{X} \leftarrow"> append(<img src="https://render.githubusercontent.com/render/math?math=x">, <img src="https://render.githubusercontent.com/render/math?math=\mathit{bboxes}">, <img src="https://render.githubusercontent.com/render/math?math=\mathit{labels}">);|
|RETURN: <img src="https://render.githubusercontent.com/render/math?math=\mathbf{X}">|

In table below we gathered the parameter values used for
generating _main_ dataset, prepared for training the SSD and SSDIR
models. An additional validation dataset the size of 10% of the
training dataset was used for evaluating SSDIR, SPAIR, and SPACE with
regard to per-object reconstructions and the downstream task. We also
present all researched values of parameters, combinations of which were
used to generate the ablation study's datasets.

|parameter | _main_ | ablation |
| :------- | :----: | :------: |
| <img src="https://render.githubusercontent.com/render/math?math=n_\mathit{images}"> | <img src="https://render.githubusercontent.com/render/math?math=50000"> | <img src="https://render.githubusercontent.com/render/math?math=10000">|
|<img src="https://render.githubusercontent.com/render/math?math=\mathit{grids}"> | <img src="https://render.githubusercontent.com/render/math?math=\left\{(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)\right\}"> | <img src="https://render.githubusercontent.com/render/math?math=\{\left\{(2, 2), (3, 3), (4, 4), (5, 5)\right\}">, </br> <img src="https://render.githubusercontent.com/render/math?math=\left\{(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)\right\}">, </br> <img src="https://render.githubusercontent.com/render/math?math=\left\{(3, 3), (4, 4), (5, 5)\right\}">, </br> <img src="https://render.githubusercontent.com/render/math?math=\left\{(3, 3), (4, 4), (5, 5), (6, 6)\right\}">, </br> <img src="https://render.githubusercontent.com/render/math?math=\left\{(4, 4), (5, 5), (6, 6)\right\}\}"> |
|<img src="https://render.githubusercontent.com/render/math?math=\mathit{ds}_\mathit{min}"> | <img src="https://render.githubusercontent.com/render/math?math=96"> | <img src="https://render.githubusercontent.com/render/math?math=\left\{96, 128, 160\right\}">|
|<img src="https://render.githubusercontent.com/render/math?math=\mathit{ds}_\mathit{max}"> | <img src="https://render.githubusercontent.com/render/math?math=384"> | <img src="https://render.githubusercontent.com/render/math?math=\left\{256, 320, 384\right\}">|
|<img src="https://render.githubusercontent.com/render/math?math={[s_x, s_y]}"> | <img src="https://render.githubusercontent.com/render/math?math=512\times512"> | <img src="https://render.githubusercontent.com/render/math?math=512\times512"> |
|<img src="https://render.githubusercontent.com/render/math?math=\Delta_\mathit{position}"> | <img src="https://render.githubusercontent.com/render/math?math=0.7"> | <img src="https://render.githubusercontent.com/render/math?math=0.7"> |
|<img src="https://render.githubusercontent.com/render/math?math=\theta_\mathit{filled}"> | <img src="https://render.githubusercontent.com/render/math?math=0.4"> | <img src="https://render.githubusercontent.com/render/math?math=0.4"> |

#### CLEVR

We used the dataset generated originally by the authors. It contains
object of <img src="https://render.githubusercontent.com/render/math?math=2"> pre-defined sizes (large and small), <img src="https://render.githubusercontent.com/render/math?math=8"> colors, <img src="https://render.githubusercontent.com/render/math?math=2">
materials and <img src="https://render.githubusercontent.com/render/math?math=3"> shapes. The locations of the objects in the scene were
processed to generate bounding boxes for training the SSD model. We
trained SSDIR, SPAIR, and SPACE using the entire training dataset, while
the validation dataset was used for evaluating each model's
reconstructions quality.

#### WIDER FACE

This dataset was used for evaluating the performance of the models in
images with a complex background when trying to focus on a particular
type of object (here: faces). The dataset contains bounding box
coordinates and hence could be used for training the SSD model directly.
We applied an additional preprocessing stage, dropping small bounding
boxes (smaller than 4% of the image) and removing images without any
bounding box. Then, the SSDIR, SPAIR, and SPACE models were trained with
the training dataset, and the validation dataset served as the
reconstructions quality benchmark.

# Training regime and hyperparameters

In this section, we summarize the hyperparameters of SSDIR used for
training models for each dataset. The batch size was tuned to fit the
GPU's memory size, whereas the other hyperparameters' optimization was
conducted using Bayesian model-based optimization. In table below
we present the hyperparameters used for each task and dataset (the
denominations of the hyperparameters match those used in the paper).

| symbol | description | MNIST | CLEVR | WIDER |
| :----- | :---------- | :---: | :---: | :---: |
| | batch size | <img src="https://render.githubusercontent.com/render/math?math=32"> | <img src="https://render.githubusercontent.com/render/math?math=32"> | <img src="https://render.githubusercontent.com/render/math?math=32"> |
| | learning rate | <img src="https://render.githubusercontent.com/render/math?math=0.0005"> | <img src="https://render.githubusercontent.com/render/math?math=0.0005"> | <img src="https://render.githubusercontent.com/render/math?math=0.0005"> |
| | clone backbone | false | true | true |
| <img src="https://render.githubusercontent.com/render/math?math=\alpha_\mathit{obj}"> | image reconstruction error coefficient | <img src="https://render.githubusercontent.com/render/math?math=10.0"> | <img src="https://render.githubusercontent.com/render/math?math=1.0"> | <img src="https://render.githubusercontent.com/render/math?math=1.0"> |
| <img src="https://render.githubusercontent.com/render/math?math=\alpha_\mathit{rec}"> | per-object reconstruction error coefficient | <img src="https://render.githubusercontent.com/render/math?math=10.0"> | <img src="https://render.githubusercontent.com/render/math?math=14.0"> | <img src="https://render.githubusercontent.com/render/math?math=10.0"> |
| <img src="https://render.githubusercontent.com/render/math?math=\alpha_\mathit{what}"> | <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_\mathit{what}"> KL loss coefficient | <img src="https://render.githubusercontent.com/render/math?math=1.0"> | <img src="https://render.githubusercontent.com/render/math?math=0.54"> | <img src="https://render.githubusercontent.com/render/math?math=0.1"> |
| <img src="https://render.githubusercontent.com/render/math?math=\alpha_\mathit{depth}"> | <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_\mathit{depth}"> KL loss coefficient | <img src="https://render.githubusercontent.com/render/math?math=1.0"> | <img src="https://render.githubusercontent.com/render/math?math=0.44"> | <img src="https://render.githubusercontent.com/render/math?math=0.4"> |
| <img src="https://render.githubusercontent.com/render/math?math=\sigma_\mathit{what}"> | <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_\mathit{what}"> standard deviation | <img src="https://render.githubusercontent.com/render/math?math=0.1"> | <img src="https://render.githubusercontent.com/render/math?math=0.27"> | <img src="https://render.githubusercontent.com/render/math?math=0.3"> |
| <img src="https://render.githubusercontent.com/render/math?math=\sigma_\mathit{depth}"> | <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_\mathit{depth}"> standard deviation | <img src="https://render.githubusercontent.com/render/math?math=0.1"> | <img src="https://render.githubusercontent.com/render/math?math=0.14"> | <img src="https://render.githubusercontent.com/render/math?math=0.1"> |
| <img src="https://render.githubusercontent.com/render/math?math=n_\mathit{hidden}"> | number of hidden layers in *what* encoder | <img src="https://render.githubusercontent.com/render/math?math=3"> | <img src="https://render.githubusercontent.com/render/math?math=3"> | <img src="https://render.githubusercontent.com/render/math?math=4"> |
| <img src="https://render.githubusercontent.com/render/math?math=\|\mathbf{z}_\mathit{what}\|"> | <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_\mathit{what}"> latent vector size | <img src="https://render.githubusercontent.com/render/math?math=64"> | <img src="https://render.githubusercontent.com/render/math?math=256"> | <img src="https://render.githubusercontent.com/render/math?math=512"> |
| | <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_\mathit{what}"> prior | <img src="https://render.githubusercontent.com/render/math?math=\mathcal{N}\left(\mathbf{0}, \mathbf{I}\right)"> | <img src="https://render.githubusercontent.com/render/math?math=\mathcal{N}\left(\mathbf{0}, \mathbf{I}\right)"> | <img src="https://render.githubusercontent.com/render/math?math=\mathcal{N}\left(\mathbf{0}, \mathbf{I}\right)"> |
| | <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}_\mathit{depth}"> prior | <img src="https://render.githubusercontent.com/render/math?math=\mathcal{N}\left(0, 1\right)"> | <img src="https://render.githubusercontent.com/render/math?math=\mathcal{N}\left(0, 1\right)"> | <img src="https://render.githubusercontent.com/render/math?math=\mathcal{N}\left(0, 1\right)"> |

## Additional reconstructions

Below, we provide additional reconstructions for the following datasets: scattered MNIST, CLEVR and WIDER FACE.
Once again, the number of reconstructions shown for each image is limited, due to the total number of objects reconstructed by each model; if the number of objects reconstructed by a model was smaller than the number of columns, we show only the ones returned by the model.

### Multi-scale MNIST

![reconstruction_mnist](assets/reconstructions_mnist.png)

### CLEVR

![reconstruction_mnist](assets/reconstructions_clevr.png)

### WIDER FACE

![reconstruction_mnist](assets/reconstructions_wider.png)
