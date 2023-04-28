{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "id": "jKdmT-0iGACI"
   },
   "outputs": [],
   "source": [
    "from tensorflow.keras.datasets import mnist\n",
    "\n",
    "def load_mnist_dataset():\n",
    "\n",
    "  # load data from tensorflow framework\n",
    "  ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data() \n",
    "\n",
    "  # Stacking train data and test data to form single array named data\n",
    "  data = np.vstack([trainData, testData]) \n",
    "\n",
    "  # Vertical stacking labels of train and test set\n",
    "  labels = np.hstack([trainLabels, testLabels]) \n",
    "\n",
    "  # return a 2-tuple of the MNIST data and labels\n",
    "  return (data, labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "id": "wp8lQh3WHD2R"
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "def load_az_dataset(datasetPath):\n",
    "\n",
    "  # List for storing data\n",
    "  data = []\n",
    "  \n",
    "  # List for storing labels\n",
    "  labels = []\n",
    "  \n",
    "  for row in open(datasetPath): #Openfile and start reading each row\n",
    "    #Split the row at every comma\n",
    "    row = row.split(\",\")\n",
    "    \n",
    "    #row[0] contains label\n",
    "    label = int(row[0])\n",
    "    \n",
    "    #Other all collumns contains pixel values make a saperate array for that\n",
    "    image = np.array([int(x) for x in row[1:]], dtype=\"uint8\")\n",
    "    \n",
    "    #Reshaping image to 28 x 28 pixels\n",
    "    image = image.reshape((28, 28))\n",
    "    \n",
    "    #append image to data\n",
    "    data.append(image)\n",
    "    \n",
    "    #append label to labels\n",
    "    labels.append(label)\n",
    "    \n",
    "  #Converting data to numpy array of type float32\n",
    "  data = np.array(data, dtype='float32')\n",
    "  \n",
    "  #Converting labels to type int\n",
    "  labels = np.array(labels, dtype=\"int\")\n",
    "  \n",
    "  return (data, labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "h4rW6MrtHQ3v",
    "outputId": "1718bf72-9bdf-4c53-a770-12ed458252a3"
   },
   "outputs": [],
   "source": [
    "(digitsData, digitsLabels) = load_mnist_dataset()\n",
    "\n",
    "(azData, azLabels) = load_az_dataset('Handwritten data/A_Z Handwritten Data.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "id": "cyIzOAgRHW4N"
   },
   "outputs": [],
   "source": [
    "# the MNIST dataset occupies the labels 0-9, so let's add 10 to every A-Z label to ensure the A-Z characters are not incorrectly labeled \n",
    "\n",
    "azLabels += 10\n",
    "\n",
    "# stack the A-Z data and labels with the MNIST digits data and labels\n",
    "\n",
    "data = np.vstack([azData, digitsData])\n",
    "labels = np.hstack([azLabels, digitsLabels])\n",
    "\n",
    "# Each image in the A-Z and MNIST digts datasets are 28x28 pixels;\n",
    "# However, the architecture we're using is designed for 32x32 images,\n",
    "# So we need to resize them to 32x32\n",
    "\n",
    "data = [cv2.resize(image, (32, 32)) for image in data]\n",
    "data = np.array(data, dtype=\"float32\")\n",
    "\n",
    "# add a channel dimension to every image in the dataset and scale the\n",
    "# pixel intensities of the images from [0, 255] down to [0, 1]\n",
    "\n",
    "data = np.expand_dims(data, axis=-1)\n",
    "data /= 255.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "id": "tZC0Li8QHn5e"
   },
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import LabelBinarizer\n",
    "le = LabelBinarizer()\n",
    "labels = le.fit_transform(labels)\n",
    "\n",
    "counts = labels.sum(axis=0)\n",
    "\n",
    "# account for skew in the labeled data\n",
    "classTotals = labels.sum(axis=0)\n",
    "classWeight = {}\n",
    "\n",
    "# loop over all classes and calculate the class weight\n",
    "for i in range(0, len(classTotals)):\n",
    "  classWeight[i] = classTotals.max() / classTotals[i]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "id": "CP1_DX40wQXQ"
   },
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "(trainX, testX, trainY, testY) = train_test_split(data,\n",
    "\tlabels, test_size=0.20, stratify=labels, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "id": "O6rSVozRHt2T"
   },
   "outputs": [],
   "source": [
    "# construct the image generator for data augmentation\n",
    "from keras.preprocessing.image import ImageDataGenerator\n",
    "aug = ImageDataGenerator(\n",
    "rotation_range=10,\n",
    "zoom_range=0.05,\n",
    "width_shift_range=0.1,\n",
    "height_shift_range=0.1,\n",
    "shear_range=0.15,\n",
    "horizontal_flip=False,\n",
    "fill_mode=\"nearest\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "id": "lIqs5O4uH0Pl"
   },
   "outputs": [],
   "source": [
    "from tensorflow import keras\n",
    "from tensorflow.keras import layers\n",
    "from keras.layers.convolutional import Conv2D\n",
    "from keras.layers.convolutional import AveragePooling2D\n",
    "from keras.layers.convolutional import MaxPooling2D\n",
    "from keras.layers.convolutional import ZeroPadding2D\n",
    "from keras.layers.core import Activation\n",
    "from keras.layers.core import Dense\n",
    "from keras.layers import Flatten\n",
    "from keras.layers import Input\n",
    "from keras.models import Model\n",
    "from keras.layers import add\n",
    "from keras.regularizers import l2\n",
    "from keras import backend as K\n",
    "\n",
    "class ResNet:\n",
    "\t@staticmethod\n",
    "\tdef residual_module(data, K, stride, chanDim, red=False,\n",
    "\t\treg=0.0001, bnEps=2e-5, bnMom=0.9):\n",
    "\t\t# the shortcut branch of the ResNet module should be\n",
    "\t\t# initialize as the input (identity) data\n",
    "\t\tshortcut = data\n",
    "\n",
    "\t\t# the first block of the ResNet module are the 1x1 CONVs\n",
    "\t\tbn1 = layers.BatchNormalization(axis=chanDim, epsilon=bnEps,\n",
    "\t\t\tmomentum=bnMom)(data)\n",
    "\t\tact1 = Activation(\"relu\")(bn1)\n",
    "\t\tconv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False,\n",
    "\t\t\tkernel_regularizer=l2(reg))(act1)\n",
    "\n",
    "\t\t# the second block of the ResNet module are the 3x3 CONVs\n",
    "\t\tbn2 = layers.BatchNormalization(axis=chanDim, epsilon=bnEps,\n",
    "\t\t\tmomentum=bnMom)(conv1)\n",
    "\t\tact2 = Activation(\"relu\")(bn2)\n",
    "\t\tconv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride,\n",
    "\t\t\tpadding=\"same\", use_bias=False,\n",
    "\t\t\tkernel_regularizer=l2(reg))(act2)\n",
    "\n",
    "\t\t# the third block of the ResNet module is another set of 1x1\n",
    "\t\t# CONVs\n",
    "\t\tbn3 = layers.BatchNormalization(axis=chanDim, epsilon=bnEps,\n",
    "\t\t\tmomentum=bnMom)(conv2)\n",
    "\t\tact3 = Activation(\"relu\")(bn3)\n",
    "\t\tconv3 = Conv2D(K, (1, 1), use_bias=False,\n",
    "\t\t\tkernel_regularizer=l2(reg))(act3)\n",
    "\n",
    "\t\t# if we are to reduce the spatial size, apply a CONV layer to\n",
    "\t\t# the shortcut\n",
    "\t\tif red:\n",
    "\t\t\tshortcut = Conv2D(K, (1, 1), strides=stride,\n",
    "\t\t\t\tuse_bias=False, kernel_regularizer=l2(reg))(act1)\n",
    "\n",
    "\t\t# add together the shortcut and the final CONV\n",
    "\t\tx = add([conv3, shortcut])\n",
    "\n",
    "\t\t# return the addition as the output of the ResNet module\n",
    "\t\treturn x\n",
    "\n",
    "\t@staticmethod\n",
    "\tdef build(width, height, depth, classes, stages, filters,\n",
    "\t\treg=0.0001, bnEps=2e-5, bnMom=0.9, dataset=\"cifar\"):\n",
    "\t\t# initialize the input shape to be \"channels last\" and the\n",
    "\t\t# channels dimension itself\n",
    "\t\tinputShape = (height, width, depth)\n",
    "\t\tchanDim = -1\n",
    "\n",
    "\t\t# if we are using \"channels first\", update the input shape\n",
    "\t\t# and channels dimension\n",
    "\t\tif K.image_data_format() == \"channels_first\":\n",
    "\t\t\tinputShape = (depth, height, width)\n",
    "\t\t\tchanDim = 1\n",
    "\n",
    "\t\t# set the input and apply BN\n",
    "\t\tinputs = Input(shape=inputShape)\n",
    "\t\tx = layers.BatchNormalization(axis=chanDim, epsilon=bnEps,\n",
    "\t\t\tmomentum=bnMom)(inputs)\n",
    "\n",
    "\t\t# check if we are utilizing the CIFAR dataset\n",
    "\t\tif dataset == \"cifar\":\n",
    "\t\t\t# apply a single CONV layer\n",
    "\t\t\tx = Conv2D(filters[0], (3, 3), use_bias=False,\n",
    "\t\t\t\tpadding=\"same\", kernel_regularizer=l2(reg))(x)\n",
    "\n",
    "\t\t# check to see if we are using the Tiny ImageNet dataset\n",
    "\t\telif dataset == \"tiny_imagenet\":\n",
    "\t\t\t# apply CONV => BN => ACT => POOL to reduce spatial size\n",
    "\t\t\tx = Conv2D(filters[0], (5, 5), use_bias=False,\n",
    "\t\t\t\tpadding=\"same\", kernel_regularizer=l2(reg))(x)\n",
    "\t\t\tx = layers.BatchNormalization(axis=chanDim, epsilon=bnEps,\n",
    "\t\t\t\tmomentum=bnMom)(x)\n",
    "\t\t\tx = Activation(\"relu\")(x)\n",
    "\t\t\tx = ZeroPadding2D((1, 1))(x)\n",
    "\t\t\tx = MaxPooling2D((3, 3), strides=(2, 2))(x)\n",
    "\n",
    "\t\t# loop over the number of stages\n",
    "\t\tfor i in range(0, len(stages)):\n",
    "\t\t\t# initialize the stride, then apply a residual module\n",
    "\t\t\t# used to reduce the spatial size of the input volume\n",
    "\t\t\tstride = (1, 1) if i == 0 else (2, 2)\n",
    "\t\t\tx = ResNet.residual_module(x, filters[i + 1], stride,\n",
    "\t\t\t\tchanDim, red=True, bnEps=bnEps, bnMom=bnMom)\n",
    "\n",
    "\t\t\t# loop over the number of layers in the stage\n",
    "\t\t\tfor j in range(0, stages[i] - 1):\n",
    "\t\t\t\t# apply a ResNet module\n",
    "\t\t\t\tx = ResNet.residual_module(x, filters[i + 1],\n",
    "\t\t\t\t\t(1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)\n",
    "\n",
    "\t\t# apply BN => ACT => POOL\n",
    "\t\tx = layers.BatchNormalization(axis=chanDim, epsilon=bnEps,\n",
    "\t\t\tmomentum=bnMom)(x)\n",
    "\t\tx = Activation(\"relu\")(x)\n",
    "\t\tx = AveragePooling2D((8, 8))(x)\n",
    "\n",
    "\t\t# softmax classifier\n",
    "\t\tx = Flatten()(x)\n",
    "\t\tx = Dense(classes, kernel_regularizer=l2(reg))(x)\n",
    "\t\tx = Activation(\"softmax\")(x)\n",
    "\n",
    "\t\t# create the model\n",
    "\t\tmodel = Model(inputs, x, name=\"resnet\")\n",
    "\n",
    "\t\t# return the constructed network architecture\n",
    "\t\treturn model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "id": "nXfO1yPEH_1o"
   },
   "outputs": [],
   "source": [
    "EPOCHS = 1\n",
    "INIT_LR = 1e-1\n",
    "BS = 512"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "s-ZuvoLbII0r",
    "outputId": "f4db6c6c-15d8-422f-8b04-25105f757ace"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "691/691 [==============================] - 2764s 4s/step - loss: 3.1050 - accuracy: 0.8020 - val_loss: 0.5086 - val_accuracy: 0.9079\n"
     ]
    }
   ],
   "source": [
    "from tensorflow.keras.optimizers.experimental import SGD\n",
    "opt = SGD(learning_rate=INIT_LR)\n",
    "\n",
    "model = ResNet.build(32, 32, 1, len(le.classes_), (3, 3, 3),\n",
    "(64, 64, 128, 256), reg=0.0005)\n",
    "\n",
    "model.compile(loss=\"categorical_crossentropy\", optimizer=opt,metrics=[\"accuracy\"])\n",
    "\n",
    "H = model.fit(\n",
    "aug.flow(trainX, trainY, batch_size=BS),\n",
    "validation_data=(testX, testY),\n",
    "steps_per_epoch=len(trainX) // BS,epochs=EPOCHS,\n",
    "class_weight=classWeight,\n",
    "verbose=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "oOIsHMMdIN_f",
    "outputId": "5262ed25-9f96-4c99-beac-6efa91653b62"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "173/173 [==============================] - 64s 364ms/step\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.20      0.81      0.32      1381\n",
      "           1       0.94      0.98      0.96      1575\n",
      "           2       0.91      0.78      0.84      1398\n",
      "           3       0.88      0.99      0.93      1428\n",
      "           4       0.88      0.90      0.89      1365\n",
      "           5       0.71      0.85      0.77      1263\n",
      "           6       0.95      0.94      0.95      1375\n",
      "           7       0.96      0.98      0.97      1459\n",
      "           8       0.94      0.94      0.94      1365\n",
      "           9       0.91      0.98      0.94      1392\n",
      "           A       0.99      0.98      0.98      2774\n",
      "           B       0.90      0.98      0.94      1734\n",
      "           C       0.97      0.98      0.97      4682\n",
      "           D       0.93      0.93      0.93      2027\n",
      "           E       0.98      0.96      0.97      2288\n",
      "           F       0.95      0.97      0.96       232\n",
      "           G       0.95      0.93      0.94      1152\n",
      "           H       0.92      0.96      0.94      1444\n",
      "           I       0.96      0.96      0.96       224\n",
      "           J       0.96      0.95      0.95      1699\n",
      "           K       0.98      0.93      0.95      1121\n",
      "           L       0.94      0.99      0.96      2317\n",
      "           M       0.99      0.98      0.99      2467\n",
      "           N       0.99      0.97      0.98      3802\n",
      "           O       0.96      0.60      0.74     11565\n",
      "           P       0.99      0.98      0.98      3868\n",
      "           Q       1.00      0.86      0.92      1162\n",
      "           R       0.98      0.96      0.97      2313\n",
      "           S       0.98      0.94      0.96      9684\n",
      "           T       0.99      0.98      0.99      4499\n",
      "           U       0.97      0.98      0.97      5802\n",
      "           V       0.99      0.98      0.98       836\n",
      "           W       0.99      0.96      0.98      2157\n",
      "           X       0.99      0.96      0.98      1254\n",
      "           Y       0.95      0.95      0.95      2172\n",
      "           Z       0.80      0.98      0.89      1215\n",
      "\n",
      "    accuracy                           0.91     88491\n",
      "   macro avg       0.92      0.94      0.92     88491\n",
      "weighted avg       0.95      0.91      0.92     88491\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "labelNames = \"0123456789\"\n",
    "\n",
    "labelNames += \"ABCDEFGHIJKLMNOPQRSTUVWXYZ\"\n",
    "\n",
    "labelNames = [l for l in labelNames]\n",
    "\n",
    "predictions = model.predict(testX, batch_size=BS)\n",
    "\n",
    "print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "id": "NMPrB-iFIWFH"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:HDF5 format does not save weights of `optimizer_experimental.Optimizer`, your optimizer will be recompiled at loading time.\n"
     ]
    }
   ],
   "source": [
    "model.save('Handwritten data/OCR_model.h5',save_format=\"h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "provenance": []
  },
  "gpuClass": "standard",
  "kernelspec": {
   "display_name": "Tensorflow (AI kit)",
   "language": "python",
   "name": "c009-intel_distribution_of_python_3_oneapi-beta05-tf"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.15"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
