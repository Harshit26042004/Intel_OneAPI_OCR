
<div align="center">
<img src="https://partner.utk.edu/wp-content/uploads/sites/77/2021/10/oneapi-lp-banner.png" >
</div>

# Intel OneAPI
<br>
oneAPI is an open, cross-industry, standards-based, unified, multiarchitecture, multi-vendor programming model that delivers a common developer experience across accelerator architectures – for faster application performance, more productivity, and greater innovation. The oneAPI initiative encourages collaboration on the oneAPI specification and compatible oneAPI implementations across the ecosystem.


One of the main advantages of using Intel OneAPI is its performance. By optimizing code for specific hardware, developers can achieve significant performance improvements compared to running code on a generic platform. This is particularly important for applications that require high-performance computing, such as machine learning and scientific simulations.


# Intel_OneAPI_OCR
<div align="center" style= width:"10px">
<img width="500px"  src="https://www.comidor.com/wp-content/uploads/2022/08/ocr-55-e1661521818617-1024x569.png" > 
</div>

 Digitization the Handwritten or Photo characters was a manual
process in before days. This was a time consuming thing and it is manually
expensive. Such handwritten or image characters are difficult to read by 
visual-impaired people. This Traditional method can be overcome with the 
help of this OCR (Optical Character Recognition) System.In this 
project,we propose a deep learning- based OCR system which will be more
accurate and efficient with the help of Intel OneAPI platform.
<br>
 The proposed system uses a convolutional neural
network(CNN)model to detect the characters from the image. The model
is trained on a large Dataset of labelled images using the OneAPI 
Platform.
<br>
 The Image is collected and organised for the classification of
characters and letters.The characters are detected with CNN models and 
trained with OneAPI platform in accurate and efficient manner , Where text
characters are extracted separately.
<br>
 Further , The trained model can predict the sample data images in
accurate and time-efficient way based on Intel machines , including CPU
and GPU's.With Add-On feature , we will also try to implement audio as an 
output , which is very useful for visual- impaired people.
<br>
 In conclusion, This OCR system using Intel OneAPI has
potential to digitalize the handwritten and predict it in a higher
accuracy and time-efficiency.

# Problem statement 
<br>
Design and develop an OCR system that can accurately recognize and convert printed or handwritten text from scanned images into editable digital text format, while maintaining the original format and layout of the document. The system should be able to handle a variety of fonts, styles, and  sizes, and have a high level of accuracy and speed in processing large volumes of documents.


<div align="center" style= width:"10px">
<img width="500px"  src="https://global-uploads.webflow.com/636bdbebfc681f083e923f81/63861eb172507319cde904f2_5f86968bee2e67ec3c263075_OCR.jpeg" > 
</div>

# Trained with Intel
<br>
With the help of Intel OneAPI we trained our model .And below is the snap of the code we developed in Intel devcloud with TENSORFLOW AI TOOLKIT
<img src="https://github.com/Harshit26042004/Intel_OneAPI_OCR/blob/main/Screenshot%202023-04-30%20192512.png" alt="sample-notebook">

# Deployment 
<br>
We deployed our model with the help of Streamlit as a web app.
Users can browse their files,Upload it ,And model will automatically detect and recognise the characters in that image.The sample of the deployment page is attached below
<img src="https://github.com/Harshit26042004/Intel_OneAPI_OCR/blob/main/Screenshot%202023-04-30%20192231.png" alt="deployment-img">
